#!/usr/bin/env python3
"""Music Inspiration Analyzer — analyse instrumental tracks and turn the analysis into
brainstorming material for musicians.

For each track: length, tempo, key timeline, mode & scale, chord progression (with Roman
numerals) and loops, song sections, groove, sound balance, energy arc — plus rule-based
ideas, an optional AI brainstorm, and MIDI files of the chord loop and variations.

Usage:
    python musicanalyze.py song1.mp3 song2.mp3 [--out ./out]
    python musicanalyze.py ./my_music_folder [--no-llm]
"""

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac"}
SR = 22050
HOP = 512

NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Key detection: Krumhansl-Schmuckler key profiles (how strongly each pitch class "belongs"
# to a key). We correlate the song's pitch-class energy (chroma) against all 24 keys.
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
KEY_NAMES = [f"{n} major" for n in MAJOR_NAMES] + [f"{n} minor" for n in MINOR_NAMES]
# Key index k: 0-11 = major with tonic k, 12-23 = minor with tonic k-12.

# Key timeline settings: estimate the key every STEP seconds from a WINDOW-second slice,
# then drop key "sections" shorter than MIN_SECTION (usually just a passing chord).
STEP, WINDOW, MIN_SECTION = 2.0, 10.0, 8.0

# Chord vocabulary for detection (intervals above the root). 7th chords get a small
# penalty so plain triads win unless the 7th is clearly there.
DETECT_QUALITIES = {"": ([0, 4, 7], 1.0), "m": ([0, 3, 7], 1.0),
                    "7": ([0, 4, 7, 10], 0.92), "m7": ([0, 3, 7, 10], 0.92)}
# Wider vocabulary accepted when reading chord names (e.g. from the AI) for MIDI export.
CHORD_INTERVALS = {"": [0, 4, 7], "maj": [0, 4, 7], "m": [0, 3, 7], "min": [0, 3, 7],
                   "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10], "maj7": [0, 4, 7, 11],
                   "dim": [0, 3, 6], "sus2": [0, 2, 7], "sus4": [0, 5, 7]}
ROMAN = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "google/gemma-4-31b-it"
ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------- small helpers

def _zscore(x, axis=-1):
    x = np.asarray(x, dtype=float)
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-9)


KEY_TEMPLATES = _zscore(np.array(
    [np.roll(MAJOR_PROFILE, k) for k in range(12)] + [np.roll(MINOR_PROFILE, k) for k in range(12)]
))  # shape (24, 12)


def _chord_templates():
    labels, rows = [], []
    for q, (ivs, w) in DETECT_QUALITIES.items():
        for root in range(12):
            v = np.zeros(12)
            v[[(root + i) % 12 for i in ivs]] = 1
            labels.append((root, q))
            rows.append(w * v / np.linalg.norm(v))
    return labels, np.array(rows)


CHORD_LABELS, CHORD_TEMPLATES = _chord_templates()


def key_scores(chroma_vec):
    """Correlation of a 12-bin chroma vector with each of the 24 keys."""
    return KEY_TEMPLATES @ _zscore(chroma_vec) / 12


def key_tonic(k):
    return k % 12


def key_is_minor(k):
    return k >= 12


def camelot(k):
    """Camelot wheel code (used by DJs/producers for harmonic mixing), e.g. B minor -> 10A."""
    base = (key_tonic(k) + 3) % 12 if key_is_minor(k) else key_tonic(k)
    return f"{(8 + 7 * base - 1) % 12 + 1}{'A' if key_is_minor(k) else 'B'}"


CAMELOT_TO_KEY = {camelot(k): k for k in range(24)}


def camelot_neighbours(k):
    code = camelot(k)
    num, letter = int(code[:-1]), code[-1]
    other = "B" if letter == "A" else "A"
    return {
        "relative": CAMELOT_TO_KEY[f"{num}{other}"],
        "up a fifth": CAMELOT_TO_KEY[f"{num % 12 + 1}{letter}"],
        "down a fifth": CAMELOT_TO_KEY[f"{(num - 2) % 12 + 1}{letter}"],
    }


LETTERS = "CDEFGAB"
LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DEGREE_OF_OFFSET = [0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6]  # semitones above tonic -> scale-degree index
ACCIDENTAL = {0: "", 1: "#", 2: "##", 11: "b", 10: "bb"}


def spell(pc, key):
    """Spell a pitch class correctly for a key: e.g. the leading tone of B minor is A#, not Bb."""
    tonic_letter = KEY_NAMES[key][0]
    letter = LETTERS[(LETTERS.index(tonic_letter) + DEGREE_OF_OFFSET[(pc - key_tonic(key)) % 12]) % 7]
    return letter + ACCIDENTAL.get((pc - LETTER_PC[letter]) % 12, "?")


def chord_name(root, q, key):
    return spell(root, key) + q


def roman(root, q, key):
    r = ROMAN[(root - key_tonic(key)) % 12]
    if q in ("m", "m7", "dim"):
        r = r.lower()
    return r + {"7": "7", "m7": "7", "maj7": "maj7", "dim": "°", "sus2": "sus2", "sus4": "sus4"}.get(q, "")


def parse_chord(name):
    """'F#m7' -> (6, 'm7', [0,3,7,10]); returns None if unparseable."""
    m = re.fullmatch(r"\s*([A-Ga-g])(##|bb|[b#]?)(maj7|maj|min|m7|m|7|dim|sus2|sus4)?\s*", str(name))
    if not m:
        return None
    root = (LETTER_PC[m.group(1).upper()] + m.group(2).count("#") - m.group(2).count("b")) % 12
    q = {"maj": "", "min": "m"}.get(m.group(3) or "", m.group(3) or "")
    return root, q, CHORD_INTERVALS[q]


def fmt_time(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def mode_of(labels):
    """Most common label; on a full tie keep the middle one."""
    c = Counter(labels).most_common()
    return labels[len(labels) // 2] if len(c) > 1 and c[0][1] == c[1][1] else c[0][0]


# ---------------------------------------------------------------- harmony

def key_timeline(chroma, duration):
    """Return [(start, end, key_index)] sections, smoothed and merged."""
    fps = SR / HOP
    times = np.arange(0, duration, STEP)
    keys = []
    for t in times:
        a = int(max(0, t + STEP / 2 - WINDOW / 2) * fps)
        b = int(min(duration, t + STEP / 2 + WINDOW / 2) * fps)
        keys.append(int(np.argmax(key_scores(chroma[:, a:max(b, a + 1)].mean(axis=1)))))
    smooth = [mode_of(keys[max(0, i - 2): i + 3]) for i in range(len(keys))]
    spans = [(t, min(t + STEP, duration), k) for t, k in zip(times, smooth)]
    return absorb_short(merge_runs(spans), MIN_SECTION)


def merge_runs(spans):
    out = []
    for s, e, k in spans:
        if out and out[-1][2] == k:
            out[-1] = (out[-1][0], e, k)
        else:
            out.append((s, e, k))
    return out


def absorb_short(spans, min_len):
    """Merge spans shorter than min_len into their longer neighbour."""
    spans = list(spans)
    while len(spans) > 1:
        short = [i for i, (s, e, _) in enumerate(spans) if e - s < min_len]
        if not short:
            break
        i = short[0]
        nbs = [spans[j] for j in (i - 1, i + 1) if 0 <= j < len(spans)]
        nb = max(nbs, key=lambda x: x[1] - x[0])
        spans[i] = (spans[i][0], spans[i][1], nb[2])
        spans = merge_runs(spans)
    return spans


def key_at(sections, t):
    for s, e, k in sections:
        if s <= t < e:
            return k
    return sections[-1][2]


def mode_and_scale(chroma, frames, key):
    """Guess the mode from characteristic scale degrees, and list the notes used most."""
    pc = chroma[:, frames].mean(axis=1)
    pc = pc / (pc.max() + 1e-9)
    t = key_tonic(key)
    rel = np.roll(pc, -t)  # rel[i] = strength of the note i semitones above the tonic
    note = lambda off: spell((t + off) % 12, key)
    if key_is_minor(key):
        if rel[11] > rel[10] * 1.15:
            mode, color = "Harmonic minor", 11
        elif rel[9] > rel[8] * 1.15:
            mode, color = "Dorian", 9
        elif rel[1] > rel[2] * 1.1:
            mode, color = "Phrygian", 1
        else:
            mode, color = "Aeolian (natural minor)", 8
    else:
        if rel[10] > rel[11] * 1.15:
            mode, color = "Mixolydian", 10
        elif rel[6] > rel[5] * 1.15:
            mode, color = "Lydian", 6
        else:
            mode, color = "Ionian (major)", 4
    top7 = sorted(np.argsort(rel)[::-1][:7])
    return {
        "mode": mode,
        "color_note": note(color),
        "color_degree": ROMAN[color] if color else "I",
        "scale_notes": [note(i) for i in top7],
        "note_strength": {note(i): round(float(rel[i]), 2) for i in range(12)},
    }


def detect_chords(chroma, rms_db, bounds, btimes):
    """One chord per beat, smoothed and merged. Returns list of dicts (named later, per key)."""
    import librosa
    sync = librosa.util.sync(chroma, bounds, aggregate=np.median, pad=False)
    loud = librosa.util.sync(rms_db[None, :], bounds, aggregate=np.mean, pad=False)[0]
    labels = []
    for i in range(sync.shape[1]):
        if loud[i] < rms_db.max() - 35:
            labels.append(-1)  # silence / no chord
            continue
        v = sync[:, i] / (np.linalg.norm(sync[:, i]) + 1e-9)
        labels.append(int(np.argmax(CHORD_TEMPLATES @ v)))
    labels = [mode_of(labels[max(0, i - 1): i + 2]) for i in range(len(labels))]
    spans = merge_runs([(btimes[i], btimes[i + 1], lab) for i, lab in enumerate(labels)])
    beat_len = float(np.median(np.diff(btimes))) if len(btimes) > 1 else 0.5
    spans = absorb_short(spans, 1.5 * beat_len)  # a chord must last ~2 beats
    chords = []
    for s, e, lab in spans:
        if lab < 0:
            continue
        root, q = CHORD_LABELS[lab]
        chords.append({"start": round(float(s), 2), "end": round(float(e), 2),
                       "beats": max(1, int(round((e - s) / beat_len))), "root": root, "q": q})
    return chords


def find_loop(names, key):
    """Find the repeating chord cycle (smallest period that repeats), rotated to start on the
    home chord if it's in the loop."""
    if len(names) < 2:
        return None
    if len(set(names)) == 1:
        return {"chords": names[:1], "repeats": len(names)}
    for p in range(2, 9):
        if len(names) < 2 * p:
            break
        match = np.mean([names[i] == names[i + p] for i in range(len(names) - p)])
        if match >= 0.6:
            loop = list(Counter(tuple(names[i:i + p]) for i in range(len(names) - p + 1)).most_common(1)[0][0])
            romans = [roman(*parse_chord(c)[:2], key) for c in loop]
            home = next((i for i, r in enumerate(romans) if r.upper().startswith("I") and
                         r.upper().rstrip("7") == "I"), 0)
            return {"chords": loop[home:] + loop[:home], "roman": romans[home:] + romans[:home],
                    "repeats": round(len(names) / p, 1)}
    return None


# ---------------------------------------------------------------- structure, rhythm, sound

def detect_structure(chroma, mfcc, rms_db, bounds, btimes, duration, key_changes=()):
    """Split the track into sections and label repeats with letters (A, B, A...).
    Boundaries within 10 s of a key change are snapped to it (key changes are more precise)."""
    import librosa
    feat = np.vstack([_zscore(librosa.util.sync(chroma, bounds, pad=False), axis=1),
                      _zscore(librosa.util.sync(mfcc, bounds, pad=False), axis=1)])
    n = feat.shape[1]
    k = int(np.clip(round(duration / 30) + 1, 2, 8))
    if n < 4 * k:
        starts = [0]
    else:
        starts = sorted(set(int(x) for x in librosa.segment.agglomerative(feat, k)))
    spans = [(btimes[a], btimes[b] if b < len(btimes) else duration, i)
             for i, (a, b) in enumerate(zip(starts, starts[1:] + [n]))]
    spans = [(s, e) for s, e, _ in absorb_short(spans, 8.0)]  # fold tiny sections into neighbours
    cuts = [e for _, e in spans[:-1]]
    for kc in key_changes:
        if cuts:
            i = int(np.argmin([abs(c - kc) for c in cuts]))
            if abs(cuts[i] - kc) <= 10:
                cuts[i] = kc
    cuts = sorted(c for c in set(cuts) if 4 < c < duration - 4)
    spans = list(zip([0.0] + cuts, cuts + [duration]))

    fps = SR / HOP
    means, energies = [], []
    for s, e in spans:
        a, b = int(s * fps), max(int(e * fps), int(s * fps) + 1)
        means.append(np.concatenate([chroma[:, a:b].mean(axis=1), _zscore(mfcc[:, a:b].mean(axis=1))]))
        energies.append(float(np.mean(rms_db[a:b])))

    # Same letter if a section is much closer to an earlier one than sections typically are.
    letters = []
    if len(means) > 2:
        M = _zscore(np.array(means), axis=0)
        d = np.linalg.norm(M[:, None] - M[None], axis=-1)
        thr = 0.5 * np.median(d[np.triu_indices(len(M), 1)])
    for i in range(len(spans)):
        lab = None
        if len(means) > 2:
            prev = [j for j in range(i) if d[i, j] < thr]
            if prev:
                lab = letters[min(prev, key=lambda j: d[i, j])]
        letters.append(lab or "ABCDEFGHIJ"[len(set(letters))])

    avg = float(np.mean(energies))
    out = []
    for (s, e), lab, en in zip(spans, letters, energies):
        level = "high" if en > avg + 2 else "low" if en < avg - 3 else "medium"
        out.append({"label": lab, "start": round(float(s), 1), "end": round(float(e), 1), "energy": level,
                    "energy_db": round(en, 1)})
    return out


def rhythm_features(y, y_perc, y_harm, beat_times, clear_beat, duration, sr):
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, units="time")
    out = {"onsets_per_sec": round(len(onsets) / duration, 2)}
    out["busyness"] = ("sparse" if out["onsets_per_sec"] < 1.5 else
                       "moderate" if out["onsets_per_sec"] < 4 else "busy")
    pe, he = float(np.sum(y_perc ** 2)), float(np.sum(y_harm ** 2))
    out["percussive_pct"] = round(100 * pe / (pe + he + 1e-9))

    if clear_beat and len(beat_times) > 4:
        local = librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=HOP, aggregate=None)
        out["tempo_variation_bpm"] = round(float(np.std(local)), 1)
        out["tempo_feel"] = ("steady (likely to a click/grid)" if out["tempo_variation_bpm"] < 2 else
                             "slightly loose" if out["tempo_variation_bpm"] < 6 else "free / rubato")
        # Only judge syncopation on the stronger half of onsets (weak ones are often note decays).
        of = librosa.time_to_frames(onsets, sr=sr, hop_length=HOP)
        strong = onsets[env[of] >= np.median(env[of])] if len(of) else onsets
        phases = []
        for t in strong:
            j = np.searchsorted(beat_times, t) - 1
            if 0 <= j < len(beat_times) - 1:
                phases.append((t - beat_times[j]) / (beat_times[j + 1] - beat_times[j]))
        ph = np.array(phases)
        if len(ph):
            sync = np.mean(((ph > 0.15) & (ph < 0.4)) | ((ph > 0.6) & (ph < 0.85)))
            out["syncopation_pct"] = round(100 * float(sync))
            out["groove"] = ("mostly on the beat" if sync < 0.15 else
                             "some syncopation" if sync < 0.35 else "heavily syncopated")
    else:
        out["tempo_feel"] = "no clear pulse (ambient / free time)"
    return out


def sound_features(y, sr, rms_db, duration):
    import librosa
    S = np.abs(librosa.stft(y, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    band_edges = {"sub (<60 Hz)": (0, 60), "bass (60–250)": (60, 250), "low-mid (250–2k)": (250, 2000),
                  "high-mid (2k–6k)": (2000, 6000), "air (>6k)": (6000, sr / 2 + 1)}
    total = S.sum() + 1e-9
    bands = {name: round(100 * float(S[(freqs >= lo) & (freqs < hi)].sum() / total), 1)
             for name, (lo, hi) in band_edges.items()}
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))

    # Energy arc, one value per second.
    fps = sr / HOP
    per_sec = np.array([rms_db[int(i * fps):int((i + 1) * fps)].mean() for i in range(int(duration))])
    rise_t, rise_db, quiet_t = None, 0.0, None
    if len(per_sec) >= 12:
        rises = [(per_sec[t:t + 2].mean() - per_sec[t - 4:t].mean(), t) for t in range(4, len(per_sec) - 2)]
        rise_db, rise_t = max(rises)
        inner = per_sec[3:-3]
        win = np.convolve(inner, np.ones(4) / 4, mode="valid")
        quiet_t = int(np.argmin(win)) + 3
    curve = np.array_split(10 ** (rms_db / 20), 24)
    curve = np.array([c.mean() for c in curve])
    audible = rms_db[rms_db > rms_db.max() - 60]
    return {
        "frequency_balance_pct": bands,
        "brightness_hz": round(centroid),
        "tone": "dark/warm" if centroid < 1500 else "balanced" if centroid < 3000 else "bright",
        "texture": "tonal / clean" if flatness < 0.02 else "mixed" if flatness < 0.1 else "noisy / textural",
        "avg_loudness_db": round(float(np.mean(audible)), 1),
        "dynamic_range_db": round(float(np.percentile(audible, 95) - np.percentile(audible, 5)), 1),
        "energy_curve": [round(float(x), 2) for x in curve / (curve.max() + 1e-9)],
        "energy_flat": bool(np.std(curve / (curve.max() + 1e-9)) < 0.08),
        "biggest_build": {"time": fmt_time(rise_t), "db": round(float(rise_db), 1)} if rise_t and rise_db > 3 else None,
        "quietest_moment": fmt_time(quiet_t) if quiet_t is not None else None,
    }


# ---------------------------------------------------------------- main analysis

def load_audio(path):
    """Load mono audio at SR. libsndfile handles mp3/wav/ogg/flac; m4a/aac go through PyAV (ffmpeg)."""
    import librosa
    try:
        return librosa.load(str(path), sr=SR, mono=True)
    except Exception:
        import av
        with av.open(str(path)) as container:
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
            chunks = []
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
        if not chunks:
            raise ValueError("no audio found in file")
        return np.concatenate(chunks).astype(np.float32), SR


def analyze(path):
    import librosa

    y, sr = load_audio(path)
    if len(y) < 2 * SR:
        raise ValueError("audio is shorter than 2 seconds")
    duration = len(y) / sr
    y_harm, y_perc = librosa.effects.hpss(y)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    clear_beat = len(beat_frames) >= 8
    if not clear_beat:  # ambient / rubato: fall back to half-second steps
        beat_frames = librosa.time_to_frames(np.arange(0.5, duration, 0.5), sr=sr, hop_length=HOP)

    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    rms_db = 20 * np.log10(librosa.feature.rms(y=y, hop_length=HOP)[0] + 1e-9)
    n = min(chroma.shape[1], mfcc.shape[1], len(rms_db))
    chroma, mfcc, rms_db = chroma[:, :n], mfcc[:, :n], rms_db[:n]
    bounds = librosa.util.fix_frames(beat_frames[beat_frames < n], x_min=0, x_max=n)
    btimes = librosa.frames_to_time(bounds, sr=sr, hop_length=HOP)
    btimes[-1] = duration
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)

    # --- key & mode
    ksec = key_timeline(chroma, duration)
    time_in_key = Counter()
    for s, e, k in ksec:
        time_in_key[k] += e - s
    main_key = time_in_key.most_common(1)[0][0]
    fps = sr / HOP
    frames = np.concatenate([np.arange(int(s * fps), min(int(e * fps), n)) for s, e, k in ksec if k == main_key])
    scores = key_scores(chroma[:, frames].mean(axis=1))
    runner_up = next(int(i) for i in np.argsort(scores)[::-1] if i != main_key)
    scale = mode_and_scale(chroma, frames, main_key)

    # --- chords
    chords = detect_chords(chroma, rms_db, bounds, btimes)
    for c in chords:
        k = key_at(ksec, (c["start"] + c["end"]) / 2)
        c["name"], c["roman"] = chord_name(c["root"], c["q"], k), roman(c["root"], c["q"], k)
    chord_time = Counter()
    for c in chords:
        chord_time[c["name"]] += c["end"] - c["start"]

    # --- chord loops, searched within each key section (loops live inside a key)
    loops = []
    for s, e, k in ksec:
        inside = [c["name"] for c in chords if s <= (c["start"] + c["end"]) / 2 < e]
        loop = find_loop(inside, k)
        if not loop:
            continue
        same = next((l for l in loops if _same_cycle(loop["chords"], l["chords"])), None)
        if same:
            same["where"].append(f"{fmt_time(s)}–{fmt_time(e)}")
        else:
            loops.append({**loop, "key": KEY_NAMES[k], "where": [f"{fmt_time(s)}–{fmt_time(e)}"]})
    for l in loops:
        l.setdefault("roman", [roman(*parse_chord(c)[:2], KEY_NAMES.index(l["key"])) for c in l["chords"]])

    # --- structure (with each section's key and main chords)
    sections = detect_structure(chroma, mfcc, rms_db, bounds, btimes, duration, [e for _, e, _ in ksec[:-1]])
    for sec in sections:
        top = Counter()
        for c in chords:
            if sec["start"] <= (c["start"] + c["end"]) / 2 < sec["end"]:
                top[c["name"]] += c["end"] - c["start"]
        sec["key"] = KEY_NAMES[key_at(ksec, (sec["start"] + sec["end"]) / 2)]
        sec["main_chords"] = [name for name, _ in top.most_common(4)]

    result = {
        "file": Path(path).name,
        "duration_sec": round(duration, 1),
        "duration": fmt_time(duration),
        "tempo_bpm": round(tempo, 1) if clear_beat else None,
        "key": KEY_NAMES[main_key],
        "key_index": int(main_key),
        "camelot": camelot(main_key),
        "key_confidence": round(float(scores[main_key]), 2),
        "runner_up_key": KEY_NAMES[runner_up],
        "key_sections": [{"start": fmt_time(s), "end": fmt_time(e), "key": KEY_NAMES[k]} for s, e, k in ksec],
        "time_in_key": {KEY_NAMES[k]: {"seconds": round(v, 1), "percent": round(100 * v / duration)}
                        for k, v in time_in_key.most_common()},
        "key_changes": len(ksec) - 1,
        **scale,
        "chords_used": [{"chord": name, "seconds": round(t, 1)} for name, t in chord_time.most_common(8)],
        "harmonic_rhythm_beats": round(float(np.mean([c["beats"] for c in chords])), 1) if chords else None,
        "chord_timeline": [{"time": fmt_time(c["start"]), "chord": c["name"], "roman": c["roman"],
                            "beats": c["beats"]} for c in chords],
        "loops": loops,
        "sections": sections,
        "rhythm": rhythm_features(y, y_perc, y_harm, beat_times, clear_beat, duration, sr),
        "sound": sound_features(y, sr, rms_db, duration),
    }
    result["ideas"] = rule_based_ideas(result)
    return result


def _same_cycle(a, b):
    """True if chord list b is a rotation of chord list a."""
    return len(a) == len(b) and any(a == b[i:] + b[:i] for i in range(len(b)))


# ---------------------------------------------------------------- rule-based ideas

def rule_based_ideas(r):
    k = r["key_index"]
    t = key_tonic(k)
    note = lambda off: spell((t + off) % 12, k)
    ideas = []

    # Mode colour
    tips = {
        "Harmonic minor": f"The raised 7th ({note(11)}) is this track's signature — it pulls hard to {note(0)}. "
                          f"Land melody phrases on {note(11)}→{note(0)}, or try the exotic {note(8)}→{note(11)} "
                          f"step in a lead line.",
        "Dorian": f"The major 6th ({note(9)}) is the Dorian colour. Feature it in a melody, or use a major IV chord "
                  f"({note(5)}) for that soulful/funky lift.",
        "Phrygian": f"The b2 ({note(1)}) gives Phrygian tension — a {note(1)} major chord resolving to {note(0)}m "
                    f"sounds cinematic/Spanish.",
        "Aeolian (natural minor)": f"Natural minor: the b6 ({note(8)}) carries the melancholy. A bVI chord "
                                   f"({note(8)}) → bVII ({note(10)}) → i is a classic epic lift.",
        "Ionian (major)": f"Bright major. Borrow the minor iv ({note(5)}m) from {note(0)} minor for a bittersweet "
                          f"turn before returning to {note(0)}.",
        "Mixolydian": f"The b7 ({note(10)}) gives a Mixolydian, bluesy-rock feel. A bVII ({note(10)}) → I "
                      f"cadence will feel natural here.",
        "Lydian": f"The #4 ({note(6)}) is the Lydian colour — a II major chord ({note(2)}) over a {note(0)} "
                  f"bass gives a dreamy, floating lift.",
    }
    ideas.append({"area": "melody/harmony", "idea": tips[r["mode"]]})

    # Borrowed chord from the parallel key
    if key_is_minor(k):
        ideas.append({"area": "harmony", "idea": f"Borrow from {note(0)} major: swap the iv ({note(5)}m) for a "
                      f"major IV ({note(5)}), or end a section on a major I ({note(0)}) — a 'Picardy' surprise."})
    else:
        ideas.append({"area": "harmony", "idea": f"Borrow from {note(0)} minor: try bVI ({note(8)}) or bVII "
                      f"({note(10)}) before the last I chord of a section."})

    # Key-change / set-list targets
    nb = camelot_neighbours(k)
    lift = (k + 2) % 12 + (12 if key_is_minor(k) else 0)
    ideas.append({"area": "structure", "idea":
                  f"Smooth key-change targets (or the next song in a set): {KEY_NAMES[nb['relative']]} (relative), "
                  f"{KEY_NAMES[nb['up a fifth']]} and {KEY_NAMES[nb['down a fifth']]} (Camelot neighbours). "
                  f"For a dramatic final-chorus lift, jump up a whole step to {KEY_NAMES[lift]}."})

    # Tempo variants
    if r["tempo_bpm"]:
        bpm = r["tempo_bpm"]
        variants = []
        if bpm >= 85:
            variants.append(f"a half-time groove (drums feel like ~{bpm / 2:.0f} BPM) for a heavier, spacious section")
        if bpm < 100:
            variants.append(f"a double-time feel at ~{bpm * 2:.0f} BPM for an energetic remix")
        ideas.append({"area": "remix", "idea": f"Same chords, new feel: try {' or '.join(variants)}."})
    else:
        ideas.append({"area": "rhythm", "idea": "There's no clear pulse — try adding a slow, soft pulse (e.g. 70 BPM) "
                      "under one section to create contrast with the free-time parts."})

    # Energy arc
    s = r["sound"]
    if s["energy_flat"]:
        mid = fmt_time(r["duration_sec"] / 2)
        ideas.append({"area": "arrangement", "idea": f"The energy stays almost flat. Create a breakdown around {mid} "
                      f"(drop the drums and bass for 4–8 bars) and bring everything back for a bigger return."})
    elif s["biggest_build"]:
        ideas.append({"area": "arrangement", "idea": f"The strongest moment is the build into {s['biggest_build']['time']} "
                      f"(+{s['biggest_build']['db']} dB). Set it up harder: a riser, a drum fill, or a bar of silence "
                      f"right before it."})
    if s["quietest_moment"]:
        ideas.append({"area": "arrangement", "idea": f"The quietest stretch is around {s['quietest_moment']} — a natural "
                      f"spot for a solo instrument, a field recording, or a new melodic motif."})

    # Groove
    g = r["rhythm"].get("groove")
    if g == "mostly on the beat":
        ideas.append({"area": "rhythm", "idea": "The groove sits mostly on the beat. Try anticipating chord changes "
                      "by an 8th note, or add a syncopated percussion layer (shaker/rim) to add swing."})
    elif g == "heavily syncopated":
        ideas.append({"area": "rhythm", "idea": "The groove is already very syncopated — a section with straight, "
                      "on-beat hits would create contrast and make the syncopation hit harder when it returns."})

    # Sound balance
    bands = s["frequency_balance_pct"]
    if bands["air (>6k)"] < 2:
        ideas.append({"area": "sound", "idea": "Very little high-end 'air' — a bright layer (hi-hats, shimmer reverb, "
                      "airy pad, or a high counter-melody) would open up the mix."})
    if bands["sub (<60 Hz)"] + bands["bass (60–250)"] < 15:
        ideas.append({"area": "sound", "idea": "The low end is light — a sub-bass or a bass line following the chord "
                      "roots would add weight."})
    return ideas


# ---------------------------------------------------------------- MIDI export

def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def write_midi(path, chords, bpm=90, beats_per_chord=4, repeats=2):
    """Write a simple 1-track MIDI file: bass root + mid-range chord, one chord per bar."""
    tpq = 480
    events = []  # (tick, on/off, note)
    tick = 0
    for _ in range(repeats):
        for root, ivs in chords:
            base = 48 + root if root >= 5 else 60 + root  # keep chords around middle C
            notes = [36 + root] + [base + i for i in ivs]
            length = beats_per_chord * tpq
            events += [(tick, 1, n) for n in notes] + [(tick + length - 10, 0, n) for n in notes]
            tick += length
    events.sort(key=lambda e: (e[0], e[1]))
    us = int(60_000_000 / max(30, min(300, bpm or 90)))
    data = b"\x00\xff\x51\x03" + us.to_bytes(3, "big")
    last = 0
    for t, on, n in events:
        data += _vlq(t - last) + bytes([0x90 if on else 0x80, n, 90 if on else 0])
        last = t
    data += b"\x00\xff\x2f\x00"
    Path(path).write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq) +
                           b"MTrk" + struct.pack(">I", len(data)) + data)


def export_midis(r, out_dir):
    """Write MIDI for each detected loop, a relative-substitution variation, and AI progressions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(r["file"]).stem
    bpm = r["tempo_bpm"] or 90
    files = []

    def save(label, names):
        parsed = [(p[0], p[2]) for p in map(parse_chord, names) if p]
        if len(parsed) < 2:
            return
        fname = f"{stem}_{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')}.mid"
        write_midi(out_dir / fname, parsed, bpm)
        files.append({"file": fname, "label": label, "chords": names})

    for i, loop in enumerate(r["loops"], 1):
        if len(loop["chords"]) < 2:
            continue
        save(f"detected loop {i}", loop["chords"])
        # Variation: swap the first chord for its relative (major <-> minor a 3rd away).
        root, q, _ = parse_chord(loop["chords"][0])
        k = KEY_NAMES.index(loop["key"])
        sub = chord_name((root - 3) % 12, "m", k) if q not in ("m", "m7") else chord_name((root + 3) % 12, "", k)
        save(f"loop {i} variation relative sub", [sub] + loop["chords"][1:])
    for i, alt in enumerate((r.get("ai") or {}).get("alt_progressions", []), 1):
        save(f"ai {i} {alt.get('name', '')}", alt.get("chords", []))
    return files


# ---------------------------------------------------------------- optional AI brainstorm

BRAINSTORM_PROMPT = """You are a creative music producer helping a musician brainstorm ideas from an
instrumental track they made. Below is automatic audio analysis of the track (JSON).

Return ONLY a JSON object, no other text:
{{
  "vibe": "<2-3 sentences on the feel the data suggests; phrase interpretations as such, e.g. 'the slow tempo and dark tone suggest...'>",
  "ideas": [{{"area": "<harmony|melody|rhythm|arrangement|sound|structure|remix>", "idea": "<one concrete, actionable suggestion that references a specific time, section, chord, note or number from the data>"}}],
  "alt_progressions": [{{"name": "<short name>", "chords": ["<chord>", "..."], "why": "<one sentence>"}}],
  "title_ideas": ["<title>", "<title>", "<title>"]
}}

RULES:
- Give 6-8 ideas, each different and specific. No generic advice like "add more layers" or "experiment".
- Do not repeat these ideas the musician already has: {existing}
- Give 2-3 alt_progressions in the track's key ({key}), 3-6 chords each, as alternatives or a contrasting
  section for the detected loop. Chord names: letter + optional b/# + optional quality from m, 7, m7, maj7, dim, sus2, sus4
  (e.g. "Bm", "F#7", "Gmaj7").
- Facts about the track must come from the data. Never claim instruments, genre or anything the data can't show.
  Automatic analysis can be wrong; if key_confidence < 0.6 treat the key as uncertain.

ANALYSIS:
{data}
"""


def make_llm():
    """Return (fn(prompt)->text, label) for whichever API key is set, or (None, None)."""
    if os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        import openai
        if os.environ.get("NVIDIA_API_KEY"):
            client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=os.environ["NVIDIA_API_KEY"])
            model, label = os.environ.get("MUSIC_MODEL", NVIDIA_MODEL), "nvidia"
        else:
            client = openai.OpenAI()
            model, label = os.environ.get("MUSIC_MODEL", OPENAI_MODEL), "openai"

        def call(prompt):
            r = client.chat.completions.create(model=model, temperature=0.8, max_tokens=3000,
                                               messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        return call, f"{label}/{model}"

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        model = os.environ.get("MUSIC_MODEL", ANTHROPIC_MODEL)

        def call(prompt):
            m = client.messages.create(model=model, max_tokens=3000, temperature=0.8,
                                       messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in m.content if b.type == "text")
        return call, f"anthropic/{model}"

    return None, None


def ai_brainstorm(r, llm):
    data = {k: v for k, v in r.items() if k not in ("chord_timeline", "ideas", "note_strength", "key_index")}
    data["chord_timeline_start"] = r["chord_timeline"][:24]
    prompt = BRAINSTORM_PROMPT.format(existing=json.dumps([i["idea"][:80] for i in r["ideas"]]),
                                      key=r["key"], data=json.dumps(data, ensure_ascii=False))
    try:
        raw = llm(prompt)
    except Exception as e:
        return {"error": f"AI brainstorm unavailable: {str(e)[:150]}"}
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw)
    a, b = raw.find("{"), raw.rfind("}")
    try:
        obj = json.loads(raw[a:b + 1])
        assert isinstance(obj, dict)
    except Exception:
        return {"error": "AI reply could not be parsed", "raw": raw.strip()[:1500]}
    return {
        "vibe": _clean(obj.get("vibe", "")),
        "ideas": [{"area": _clean(i.get("area", "idea")), "idea": _clean(i["idea"])}
                  for i in obj.get("ideas", []) if isinstance(i, dict) and i.get("idea")],
        "alt_progressions": [{"name": _clean(p.get("name", "")), "why": _clean(p.get("why", "")),
                              "chords": [_clean(c) for c in p["chords"]]}
                             for p in obj.get("alt_progressions", [])
                             if isinstance(p, dict) and isinstance(p.get("chords"), list)],
        "title_ideas": [_clean(t) for t in obj.get("title_ideas", [])][:5],
    }


def _clean(text):
    """AI text ends up in markdown/HTML reports: neutralise HTML tags and markdown links."""
    text = str(text).strip().replace("<", "‹").replace(">", "›").replace("`", "'")
    return text.replace("](", "] (")


# ---------------------------------------------------------------- cross-track

def pairings(results):
    """Which tracks blend well (key-compatible on the Camelot wheel and similar tempo)."""
    pairs = []
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            ca, cb = camelot(a["key_index"]), camelot(b["key_index"])
            na, nb = int(ca[:-1]), int(cb[:-1])
            if ca == cb:
                rel = "same key"
            elif na == nb:
                rel = "relative keys"
            elif ca[-1] == cb[-1] and (na - nb) % 12 in (1, 11):
                rel = "neighbouring keys"
            else:
                continue
            tempo = ""
            if a["tempo_bpm"] and b["tempo_bpm"]:
                ratio = b["tempo_bpm"] / a["tempo_bpm"]
                if abs(ratio - 1) < 0.06:
                    tempo = "similar tempo"
                elif abs(ratio - 2) < 0.12 or abs(ratio - 0.5) < 0.03:
                    tempo = "half/double tempo"
            pairs.append({"a": a["file"], "b": b["file"], "keys": f"{a['key']} ({ca}) ↔ {b['key']} ({cb})",
                          "relation": rel, "tempo": tempo or "different tempo"})
    return pairs


# ---------------------------------------------------------------- report

BARS = "▁▂▃▄▅▆▇█"


def sparkline(values):
    return "".join(BARS[min(len(BARS) - 1, int(v * (len(BARS) - 1) + 0.5))] for v in values)


def bar(pct, width=20):
    return "█" * int(round(pct / 100 * width))


def track_report(r):
    L = [f"## 🎵 {r['file']}", ""]
    if "error" in r:
        return L + [f"⚠️ Could not analyse this file: {r['error']}", ""]
    rh, s = r["rhythm"], r["sound"]
    tempo = f"{r['tempo_bpm']:.0f} BPM" if r["tempo_bpm"] else "no clear beat"
    L += [f"**{r['duration']}** · **{tempo}** · **{r['key']}** ({r['camelot']}) · **{r['mode']}** · "
          f"{rh['tempo_feel']} · {s['tone']}", ""]

    # Harmony
    L += ["### Harmony", "",
          f"- **Key:** {r['key']} (confidence {r['key_confidence']}; runner-up {r['runner_up_key']})",
          f"- **Mode:** {r['mode']} — colour note **{r['color_note']}**",
          f"- **Notes used most (scale):** {' '.join(r['scale_notes'])}",
          f"- **Chords used most:** " + ", ".join(f"{c['chord']} ({fmt_time(c['seconds'])})" for c in r["chords_used"]),
          ]
    if r["harmonic_rhythm_beats"]:
        L.append(f"- **Harmonic rhythm:** a chord change every ~{r['harmonic_rhythm_beats']} beats")
    L += ["", "**Time in each key**", "", "| Key | Time | Share |", "|-----|------|-------|"]
    for k, v in r["time_in_key"].items():
        L.append(f"| {k} | {fmt_time(v['seconds'])} | {v['percent']}% |")
    if r["key_changes"]:
        L += ["", "**Key timeline:** " + " → ".join(f"{s_['key']} ({s_['start']}–{s_['end']})" for s_ in r["key_sections"])]
    if r["loops"]:
        L += ["", "**Chord loops**", ""]
        for lp in r["loops"]:
            L.append(f"- `{' – '.join(lp['chords'])}`  ({' – '.join(lp['roman'])} in {lp['key']}) "
                     f"· {', '.join(lp['where'])} · ×{lp['repeats']}")
    if r["chord_timeline"]:
        L += ["", "<details><summary>Full chord timeline</summary>", "",
              " · ".join(f"{c['time']} {c['chord']}" for c in r["chord_timeline"]), "", "</details>"]

    # Structure
    L += ["", "### Structure", "", "| Section | Time | Length | Energy | Key | Main chords |",
          "|---------|------|--------|--------|-----|-------------|"]
    for sec in r["sections"]:
        L.append(f"| **{sec['label']}** | {fmt_time(sec['start'])}–{fmt_time(sec['end'])} | "
                 f"{fmt_time(sec['end'] - sec['start'])} | {sec['energy']} | {sec['key']} | "
                 f"{', '.join(sec['main_chords']) or '—'} |")
    L += ["", f"Form: **{' '.join(sec['label'] for sec in r['sections'])}**"]

    # Rhythm & sound
    L += ["", "### Groove & sound", "",
          f"- **Pulse:** {rh['tempo_feel']}" + (f" (±{rh['tempo_variation_bpm']} BPM)" if "tempo_variation_bpm" in rh else ""),
          f"- **Rhythm:** {rh['busyness']} ({rh['onsets_per_sec']} notes/hits per sec)"
          + (f", {rh['groove']} ({rh['syncopation_pct']}% off-beat)" if "groove" in rh else ""),
          f"- **Percussive vs tonal:** {rh['percussive_pct']}% percussive energy",
          f"- **Tone:** {s['tone']} ({s['brightness_hz']} Hz centroid), {s['texture']}",
          f"- **Loudness:** avg {s['avg_loudness_db']} dBFS, dynamic range {s['dynamic_range_db']} dB",
          f"- **Energy arc:** `{sparkline(s['energy_curve'])}`"
          + (f" — biggest build into **{s['biggest_build']['time']}**" if s["biggest_build"] else "")
          + (f", quietest around **{s['quietest_moment']}**" if s["quietest_moment"] else ""),
          "", "**Frequency balance**", "", "```"]
    for name, pct in s["frequency_balance_pct"].items():
        L.append(f"{name:<18} {bar(pct):<20} {pct:>5}%")
    L.append("```")

    # Ideas
    L += ["", "### 💡 Ideas to try", ""]
    for i in r["ideas"]:
        L.append(f"- **{i['area']}:** {i['idea']}")

    ai = r.get("ai")
    if ai:
        L += ["", "### 🤖 AI brainstorm", ""]
        if ai.get("error"):
            L.append(f"_{ai['error']}_")
        else:
            if ai["vibe"]:
                L += [f"> {ai['vibe']}", ""]
            for i in ai["ideas"]:
                L.append(f"- **{i.get('area', 'idea')}:** {i['idea']}")
            if ai["alt_progressions"]:
                L += ["", "**Alternative progressions**", ""]
                for p in ai["alt_progressions"]:
                    L.append(f"- *{p.get('name', '')}*: `{' – '.join(map(str, p['chords']))}` — {p.get('why', '')}")
            if ai["title_ideas"]:
                L += ["", "**Title ideas:** " + " · ".join(f"_{t}_" for t in ai["title_ideas"])]

    if r.get("midi"):
        L += ["", "### 🎹 MIDI sketches (drag into your DAW)", ""]
        for m in r["midi"]:
            L.append(f"- `midi/{m['file']}` — {m['label']}: {' – '.join(m['chords'])}")
    L.append("")
    return L


def build_report(results):
    ok = [r for r in results if "error" not in r]
    L = ["# 🎵 Music Inspiration Report", "", f"_{len(results)} track(s) analysed_", ""]
    if ok:
        L += ["| Track | Length | Tempo | Key | Camelot | Mode | Form | Energy |",
              "|-------|--------|-------|-----|---------|------|------|--------|"]
        for r in ok:
            tempo = f"{r['tempo_bpm']:.0f}" if r["tempo_bpm"] else "—"
            L.append(f"| {r['file']} | {r['duration']} | {tempo} | {r['key']} | {r['camelot']} | {r['mode']} | "
                     f"{' '.join(s['label'] for s in r['sections'])} | `{sparkline(r['sound']['energy_curve'])}` |")
        L.append("")
    for r in results:
        L += track_report(r)

    if len(ok) > 1:
        L += ["## 🔗 Tracks that blend well", "",
              "_Key-compatible on the Camelot wheel — good for medleys, mashups, DJ transitions or a set list._", ""]
        pairs = pairings(ok)
        if pairs:
            for p in pairs:
                L.append(f"- **{p['a']}** + **{p['b']}**: {p['keys']} — {p['relation']}, {p['tempo']}")
        else:
            L.append("- No key-compatible pairs — these tracks would contrast rather than blend.")
        L.append("")

    L += ["---",
          "_All analysis is automatic and approximate. Key detection is ~70–80% accurate on real music "
          "(often confusing relative keys like B minor ↔ D major); chord detection works best on clear, "
          "sustained harmony and simplifies to major/minor/7th chords; tempo can come out at half or "
          "double the felt tempo; section letters are a rough guide. Use it as a starting point for ideas, "
          "not as a transcription._"]
    return "\n".join(L)


# ---------------------------------------------------------------- main

def collect_files(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        elif p.exists():
            files.append(p)
        else:
            print(f"  ! Not found: {p}", file=sys.stderr)
    return files


def run_analysis(files, out, llm=None, progress=None):
    """Analyse files, write music_report.md / music_results.json / midi/*.mid into `out`.
    `progress(message)` is called as work proceeds. Returns (results, report_markdown)."""
    progress = progress or (lambda msg: None)
    out = Path(out)
    results = []
    for i, f in enumerate(files, 1):
        f = Path(f)
        progress(f"[{i}/{len(files)}] Analysing {f.name}...")
        try:
            r = analyze(f)
            if llm:
                progress(f"[{i}/{len(files)}] Brainstorming ideas for {f.name}...")
                r["ai"] = ai_brainstorm(r, llm)
            r["midi"] = export_midis(r, out / "midi")
        except Exception as e:  # unreadable / corrupt file — keep going
            progress(f"  ! {f.name}: {e}")
            r = {"file": f.name, "error": "unreadable or unsupported audio" if "Error opening" in str(e)
                 or "Invalid data" in str(e) else str(e)[:200]}
        results.append(r)

    report = build_report(results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "music_report.md").write_text(report, encoding="utf-8")
    (out / "music_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results, report


def main():
    ap = argparse.ArgumentParser(description="Analyse instrumental tracks and generate ideas for musicians.")
    ap.add_argument("paths", nargs="+", help="audio files and/or folders")
    ap.add_argument("--out", default=".", help="output folder for the report, JSON and MIDI files")
    ap.add_argument("--no-llm", action="store_true", help="skip the AI brainstorm")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    files = collect_files(args.paths)
    if not files:
        sys.exit("No audio files found.")

    llm, llm_label = (None, None) if args.no_llm else make_llm()
    if llm:
        print(f"AI brainstorm via {llm_label} (only the analysis numbers are sent, never the audio).", file=sys.stderr)
    elif not args.no_llm:
        print("No NVIDIA/Anthropic/OpenAI key set — skipping the AI brainstorm.", file=sys.stderr)

    out = Path(args.out)
    _, report = run_analysis(files, out, llm, progress=lambda msg: print(msg, file=sys.stderr))
    print(report)
    print(f"\nWrote {out / 'music_report.md'}, {out / 'music_results.json'} and MIDI files in {out / 'midi'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
