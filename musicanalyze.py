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
import html
import json
import os
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import instruments

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

# Key timeline settings: estimate the key every STEP seconds from a WINDOW-second slice.
# Changing key costs KEY_SWITCH_PENALTY (Viterbi smoothing), so the key only changes when the
# evidence stays different for a while; sections shorter than MIN_SECTION are then dropped.
STEP, WINDOW, MIN_SECTION = 2.0, 10.0, 8.0
KEY_SWITCH_PENALTY = 0.6

# Chord vocabulary for detection (intervals above the root). 7th and power chords get a small
# penalty so plain triads win unless the 7th is clearly there / the third is clearly missing.
DETECT_QUALITIES = {"": ([0, 4, 7], 1.0), "m": ([0, 3, 7], 1.0),
                    "7": ([0, 4, 7, 10], 0.92), "m7": ([0, 3, 7, 10], 0.92),
                    "5": ([0, 7], 0.86)}
CHORD_SWITCH_PENALTY = 0.12
# Wider vocabulary accepted when reading chord names (e.g. from the AI) for MIDI export.
# The longest matching quality wins, so "m7b5" beats "m7" and "m".
CHORD_INTERVALS = {"": [0, 4, 7], "maj": [0, 4, 7], "M": [0, 4, 7], "m": [0, 3, 7], "min": [0, 3, 7],
                   "-": [0, 3, 7], "5": [0, 7], "6": [0, 4, 7, 9], "m6": [0, 3, 7, 9],
                   "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10], "maj7": [0, 4, 7, 11], "M7": [0, 4, 7, 11],
                   "9": [0, 4, 7, 10, 14], "m9": [0, 3, 7, 10, 14], "maj9": [0, 4, 7, 11, 14],
                   "add9": [0, 4, 7, 14], "madd9": [0, 3, 7, 14],
                   "dim": [0, 3, 6], "dim7": [0, 3, 6, 9], "m7b5": [0, 3, 6, 10], "aug": [0, 4, 8],
                   "+": [0, 4, 8], "sus2": [0, 2, 7], "sus4": [0, 5, 7], "sus": [0, 5, 7],
                   "7sus4": [0, 5, 7, 10], "11": [0, 4, 7, 10, 14, 17], "13": [0, 4, 7, 10, 14, 21]}

# Modes, as intervals above the tonic, with the "colour" degree that defines each one.
MODES = {
    "Ionian (major)": ([0, 2, 4, 5, 7, 9, 11], 4),
    "Lydian": ([0, 2, 4, 6, 7, 9, 11], 6),
    "Mixolydian": ([0, 2, 4, 5, 7, 9, 10], 10),
    "Phrygian dominant": ([0, 1, 4, 5, 7, 8, 10], 1),
    "Dorian": ([0, 2, 3, 5, 7, 9, 10], 9),
    "Aeolian (natural minor)": ([0, 2, 3, 5, 7, 8, 10], 8),
    "Harmonic minor": ([0, 2, 3, 5, 7, 8, 11], 11),
    "Phrygian": ([0, 1, 3, 5, 7, 8, 10], 1),
}
ROMAN = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "google/gemma-4-31b-it"
# Tried in order if the main model times out or has been retired (comma-separated, overridable
# with MUSIC_FALLBACK_MODELS).
NVIDIA_FALLBACK_MODELS = "nvidia/nemotron-3-super-120b-a12b,openai/gpt-oss-20b"
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
    is_minor = q.startswith("m") and not q.startswith("maj")
    if is_minor or q in ("dim", "dim7"):
        r = r.lower()
    special = {"": "", "m": "", "dim": "°", "dim7": "°7", "m7b5": "ø7", "aug": "+"}
    return r + special.get(q, q[1:] if is_minor else q)


def parse_chord(name):
    """'F#m7' -> (6, 'm7', [0,3,7,10]). Unknown extensions fall back to the longest known
    quality prefix ('B7alt' -> B7, 'Em9/G' -> Em9). Returns None if there's no recognisable root."""
    m = re.fullmatch(r"\s*([A-Ga-g])(##|bb|[b#]?)([^/\s]*)(?:/[A-Ga-g][b#]?)?\s*", str(name))
    if not m:
        return None
    root = (LETTER_PC[m.group(1).upper()] + m.group(2).count("#") - m.group(2).count("b")) % 12
    q = max((k for k in CHORD_INTERVALS if m.group(3).startswith(k)), key=len, default="")
    q = {"maj": "", "M": "", "min": "m", "-": "m", "M7": "maj7", "+": "aug", "sus": "sus4"}.get(q, q)
    return root, q, CHORD_INTERVALS[q]


def fmt_time(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def mode_of(labels):
    """Most common label; on a full tie keep the middle one."""
    c = Counter(labels).most_common()
    return labels[len(labels) // 2] if len(c) > 1 and c[0][1] == c[1][1] else c[0][0]


# ---------------------------------------------------------------- harmony

def viterbi(scores, penalty):
    """Best label path through a (time x labels) score matrix, where switching label costs
    `penalty`. Smooths out flicker far better than a majority vote."""
    T, K = scores.shape
    dp, back = scores[0].copy(), np.zeros((T, K), dtype=int)
    for t in range(1, T):
        best = int(np.argmax(dp))
        switch = dp[best] - penalty
        back[t] = np.where(dp >= switch, np.arange(K), best)
        dp = np.maximum(dp, switch) + scores[t]
    path = [int(np.argmax(dp))]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def key_timeline(chroma, duration):
    """Return [(start, end, key_index)] sections, smoothed and merged."""
    fps = SR / HOP
    times = np.arange(0, duration, STEP)
    scores = []
    for t in times:
        a = int(max(0, t + STEP / 2 - WINDOW / 2) * fps)
        b = int(min(duration, t + STEP / 2 + WINDOW / 2) * fps)
        scores.append(key_scores(chroma[:, a:max(b, a + 1)].mean(axis=1)))
    keys = viterbi(np.array(scores), KEY_SWITCH_PENALTY)
    spans = [(t, min(t + STEP, duration), k) for t, k in zip(times, keys)]
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


def mode_and_scale(chroma, frames, tonic):
    """Fit the pitch-class profile (tonic fixed) against each mode's scale. Also detects a
    drone/pedal on the tonic and a mix of major and minor thirds."""
    pc = chroma[:, frames].mean(axis=1)
    rel = np.roll(pc / (pc.max() + 1e-9), -tonic)  # rel[i] = strength of the note i semitones above tonic
    # Fit on degrees 1-11 only: a loud tonic (drone) would otherwise dominate the correlation.
    fits = {}
    for name, (ivs, _) in MODES.items():
        tmpl = np.zeros(12)
        tmpl[ivs] = 1
        fits[name] = float(_zscore(rel[1:]) @ _zscore(tmpl[1:]) / 11)
        if name in ("Ionian (major)", "Aeolian (natural minor)"):
            fits[name] += 0.04  # prefer the plain label unless an exotic mode clearly fits better
    ranked = sorted(fits, key=fits.get, reverse=True)
    mode, runner_up = ranked[0], ranked[1]
    margin = fits[mode] - fits[runner_up]
    if margin <= 0.06:  # unsure: report the plain major/minor with the same third, mention the exotic one
        plain = "Aeolian (natural minor)" if 3 in MODES[mode][0] else "Ionian (major)"
        if mode != plain:
            mode, runner_up = plain, mode
    key = tonic + (12 if 3 in MODES[mode][0] else 0)
    note = lambda off: spell((tonic + off) % 12, key)
    others = np.delete(rel, 0)
    lo, hi = sorted((rel[3], rel[4]))
    return {
        "key_index": key,
        "mode": mode,
        "mode_confidence": "high" if margin > 0.15 else "medium" if margin > 0.06 else "low",
        "mode_runner_up": runner_up,
        "color_note": note(MODES[mode][1]),
        "scale_notes": [note(i) for i in MODES[mode][0]],
        # Drone/pedal: the tonic sounds strongly in nearly every moment, whatever the chord.
        "drone_pct": round(100 * float(np.mean(chroma[tonic, frames] >= 0.5))),
        "drone": bool(np.mean(chroma[tonic, frames] >= 0.5) >= 0.75),
        "third_mix": bool(lo > 0.6 * hi and lo > np.median(others)),
        "thirds": f"{note(3)} / {note(4)}",
        "note_strength": {note(i): round(float(rel[i]), 2) for i in range(12)},
    }


def detect_chords(chroma, rms_db, bounds, btimes):
    """One chord per beat, Viterbi-smoothed and merged. Returns list of dicts (named later, per key)."""
    import librosa
    sync = librosa.util.sync(chroma, bounds, aggregate=np.median, pad=False)
    loud = librosa.util.sync(rms_db[None, :], bounds, aggregate=np.mean, pad=False)[0]
    v = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-9)
    scores = (CHORD_TEMPLATES @ v).T  # (beats, chords)
    silent = loud < rms_db.max() - 35
    scores = np.hstack([scores, np.where(silent, 1.0, 0.0)[:, None]])  # last column = no chord
    labels = viterbi(scores, CHORD_SWITCH_PENALTY)
    labels = [-1 if lab == len(CHORD_LABELS) else lab for lab in labels]
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
        # Steadiness from the gaps between detected beats (ignoring missed/extra beats).
        ibi = np.diff(beat_times)
        ibi = ibi[(ibi > 0.6 * np.median(ibi)) & (ibi < 1.6 * np.median(ibi))]
        cv = float(np.std(ibi) / np.mean(ibi)) if len(ibi) > 3 else 0.0
        out["tempo_variation_pct"] = round(100 * cv, 1)
        out["tempo_feel"] = ("steady (likely to a click/grid)" if cv < 0.025 else
                             "steady, played by feel" if cv < 0.07 else
                             "loose / pushing and pulling" if cv < 0.12 else "free / rubato")
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
    edge = 8  # ignore the first/last seconds: the recording starting/stopping isn't a "build"
    if len(per_sec) >= 2 * edge + 12:
        rises = [(per_sec[t:t + 4].mean() - per_sec[t - 6:t].mean(), t) for t in range(edge, len(per_sec) - edge)]
        rise_db, rise_t = max(rises)
        win = np.convolve(per_sec[edge:-edge], np.ones(6) / 6, mode="valid")
        quiet_t = int(np.argmin(win)) + edge + 3  # centre of the quietest 6-second stretch
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
    # Split off the drums so they don't blur the harmony, and analyse each part on its own.
    y_perc, y_harm, separation = instruments.separate(y, sr)  # drums, everything else

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    clear_beat = len(beat_frames) >= 8
    if not clear_beat:  # ambient / rubato: fall back to half-second steps
        beat_frames = librosa.time_to_frames(np.arange(0.5, duration, 0.5), sr=sr, hop_length=HOP)

    # Instruments are often not at A440 (phone demos, detuned guitars). Correct for it, or every
    # note smears across two semitones and keys/chords come out wrong.
    tuning = float(librosa.estimate_tuning(y=y_harm, sr=sr))
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP, tuning=tuning)
    mfcc =librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    rms_db = 20 * np.log10(librosa.feature.rms(y=y, hop_length=HOP)[0] + 1e-9)
    n = min(chroma.shape[1], mfcc.shape[1], len(rms_db))
    chroma, mfcc, rms_db = chroma[:, :n], mfcc[:, :n], rms_db[:n]
    bounds = librosa.util.fix_frames(beat_frames[beat_frames < n], x_min=0, x_max=n)
    btimes = librosa.frames_to_time(bounds, sr=sr, hop_length=HOP)
    btimes[-1] = duration
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)

    # --- key & mode
    ksec = key_timeline(chroma, duration)
    # The home note (tonic) is reliable; major vs minor often isn't (a drone, power chords, or
    # blues-style mixing of both thirds). So find the main tonic first, then let the mode fit
    # over all of its sections decide major vs minor, and relabel those sections consistently.
    tonic_time = Counter()
    for s, e, k in ksec:
        tonic_time[key_tonic(k)] += e - s
    main_tonic = tonic_time.most_common(1)[0][0]
    fps = sr / HOP
    frames = np.concatenate([np.arange(int(s * fps), min(int(e * fps), n))
                             for s, e, k in ksec if key_tonic(k) == main_tonic])
    scale = mode_and_scale(chroma, frames, main_tonic)
    main_key = scale.pop("key_index")
    ksec = merge_runs([(s, e, main_key if key_tonic(k) == main_tonic else k) for s, e, k in ksec])
    time_in_key = Counter()
    for s, e, k in ksec:
        time_in_key[k] += e - s
    scores = key_scores(chroma[:, frames].mean(axis=1))
    runner_up = next(int(i) for i in np.argsort(scores)[::-1] if i != main_key)
    if scores[main_key] < 0.6 and scale["mode_confidence"] == "high":
        scale["mode_confidence"] = "medium"  # the mode can't be surer than the key it's built on

    # --- chords
    chords = detect_chords(chroma, rms_db, bounds, btimes)
    bass = instruments.bass_notes(y_harm, sr, btimes, tuning)  # one note (or None) per beat span
    for c in chords:
        k = key_at(ksec, (c["start"] + c["end"]) / 2)
        c["name"], c["roman"] = chord_name(c["root"], c["q"], k), roman(c["root"], c["q"], k)
        # If the bass mostly plays something other than the root under this chord: slash chord (A/E).
        under = [bass[i] % 12 for i in range(len(bass)) if bass[i] is not None
                 and c["start"] <= (btimes[i] + btimes[i + 1]) / 2 < c["end"]]
        if under:
            pc = Counter(under).most_common(1)[0][0]
            tones = {(c["root"] + iv) % 12 for iv in CHORD_INTERVALS[c["q"]]}
            meaningful = pc in tones or pc == key_tonic(k)  # inversion, or pedal on the home note
            c["slash"] = c["name"] + "/" + spell(pc, k) if pc != c["root"] and meaningful else c["name"]
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

    # --- structure (with each section's key, main chords and which instruments play)
    drum_rms = librosa.feature.rms(y=y_perc, hop_length=HOP)[0]
    sections = detect_structure(chroma, mfcc, rms_db, bounds, btimes, duration, [e for _, e, _ in ksec[:-1]])
    for sec in sections:
        top = Counter()
        for c in chords:
            if sec["start"] <= (c["start"] + c["end"]) / 2 < sec["end"]:
                top[c["name"]] += c["end"] - c["start"]
        sec["key"] = KEY_NAMES[key_at(ksec, (sec["start"] + sec["end"]) / 2)]
        sec["main_chords"] = [name for name, _ in top.most_common(4)]
        a, b = int(sec["start"] * sr / HOP), max(int(sec["end"] * sr / HOP), int(sec["start"] * sr / HOP) + 1)
        sec["drums_level"] = float(np.mean(drum_rms[a:b] ** 2)) if a < len(drum_rms) else 0.0
        in_sec = [n for i, n in enumerate(bass) if sec["start"] <= (btimes[i] + btimes[i + 1]) / 2 < sec["end"]]
        sec["bass"] = bool(in_sec) and sum(n is not None for n in in_sec) >= 0.4 * len(in_sec)
    loudest_drums = max((sec["drums_level"] for sec in sections), default=0.0)
    for sec in sections:
        sec["drums"] = loudest_drums > 0 and sec.pop("drums_level") >= 0.1 * loudest_drums

    result = {
        "file": Path(path).name,
        "duration_sec": round(duration, 1),
        "duration": fmt_time(duration),
        "tuning_cents": int(round(tuning * 100)),
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
                            "with_bass": c.get("slash", c["name"]), "beats": c["beats"]} for c in chords],
        "loops": loops,
        "sections": sections,
        "rhythm": rhythm_features(y, y_perc, y_harm, beat_times, clear_beat, duration, sr),
        "sound": sound_features(y, sr, rms_db, duration),
        "separation": separation,
        "drums": instruments.drum_analysis(y_perc, sr, 256, beat_times, float(np.sum(y ** 2)))
        if clear_beat else {"present": False},
        "bass": bass_summary(bass, btimes, chords, main_key, duration),
    }
    result["ideas"] = rule_based_ideas(result)
    return result


def note_name(midi, key):
    return f"{spell(midi % 12, key)}{midi // 12 - 1}"


def bass_summary(notes, btimes, chords, key, duration):
    """Describe the bass line: main notes, range, how it moves, whether it follows the chords."""
    voiced = [(i, n) for i, n in enumerate(notes) if n is not None]
    if len(voiced) < max(8, 0.2 * len(notes)):
        return {"present": False}
    pcs = Counter(n % 12 for _, n in voiced)
    pairs = [(a, b) for (i, a), (j, b) in zip(voiced, voiced[1:]) if j == i + 1]
    moves = [abs(b - a) for a, b in pairs]
    stay = np.mean([m == 0 for m in moves]) if moves else 0.0
    step = np.mean([0 < m <= 2 for m in moves]) if moves else 0.0
    leap = np.mean([m > 2 for m in moves]) if moves else 0.0
    root_hits = []
    for i, n in voiced:
        mid = (btimes[i] + btimes[i + 1]) / 2
        c = next((c for c in chords if c["start"] <= mid < c["end"]), None)
        if c:
            root_hits.append(n % 12 == c["root"])
    follows = float(np.mean(root_hits)) if root_hits else 0.0
    top_pc, top_n = pcs.most_common(1)[0]
    top_share = top_n / len(voiced)

    # A riff: the most common 4-beat (one bar) note pattern, if it repeats enough and moves.
    bars = [tuple(notes[i:i + 4]) for i in range(0, len(notes) - 3, 4)]
    bars = [b for b in bars if all(n is not None for n in b)]
    riff = None
    if bars:
        pattern, count = Counter(bars).most_common(1)[0]
        if count >= 3 and count >= 0.25 * len(bars) and len(set(pattern)) > 1:
            riff = [spell(n % 12, key) for n in pattern]

    if top_share >= 0.55 and stay >= 0.5:
        style = f"pedal — holds {spell(top_pc, key)} under changing chords"
    elif follows >= 0.6:
        style = "plays the chord roots"
    elif riff:
        style = "riff-based"
    elif step >= 0.35:
        style = "melodic / walking (moves stepwise)"
    else:
        style = "mixed (roots, jumps and passing notes)"
    lo, hi = (int(v) for v in np.percentile([n for _, n in voiced], [10, 90]))  # ignore stray octave errors
    return {
        "present": True,
        "plays_pct": round(100 * len(voiced) / len(notes)),
        "main_notes": [{"note": spell(pc, key), "pct": round(100 * c / len(voiced))} for pc, c in pcs.most_common(5)],
        "range": f"{note_name(lo, key)}–{note_name(hi, key)}",
        "style": style,
        "stays_pct": round(100 * stay),
        "steps_pct": round(100 * step),
        "leaps_pct": round(100 * leap),
        "follows_roots_pct": round(100 * follows),
        "riff": riff,
        "pedal_note": spell(top_pc, key) if top_share >= 0.55 and stay >= 0.5 else None,
    }


def _same_cycle(a, b):
    """True if chord list b is a rotation of chord list a."""
    return len(a) == len(b) and any(a == b[i:] + b[:i] for i in range(len(b)))


# ---------------------------------------------------------------- rule-based ideas

def rule_based_ideas(r):
    k = r["key_index"]
    t = key_tonic(k)
    note = lambda off: spell((t + off) % 12, k)
    ideas = []

    used = {c["chord"] for c in r["chord_timeline"]}
    roots_used = {parse_chord(c)[0] for c in used if parse_chord(c)}

    def chord(off, q=""):
        """Chord on a scale degree, spelled for the key, e.g. chord(5, 'm') -> 'Am' in E."""
        return spell((t + off) % 12, k) + q

    # Tuning: matters as soon as the musician plays along with anything else.
    cents = r.get("tuning_cents", 0)
    if abs(cents) >= 15:
        ideas.append({"area": "tuning", "idea":
                      f"Your instrument is tuned about {abs(cents)} cents {'sharp' if cents > 0 else 'flat'} of "
                      f"standard pitch (A = {440 * 2 ** (cents / 1200):.0f} Hz). The MIDI sketches below include a "
                      f"pitch bend so they match this recording; for anything else (other players, samples, synths) "
                      f"either retune to A440 or detune those by {cents:+d} cents."})

    # Mode colour (only stated firmly when the fit is clear)
    tips = {
        "Harmonic minor": f"The raised 7th ({note(11)}) is the signature — it pulls hard to {note(0)}. "
                          f"Land melody phrases on {note(11)}→{note(0)}, or try the exotic {note(8)}→{note(11)} "
                          f"step in a lead line.",
        "Dorian": f"The major 6th ({note(9)}) is the Dorian colour. Feature it in a melody, or use a major IV chord "
                  f"({chord(5)}) for that soulful/funky lift.",
        "Phrygian": f"The b2 ({note(1)}) gives Phrygian tension — a {chord(1)} chord resolving to {chord(0, 'm')} "
                    f"sounds dark and cinematic.",
        "Phrygian dominant": f"Phrygian dominant (the 'Spanish'/flamenco scale): the b2 ({note(1)}) against the major "
                             f"3rd ({note(4)}). The classic move is {chord(8)} → {chord(1)} → {chord(0)} "
                             f"(Andalusian cadence), or a {chord(1)}–{chord(0)} vamp.",
        "Aeolian (natural minor)": f"Natural minor: the b6 ({note(8)}) carries the melancholy. A bVI ({chord(8)}) "
                                   f"→ bVII ({chord(10)}) → i lift is a classic epic move.",
        "Ionian (major)": f"Plain major, so the strongest colour is the leading tone ({note(11)}) → {note(0)} at "
                          f"phrase ends, or a V ({chord(7)}) → I cadence to land sections.",
        "Mixolydian": f"The b7 ({note(10)}) gives a Mixolydian, bluesy-rock feel. A bVII ({chord(10)}) → I "
                      f"cadence will feel natural here.",
        "Lydian": f"The #4 ({note(6)}) is the Lydian colour — a II major chord ({chord(2)}) over a {note(0)} "
                  f"bass gives a dreamy, floating lift.",
    }
    if r.get("mode_confidence") == "low":
        ideas.append({"area": "melody/harmony", "idea":
                      f"The scale is ambiguous between {r['mode']} and {r['mode_runner_up']} — the notes that "
                      f"would decide it barely appear. That's an opportunity: commit with a melody note. "
                      + tips[r["mode"]]})
    else:
        ideas.append({"area": "melody/harmony", "idea": tips[r["mode"]]})

    # Drone / pedal tone
    bass = r.get("bass") or {}
    if bass.get("pedal_note") == note(0):
        ideas.append({"area": "bass", "idea":
                      f"The bass holds {note(0)} under almost every chord (a pedal). Release it in one section: let "
                      f"the bass follow the chord roots ({chord(5)} under {chord(5)}, {chord(9, 'm')} under "
                      f"{chord(9, 'm')}) so the harmony suddenly moves — then return to the {note(0)} pedal for home."})
    elif r.get("drone"):
        heard = {c.get("with_bass") for c in r["chord_timeline"]}
        fresh = [f"{chord(off, q)}/{note(0)}" for off, q in ((2, ""), (10, ""), (7, ""), (5, ""), (9, "m"))
                 if f"{chord(off, q)}/{note(0)}" not in heard][:2]
        ideas.append({"area": "harmony", "idea":
                      f"{note(0)} rings through almost everything (a drone/pedal). Put new chords on top of it — "
                      f"{' or '.join(fresh)} — for tension without leaving home, or drop the drone for one "
                      f"section so its return hits harder."})
    if bass.get("present") and bass.get("follows_roots_pct", 0) >= 70:
        ideas.append({"area": "bass", "idea": "The bass mostly plays chord roots. Add approach notes: on the last 8th "
                      "before each chord change, play a note a half-step above or below the next root."})

    # Major/minor thirds both present
    if r.get("third_mix"):
        ideas.append({"area": "melody", "idea":
                      f"Both thirds ({r['thirds']}) show up — a bluesy major/minor blur. Bend from the minor to the "
                      f"major third in a lead line, or keep one section strictly major and another strictly minor."})

    # Borrowed chords from the parallel key — only suggest what isn't already in the track
    if key_is_minor(k):
        options = [(5, "", "a major IV"), (7, "", "a major V for a stronger pull home"),
                   (0, "", "a major I to end a section (a 'Picardy third' surprise)")]
        source = f"{note(0)} major"
    else:
        options = [(5, "m", "a minor iv"), (8, "", "a bVI"), (10, "", "a bVII"), (3, "", "a bIII")]
        source = f"{note(0)} minor"
    fresh = [(off, q, label) for off, q, label in options if chord(off, q) not in used]
    already = [chord(off, q) for off, q, _ in options if chord(off, q) in used]
    if fresh:
        text = f"Borrow from {source}: try " + ", or ".join(f"{label} — {chord(off, q)}" for off, q, label in fresh[:2])
        text += "." + (f" You already use {', '.join(already)} from there, so this colour suits the track." if already else "")
        ideas.append({"area": "harmony", "idea": text})

    # A chord from the scale the track never touches
    scale_degrees = MODES[r["mode"]][0]
    unused = [off for off in scale_degrees[1:] if (t + off) % 12 not in roots_used]
    if unused:
        off = unused[0]
        has = lambda iv: (off + iv) % 12 in scale_degrees
        q = "dim" if has(3) and has(6) and not has(7) else "m" if has(3) and not has(4) else ""
        numeral = roman((t + off) % 12, q, k)
        ideas.append({"area": "harmony", "idea":
                      f"The track never visits the {numeral} chord ({chord(off, q)}), which fits the scale — a fresh "
                      f"place to go for a bridge or a new section."})

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
    build, quiet = s["biggest_build"], s["quietest_moment"]
    to_sec = lambda ts: int(ts.split(":")[0]) * 60 + int(ts.split(":")[1])
    if build and quiet and abs(to_sec(build["time"]) - to_sec(quiet)) <= 10:
        # The quietest moment and the biggest build are one event: a dip and a comeback.
        ideas.append({"area": "arrangement", "idea": f"The biggest contrast is the dip around {quiet} that comes back "
                      f"+{build['db']} dB by {build['time']}. Make it a real breakdown: strip it to one instrument for "
                      f"4–8 bars, then hit the return with everything (a riser or a bar of silence just before helps)."})
        quiet = None
    elif build and not s["energy_flat"]:
        ideas.append({"area": "arrangement", "idea": f"The strongest moment is the build into {build['time']} "
                      f"(+{build['db']} dB). Set it up harder: a riser, a drum fill, or a bar of silence "
                      f"right before it."})
    if quiet:
        ideas.append({"area": "arrangement", "idea": f"The quietest stretch is around {s['quietest_moment']} — a natural "
                      f"spot for a solo instrument, a field recording, or a new melodic motif."})

    # Drums
    d = r.get("drums") or {}
    if d.get("grid"):
        traits = d.get("traits", [])
        changes = [fmt_time(sec["start"]) for sec in r["sections"][1:]]
        if not d.get("fills") and changes:
            ideas.append({"area": "drums", "idea": f"The groove runs without obvious fills. Mark the section changes "
                          f"({', '.join(changes[:4])}) with a one-bar fill or a crash on the downbeat."})
        if any("backbeat" in t for t in traits):
            quiet = next((sec["label"] for sec in r["sections"] if sec["energy"] == "low"), None)
            ideas.append({"area": "drums", "idea": "Classic backbeat (snare on 2 and 4). For contrast, play one "
                          f"section{' (e.g. ' + quiet + ')' if quiet else ''} half-time — snare only on 3 — so the "
                          "return to the backbeat lifts."})
        if not any("syncopated" in t or "tresillo" in t for t in traits):
            ideas.append({"area": "drums", "idea": "The kick stays on the main beats. Add one push — a kick on the "
                          "'a' of 2 or the '&' of 3 — for more bounce without changing the feel."})
        hats = d.get("hats") or ""
        if hats and hats.count("h") <= 5:
            ideas.append({"area": "drums", "idea": "The cymbal/hi-hat marks only the main beats. Switch it to steady "
                          "8ths (or 16ths) in the loudest section to raise the energy."})

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
    if bands["sub (<60 Hz)"] < 1 and bands["air (>6k)"] < 1:
        ideas.append({"area": "sound", "idea": "Almost nothing below 60 Hz or above 6 kHz — typical of a phone/voice-memo "
                      "recording, so judge the mix from a proper recording. When producing it, those empty ranges are "
                      "space to fill: a sub-bass under the roots, and something airy on top (shimmer, hi-hats, a high "
                      "counter-melody)."})
    elif bands["air (>6k)"] < 2:
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


def write_midi(path, chords, bpm=90, beats_per_chord=4, repeats=2, cents=0):
    """Write a simple 1-track MIDI file: bass root + mid-range chord, one chord per bar.
    `cents` adds a pitch bend so it plays in tune with a recording that isn't at A440
    (assumes the usual ±2 semitone bend range)."""
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
    if cents:
        bend = int(np.clip(8192 + cents / 200 * 8192, 0, 16383))
        data += b"\x00" + bytes([0xE0, bend & 0x7F, bend >> 7])
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
    stem = re.sub(r"[^\w.\-]+", "_", Path(r["file"]).stem)
    bpm = r["tempo_bpm"] or 90
    cents = r.get("tuning_cents", 0) if abs(r.get("tuning_cents", 0)) >= 10 else 0
    files = []

    def save(label, names):
        parsed = [(p[0], p[2]) for p in map(parse_chord, names) if p]
        if len(parsed) < 2:
            return
        fname = f"{stem}_{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')}.mid"
        write_midi(out_dir / fname, parsed, bpm, cents=cents)
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
    if not r["loops"]:
        # No strict loop: sketch the 4 most-used chords in the order the track first plays them.
        top = [c["chord"] for c in r["chords_used"][:4]]
        first_seen = list(dict.fromkeys(c["chord"] for c in r["chord_timeline"] if c["chord"] in top))
        save("main chords", first_seen)
    for i, alt in enumerate((r.get("ai") or {}).get("alt_progressions", []), 1):
        save(f"ai {i} {alt.get('name', '')}", alt.get("chords", []))
    return files


# ---------------------------------------------------------------- optional AI brainstorm

BRAINSTORM_PROMPT = """You are a creative music producer helping a musician brainstorm ideas from an
instrumental track they made. Below is automatic audio analysis of the track (JSON).

Return ONLY a JSON object, no other text:
{{
  "vibe": "<at most 2 sentences on the feel the data suggests; phrase interpretations as such, e.g. 'the slow tempo and dark tone suggest...'>",
  "ideas": [{{"area": "<harmony|melody|rhythm|arrangement|sound|structure|remix>", "idea": "<one concrete, actionable suggestion that references a specific time, section, chord, note or number from the data>"}}],
  "alt_progressions": [{{"name": "<short name>", "chords": ["<chord>", "..."], "why": "<one sentence>"}}],
  "title_ideas": ["<title>", "<title>", "<title>"]
}}

RULES:
- Give 6 ideas, each different, specific and under 40 words. No generic advice like "add more layers" or "experiment".
- Keep the whole reply short (under 600 words) so it isn't cut off.
- Do not repeat these ideas the musician already has: {existing}
- Give 2-3 alt_progressions in the track's key ({key}), 3-6 chords each, as alternatives or a contrasting
  section for the detected loop. Chord names: letter + optional b/# + optional quality from m, 7, m7, maj7, dim, sus2, sus4
  (e.g. "Bm", "F#7", "Gmaj7").
- Facts about the track must come from the data. Never claim instruments, genre or anything the data can't show.
  Automatic analysis can be wrong; if key_confidence < 0.6 treat the key as uncertain, and if
  mode_confidence is "low" don't build ideas on the exact mode.
- When you mention a time or a section, copy its start/end exactly from the data.
- Prefer chords the track doesn't already use (see chords_used) for alternative progressions.

ANALYSIS:
{data}
"""


def make_llm():
    """Return (fn(prompt)->text, label) for whichever API key is set, or (None, None)."""
    if os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        import openai
        if os.environ.get("NVIDIA_API_KEY"):
            client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=os.environ["NVIDIA_API_KEY"],
                                   timeout=90, max_retries=1)
            models = [os.environ.get("MUSIC_MODEL", NVIDIA_MODEL)] + [
                m.strip() for m in os.environ.get("MUSIC_FALLBACK_MODELS", NVIDIA_FALLBACK_MODELS).split(",") if m.strip()]
            label = "nvidia"
        else:
            client = openai.OpenAI(timeout=90, max_retries=1)
            models, label = [os.environ.get("MUSIC_MODEL", OPENAI_MODEL)], "openai"

        def call(prompt):
            # NVIDIA's free endpoints sometimes time out (504), retire models (410), or (reasoning
            # models) spend the whole budget thinking and return nothing. Fall through to the next
            # model in all of those cases instead of failing the brainstorm.
            problems = []
            for model in dict.fromkeys(models):
                try:
                    r = client.chat.completions.create(model=model, temperature=0.8, max_tokens=4096,
                                                       messages=[{"role": "user", "content": prompt}])
                except (openai.APIStatusError, openai.APITimeoutError, openai.APIConnectionError) as e:
                    if getattr(e, "status_code", 500) in (400, 401, 403):  # bad request / key: don't hammer
                        raise
                    problems.append(f"{model}: {type(e).__name__} {getattr(e, 'status_code', '')}".strip())
                    continue
                text = r.choices[0].message.content or ""
                if "{" in text:
                    call.model_used = model
                    return text
                problems.append(f"{model}: empty reply")
            raise RuntimeError("no model gave an answer (" + "; ".join(problems) + ")")
        return call, f"{label}/{models[0]}"

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        model = os.environ.get("MUSIC_MODEL", ANTHROPIC_MODEL)

        def call(prompt):
            m = client.messages.create(model=model, max_tokens=3000, temperature=0.8,
                                       messages=[{"role": "user", "content": prompt}])
            call.model_used = model
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
    obj = _parse_json_reply(raw)
    if obj is None:
        return {"error": "AI reply could not be parsed", "raw": raw.strip()[:1500]}
    return {
        "model": getattr(llm, "model_used", None),
        "vibe": _clean(obj.get("vibe", "")),
        "ideas": [{"area": _clean(i.get("area", "idea")), "idea": _clean(i["idea"])}
                  for i in obj.get("ideas", []) if isinstance(i, dict) and i.get("idea")],
        "alt_progressions": [{"name": _clean(p.get("name", "")), "why": _clean(p.get("why", "")),
                              "chords": [_clean(c) for c in p["chords"]]}
                             for p in obj.get("alt_progressions", [])
                             if isinstance(p, dict) and isinstance(p.get("chords"), list)],
        "title_ideas": [_clean(t) for t in obj.get("title_ideas", [])][:5],
    }


def _parse_json_reply(raw):
    """Parse the AI's JSON. If the reply was cut off mid-way, keep everything up to the last
    complete item by closing the open brackets, so a long answer still yields its ideas."""
    a = raw.find("{")
    if a < 0:
        return None
    text = raw[a:raw.rfind("}") + 1] if raw.rfind("}") > a else raw[a:]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    body = raw[a:]
    for cut in [i for i, ch in enumerate(body) if ch == "}"][::-1][:200]:
        head = body[:cut + 1]
        # Close whatever is still open, innermost first.
        stack = []
        in_str = esc = False
        for ch in head:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()
        if in_str:
            continue
        try:
            obj = json.loads(head + "".join(reversed(stack)))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


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

# Plain-English explanations, shown as hover tooltips and in the "How to read this report" guide.
GLOSSARY = {
    "Length": "Duration of the track (minutes:seconds).",
    "Tempo": "Speed in beats per minute (BPM). Automatic detection occasionally reads half or double what you feel.",
    "Key": "The home note and scale the music settles on, e.g. E major. Major usually sounds brighter, minor darker.",
    "Camelot": "A DJ/producer code for keys: numbers 1-12 around a wheel, A = minor, B = major. Tracks with the same "
               "code, one number apart (same letter), or the same number with the other letter blend smoothly.",
    "Mode": "The flavour of the scale: which notes are used around the home note. Hover a mode name to hear what "
            "it's like in words.",
    "Form": "The order of the song's sections. Each letter is a section; the same letter again means it sounds like "
            "an earlier one (e.g. A B A C = verse, chorus, verse, bridge). Detected automatically, so approximate.",
    "Energy": "Loudness across the track from start (left) to end (right) in 24 slices; taller bars are louder. "
              "Shows builds, drops and breakdowns at a glance.",
    "Confidence": "How strongly the notes match the key (0-1). Above 0.75 is clear; below 0.6 the harmony is "
                  "ambiguous (modal music, drones, power chords).",
    "Colour note": "The note that gives this mode its character. Feature it in a melody to make the mode obvious.",
    "Roman numerals": "Chords named by their place in the key: I = the home chord, IV and V = built on the 4th and "
                      "5th notes. Upper case = major, lower case = minor, so 'i - iv - V' is the same progression "
                      "in any key.",
    "Harmonic rhythm": "How often the chord changes.",
    "Drone / pedal": "A note that keeps sounding while the chords change above it.",
    "Slash chord": "A/E means an A chord with E as the lowest (bass) note.",
    "Tuning": "How far the instrument is from standard pitch (A = 440 Hz), in cents. 100 cents = one semitone.",
    "Drum grid": "One bar split into 16 sixteenth-notes (1 e & a 2 e & a 3 e & a 4 e & a). K = kick-heavy hit, "
                 "S = snare-heavy hit, X = strong hit (kick and snare), x = lighter hit, . = usually nothing. "
                 "h = hi-hat/cymbal.",
    "Groove repetition": "How closely the bars repeat the main drum groove: tight = loop-like, loose = lots of "
                         "variation.",
    "Syncopation": "Hits that fall between the beats instead of on them; makes a groove feel pushed or funky.",
    "Percussive": "Share of the sound energy coming from drums/percussion rather than sustained notes.",
    "Frequency balance": "Share of energy in each range: sub (felt more than heard), bass, low-mids (body), "
                         "high-mids (presence), air (sparkle).",
    "dBFS": "Loudness relative to the digital maximum (0 dBFS); more negative = quieter.",
    "Dynamic range": "The difference between the loud and quiet parts, in dB. Bigger = more contrast.",
}
MODE_INFO = {
    "Ionian (major)": "The ordinary major scale: bright, stable, resolved.",
    "Lydian": "Major with a raised 4th: dreamy, floating, film-score.",
    "Mixolydian": "Major with a flat 7th: bluesy, rock, folk, laid-back.",
    "Phrygian dominant": "Flat 2nd with a major 3rd: Spanish / flamenco / Middle-Eastern.",
    "Dorian": "Minor with a raised 6th: soulful, jazzy, funky, less sad than plain minor.",
    "Aeolian (natural minor)": "The ordinary minor scale: sad, serious, emotional.",
    "Harmonic minor": "Minor with a raised 7th: dramatic, classical, exotic pull to home.",
    "Phrygian": "Minor with a flat 2nd: dark, tense, metal / cinematic.",
}


def tip(text, term=None, explanation=None):
    """Wrap text in a hover tooltip (an <abbr>). Works in the web app and most Markdown viewers."""
    title = explanation or GLOSSARY[term or text]
    return f'<abbr title="{html.escape(title, quote=True)}">{text}</abbr>'


def mode_tip(mode):
    return tip(mode, explanation=MODE_INFO.get(mode, mode))


def guide():
    items = "".join(f"<li><b>{html.escape(k)}</b> — {html.escape(v)}</li>" for k, v in GLOSSARY.items())
    modes = "".join(f"<li><b>{html.escape(k)}</b> — {html.escape(v)}</li>" for k, v in MODE_INFO.items())
    return ["<details><summary>📖 How to read this report</summary>", "",
            f"<ul>{items}</ul><p><b>Modes</b></p><ul>{modes}</ul>", "", "</details>", ""]


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
    L += [f"**{r['duration']}** · **{tempo}** · **{tip(r['key'], 'Key')}** "
          f"({tip(r['camelot'], 'Camelot')}) · **{mode_tip(r['mode'])}** · {rh['tempo_feel']} · {s['tone']}", ""]

    # Harmony
    conf = r["key_confidence"]
    conf_note = "clear" if conf >= 0.75 else "fairly clear" if conf >= 0.6 else "low — the harmony is ambiguous"
    mode_note = ("" if r["mode_confidence"] == "high" else
                 f"; could also be {r['mode_runner_up']}" if r["mode_confidence"] == "medium" else
                 f"; uncertain — {r['mode_runner_up']} fits almost as well")
    L += ["### Harmony", "",
          f"- **{tip('Key')}:** {r['key']} ({tip('confidence', 'Confidence')} {conf}: {conf_note}; "
          f"runner-up {r['runner_up_key']})",
          f"- **{tip('Mode')}:** {mode_tip(r['mode'])} ({r['mode_confidence']} confidence{mode_note}) — "
          f"{tip('colour note', 'Colour note')} **{r['color_note']}**",
          f"- **Scale:** {' '.join(r['scale_notes'])}",
          ]
    if r.get("drone"):
        who = " (the bass holds it)" if (r.get("bass") or {}).get("pedal_note") == r["scale_notes"][0] else ""
        L.append(f"- **{tip('Drone / pedal')}:** {r['scale_notes'][0]} sounds through {r['drone_pct']}% of the "
                 f"track{who}")
    if r.get("third_mix"):
        L.append(f"- **Major/minor blur:** both thirds ({r['thirds']}) are prominent")
    cents = r.get("tuning_cents", 0)
    if abs(cents) >= 10:
        L.append(f"- **{tip('Tuning')}:** about {abs(cents)} cents {'sharp' if cents > 0 else 'flat'} of A440 "
                 f"(A ≈ {440 * 2 ** (cents / 1200):.0f} Hz); the analysis corrects for this")
    L.append(f"- **Chords used most:** " + ", ".join(f"{c['chord']} ({fmt_time(c['seconds'])})" for c in r["chords_used"]))
    if r["harmonic_rhythm_beats"]:
        L.append(f"- **{tip('Harmonic rhythm')}:** a chord change every ~{r['harmonic_rhythm_beats']} beats")
    L += ["", "**Time in each key**", "", "| Key | Time | Share |", "|-----|------|-------|"]
    for k, v in r["time_in_key"].items():
        L.append(f"| {k} | {fmt_time(v['seconds'])} | {v['percent']}% |")
    if r["key_changes"]:
        L += ["", "**Key timeline:** " + " → ".join(f"{s_['key']} ({s_['start']}–{s_['end']})" for s_ in r["key_sections"])]
    if r["loops"]:
        L += ["", "**Chord loops**", ""]
        for lp in r["loops"]:
            L.append(f"- `{' – '.join(lp['chords'])}`  ({tip(' – '.join(lp['roman']), 'Roman numerals')} "
                     f"in {lp['key']}) "
                     f"· {', '.join(lp['where'])} · ×{lp['repeats']}")
    if r["chord_timeline"]:
        L += ["", "<details><summary>Full chord timeline (with bass notes as slash chords)</summary>", "",
              " · ".join(f"{c['time']} {c.get('with_bass', c['chord'])}" for c in r["chord_timeline"]), "",
              "</details>"]

    # Structure
    L += ["", "### Structure", "",
          f"| Section | Time | Length | Energy | Key | Main chords | Drums | Bass |",
          "|---------|------|--------|--------|-----|-------------|-------|------|"]
    for sec in r["sections"]:
        L.append(f"| **{sec['label']}** | {fmt_time(sec['start'])}–{fmt_time(sec['end'])} | "
                 f"{fmt_time(sec['end'] - sec['start'])} | {sec['energy']} | {sec['key']} | "
                 f"{', '.join(sec['main_chords']) or '—'} | {'✓' if sec.get('drums') else '—'} | "
                 f"{'✓' if sec.get('bass') else '—'} |")
    L += ["", f"{tip('Form')}: **{' '.join(sec['label'] for sec in r['sections'])}**"]
    L += instruments_report(r)

    # Rhythm & sound
    L += ["", "### Groove & sound", "",
          f"- **Pulse:** {rh['tempo_feel']}"
          + (f" (beat-to-beat variation {rh['tempo_variation_pct']}%)" if "tempo_variation_pct" in rh else ""),
          f"- **Rhythm:** {rh['busyness']} ({rh['onsets_per_sec']} notes/hits per sec)"
          + (f", {rh['groove']} ({rh['syncopation_pct']}% {tip('off-beat', 'Syncopation')})" if "groove" in rh else ""),
          f"- **{tip('Percussive', 'Percussive')} vs tonal:** {rh['percussive_pct']}% percussive energy",
          f"- **Tone:** {s['tone']} ({s['brightness_hz']} Hz centroid), {s['texture']}",
          f"- **Loudness:** avg {s['avg_loudness_db']} {tip('dBFS')}, {tip('dynamic range', 'Dynamic range')} "
          f"{s['dynamic_range_db']} dB",
          f"- **{tip('Energy arc', 'Energy')}:** `{sparkline(s['energy_curve'])}`"
          + (f" — biggest build into **{s['biggest_build']['time']}**" if s["biggest_build"] else "")
          + (f", quietest around **{s['quietest_moment']}**" if s["quietest_moment"] else ""),
          "", f"**{tip('Frequency balance')}**", "", "```"]
    for name, pct in s["frequency_balance_pct"].items():
        L.append(f"{name:<18} {bar(pct):<20} {pct:>5}%")
    L.append("```")

    # Ideas
    L += ["", "### 💡 Ideas to try", ""]
    for i in r["ideas"]:
        L.append(f"- **{i['area']}:** {i['idea']}")

    ai = r.get("ai")
    if ai:
        L += ["", "### 🤖 AI brainstorm", ""] + ([f"_Suggestions by {ai['model']}._", ""] if ai.get("model") else [])
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


def instruments_report(r):
    d, b = r.get("drums") or {}, r.get("bass") or {}
    if not d.get("present") and not b.get("present"):
        return []
    how = ("separated with Demucs (AI source separation)" if r.get("separation") == "demucs" else
           "separated with a lighter harmonic/percussive split (Demucs not installed)")
    L = ["", "### Instruments", "", f"_Drums and the rest were {how}._", ""]
    if d.get("present") and d.get("grid"):
        where = [sec["label"] for sec in r["sections"] if sec.get("drums")]
        parts = [f"plays in sections {' '.join(where)}" if where else "plays throughout",
                 f"{tip('groove repetition', 'Groove repetition')}: {d['repetition']}"]
        parts += d.get("traits", [])
        if d.get("fills"):
            parts.append("likely fills at " + ", ".join(fmt_time(t) for t in d["fills"]))
        g = d["grid"]
        rows = ["         1 e & a 2 e & a 3 e & a 4 e & a", "Drums    " + " ".join(g)]
        if d.get("hats"):
            rows.append("Cymbals  " + " ".join(d["hats"]))
        L += [f"**🥁 Drums** — " + " · ".join(parts), "", f"Main groove ({tip('how to read', 'Drum grid')}):", "",
              "```", *rows, "```", ""]
    elif d.get("present"):
        L += ["**🥁 Drums** — present, but no clear repeating pattern was found.", ""]
    if b.get("present"):
        notes = ", ".join(f"{n['note']} {n['pct']}%" for n in b["main_notes"])
        parts = [f"plays {b['plays_pct']}% of the time", f"range {b['range']}", f"style: {b['style']}",
                 f"main notes: {notes}", f"on the chord root {b['follows_roots_pct']}% of the time"]
        if b.get("riff"):
            parts.append(f"repeating riff: `{' '.join(b['riff'])}`")
        L += [f"**🎸 Bass** — " + " · ".join(parts), ""]
    else:
        L += ["**🎸 Bass** — no clear bass line detected.", ""]
    return L


def build_report(results):
    ok = [r for r in results if "error" not in r]
    L = ["# 🎵 Music Inspiration Report", "", f"_{len(results)} track(s) analysed_", ""]
    if ok:
        heads = ["Length", "Tempo", "Key", "Camelot", "Mode", "Form", "Energy"]
        L += ["| Track | " + " | ".join(tip(h) for h in heads) + " |",
              "|-------|" + "|".join("-" * (len(h) + 2) for h in heads) + "|"]
        for r in ok:
            tempo = f"{r['tempo_bpm']:.0f}" if r["tempo_bpm"] else "—"
            L.append(f"| {r['file']} | {r['duration']} | {tempo} | {r['key']} | {r['camelot']} | {mode_tip(r['mode'])} | "
                     f"{' '.join(s['label'] for s in r['sections'])} | `{sparkline(r['sound']['energy_curve'])}` |")
        L.append("")
        L += guide()
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
