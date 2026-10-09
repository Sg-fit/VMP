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

import inspiration
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

# Any OpenAI-compatible endpoint works (e.g. a self-hosted model); NVIDIA is the default.
NVIDIA_BASE_URL = os.environ.get("MUSIC_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Measured with --bench-llm on the free tier (Oct 2026): gpt-oss-20b 17 s, nemotron-3-super 22 s (both
# with their "think less" switch); gemma-4-31b timed out at 120 s, so it's no longer a default.
NVIDIA_MODEL = "openai/gpt-oss-20b"
# Tried in order if the main model times out or has been retired (comma-separated, overridable
# with MUSIC_FALLBACK_MODELS).
NVIDIA_FALLBACK_MODELS = "nvidia/nemotron-3-super-120b-a12b"
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
        if q in ("", "m"):
            # Don't claim major or minor without hearing a third: if neither third is clearly
            # present (open fifths, or a quiet melody note), call it a power chord.
            cols = [i for i in range(len(btimes) - 1) if s <= btimes[i] < e and i < sync.shape[1]]
            if cols:
                v = sync[:, cols].mean(axis=1)
                third = max(v[(root + 3) % 12], v[(root + 4) % 12])
                # Compare the third with the background level of notes that aren't in the chord (not with
                # the root, which a bass doubling makes very loud). Measured on chords with known
                # content: real thirds are >= 5x the background, missing ones <= 1.5x.
                chordish = {root, (root + 3) % 12, (root + 4) % 12, (root + 7) % 12, (root + 10) % 12, (root + 11) % 12}
                floor = float(np.median([v[i] for i in range(12) if i not in chordish])) + 1e-9
                if third < 3 * floor:
                    q = "5"
        chords.append({"start": round(float(s), 2), "end": round(float(e), 2),
                       "beats": max(1, int(round((e - s) / beat_len))), "root": root, "q": q})
    return chords


def detect_meter(rms_db, chroma, bounds, clear_beat):
    """3/4 or 4/4: do strong beats and chord changes recur every 3 beats or every 4?
    Only says 3/4 when the evidence is clearly stronger; otherwise 4/4 (the most common)."""
    import librosa
    default = {"beats_per_bar": 4, "label": "4/4", "confidence": "assumed"}
    if not clear_beat or len(bounds) < 14:
        return default
    loud = librosa.util.sync(rms_db[None, :], bounds, aggregate=np.max, pad=False)[0]
    accent = np.r_[0, np.diff(loud)]                      # rise in loudness at each beat
    C = librosa.util.sync(chroma, bounds, aggregate=np.median, pad=False)
    C = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    change = np.r_[0, 1 - np.sum(C[:, 1:] * C[:, :-1], axis=0)]  # harmonic change at each beat
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    feat = z(accent) + z(change)
    score = {m: max(feat[p::m].mean() - np.delete(feat, np.arange(p, len(feat), m)).mean() for p in range(m))
             for m in (3, 4)}
    if score[3] > score[4] + 0.3 and score[3] > 0.6:
        return {"beats_per_bar": 3, "label": "3/4", "confidence": "detected",
                "scores": {k: round(float(v), 2) for k, v in score.items()}}
    conf = "detected" if score[4] > score[3] + 0.3 else "assumed"
    return {"beats_per_bar": 4, "label": "4/4", "confidence": conf,
            "scores": {k: round(float(v), 2) for k, v in score.items()}}


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


def _analyze_audio(path):
    """The expensive part: everything that depends only on the audio (cacheable)."""
    import librosa

    y, sr = load_audio(path)
    if len(y) < 2 * SR:
        raise ValueError("audio is shorter than 2 seconds")
    duration = len(y) / sr
    # Split off the drums so they don't blur the harmony, and analyse each part on its own.
    y_perc, y_harm, separation = instruments.separate(y, sr)  # drums, everything else

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beat_frames) < 0.5 * duration * tempo / 60:
        # The default onset curve (a median across frequencies) is built for drums and nearly erases
        # soft onsets such as bowed strings or sustained piano; the standard (mean) curve keeps them.
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
        tempo2, frames2 = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP)
        if len(frames2) > len(beat_frames):
            tempo, beat_frames = float(np.atleast_1d(tempo2)[0]), frames2
    clear_beat = len(beat_frames) >= 8
    tracked_frames = beat_frames  # beats actually heard (the meter is judged on these only)
    if clear_beat:
        # The tracker often stops at the last clear hit; keep counting beats to the end so a final
        # ringing chord isn't lumped into one long "silent" span.
        period = int(np.median(np.diff(beat_frames)))
        end = librosa.time_to_frames(duration, sr=sr, hop_length=HOP)
        extra = np.arange(beat_frames[-1] + period, end - period // 2, period)
        beat_frames = np.concatenate([beat_frames, extra]).astype(int)
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

    # A key read as "X Mixolydian" only because of a prominent X7 chord that sits next to the chord a
    # fourth above it is, musically, that chord's key with X7 as its V7 (e.g. A - E7 is A major, not
    # E Mixolydian). Re-home the key in that case.
    if scale["mode"] == "Mixolydian" and chords:
        dom = sum(c["end"] - c["start"] for c in chords if c["q"] == "7" and c["root"] == main_tonic)
        target = (main_tonic + 5) % 12
        resolves = any(a["q"] == "7" and a["root"] == main_tonic and b["root"] == target or
                       b["q"] == "7" and b["root"] == main_tonic and a["root"] == target
                       for a, b in zip(chords, chords[1:]))
        if dom >= 0.15 * duration and resolves:
            old_key = main_key
            frames_t = np.concatenate([np.arange(int(s * fps), min(int(e * fps), n))
                                       for s, e, k in ksec if key_tonic(k) == main_tonic])
            scale = mode_and_scale(chroma, frames_t, target)
            main_key = scale.pop("key_index")
            main_tonic = target
            ksec = merge_runs([(s, e, main_key if k == old_key else k) for s, e, k in ksec])
            time_in_key = Counter()
            for s, e, k in ksec:
                time_in_key[k] += e - s
            scores = key_scores(chroma[:, frames_t].mean(axis=1))
            runner_up = old_key
            scale["mode_confidence"] = "medium" if scale["mode_confidence"] == "high" else scale["mode_confidence"]

    meter = detect_meter(rms_db, chroma,
                         librosa.util.fix_frames(tracked_frames[tracked_frames < n], x_min=0, x_max=n), clear_beat)
    bass, bass_line = instruments.bass_notes(y_harm, sr, btimes, tuning)  # per beat + note-by-note
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
    # Bass presence from low-end energy in the drumless audio (pitch tracking alone misses quiet
    # or busy bass on phone recordings): 40-200 Hz power and its ratio to the 200-2000 Hz band.
    spec = np.abs(librosa.stft(y_harm, n_fft=4096, hop_length=HOP)) ** 2
    spec_f = librosa.fft_frequencies(sr=sr, n_fft=4096)
    low_band = spec[(spec_f >= 40) & (spec_f < 200)].sum(axis=0)
    mid_band = spec[(spec_f >= 200) & (spec_f < 2000)].sum(axis=0)
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
        sec["bass_low_db"] = float(10 * np.log10(np.mean(low_band[a:b]) + 1e-12))
        sec["bass_ratio_db"] = float(10 * np.log10(np.mean(low_band[a:b]) / (np.mean(mid_band[a:b]) + 1e-12) + 1e-12))
    loudest_drums = max((sec["drums_level"] for sec in sections), default=0.0)
    if float(np.sum(y_perc ** 2)) < 0.03 * float(np.sum(y ** 2)):
        loudest_drums = 0.0  # no real drums anywhere (same rule as instruments.drum_analysis)
    loudest_low = max((sec["bass_low_db"] for sec in sections), default=0.0)
    for sec in sections:
        sec["drums"] = loudest_drums > 0 and sec.pop("drums_level") >= 0.1 * loudest_drums
        # Bass plays here if the low end is within 12 dB of its loudest section and isn't swamped by mids.
        sec["bass"] = sec.pop("bass_low_db") >= loudest_low - 12 and sec.pop("bass_ratio_db") >= -9

    result = {
        "file": Path(path).name,
        "duration_sec": round(duration, 1),
        "duration": fmt_time(duration),
        "tuning_cents": int(round(tuning * 100)),
        "tempo_bpm": round(tempo, 1) if clear_beat else None,
        "meter": meter,
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
        "harmonic_rhythm_beats": round(float(np.mean([c["beats"] for c in chords])), 1) if chords and clear_beat else None,
        "harmonic_rhythm_sec": round(float(np.mean([c["end"] - c["start"] for c in chords])), 1) if chords else None,
        "chord_timeline": [{"time": fmt_time(c["start"]), "chord": c["name"], "roman": c["roman"],
                            "with_bass": c.get("slash", c["name"]), "beats": c["beats"]} for c in chords],
        "loops": loops,
        "sections": sections,
        "rhythm": rhythm_features(y, y_perc, y_harm, beat_times, clear_beat, duration, sr),
        "sound": sound_features(y, sr, rms_db, duration),
        "separation": separation,
        "drums": instruments.drum_analysis(y_perc, sr, 256, beat_times, float(np.sum(y ** 2)),
                                           beats_per_bar=meter["beats_per_bar"])
        if clear_beat else {"present": False},
        "bass": bass_summary(bass, btimes, chords, main_key, duration),
    }
    if not any(sec["bass"] for sec in sections):  # no low end anywhere: the "bass notes" were chord tones
        result["bass"] = {"present": False}
        bass_line = []
    loudest = max((n[3] for n in bass_line), default=0.0)
    result["bass_transcription"] = [  # note-by-note, for the bass MIDI file
        {"start": round(a, 3), "end": round(e, 3), "midi": n, "velocity": int(np.clip(110 + 2.5 * (db - loudest), 35, 120))}
        for a, e, n, db in bass_line]

    # --- best-effort top line (the musician's typed melody is added later, in analyze())
    top, grid = instruments.top_line(y_harm, sr, btimes, tuning)
    result["top_line"] = top_line_summary(top, grid, chords, main_key)
    return result, {"chroma": chroma.astype(np.float32), "bounds": np.asarray(bounds)}


# Bump when the analysis changes, so cached results from older code aren't reused.
ANALYSIS_VERSION = "2026-10-10a"


def _cache_key(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return f"{h.hexdigest()[:32]}-{ANALYSIS_VERSION}-{instruments.separation_method()}"


def analyze(path, melody_text=None, cache_dir=None):
    """Analyse one track. The audio analysis (the slow part) is cached by file content, so
    re-running the same recording - e.g. to add a melody or retry the AI - is near-instant."""
    import pickle
    result = extras = None
    cache_file = None
    if cache_dir:
        cache_file = Path(cache_dir) / f"{_cache_key(path)}.pkl"
        try:
            with open(cache_file, "rb") as f:
                result, extras = pickle.load(f)
            result = {**result, "file": Path(path).name, "cached": True}
        except (OSError, pickle.PickleError, EOFError, ValueError):
            result = None
    if result is None:
        result, extras = _analyze_audio(path)
        if cache_file:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    pickle.dump((result, extras), f)
                tmp.replace(cache_file)
            except OSError:
                pass  # caching is an optimisation only

    main_key = result["key_index"]
    if melody_text:
        result["melody"] = inspiration.analyse_melody(melody_text, key_tonic(main_key), result["mode"], MODES,
                                                      lambda pc: spell(pc, main_key))
        if "error" not in result["melody"]:
            pcs = [n % 12 for n in result["melody"]["variations_midi"]["original"]]
            result["melody"]["check"] = check_melody(extras["chroma"], extras["bounds"], pcs, main_key)
    result["references"] = shared_dna(result)
    result["ideas"] = rule_based_ideas(result)
    return result


def prune_cache(cache_dir, max_age_days=30):
    """Delete cache entries not used for a while (called occasionally)."""
    import time as _time
    cutoff = _time.time() - max_age_days * 86400
    for p in Path(cache_dir).glob("*.pkl"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def check_melody(chroma, bounds, pcs, key):
    """Double-check a typed melody against the recording.

    1. Presence: does each note actually sound in the track (pitch-class strength)?
    2. Order: does the sequence fit the audio better than shuffled versions of the same notes,
       stepping one note per beat? (On demos where chords/drones dominate, it often can't tell.)"""
    import librosa
    profile = chroma.mean(axis=1)
    profile = profile / (profile.max() + 1e-9)
    rank = {pc: int(np.sum(profile > profile[pc])) + 1 for pc in range(12)}
    notes = [{"note": spell(pc, key), "strength": round(float(profile[pc]), 2), "rank": rank[pc]}
             for pc in dict.fromkeys(pcs)]
    weak = [n["note"] for n in notes if n["strength"] < 0.35]

    C = librosa.util.sync(chroma, bounds, aggregate=np.mean, pad=False)
    C = C / (C.max(axis=0, keepdims=True) + 1e-9)
    k = len(pcs)

    def fit(seq):
        idx = np.array(seq)
        vals = [C[idx, np.arange(t, t + k)].mean() for t in range(C.shape[1] - k)]
        return float(np.percentile(vals, 95)) if vals else 0.0
    order_pct = None
    if C.shape[1] > 2 * k and len(set(pcs)) > 1:
        rng = np.random.default_rng(0)
        mine = fit(pcs)
        shuffled = [fit(list(rng.permutation(pcs))) for _ in range(60)]
        order_pct = round(100 * float(np.mean([mine > x for x in shuffled])))

    if weak:
        verdict = (f"{', '.join(weak)} {'is' if len(weak) == 1 else 'are'} very quiet in the recording. If you're "
                   f"sure {'it is' if len(weak) == 1 else 'they are'} there, {'it is' if len(weak) == 1 else 'they are'} "
                   f"probably just mixed low; if not, double-check.")
    else:
        verdict = "All the notes you typed clearly sound in the recording."
    if order_pct is None:
        pass
    elif order_pct >= 80:
        verdict += " The order fits the audio better than shuffled versions — it looks right."
    else:
        verdict += (" The recording can't confirm the order or rhythm (it fits no better than shuffles of the "
                    "same notes) — typical when chords or a held note dominate a phone recording, so trust "
                    "your ears on the order.")
    return {"notes": notes, "weak_notes": weak, "order_beats_shuffles_pct": order_pct, "verdict": verdict}


def top_line_summary(notes, grid, chords, key):
    """Describe the loudest pitched line, and say honestly when it's just following the chords."""
    voiced = [(grid[i], n) for i, n in enumerate(notes) if n is not None]
    if len(voiced) < 16:
        return {"present": False}
    in_chord = []
    for t, n in voiced:
        c = next((c for c in chords if c["start"] <= t < c["end"]), None)
        if c:
            in_chord.append(n % 12 in {(c["root"] + iv) % 12 for iv in CHORD_INTERVALS[c["q"]]})
    chord_share = float(np.mean(in_chord)) if in_chord else 0.0
    events = [n for i, n in enumerate(notes) if n is not None and (i == 0 or n != notes[i - 1])]
    names = [spell(n % 12, key) for n in events]
    motifs = Counter(tuple(names[i:i + 4]) for i in range(len(names) - 3))
    motifs = [{"notes": list(m), "times": c} for m, c in motifs.most_common(3) if c >= 3 and len(set(m)) > 1]
    lo, hi = (int(v) for v in np.percentile([n for _, n in voiced], [10, 90]))
    pcs = Counter(spell(n % 12, key) for _, n in voiced)
    return {
        "present": True,
        "chord_tone_pct": round(100 * chord_share),
        "follows_chords": chord_share >= 0.75,
        "range": f"{note_name(lo, key)}–{note_name(hi, key)}",
        "main_notes": [{"note": k, "pct": round(100 * v / len(voiced))} for k, v in pcs.most_common(5)],
        "motifs": motifs,
    }


def shared_dna(r):
    """What the track has in common with well-known music: progressions, mode, groove, melody."""
    out = {"progressions": [], "mode": None, "groove": None, "melody": []}
    loops = [l["chords"] for l in r["loops"] if len(l["chords"]) >= 2]
    cyclic = bool(loops)
    if not loops:  # no repeating loop: the 4 most-used chords in the order they first appear (no wrap-around)
        top = [c["chord"] for c in r["chords_used"][:4]]
        first = list(dict.fromkeys(c["chord"] for c in r["chord_timeline"] if c["chord"] in top))
        loops = [first] if len(first) >= 2 else []
    seen = set()
    for names in loops:
        parsed = [parse_chord(n) for n in names]
        for m in inspiration.match_progression([(p[0], p[1]) for p in parsed if p], cyclic=cyclic):
            if m["progression"] not in seen:
                seen.add(m["progression"])
                out["progressions"].append({**m, "your_chords": names})
    if r["mode_confidence"] != "low" and r["mode"] in inspiration.MODE_REFERENCES:
        out["mode"] = {"mode": r["mode"], "examples": inspiration.MODE_REFERENCES[r["mode"]]}
    out["groove"] = inspiration.groove_reference((r.get("drums") or {}).get("traits", []))
    out["melody"] = [d for d in (r.get("melody") or {}).get("devices", []) if "“" in d["reference"]]
    return out


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
        "tracked_pct": round(100 * len(voiced) / len(notes)),
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


def _midi_header(bpm, cents, beats_per_bar=4, program=None):
    """Tempo, time signature, optional instrument (GM program) and tuning pitch bend."""
    us = int(60_000_000 / max(30, min(300, bpm or 90)))
    data = b"\x00\xff\x51\x03" + us.to_bytes(3, "big")
    data += b"\x00\xff\x58\x04" + bytes([beats_per_bar, 2, 24, 8])  # time signature, e.g. 3/4 or 4/4
    if program is not None:
        data += b"\x00" + bytes([0xC0, program])
    if cents:
        bend = int(np.clip(8192 + cents / 200 * 8192, 0, 16383))
        data += b"\x00" + bytes([0xE0, bend & 0x7F, bend >> 7])
    return data


def write_notes_midi(path, notes, bpm, cents=0, beats_per_bar=4, program=33):
    """Note-by-note MIDI at the real times: notes = [(start s, end s, midi, velocity)]. The tempo is the
    track's, so the notes line up with the recording when both start at 0:00. Program 33 = bass (finger)."""
    tpq = 480
    to_tick = lambda sec: int(round(sec * (bpm or 90) / 60 * tpq))
    events = []
    for a, e, n, v in notes:
        events += [(to_tick(a), 1, n, v), (max(to_tick(a) + 1, to_tick(e)), 0, n, 0)]
    events.sort(key=lambda x: (x[0], x[1]))
    data, last = _midi_header(bpm, cents, beats_per_bar, program), 0
    for t, on, n, v in events:
        data += _vlq(t - last) + bytes([0x90 if on else 0x80, n, v])
        last = t
    data += b"\x00\xff\x2f\x00"
    Path(path).write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq) +
                           b"MTrk" + struct.pack(">I", len(data)) + data)


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
    data, last = _midi_header(bpm, cents, beats_per_chord), 0
    for t, on, n in events:
        data += _vlq(t - last) + bytes([0x90 if on else 0x80, n, 90 if on else 0])
        last = t
    data += b"\x00\xff\x2f\x00"
    Path(path).write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq) +
                           b"MTrk" + struct.pack(">I", len(data)) + data)


def write_melody_midi(path, notes, bpm=90, cents=0, beats_per_note=1, repeats=2):
    """One note per beat (a sketch of the melody's pitches, not its exact rhythm)."""
    tpq, data, last, tick = 480, b"", 0, 0
    us = int(60_000_000 / max(30, min(300, bpm or 90)))
    data = b"\x00\xff\x51\x03" + us.to_bytes(3, "big")
    if cents:
        bend = int(np.clip(8192 + cents / 200 * 8192, 0, 16383))
        data += b"\x00" + bytes([0xE0, bend & 0x7F, bend >> 7])
    events = []
    for _ in range(repeats):
        for n in notes:
            events += [(tick, 1, n), (tick + beats_per_note * tpq - 10, 0, n)]
            tick += beats_per_note * tpq
    for t, on, n in events:
        data += _vlq(t - last) + bytes([0x90 if on else 0x80, n, 96 if on else 0])
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
    bar = (r.get("meter") or {}).get("beats_per_bar", 4)  # one chord = one full bar (3 beats in 3/4)
    files = []
    line = r.get("bass_transcription") or []
    if line:
        fname = f"{stem}_bass_transcription.mid"
        write_notes_midi(out_dir / fname, [(n["start"], n["end"], n["midi"], n["velocity"]) for n in line],
                         bpm, cents=cents, beats_per_bar=bar)
        files.append({"file": fname, "label": f"bass, note by note ({len(line)} notes at their real times)",
                      "chords": []})

    def save(label, names):
        parsed = [(p[0], p[2]) for p in map(parse_chord, names) if p]
        if len(parsed) < 2:
            return
        fname = f"{stem}_{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')}.mid"
        write_midi(out_dir / fname, parsed, bpm, beats_per_chord=bar, cents=cents)
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
    mel = r.get("melody") or {}
    for label, notes in (mel.get("variations_midi") or {}).items():
        fname = f"{stem}_melody_{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')}.mid"
        write_melody_midi(out_dir / fname, notes, bpm, cents=cents)
        files.append({"file": fname, "label": f"melody — {label}", "chords": mel["variations"][label]})
    for i, alt in enumerate((r.get("ai") or {}).get("alt_progressions", []), 1):
        save(f"ai {i} {alt.get('name', '')}", alt.get("chords", []))
    return files


# ---------------------------------------------------------------- optional AI brainstorm

BRAINSTORM_PROMPT = """You are a creative music producer helping a musician develop an instrumental demo.
Below is an automatic analysis of the track. Reply with ONLY this JSON (no other text):
{{"vibe": "<1-2 sentences on the feel the data suggests>",
 "ideas": [{{"area": "<harmony|melody|rhythm|arrangement|sound|structure>", "idea": "<under 30 words, specific: cite a time, section, chord or note>"}}],
 "alt_progressions": [{{"name": "<short>", "chords": ["<chord>", "..."], "why": "<one short sentence>"}}],
 "title_ideas": ["<title>", "<title>", "<title>"],
 "references": [{{"artist": "<artist>", "track": "<recording>", "shared": "<the one specific element it shares>"}}]}}

Rules: exactly 5 ideas, different from each other and from the "already suggested" list; no generic advice.
At most one rhythm idea and one sound/production idea; favour harmony, melody, arrangement and structure.
Only cite times that appear in the analysis (chord timeline, section starts, key moments) - never invent one.
When explaining a chord, be musically exact (e.g. a dominant 7th leads to the chord a fifth below it).
Titles: evocative but specific to this track; avoid stock words like "Midnight", "Echo", "Pulse".
2 alt_progressions in {key}, 3-6 chords each (names like Bm, F#7, Gmaj7), preferring chords the track doesn't use.
2 references only if you are certain the recording exists and really shares that element; otherwise [].
Use only facts given below; the analysis is approximate, so don't build on anything marked uncertain.

ANALYSIS
{summary}
"""

# How long to wait for the main model before also asking a backup model in parallel, and the
# overall limit for one brainstorm. Results are shown before the AI finishes, so the limit can be
# generous: NVIDIA's free models often need 60-90+ seconds.
AI_HEDGE_SECONDS = float(os.environ.get("MUSIC_AI_HEDGE_SECONDS", 30))
AI_DEADLINE_SECONDS = float(os.environ.get("MUSIC_AI_DEADLINE_SECONDS", 150))


def ai_summary(r):
    """A compact plain-text digest of the analysis (~1.5k characters instead of ~10k of JSON):
    smaller prompts are answered much faster and cut the chance of time-outs."""
    rh, s, d, b = r["rhythm"], r["sound"], r.get("drums") or {}, r.get("bass") or {}
    lines = [
        f"Length {r['duration']}, {r['tempo_bpm'] or 'free'} BPM ({rh['tempo_feel']}), {s['tone']} tone.",
        f"Key {r['key']} (confidence {r['key_confidence']}{', uncertain' if r['key_confidence'] < 0.6 else ''}); "
        f"mode {r['mode']} ({r['mode_confidence']} confidence); scale {' '.join(r['scale_notes'])}."
        + (f" Tuned {r['tuning_cents']:+d} cents." if abs(r.get('tuning_cents', 0)) >= 10 else ""),
        "Sections: " + "; ".join(f"{x['label']} {fmt_time(x['start'])}-{fmt_time(x['end'])} {x['energy']} energy "
                                 f"[{', '.join(x['main_chords'][:3])}]" for x in r["sections"]) + ".",
        "Most used chords: " + ", ".join(c["chord"] for c in r["chords_used"][:6]) + ".",
        "Chord timeline: " + ", ".join(f"{c['time']} {c['chord']}" for c in r["chord_timeline"][:28])
        + (" ..." if len(r["chord_timeline"]) > 28 else "") + ".",
    ]
    if r["loops"]:
        lines.append("Chord loops: " + "; ".join(f"{' - '.join(l['chords'])} ({' - '.join(l['roman'])})"
                                                 for l in r["loops"]) + ".")
    if d.get("grid"):
        lines.append(f"Drums: {', '.join(d.get('traits') or ['groove'])}; {d['repetition']} repetition; "
                     f"16-step grid {d['grid']} (K kick, S snare, X both)"
                     + (f"; fills at {', '.join(fmt_time(t) for t in d['fills'])}" if d.get("fills") else "; no fills")
                     + ".")
    if b.get("present"):
        lines.append(f"Bass: {b['style']}; main notes " + ", ".join(f"{n['note']} {n['pct']}%" for n in b["main_notes"][:3])
                     + ".")
    moments = _key_moments(r)
    if moments:
        lines.append("Key moments: " + "; ".join(f"{t} {what}" for t, what in moments) + ".")
    if s.get("biggest_build"):
        lines.append(f"Energy: biggest build into {s['biggest_build']['time']} (+{s['biggest_build']['db']} dB); "
                     f"quietest around {s['quietest_moment']}.")
    m = r.get("melody") or {}
    if m.get("notes"):
        lines.append(f"Melody typed by the musician: {' '.join(m['notes'])} (degrees {' '.join(m['degrees'])}; "
                     f"{m['shape']}).")
    refs = r.get("references") or {}
    known = [p["progression"] for p in refs.get("progressions", [])] + ([refs["mode"]["mode"]] if refs.get("mode") else [])
    if known:
        lines.append("Already compared with: " + "; ".join(known) + ".")
    lines.append("Already suggested: " + " | ".join(i["idea"][:60] for i in r["ideas"][:8]) + ".")
    return "\n".join(lines)


def _key_moments(r):
    """Times worth citing: section starts, drum fills, the biggest build and the quietest moment."""
    out = [(fmt_time(x["start"]), f"section {x['label']} starts") for x in r["sections"][1:]]
    out += [(fmt_time(t), "drum fill") for t in (r.get("drums") or {}).get("fills", [])]
    snd = r["sound"]
    if snd.get("biggest_build"):
        out.append((snd["biggest_build"]["time"], "biggest build"))
    if snd.get("quietest_moment"):
        out.append((snd["quietest_moment"], "quietest moment"))
    return sorted(out, key=lambda x: _to_sec(x[0]))


def _to_sec(ts):
    m, s = ts.split(":")
    return int(m) * 60 + int(s)


def _check_times(text, r, tolerance=3):
    """Flag an idea whose cited time doesn't fit the analysis: nothing happens then, or the time is
    named as part of a section (e.g. "1:45, just before section C") that is nowhere near it."""
    times = re.findall(r"\b(\d{1,2}:\d{2})\b", text)
    if not times:
        return text
    known = [_to_sec(c["time"]) for c in r["chord_timeline"]] + [_to_sec(t) for t, _ in _key_moments(r)]
    known += [x["start"] for x in r["sections"]] + [x["end"] for x in r["sections"]]
    problems = [f"nothing in the analysis happens at {t}" for t in times
                if not any(abs(_to_sec(t) - k) <= tolerance for k in known)]
    labels = {a or b for a, b in re.findall(r"\bsection ([A-J])\b|\b([A-J]) section\b", text)}
    for lab in labels:
        spans = [(x["start"], x["end"]) for x in r["sections"] if x["label"] == lab]
        if spans and not any(s0 - 5 <= _to_sec(t) <= e0 + 5 for t in times for s0, e0 in spans):
            where = ", ".join(f"{fmt_time(s0)}-{fmt_time(e0)}" for s0, e0 in spans)
            problems.append(f"section {lab} is at {where}")
    # "<chord> at m:ss": is that chord (same root) really playing around then?
    # Skip chords the AI suggests adding: the verb governs the chord directly ("play A7", "add a G#7").
    # With a preposition in between ("add G#7 before C#m"), the second chord is a claim to check.
    suggest = re.compile(r"\b(?:insert|add|play|use|try|layer|introduce|swap|put|with)\s+"
                         r"(?:(?!(?:before|after|over|under|on|into|to|than|from|of|at|in|the)\b)[\w#'-]+\s+){0,3}$", re.I)
    for m_ in re.finditer(r"\b([A-G][b#]?(?:maj7|m7|m|7|5)?)(?:\s+chord)?\s+at\s+(\d{1,2}:\d{2})\b", text):
        name, t = m_.group(1), m_.group(2)
        if suggest.search(text[max(0, m_.start() - 30):m_.start()]):
            continue  # a chord the AI suggests adding, not a claim about the recording
        want = parse_chord(name)
        if not want:
            continue
        near = [c for c in r["chord_timeline"] if abs(_to_sec(c["time"]) - _to_sec(t)) <= tolerance]
        if not any((parse_chord(c["chord"]) or (None,))[0] == want[0] for c in near):
            when = [c["time"] for c in r["chord_timeline"] if (parse_chord(c["chord"]) or (None,))[0] == want[0]][:3]
            problems.append(f"{name} isn't playing at {t}" + (f" (it's at {', '.join(when)})" if when else ""))
    return text + (f" (⚠ check this: {'; '.join(problems)})" if problems else "")


def make_llm():
    """Return (fn(prompt, info=None) -> text, label) for whichever API key is set, or (None, None).
    fn records which model answered in info["model"] (thread-safe: tracks are brainstormed in parallel)."""
    if os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        import openai
        if os.environ.get("NVIDIA_API_KEY"):
            client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=os.environ["NVIDIA_API_KEY"],
                                   timeout=AI_DEADLINE_SECONDS, max_retries=0)
            models = [os.environ.get("MUSIC_MODEL", NVIDIA_MODEL)] + [
                m.strip() for m in os.environ.get("MUSIC_FALLBACK_MODELS", NVIDIA_FALLBACK_MODELS).split(",") if m.strip()]
            label = "nvidia"
        else:
            client = openai.OpenAI(timeout=AI_DEADLINE_SECONDS, max_retries=0)
            models, label = [os.environ.get("MUSIC_MODEL", OPENAI_MODEL)], "openai"
        models = list(dict.fromkeys(models))

        def ask(model, prompt):
            return _chat(client, model, prompt)

        def call(prompt, info=None):
            return _race(models, lambda model: ask(model, prompt), info,
                         fatal=lambda e: getattr(e, "status_code", None) in (400, 401, 403),
                         complete=_complete_answer)
        return call, f"{label}/{models[0]}"

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic(timeout=AI_DEADLINE_SECONDS, max_retries=0)
        model = os.environ.get("MUSIC_MODEL", ANTHROPIC_MODEL)

        def call(prompt, info=None):
            m = client.messages.create(model=model, max_tokens=1200, temperature=0.7,
                                       messages=[{"role": "user", "content": prompt}])
            if info is not None:
                info["model"] = model
            return "".join(b.text for b in m.content if b.type == "text")
        return call, f"anthropic/{model}"

    return None, None


# "Reasoning" models think before answering; with a small token budget they can use it all up and
# return nothing. Each family has a documented plain-text switch to keep the thinking short.
REASONING_HINTS = {"gpt-oss": "Reasoning: low", "nemotron": "/no_think", "qwen3": "/no_think"}


def _chat(client, model, prompt, quick=True):
    """One chat request (OpenAI-compatible). quick=True asks reasoning models to skip long thinking
    and gives them more room to answer."""
    hint = next((h for k, h in REASONING_HINTS.items() if k in model.lower()), None)
    messages = ([{"role": "system", "content": hint}] if hint and quick else []) +         [{"role": "user", "content": prompt}]
    r = client.chat.completions.create(model=model, temperature=0.7, max_tokens=4096 if hint else 1200,
                                       messages=messages)
    return r.choices[0].message.content or ""


def _complete_answer(text):
    """A brainstorm reply is complete when it has several ideas and at least one progression
    (a reply cut off at the token limit usually loses the progressions at the end)."""
    obj = _parse_json_reply(re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL))
    return bool(obj) and len(obj.get("ideas") or []) >= 3 and bool(obj.get("alt_progressions"))


def _race(models, ask, info, fatal, complete=lambda text: True):
    """Ask models[0]; if it hasn't answered after AI_HEDGE_SECONDS (or fails), also ask the next
    model - in parallel, not after it - and take the first complete answer. A partial answer is kept
    as a last resort while the others finish. Gives up at AI_DEADLINE_SECONDS."""
    import concurrent.futures as cf
    import time as _time
    pool = cf.ThreadPoolExecutor(max_workers=len(models))
    start, nxt, pending, problems = _time.time(), 0, {}, []
    partial = None  # (text, model): a usable but incomplete answer, used only if nothing better arrives

    def launch():
        nonlocal nxt
        pending[pool.submit(ask, models[nxt])] = models[nxt]
        nxt += 1
    try:
        launch()
        while pending:
            left = AI_DEADLINE_SECONDS - (_time.time() - start)
            if left <= 0:
                break
            wait = min(left, AI_HEDGE_SECONDS) if nxt < len(models) else left
            done, _ = cf.wait(pending, timeout=wait, return_when=cf.FIRST_COMPLETED)
            for f in done:
                model = pending.pop(f)
                try:
                    text = f.result()
                except Exception as e:  # time-out, 5xx, retired model...
                    if fatal(e):
                        raise
                    problems.append(f"{model}: {type(e).__name__} {getattr(e, 'status_code', '') or ''}".strip())
                    continue
                if "{" in text and complete(text):
                    if info is not None:
                        info["model"] = model
                    return text
                if "{" in text:
                    partial = partial or (text, model)
                    problems.append(f"{model}: incomplete reply")
                else:
                    problems.append(f"{model}: empty reply")
            # Nothing usable yet (one failed, or the hedge timer ran out): bring in the next model.
            if nxt < len(models):
                launch()
        if partial:
            if info is not None:
                info["model"] = partial[1]
                info["partial"] = True
            return partial[0]
        problems += [f"{m}: no answer within {AI_DEADLINE_SECONDS:.0f}s" for m in pending.values()]
        raise RuntimeError("no model answered in time (" + "; ".join(problems) + ")")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # don't wait for slow stragglers


def ai_brainstorm(r, llm):
    prompt = BRAINSTORM_PROMPT.format(key=r["key"], summary=ai_summary(r))
    info = {}
    try:
        raw = llm(prompt, info)
    except Exception as e:
        return {"error": f"AI brainstorm unavailable: {str(e)[:220]}"}
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw)
    obj = _parse_json_reply(raw)
    if obj is None:
        return {"error": "AI reply could not be parsed", "raw": raw.strip()[:1500]}
    return {
        "model": info.get("model"),
        "partial": bool(info.get("partial")),
        "vibe": _clean(obj.get("vibe", "")),
        "ideas": [{"area": _clean(i.get("area", "idea")), "idea": _check_times(_clean(i["idea"]), r)}
                  for i in obj.get("ideas", []) if isinstance(i, dict) and i.get("idea")],
        "alt_progressions": [{"name": _clean(p.get("name", "")), "why": _clean(p.get("why", "")),
                              "chords": [_clean(c) for c in p["chords"]]}
                             for p in obj.get("alt_progressions", [])
                             if isinstance(p, dict) and isinstance(p.get("chords"), list)],
        "title_ideas": [_clean(t) for t in obj.get("title_ideas", [])][:5],
        "references": [{k: _clean(x.get(k, "")) for k in ("artist", "track", "shared")}
                       for x in obj.get("references", []) if isinstance(x, dict) and x.get("track")][:3],
    }


BENCH_SUMMARY = """Length 2:43, 83 BPM (steady, played by feel), dark/warm tone.
Key E major (confidence 0.83); mode Ionian (major) (medium confidence); scale E F# G# A B C# D#.
Sections: A 0:00-1:00 medium energy [A, E, Em]; B 1:00-1:14 high energy [Bm, C#m7, B]; A 1:14-2:11 medium energy [E, A, Em]; C 2:11-2:43 medium energy [E, C#m].
Most used chords: E, A, Em, C#m, B, Bm.
Chord loops: Em - A (i - IV).
Drums: backbeat (snare on 2 and 4); tight repetition; 16-step grid X...S...X...S... (K kick, S snare, X both); no fills.
Bass: plays the chord roots; main notes E 73%, A 16%, C# 3%.
Energy: biggest build into 2:17 (+14.8 dB); quietest around 2:13."""


def bench_llm():
    """Time each configured model on a realistic prompt - with and without the 'think less' switch for
    reasoning models - and print the .env settings for the fastest reliable setup."""
    import math
    import time as _time
    if not os.environ.get("NVIDIA_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        if os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("--bench-llm compares NVIDIA/OpenAI models; with Anthropic there is only one model to use.")
        sys.exit("Set NVIDIA_API_KEY or OPENAI_API_KEY first.")
    import openai
    prompt = BRAINSTORM_PROMPT.format(key="E major", summary=BENCH_SUMMARY)
    if os.environ.get("NVIDIA_API_KEY"):
        client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=os.environ["NVIDIA_API_KEY"], timeout=120, max_retries=0)
        names = [os.environ.get("MUSIC_MODEL", NVIDIA_MODEL)] +             [m.strip() for m in (NVIDIA_FALLBACK_MODELS + "," + os.environ.get("MUSIC_BENCH_MODELS", "")).split(",")]
    else:
        client = openai.OpenAI(timeout=120, max_retries=0)
        names = [os.environ.get("MUSIC_MODEL", OPENAI_MODEL)] + os.environ.get("MUSIC_BENCH_MODELS", "").split(",")
    models = [m for m in dict.fromkeys(n.strip() for n in names) if m]
    print(f"Testing {len(models)} model(s), one request at a time ({len(prompt)}-character prompt, 120 s limit).\n")
    results = []
    for model in models:
        reasoning = any(k in model.lower() for k in REASONING_HINTS)
        for quick in ([True, False] if reasoning else [True]):
            label = model + ("  (think less)" if reasoning and quick else "  (normal)" if reasoning else "")
            t = _time.time()
            try:
                text = _chat(client, model, prompt, quick=quick)
                secs = _time.time() - t
                usable = _parse_json_reply(text) is not None
                note = "usable" if usable else ("EMPTY reply" if not text.strip() else "reply not usable")
            except Exception as e:
                secs, usable = _time.time() - t, False
                note = f"{type(e).__name__} {getattr(e, 'status_code', '') or ''}".strip()
            print(f"  {label:55} {secs:6.1f}s   {note}")
            results.append((model, quick, secs, usable))
    good = sorted([r for r in results if r[3]], key=lambda r: r[2])
    if not good:
        print("\nNo model gave a usable answer. NVIDIA's free models may be overloaded right now - try again in a "
              "while, try others with MUSIC_BENCH_MODELS=model1,model2, or use an OpenAI/Anthropic key.")
        return
    order = list(dict.fromkeys(r[0] for r in good))
    # Real jobs ask for two tracks at once and free endpoints slow down under load: leave headroom.
    deadline = min(180, max(90, math.ceil(good[0][2] * 4 / 10) * 10))
    print("\nRecommended .env settings (then run: docker compose up -d):\n")
    print(f"MUSIC_MODEL={order[0]}")
    if len(order) > 1:
        print(f"MUSIC_FALLBACK_MODELS={','.join(order[1:])}")
    print(f"MUSIC_AI_DEADLINE_SECONDS={deadline}")


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
    "Scale degree": "A note's position in the key: 1 is the home note, 5 the fifth above it, b6 a flattened "
                    "sixth, and so on. Degrees describe a melody independently of the key.",
    "Top line": "The loudest pitched part, found automatically. In a full-band phone recording this is often the "
                "chord instrument rather than the melody.",
    "Variations": "Classic ways to develop a motif: inversion flips its intervals upside down, retrograde plays "
                  "it backwards, a sequence repeats it a step higher, an answer phrase resolves to the home note.",
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
    meter = r.get("meter") or {}
    if r["tempo_bpm"] and meter.get("label"):
        tempo += f" in {meter['label']}" + (" (waltz feel)" if meter["beats_per_bar"] == 3 else "") + \
            (" (assumed)" if meter.get("confidence") == "assumed" else "")
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
    elif r.get("harmonic_rhythm_sec"):
        L.append(f"- **{tip('Harmonic rhythm')}:** a chord change every ~{r['harmonic_rhythm_sec']} seconds")
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
    L += melody_report(r)
    L += dna_report(r)

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
        L += ["", "### 🤖 AI brainstorm", ""] + ([f"_Suggestions by {ai['model']}"
                                                + (" (its reply was cut short, so some parts may be missing)" if ai.get("partial") else "")
                                                + "._", ""] if ai.get("model") else [])
        if ai.get("pending"):
            L.append("_⏳ The AI is still writing ideas for this track — they'll appear here in a moment._")
        elif ai.get("error"):
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
            L.append(f"- `midi/{m['file']}` — {m['label']}" + (f": {' – '.join(m['chords'])}" if m["chords"] else ""))
    L.append("")
    return L


def melody_report(r):
    L = ["", "### 🎶 Melody", ""]
    m = r.get("melody")
    if m and m.get("error"):
        L += [f"_{m['error']}_", ""]
    elif m:
        chk = m.get("check") or {}
        L += [f"**Your melody:** `{' '.join(m['notes'])}`", ""]
        if chk:
            strengths = ", ".join(f"{n['note']} (#{n['rank']} of 12)" for n in chk["notes"])
            L += [f"> **Checked against the recording:** {chk['verdict']} "
                  f"_Note strength rank in this track: {strengths}._", ""]
        L += [
              f"- **{tip('Scale degrees', 'Scale degree')}:** {' '.join(m['degrees'])} (1 = {r['scale_notes'][0]}, the home note)",
              f"- **Fits:** {', '.join(mode_tip(x) for x in m['fits_modes']) or 'no single mode — it mixes notes from the major and minor scales on ' + r['scale_notes'][0]}"
              + (f" · outside the detected scale: {', '.join(m['outside_detected_scale'])}" if m["outside_detected_scale"] else ""),
              f"- **Shape:** {m['shape']} · range {m['range']} · {m['steps_pct']}% steps, {m['leaps_pct']}% leaps",
              f"- **Moves:** {', '.join(m['intervals'])}"]
        if m["harmonize"]:
            L.append("- **Chords that contain each note:** " +
                     " · ".join(f"{n} → {', '.join(cs) or '—'}" for n, cs in m["harmonize"].items()))
        if m["devices"]:
            L += ["", "**Melodic devices**", ""]
            L += [f"- {d['device']} — {d['reference']}" for d in m["devices"]]
        L += ["", f"**{tip('Variations', 'Variations')}** (MIDI files below)", ""]
        L += [f"- {name}: `{' '.join(notes)}`" for name, notes in m["variations"].items()]
        L.append("")
    t = r.get("top_line") or {}
    if t.get("present"):
        motif = "; ".join(f"`{' '.join(x['notes'])}` ×{x['times']}" for x in t["motifs"]) or "none clearly repeating"
        main = ", ".join(f"{n['note']} {n['pct']}%" for n in t["main_notes"])
        verdict = ("it's {0}% chord notes, so it's most likely following the chords rather than a separate "
                   "melody".format(t["chord_tone_pct"]) if t["follows_chords"] else
                   f"{t['chord_tone_pct']}% chord notes — likely a real melodic line")
        L += [f"**{tip('Top line', 'Top line')} (automatic):** range {t['range']} · main notes {main} · "
              f"repeating 4-note figures: {motif} · {verdict}."]
        if not m:
            L += ["", "_For a precise melody analysis, type the melody or riff when uploading "
                  "(e.g. `A E F E A E D E`)._"]
        L.append("")
    return L


def dna_report(r):
    d = r.get("references") or {}
    ai_refs = (r.get("ai") or {}).get("references") or []
    if not (d.get("progressions") or d.get("mode") or d.get("groove") or d.get("melody") or ai_refs):
        return []
    L = ["", "### 🎧 Shared DNA with well-known music", "",
         "_Not “sounds like” — specific building blocks this track shares with recordings worth studying._", ""]
    for p in d.get("progressions", []):
        how = "uses the same loop as" if p["match"] == "same loop" else "contains"
        L.append(f"- **Chords:** your `{' – '.join(p['your_chords'])}` {how} **{p['progression']}** — "
                 + ", ".join(p["examples"]))
    if d.get("mode"):
        L.append(f"- **Scale:** {mode_tip(d['mode']['mode'])} is the sound of " + ", ".join(d["mode"]["examples"]))
    if d.get("groove"):
        L.append(f"- **Groove:** {d['groove']}")
    for m in d.get("melody", []):
        L.append(f"- **Melody:** {m['device']} — {m['reference']}")
    if ai_refs:
        L += ["", "_AI suggestions — double-check before relying on them:_", ""]
        L += [f"- **{x.get('artist', '?')} — “{x.get('track', '?')}”:** {x.get('shared', '')}" for x in ai_refs]
    L.append("")
    return L


def instruments_report(r):
    d, b = r.get("drums") or {}, r.get("bass") or {}
    if not d.get("present") and not b.get("present"):
        return []
    how = ("separated with Demucs (AI source separation)" if r.get("separation") == "demucs" else
           "separated with a lighter harmonic/percussive split (Demucs AI separation not in use)")
    L = ["", "### Instruments", "", f"_Drums and the rest were {how}._", ""]
    if d.get("present") and d.get("grid"):
        where = [sec["label"] for sec in r["sections"] if sec.get("drums")]
        parts = [f"plays in sections {' '.join(where)}" if where else "plays throughout",
                 f"{tip('groove repetition', 'Groove repetition')}: {d['repetition']}"]
        parts += d.get("traits", [])
        if d.get("fills"):
            parts.append("likely fills at " + ", ".join(fmt_time(t) for t in d["fills"]))
        g = d["grid"]
        counts = " ".join(f"{b + 1} e & a" for b in range(len(g) // 4))  # 4 beats in 4/4, 3 in 3/4
        rows = ["         " + counts, "Drums    " + " ".join(g)]
        if d.get("hats"):
            rows.append("Cymbals  " + " ".join(d["hats"]))
        L += [f"**🥁 Drums** — " + " · ".join(parts), "", f"Main groove ({tip('how to read', 'Drum grid')}):", "",
              "```", *rows, "```", ""]
    elif d.get("present"):
        L += ["**🥁 Drums** — present, but no clear repeating pattern was found.", ""]
    if b.get("present"):
        notes = ", ".join(f"{n['note']} {n['pct']}%" for n in b["main_notes"])
        where = [sec["label"] for sec in r["sections"] if sec.get("bass")]
        parts = [f"plays in sections {' '.join(where)}" if where else "barely audible",
                 f"clear notes on {b['tracked_pct']}% of beats", f"range {b['range']}", f"style: {b['style']}",
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
          "not as a transcription._", "", f"_Analyzer version {CODE_VERSION}._"]
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


def _timed_brainstorm(r, llm):
    """ai_brainstorm, plus one log line per track (visible in `docker compose logs`)."""
    import time as _time
    t0 = _time.time()
    ai = ai_brainstorm(r, llm)
    outcome = f"ideas from {ai.get('model')}" if not ai.get("error") else f"FAILED: {ai['error']}"
    print(f"[ai] {r['file']}: {outcome} ({_time.time() - t0:.1f}s)", file=sys.stderr, flush=True)
    return ai


DEFAULT_CACHE_DIR = Path(os.environ.get("MUSIC_CACHE_DIR", Path.home() / ".cache" / "music-analyzer"))


def _code_version():
    import hashlib
    h = hashlib.sha256()
    for name in ("musicanalyze.py", "instruments.py", "inspiration.py"):
        try:
            h.update(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n"))
        except OSError:
            pass
    return h.hexdigest()[:8]


CODE_VERSION = _code_version()


def _write_outputs(results, out):
    report = build_report(results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "music_report.md").write_text(report, encoding="utf-8")
    (out / "music_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_analysis(files, out, llm=None, progress=None, melody_text=None, on_analysis_done=None,
                 cache_dir=DEFAULT_CACHE_DIR):
    """Analyse files, write music_report.md / music_results.json / midi/*.mid into `out`.

    Built to keep the musician waiting as little as possible:
    - the slow audio analysis is cached by file content (re-runs are near-instant);
    - each track's AI brainstorm runs in the background while the next track is analysed;
    - on_analysis_done(report) is called as soon as the analysis is finished, before the AI
      ideas arrive, so a UI can show results immediately and add the AI section later.
    Returns (results, report_markdown)."""
    import concurrent.futures as cf
    progress = progress or (lambda msg: None)
    out = Path(out)
    results, pending = [], []
    ai_pool = cf.ThreadPoolExecutor(max_workers=2) if llm else None  # free endpoints queue bursts
    try:
        for i, f in enumerate(files, 1):
            f = Path(f)
            progress(f"[{i}/{len(files)}] Analysing {f.name}...")
            try:
                r = analyze(f, inspiration.melody_for(f.name, melody_text), cache_dir=cache_dir)
                r["analyzer_version"] = CODE_VERSION
                r["midi"] = export_midis(r, out / "midi")
                if ai_pool:
                    pending.append((r, ai_pool.submit(_timed_brainstorm, r, llm)))  # overlaps the next track
            except Exception as e:  # unreadable / corrupt file — keep going
                progress(f"  ! {f.name}: {e}")
                r = {"file": f.name, "error": "unreadable or unsupported audio" if "Error opening" in str(e)
                     or "Invalid data" in str(e) else str(e)[:200]}
            results.append(r)

        if pending:
            for r, _ in pending:
                r["ai"] = {"pending": True}
            report = _write_outputs(results, out)
            if on_analysis_done:
                on_analysis_done(report)
            for n, (r, fut) in enumerate(pending, 1):
                progress(f"Analysis done — waiting for AI ideas ({n}/{len(pending)})...")
                r["ai"] = fut.result()
                r["midi"] = export_midis(r, out / "midi")  # adds MIDI for the AI's progressions
    finally:
        if ai_pool:
            ai_pool.shutdown(wait=False, cancel_futures=True)
    if cache_dir:
        prune_cache(cache_dir)
    report = _write_outputs(results, out)
    return results, report


def main():
    ap = argparse.ArgumentParser(description="Analyse instrumental tracks and generate ideas for musicians.")
    ap.add_argument("paths", nargs="*", help="audio files and/or folders")
    ap.add_argument("--out", default=".", help="output folder for the report, JSON and MIDI files")
    ap.add_argument("--no-llm", action="store_true", help="skip the AI brainstorm")
    ap.add_argument("--melody", help="melody/riff notes to analyse, e.g. \"A E F E A E D E\" "
                                     "(prefix with part of a file name to target one track: \"copy 4: A E F E\")")
    ap.add_argument("--no-cache", action="store_true", help="re-analyse even if this recording was analysed before")
    ap.add_argument("--bench-llm", action="store_true", help="time the AI models and recommend the fastest")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if args.bench_llm:
        bench_llm()
        return
    files = collect_files(args.paths)
    if not files:
        sys.exit("No audio files found.")

    llm, llm_label = (None, None) if args.no_llm else make_llm()
    if llm:
        print(f"AI brainstorm via {llm_label} (only the analysis numbers are sent, never the audio).", file=sys.stderr)
    elif not args.no_llm:
        print("No NVIDIA/Anthropic/OpenAI key set — skipping the AI brainstorm.", file=sys.stderr)

    out = Path(args.out)
    _, report = run_analysis(files, out, llm, progress=lambda msg: print(msg, file=sys.stderr),
                             melody_text=args.melody, cache_dir=None if args.no_cache else DEFAULT_CACHE_DIR)
    print(report)
    print(f"\nWrote {out / 'music_report.md'}, {out / 'music_results.json'} and MIDI files in {out / 'midi'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
