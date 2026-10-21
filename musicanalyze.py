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


def _private_numba_cache():
    """Give this process its own numba cache folder. Processes sharing one folder (the app's worker warming
    up while the deploy self-test runs) can load code another process is still writing and crash inside
    librosa's beat tracker with a segmentation fault. Must run before numba is imported."""
    import atexit
    import shutil
    base = os.environ.get("NUMBA_CACHE_DIR")
    if not base or "numba" in sys.modules:
        return
    mine = os.path.join(base, f"pid{os.getpid()}")
    os.environ["NUMBA_CACHE_DIR"] = mine
    atexit.register(shutil.rmtree, mine, True)


_private_numba_cache()

import electronic
import inspiration
import instruments
import notation

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
# chord_clarity() below this: the chord's notes are no louder than the other notes around them (shown with a ?)
UNCLEAR_CHORD = 1.0
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
                       "beats": max(1, int(round((e - s) / beat_len))), "root": root, "q": q,
                       "clarity": round(chord_clarity(sync, btimes, s, e, root, q), 2)})
    return chords


def chord_clarity(sync, btimes, s, e, root, q):
    """How much the chord's own notes stand out from the other notes over its span: the quietest chord tone
    divided by the typical (median) non-chord note. A strummed or held chord is well above 1; a melodic run or a
    blues lick, where five or six notes sound about equally, sits near 1 - a chord name there is a guess."""
    cols = [i for i in range(len(btimes) - 1) if s <= btimes[i] < e and i < sync.shape[1]]
    if not cols:
        return 0.0
    v = sync[:, cols].mean(axis=1)
    ivs = DETECT_QUALITIES.get(q, ([0, 7], 1.0))[0]
    core = {(root + i) % 12 for i in ivs[:3]}  # root, third, fifth: a 7th is often played quietly, or left out
    others = [v[i] for i in range(12) if (i - root) % 12 not in ivs]
    return float(min(v[i] for i in core) / (np.median(others) + 1e-9))


def detect_meter(rms_db, chroma, bounds, clear_beat, allow_odd=True):
    """3/4, 4/4, 5/4 or 7/4: do strong beats and chord changes recur every 3, 4, 5 or 7 beats?
    Only leaves 4/4 (the most common) when another meter is clearly stronger."""
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
    # Odd meters (5/4, 7/4) are only considered with enough bars to see them repeat.
    candidates = (3, 4, 5, 7) if allow_odd and len(feat) >= 8 * 7 else (3, 4)
    by_phase = {m: [feat[p::m].mean() - np.delete(feat, np.arange(p, len(feat), m)).mean() for p in range(m)]
                for m in candidates}
    score = {m: max(v) for m, v in by_phase.items()}
    m = 3 if score[3] > score[4] + 0.3 and score[3] > 0.6 else 4
    for odd in (5, 7):  # rare, so they must beat every other meter clearly
        if odd in score and score[odd] > 0.8 and all(score[odd] > v + 0.4 for k_, v in score.items() if k_ != odd):
            m = odd
    others = [v for k_, v in score.items() if k_ != m]
    conf = "detected" if m != 4 or score[4] > max(others) + 0.3 else "assumed"
    # Beat segment i starts at bounds[i]; the strongest phase marks the first beat of each bar.
    phase = int(np.argmax(by_phase[m]))
    return {"beats_per_bar": m, "label": f"{m}/4", "confidence": conf,
            "scores": {k: round(float(v), 2) for k, v in score.items()},
            "downbeat_frame": int(bounds[phase + m]) if phase + m < len(bounds) else None}


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
    # About one candidate boundary per 20 s: a 4-5 minute song needs ~14 to find a solo or an outro (with 8, a
    # chorus + solo + final chorus came out as one section). Neighbours that turn out alike are merged below.
    k = int(np.clip(round(duration / 20) + 1, 2, 14))
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
        letters.append(lab or "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[min(25, len(set(letters)))])

    # Neighbouring sections with the same letter are one section (the clustering over-split it).
    merged = []
    for (s, e), lab, en in zip(spans, letters, energies):
        if merged and merged[-1][2] == lab:
            ps, pe, _, pen = merged[-1]
            merged[-1] = (ps, e, lab, (pen * (pe - ps) + en * (e - s)) / (e - ps))
        else:
            merged.append((s, e, lab, en))
    spans, energies = [(s, e) for s, e, _, _ in merged], [m[3] for m in merged]
    order = list(dict.fromkeys(m[2] for m in merged))  # re-letter A, B, C... in order of appearance
    letters = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[order.index(m[2])] for m in merged]

    avg = float(np.mean(energies))
    out = []
    for (s, e), lab, en in zip(spans, letters, energies):
        level = "high" if en > avg + 2 else "low" if en < avg - 3 else "medium"
        out.append({"label": lab, "start": round(float(s), 1), "end": round(float(e), 1), "energy": level,
                    "energy_db": round(en, 1)})
    return out


SPLIT_MIN_SEC = 8.0   # the drums stopping (or starting) for at least this long is a section change of its own
REFINE_SEC = 4.0      # a boundary may move this far to the sharpest change in level nearby
REFINE_GAIN_DB = 1.5  # ...when the change there is clearly sharper than where the boundary was
REFINE_MIN_DB = 6.0   # ...and is a real level event: 6 dB summed over drums, low end and harmony (a part coming in)
MIN_SECTION_SEC = 4.0


def refine_sections(sections, curves, fps, btimes, duration, rms_db, drums=True, key_changes=()):
    """Sections come from how the harmony and timbre repeat, which can miss what dance music is built from: the
    drums dropping out for a breakdown while the chords carry on, and changes that land a bar either side of
    where the clustering put them. So: split a section where the drums stop or start for SPLIT_MIN_SEC or more,
    then move each boundary to the sharpest change in level (drums, low end, harmony) within REFINE_SEC, snapped
    to a beat. A boundary on a key change stays (detect_structure snapped it there: key changes are more precise).
    curves: per-frame power of the drums, the low end and the harmony."""
    if len(sections) < 1 or duration < 4 * SPLIT_MIN_SEC:
        return sections
    win = max(1, int(0.5 * fps))
    smooth = [np.convolve(c, np.ones(win) / win, mode="same") for c in curves]
    db = [10 * np.log10(c + 1e-12) for c in smooth]
    cuts = [x["start"] for x in sections[1:]]
    drum_cuts = set()
    if drums:
        d = smooth[0]
        on = d >= 0.1 * np.percentile(d, 90)  # within 10 dB of the drums' usual level
        runs, start = [], 0
        for i in range(1, len(on) + 1):
            if i == len(on) or on[i] != on[start]:
                runs.append([start, i, bool(on[start])])
                start = i
        merged = []
        for r in runs:  # runs shorter than SPLIT_MIN_SEC belong to their neighbours
            if merged and (r[1] - r[0] < SPLIT_MIN_SEC * fps or r[2] == merged[-1][2]):
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        if merged and merged[0][1] - merged[0][0] < SPLIT_MIN_SEC * fps and len(merged) > 1:
            merged[1][0] = merged[0][0]
            merged = merged[1:]
        for a, b in zip(merged, merged[1:]):
            t = a[1] / fps
            if a[2] != b[2] and not any(abs(t - c) < REFINE_SEC for c in cuts):
                cuts.append(t)
                drum_cuts.add(t)
        cuts.sort()

    def novelty(i):
        w = int(2 * fps)
        if i - w < 0 or i + w > len(db[0]):
            return 0.0
        return float(sum(abs(c[i:i + w].mean() - c[i - w:i].mean()) for c in db))
    moved = []
    for ci, t in enumerate(cuts):
        i = int(t * fps)
        lo, hi = max(0, int((t - REFINE_SEC) * fps)), min(len(db[0]), int((t + REFINE_SEC) * fps))
        cand = range(lo, hi, max(1, int(fps / 20)))
        best = max(cand, key=novelty, default=i)
        from_drums = t in drum_cuts
        on_key_change = any(abs(t - k) < 0.5 for k in key_changes)
        if not on_key_change and novelty(best) >= max(novelty(i) + REFINE_GAIN_DB, REFINE_MIN_DB):
            t = best / fps
        if not on_key_change:
            t = float(btimes[int(np.argmin(np.abs(btimes - t)))]) if len(btimes) else t
        # a split made because the drums stopped must still leave SPLIT_MIN_SEC either side once it has moved
        least = SPLIT_MIN_SEC if from_drums else MIN_SECTION_SEC
        nxt = cuts[ci + 1] if ci + 1 < len(cuts) else duration
        if least <= t <= duration - least and (not moved or t - moved[-1] >= least) \
                and (not from_drums or nxt - t >= least):
            moved.append(t)
    # rebuild the sections on the new boundaries; each piece keeps the letter of the old section it mostly
    # overlaps, and a piece split off from a longer one gets a letter of its own
    bounds = [0.0] + moved + [duration]
    out, used = [], {x["label"] for x in sections}
    owner = []
    for s_, e_ in zip(bounds[:-1], bounds[1:]):
        best = max(sections, key=lambda x: min(e_, x["end"]) - max(s_, x["start"]))
        owner.append(best)
    for k, ((s_, e_), sec) in enumerate(zip(zip(bounds[:-1], bounds[1:]), owner)):
        label = sec["label"]
        same = [j for j, o in enumerate(owner) if o is sec]
        if len(same) > 1 and k != max(same, key=lambda j: bounds[j + 1] - bounds[j]):
            label = next(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used)
            used.add(label)
        a, b = int(s_ * fps), max(int(e_ * fps), int(s_ * fps) + 1)
        out.append({"label": label, "start": round(s_, 1), "end": round(e_, 1),
                    "energy_db": round(float(np.mean(rms_db[a:b])), 1)})
    order = list(dict.fromkeys(x["label"] for x in out))  # letters in order of appearance again
    avg = float(np.mean([x["energy_db"] for x in out]))
    for x in out:
        x["label"] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[order.index(x["label"])]
        x["energy"] = "high" if x["energy_db"] > avg + 2 else "low" if x["energy_db"] < avg - 3 else "medium"
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


OCTAVE_CENTRES = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000)


def sound_features(y, sr, rms_db, duration):
    import librosa
    S = np.abs(librosa.stft(y, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    band_edges = {"sub (<60 Hz)": (0, 60), "bass (60–250)": (60, 250), "low-mid (250–2k)": (250, 2000),
                  "high-mid (2k–6k)": (2000, 6000), "air (>6k)": (6000, sr / 2 + 1)}
    total = S.sum() + 1e-9
    bands = {name: round(100 * float(S[(freqs >= lo) & (freqs < hi)].sum() / total), 1)
             for name, (lo, hi) in band_edges.items()}
    # Brightness from the magnitude spectrum (S is power): the power-weighted centre sits at 400-500 Hz for almost
    # any music, so every track read "dark/warm". This is also the measure the per-section mood table uses.
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)))
    # The tonal balance in octaves (the shape reference plugins draw), each band's share of the total in dB: a
    # level-independent picture, so two tracks of different loudness can be laid over each other.
    octaves = {f"{c:g}": round(float(10 * np.log10(S[(freqs >= c / 2 ** 0.5) & (freqs < c * 2 ** 0.5)].sum() / total
                                                    + 1e-9)), 1)
               for c in OCTAVE_CENTRES if c * 2 ** 0.5 <= sr / 2 * 1.05}
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
        "octave_db": octaves,
        "brightness_hz": round(centroid),
        "tone": "dark/warm" if centroid < 1500 else "balanced" if centroid < 3000 else "bright",
        "texture": "tonal / clean" if flatness < 0.02 else "mixed" if flatness < 0.1 else "noisy / textural",
        "avg_loudness_db": round(float(np.mean(audible)), 1),
        "dynamic_range_db": round(float(np.percentile(audible, 95) - np.percentile(audible, 5)), 1),
        "energy_curve": [round(float(x), 2) for x in curve / (curve.max() + 1e-9)],
        # the same, finer, for the timeline picture (loudness in dB above the quietest audible level, 0-1)
        "energy_line": [round(float(x), 3) for x in np.clip(
            (np.array([c.mean() for c in np.array_split(rms_db, 160)]) - (rms_db.max() - 40)) / 40, 0, 1)],
        "energy_flat": bool(np.std(curve / (curve.max() + 1e-9)) < 0.08),
        "biggest_build": {"time": fmt_time(rise_t), "db": round(float(rise_db), 1)} if rise_t and rise_db > 3 else None,
        "quietest_moment": fmt_time(quiet_t) if quiet_t is not None else None,
    }


# ---------------------------------------------------------------- main analysis

def load_audio_channels(path):
    """All channels at the file's own sample rate, as (channels, samples) float32 - for loudness, which has to be
    measured on what listeners actually hear, not on the mono 22 kHz copy the analysis uses."""
    import librosa
    try:
        y, sr = librosa.load(str(path), sr=None, mono=False)
        return np.atleast_2d(y).astype(np.float32), int(sr)
    except Exception:
        import av
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            sr = stream.rate
            resampler = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=sr)
            chunks = [out.to_ndarray() for frame in container.decode(audio=0) for out in resampler.resample(frame)]
            chunks += [out.to_ndarray() for out in resampler.resample(None)]
        if not chunks:
            raise ValueError("no audio found in file")
        return np.concatenate(chunks, axis=1).astype(np.float32), int(sr)


def _biquad(kind, G, Q, fc, sr):
    """The two filters of ITU-R BS.1770 'K-weighting', designed for any sample rate."""
    A, w0 = 10 ** (G / 40), 2 * np.pi * fc / sr
    alpha, c = np.sin(w0) / (2 * Q), np.cos(w0)
    if kind == "shelf":
        b = [A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * c),
             A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * c),
             (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * alpha]
    else:  # high-pass
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def _kweighted_cumsum(ch, sr):
    """Running sum of the K-weighted power of all channels (every channel weight is 1.0 for mono and stereo)."""
    from scipy.signal import lfilter
    z = np.zeros(ch.shape[1])
    for x in ch:
        for kind, G, Q, fc in (("shelf", 4.0, 1 / np.sqrt(2), 1500.0), ("highpass", 0.0, 0.5, 38.0)):
            b, a = _biquad(kind, G, Q, fc, sr)
            x = lfilter(b, a, x)
        z += x ** 2
    return np.concatenate([[0.0], np.cumsum(z)])


def _window_power(csum, sr, length, step=0.1):
    win, hop = int(length * sr), int(step * sr)
    if len(csum) - 1 < win:
        return np.array([])
    starts = np.arange(0, len(csum) - 1 - win + 1, hop)
    return (csum[starts + win] - csum[starts]) / win


def integrated_lufs(ch, sr, csum=None):
    """Integrated loudness (LUFS) per ITU-R BS.1770-4 / EBU R128: K-weighting, 400 ms blocks every 100 ms,
    an absolute gate at -70 LUFS and a relative gate 10 LU below the gated mean. None for silence."""
    power = _window_power(_kweighted_cumsum(ch, sr) if csum is None else csum, sr, 0.4)
    loud = -0.691 + 10 * np.log10(power + 1e-12)
    gated = power[loud > -70]
    if not len(gated):
        return None
    rel = -0.691 + 10 * np.log10(gated.mean()) - 10
    return float(-0.691 + 10 * np.log10(power[(loud > -70) & (loud > rel)].mean()))


def loudness_range(ch, sr, csum=None):
    """Loudness range (LU) per EBU Tech 3342: short-term loudness (3 s windows, 10 per second), an absolute gate
    at -70 LUFS and a relative gate 20 LU below their mean, then the spread from the 10th to the 95th
    percentile - how far the level moves across the track, ignoring the quietest and loudest moments."""
    power = _window_power(_kweighted_cumsum(ch, sr) if csum is None else csum, sr, 3.0)
    loud = -0.691 + 10 * np.log10(power + 1e-12)
    gated = power[loud > -70]
    if len(gated) < 2:
        return None
    rel = -0.691 + 10 * np.log10(gated.mean()) - 20
    keep = loud[(loud > -70) & (loud > rel)]
    return float(np.percentile(keep, 95) - np.percentile(keep, 10)) if len(keep) >= 2 else None


def stereo_correlation(ch):
    """Correlation of left and right (+1 = mono, 0 = unrelated, below 0 = out of phase: parts cancel when the mix
    is played in mono, on a phone speaker or a club PA). None for a mono file."""
    if ch.shape[0] < 2:
        return None
    left, right = ch[0].astype(np.float64), ch[1].astype(np.float64)
    den = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
    return float(np.sum(left * right) / den) if den > 0 else None


def true_peak_db(ch, sr):
    """Peak level between the samples too (what a DAC or an mp3/AAC encoder reconstructs): 4x oversampling below
    96 kHz, as BS.1770 recommends. In dBTP (dB relative to digital full scale)."""
    from scipy.signal import resample_poly
    up = 4 if sr < 96000 else 2
    peak, chunk, pad = 0.0, sr * 10, 512
    for x in ch:
        for i in range(0, len(x), chunk):  # in pieces to keep memory flat
            # Pad each piece and keep only its middle: the oversampling filter rings where a piece is cut off,
            # which read a steady tone 0.5 dB too high.
            a = max(0, i - pad)
            seg = resample_poly(x[a: i + chunk + pad], up, 1)
            mid = seg[(i - a) * up: (i - a + min(chunk, len(x) - i)) * up]
            if len(mid):
                peak = max(peak, float(np.max(np.abs(mid))))
    return float(20 * np.log10(peak + 1e-12))


SPOTIFY_TARGET_LUFS = -14.0  # Spotify's normal playback level (support.spotify.com, "Loudness normalization")


def streaming_loudness(path):
    """Integrated loudness and true peak as streaming services measure them, and what Spotify's normalisation
    would do to the track: turn louder masters down to -14 LUFS; turn softer ones up, but only as far as leaves
    the true peak at -1 dBTP (their published rule). A mono file is measured as it plays: on both speakers."""
    ch, sr = load_audio_channels(path)
    corr = stereo_correlation(ch)
    widths = electronic.band_widths(ch, sr)
    mono = ch.shape[0] == 1
    if mono:
        ch = np.vstack([ch, ch])
    ch = ch[:2]  # BS.1770 weights for surround channels differ; music uploads are mono or stereo
    csum = _kweighted_cumsum(ch, sr)
    lufs = integrated_lufs(ch, sr, csum)
    if lufs is None:
        return None
    lra = loudness_range(ch, sr, csum)
    tp = true_peak_db(ch, sr)
    if lufs > SPOTIFY_TARGET_LUFS:
        gain = SPOTIFY_TARGET_LUFS - lufs
    else:
        gain = max(0.0, min(SPOTIFY_TARGET_LUFS - lufs, -1.0 - tp))
    return {"lufs": round(lufs, 1), "true_peak_db": round(tp, 1), "mono_file": mono,
            "loudness_range_lu": None if lra is None else round(lra, 1),
            "stereo_correlation": None if corr is None else round(corr, 2), "band_widths": widths,
            "plr_db": round(tp - lufs, 1),  # peak-to-loudness ratio: low = heavily limited, high = punchy/dynamic
            "spotify_gain_db": round(gain, 1), "spotify_plays_at": round(lufs + gain, 1)}


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


# Chord forms, one entry per bar: the bar's root relative to the home chord (0 = I, 5 = IV, 7 = V).
BLUES_FORMS = {
    "12-bar blues": [0, 0, 0, 0, 5, 5, 0, 0, 7, 5, 0, 0],
    "12-bar blues (quick change: IV in bar 2)": [0, 5, 0, 0, 5, 5, 0, 0, 7, 5, 0, 0],
    "12-bar blues (V in bars 9 and 10)": [0, 0, 0, 0, 5, 5, 0, 0, 7, 7, 0, 0],
    "12-bar blues (bVI–V in bars 9 and 10)": [0, 0, 0, 0, 5, 5, 0, 0, 8, 7, 0, 0],  # the usual minor blues
    "12-bar blues (ii–V in bars 9 and 10)": [0, 0, 0, 0, 5, 5, 0, 0, 2, 7, 0, 0],   # jazz blues
    "8-bar blues": [0, 7, 5, 5, 0, 7, 0, 7],                                        # 'Key to the Highway'
    "8-bar blues (I–IV start)": [0, 0, 5, 5, 0, 7, 0, 7],
    "16-bar blues (8 bars on I)": [0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 7, 5, 0, 0],
    "16-bar blues (extended turnaround)": [0, 0, 0, 0, 5, 5, 0, 0, 7, 5, 7, 5, 7, 5, 0, 0],
}
TWELVE_BAR = {k: v for k, v in BLUES_FORMS.items() if len(v) == 12}
V_SUBS = (7, 2, 8)  # a bar on V may hold ii (jazz) or bVI (minor blues) instead


def _chord_spans(r):
    tl = [c for c in r.get("chord_timeline") or [] if "start" in c and parse_chord(c["chord"])]
    ends = [c["start"] for c in tl[1:]] + [r["duration_sec"]]
    return [(c["start"], e, parse_chord(c["chord"])[0]) for c, e in zip(tl, ends)]


BAR_STRETCH = (0.0, -0.01, 0.01, -0.02, 0.02, -0.03, 0.03)


def _bar_grids(r, spans):
    """Candidate bar grids. Bar length: the one the tempo gives, and double and half in case the beat was counted
    at the wrong speed (the 'family'), each nudged by up to 3%: a tempo off by a percent or two slides a long form
    off its bar lines by the third or fourth chorus. Bar lines: every 1/8 bar from the detected downbeat. Yields
    (bar_sec, first_bar_line, [root of each bar], how cleanly chords change on the bar lines, family, nudge)."""
    beat_bar = (r.get("meter") or {}).get("beats_per_bar", 4) * 60 / r["tempo_bpm"]
    down = r.get("downbeat_sec") or 0.0
    for family, nudge in ((f, n) for f in (1.0, 2.0, 0.5) for n in BAR_STRETCH):
        bar = beat_bar * family * (1 + nudge)
        if not 0.8 <= bar <= 8:
            continue
        # Half/double length stands for "the beat was counted at the wrong speed", so the tempo it implies must be
        # one music is actually played at. Doubling a 90 BPM ballad's bars means 45 BPM: no correction, just a
        # longer grid that makes one pass of anything look like most of the song.
        if family != 1.0 and not TACTUS_MIN <= (r.get("meter") or {}).get("beats_per_bar", 4) * 60 / bar <= TACTUS_MAX:
            continue
        for q in range(8):
            start = (down + q * bar / 8) % bar
            roots, purity = [], []
            for i in range(int((r["duration_sec"] - start) // bar)):
                a, b = start + i * bar, start + (i + 1) * bar
                cover = Counter()
                for s, e, root in spans:
                    overlap = min(b, e) - max(a, s)
                    if overlap > 0:
                        cover[root] += overlap
                roots.append(cover.most_common(1)[0][0] if cover else None)
                purity.append(cover.most_common(1)[0][1] / bar if cover else 0)
            yield bar, start, roots, float(np.mean(purity)) if purity else 0.0, family, abs(nudge)


def _pick_family(best):
    """best: {family: (score, bars it was measured on, candidate)}. The bar length the tempo gives wins unless half or
    double fits better by more than chance: two standard errors of the difference between two hit rates. (A 12-bar
    at 152 BPM also half-fits as a 24-bar at 76; with half as many bars that reading is far less certain.)"""
    if 1.0 not in best:
        return max(best.values(), key=lambda x: x[0])[2] if best else None
    p1, n1, choice = best[1.0]
    top = p1
    for fam, (p2, n2, cand) in best.items():
        if fam == 1.0:
            continue
        se = np.sqrt(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1))
        if p2 - p1 > 2 * se and p2 > top:
            top, choice = p2, cand
    return choice


def _harmonic_loop_sec(spans, step=0.1):
    """The shortest time after which the chord roots repeat almost exactly (>= 90% of the time), measured on the
    chord timeline itself rather than on a bar grid, so a slightly wrong bar length can't hide a short loop.
    None if nothing repeats that well, or the piece is one chord (the bar-grid check handles drones)."""
    if not spans:
        return None
    grid = np.arange(spans[0][0], spans[-1][1], step)
    roots = np.full(len(grid), -1)
    for s_, e_, root in spans:
        roots[(grid >= s_) & (grid < e_)] = root
    if len(set(roots[roots >= 0].tolist())) < 2:
        return None
    for lag in range(int(1.5 / step), len(grid) // 2):
        a, b = roots[:-lag], roots[lag:]
        ok = (a >= 0) & (b >= 0)
        if ok.sum() > 0.4 * len(grid) and np.mean(a[ok] == b[ok]) >= 0.9:
            return lag * step
    return None


def _anchors(tpl):
    """The bars where the harmony moves (bar 1, and every bar whose chord differs from the one before)."""
    return [0] + [i for i in range(1, len(tpl)) if tpl[i] != tpl[i - 1]]


def detect_blues_form(r):
    """8-, 12- and 16-bar blues. Bar-by-bar chord roots are compared with the classic layouts, trying every
    starting bar and half/double bar lengths. Works from roots only, so power chords (D5, G5: blues guitar often
    leaves out the third) still count. Returns None, or {name, bars, template, bar_sec, chorus_sec,
    choruses: [start s], match_pct}."""
    spans = _chord_spans(r)
    if len(spans) < 6 or not r.get("tempo_bpm"):
        return None
    tonics = {key_tonic(r["key_index"])}
    longest = Counter()
    for s, e, root in spans:
        longest[root] += e - s
    tonics.add(longest.most_common(1)[0][0])
    loop_sec = _harmonic_loop_sec(spans)
    best = {}
    for bar, start, roots, clean, family, nudge in _bar_grids(r, spans):
        # How exactly the piece repeats after each number of bars. A piece that repeats with a period shorter than a
        # form (a drone, a 4- or 8-bar loop) is that loop, not the form, however well it lines up on average.
        valid = lambda i, lag: roots[i] is not None and roots[i + lag] is not None
        repeats = {lag: np.mean([roots[i] == roots[i + lag] for i in range(len(roots) - lag) if valid(i, lag)] or [0])
                   for lag in range(1, 16)}
        for tonic in tonics:
            rel = [None if x is None else (x - tonic) % 12 for x in roots]
            for name, tpl in BLUES_FORMS.items():
                L = len(tpl)
                if len(rel) < 2 * L or max(repeats[lag] for lag in range(1, L)) >= 0.9:
                    continue  # need two full choruses, and no shorter cycle
                if loop_sec and loop_sec < 0.9 * L * bar:
                    continue  # the chords as heard repeat sooner than this form would: a short loop, on any grid
                pos = lambda i: (i - phase) % L
                hit = lambda i: (rel[i] == tpl[pos(i)] or (pos(i) == L - 1 and rel[i] == 7)  # turnaround on V
                                 or (tpl[pos(i)] == 7 and rel[i] in V_SUBS))
                for phase in range(L):
                    idx = range(phase, len(rel) - (len(rel) - phase) % L)  # whole choruses only
                    if len(idx) < 2 * L:
                        continue
                    score = np.mean([hit(i) for i in idx])
                    moves = np.mean([hit(i) for i in idx if pos(i) in _anchors(tpl)])
                    home = np.mean([hit(i) for i in idx if tpl[pos(i)] == 0 and pos(i) < L // 2])
                    clear = sum(np.mean([hit(i) for i in range(c0, c0 + L)]) >= 0.75 for c0 in idx[::L])
                    # A blues lives on I, IV and V (plus ii / bVI substitutes); a loop with, say, a vi chord in a
                    # quarter of its bars (I-V-vi-IV pop) isn't one, even if it lines up on average.
                    foreign = np.mean([rel[i] not in (0, 5, 7, 2, 8) for i in idx if rel[i] is not None])
                    if foreign > 0.15 or (L == 8 and score < 0.8):
                        continue
                    # A drone, or a short loop that fits on average, isn't a blues: the bars where the harmony
                    # moves must match, the opening bars must sit at home, and at least two choruses must
                    # clearly fit (an intro or ending chorus is often played freely).
                    if moves < 0.6 or home < 0.6 or clear < 2:
                        continue
                    # A chorus that began just before the recording (first bar cut off) still counts if its other
                    # bars fit: chord changes are often detected a beat early.
                    early = phase - L >= -1 and np.mean([hit(i) for i in range(max(0, phase - L), phase)]) >= 0.7
                    cand = (round(float(score), 2), round(clean, 2), L == 12, -nudge, name, tpl, bar,
                            start + (phase - L * early) * bar, len(idx) // L + early)
                    if family not in best or cand[:4] > best[family][2][:4]:
                        best[family] = (float(score), len(idx), cand)
    best = _pick_family(best)
    if not best or best[0] < 0.7:
        return None
    score, _, _, _, name, tpl, bar, first, count = best
    if key_is_minor(r["key_index"]):
        name = name.replace("-bar blues", "-bar minor blues")
    L = len(tpl)
    return {"name": name, "bars": L, "template": tpl, "bar_sec": round(bar, 2), "chorus_sec": round(L * bar, 1),
            "match_pct": round(100 * score),
            "choruses": [round(float(max(0.0, first + k * L * bar)), 1) for k in range(count)]}


detect_twelve_bar = detect_blues_form  # older name


def detect_song_form(r):
    """AABA, the 32-bar song form of standards and classic pop ('Over the Rainbow', 'Yesterday'): four 8-bar
    sections where the 1st, 2nd and 4th share their chords and the 3rd (the bridge) differs. Found by comparing
    8-bar stretches of bar-by-bar chord roots. Returns None, or {name, bar_sec, section_sec, starts: [A, A, B, A
    start s], cycles, match_pct}."""
    spans = _chord_spans(r)
    if len(spans) < 8 or not r.get("tempo_bpm"):
        return None
    same = lambda a, b: float(np.mean([x == y and x is not None for x, y in zip(a, b)]))

    def aaba(chunks, k):  # how clearly chunks k..k+3 are A A B A (A sections alike, the bridge different)
        a1, a2, b, a3 = chunks[k:k + 4]
        if len({x for x in a1 if x is not None}) < 2:  # a section on one chord isn't a song form
            return None
        a_sim = min(same(a1, a2), same(a1, a3), same(a2, a3))
        b_sim = max(same(a1, b), same(a2, b), same(a3, b))
        return a_sim - b_sim if a_sim >= 0.75 and b_sim <= 0.5 else None

    best = {}
    for bar, start, roots, clean, family, nudge in _bar_grids(r, spans):
        if len(roots) < 32:
            continue
        for phase in range(8):
            chunks = [roots[i:i + 8] for i in range(phase, len(roots) - 7, 8)]
            for k in range(len(chunks) - 3):
                margin = aaba(chunks, k)
                if margin is None:
                    continue
                # follow the form forward: each further AABA with the same A section is another cycle
                cycles, j = 1, k + 4
                while j + 3 < len(chunks) and aaba(chunks, j) is not None and same(chunks[j], chunks[k]) >= 0.75:
                    cycles, j = cycles + 1, j + 4
                coverage = cycles * 32 / len(roots)
                # The 32-bar form has to be what the piece is made of, not one verse-verse-chorus-verse stretch of a
                # longer song: its cycles must cover most of the bars (an intro or ending may be extra).
                if coverage < 0.6:
                    continue
                cand = (cycles, round(margin, 2), round(clean, 2), -nudge, -k, bar, start + (phase + 8 * k) * bar)
                if family not in best or cand[:5] > best[family][2][:5]:
                    # how well it fits, on the same 0-1 scale as the blues hit rate, over the bars it covers
                    best[family] = (min(1.0, 0.5 + margin), cycles * 32, cand)
    best = _pick_family(best)
    if not best:
        return None
    cycles, margin, _, _, _, bar, first = best
    return {"name": "AABA (32-bar song form)", "bar_sec": round(bar, 2), "section_sec": round(8 * bar, 1),
            "starts": [round(float(first + j * 8 * bar), 1) for j in range(4)], "cycles": cycles,
            "match_pct": round(100 * min(1.0, 0.5 + margin))}


# One chord per bar, as (semitones above the home note, quality).
BLUES_VARIATIONS = [
    ("quick-change blues with dominant 7ths",
     [(0, "7"), (5, "7"), (0, "7"), (0, "7"), (5, "7"), (5, "7"), (0, "7"), (0, "7"), (7, "7"), (5, "7"), (0, "7"), (7, "7")],
     "The classic blues sound: every chord a dominant 7th, the IV chord in bar 2, and V in bar 12 to turn around."),
    ("jazz blues",
     [(0, "7"), (5, "7"), (0, "7"), (7, "m7"), (5, "7"), (6, "dim7"), (0, "7"), (9, "7"), (2, "m7"), (7, "7"), (0, "7"), (7, "7")],
     "Bebop-style: a ii–V into the IV chord (bar 4), a passing diminished chord (bar 6), and a VI7–ii–V back home."),
]


MINOR_BLUES_VARIATIONS = [
    ("minor blues",
     [(0, "m7"), (5, "m7"), (0, "m7"), (0, "m7"), (5, "m7"), (5, "m7"), (0, "m7"), (0, "m7"), (8, "7"), (7, "7"), (0, "m7"), (7, "7")],
     "The 'Thrill Is Gone' sound: minor 7ths on i and iv, then bVI7 to V7 in bars 9-10 for a dark pull home."),
    ("minor blues with a ii–V",
     [(0, "m7"), (0, "m7"), (0, "m7"), (0, "m7"), (5, "m7"), (5, "m7"), (0, "m7"), (0, "m7"), (2, "m7b5"), (7, "7"), (0, "m7"), (7, "7")],
     "Jazz minor blues: a half-diminished ii to a dominant V in bars 9-10, the classic minor-key turnaround."),
]


def blues_variations(r):
    """Variations of the detected blues form, spelled in the track's key: for a 12-bar, the classic dominant-7th and
    jazz versions (minor-blues versions in a minor key); for an 8- or 16-bar, the same form with 7th chords."""
    k = r["key_index"]
    t = key_tonic(k)
    form = r.get("blues_form") or {"bars": 12}
    if form["bars"] == 12:
        table = MINOR_BLUES_VARIATIONS if key_is_minor(k) else BLUES_VARIATIONS
    else:
        minor = key_is_minor(k)
        q = lambda off: "m7" if minor and off in (0, 5) else "7"
        table = [(f"{form['name']} with 7th chords", [(off, q(off)) for off in form["template"]],
                 "The same form with the chords blues players reach for: " +
                 ("minor 7ths on i and iv, a dominant V7." if minor else "a dominant 7th on every chord.")),
                 ]
    return [{"name": name, "why": why, "chords": [chord_name((t + off) % 12, q, k) for off, q in bars]}
            for name, bars, why in table]


def fits_form(chords, template, tonic):
    """Do these chords (one per bar) follow the form? Checks the bars where the form's harmony moves (bar 1, and
    each change of chord), allowing a quarter of them to be substitutions (a V bar may also be ii or bVI)."""
    roots = [(parse_chord(c) or (None,))[0] for c in chords]
    if len(roots) != len(template) or None in roots:
        return False
    rel = [(x - tonic) % 12 for x in roots]
    ok = [rel[i] == template[i] or (template[i] == 7 and rel[i] in V_SUBS) for i in _anchors(template)]
    return sum(ok) >= 0.75 * len(ok)


def fits_twelve_bar(chords, tonic):
    return fits_form(chords, BLUES_FORMS["12-bar blues"], tonic)


def key_fifths(k):
    """Key signature as MusicXML 'fifths' (+sharps / -flats), spelled the same way as the notes."""
    steps = [0, 2, 3, 5, 7, 8, 10] if key_is_minor(k) else [0, 2, 4, 5, 7, 9, 11]
    names = [spell((key_tonic(k) + s) % 12, k) for s in steps]
    return sum(n.count("#") for n in names) - sum(n.count("b") for n in names)


def bass_sheet(r, chords):
    """The bass transcription written out on the beat grid (bars of the detected meter), with the
    chord symbols above it. None without a clear beat or a bass line."""
    if not r.get("beat_times") or not r.get("bass_transcription"):
        return None
    notes = [(n["start"], n["end"], n["midi"], n["velocity"]) for n in r["bass_transcription"]]
    return notation.quantize(notes, chords, r["beat_times"], r["meter"]["beats_per_bar"], r["downbeat_sec"],
                             r["duration_sec"])


# How positive each mode tends to sound (0 = dark, 1 = bright), for the feel estimate.
MODE_BRIGHTNESS = {"Lydian": 0.8, "Ionian (major)": 0.75, "Mixolydian": 0.65, "Dorian": 0.45,
                   "Phrygian dominant": 0.3, "Aeolian (natural minor)": 0.3, "Harmonic minor": 0.25, "Phrygian": 0.2}


def mood_words(arousal, valence):
    if arousal >= 60:
        return "uplifting, energetic" if valence >= 55 else "driving, intense" if valence >= 40 else "dark, dramatic"
    if arousal >= 40:
        return "warm, upbeat" if valence >= 55 else "steady, bittersweet" if valence >= 40 else "brooding, tense"
    return "calm, peaceful" if valence >= 55 else "reflective, wistful" if valence >= 40 else "sad, melancholic"


def feel_analysis(y, sr, rms_db, sections, chords, r):
    """Energy (arousal) and mood (valence) per section, measured on the audio itself.
    Energy: loudness, how many notes/hits per second, brightness and tempo.
    Mood: the mode, the share of minor chords, brightness and tempo.
    A rule-of-thumb model from music-emotion research, not a trained classifier."""
    import librosa
    clip = lambda x, lo, hi: float(np.clip((x - lo) / (hi - lo), 0, 1))
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="time")
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    tempo = r["tempo_bpm"] or 70  # no clear beat reads as slow
    tempo_n = clip(tempo, 60, 160)
    loudest = max((s["energy_db"] for s in sections), default=0.0)
    fps = sr / HOP
    out = []
    for s in sections:
        a, b = int(s["start"] * fps), max(int(s["end"] * fps), int(s["start"] * fps) + 1)
        length = max(0.1, s["end"] - s["start"])
        rate = float(np.sum((onsets >= s["start"]) & (onsets < s["end"]))) / length
        bright = float(np.mean(centroid[a:b])) if a < len(centroid) else float(np.mean(centroid))
        loud_rel = s["energy_db"] - loudest
        chord_t = [(c["end"] - c["start"], c["q"].startswith("m") and not c["q"].startswith("maj"))
                   for c in chords if s["start"] <= (c["start"] + c["end"]) / 2 < s["end"]]
        total = sum(t for t, _ in chord_t)
        minor = sum(t for t, m in chord_t if m) / total if total else 0.5
        if s["key"] == r["key"]:
            base = MODE_BRIGHTNESS.get(r["mode"], 0.5)
        else:
            base = 0.3 if "minor" in s["key"] else 0.7
        arousal = 100 * (0.25 * clip(loud_rel, -15, 0) + 0.1 * clip(s["energy_db"], -35, -12) +
                         0.25 * clip(rate, 0.5, 5) + 0.2 * clip(bright, 800, 3500) + 0.2 * tempo_n)
        valence = 100 * (0.4 * base + 0.3 * (1 - minor) + 0.15 * clip(bright, 800, 3500) + 0.15 * tempo_n)
        out.append({"label": s["label"], "start": s["start"], "end": s["end"],
                    "energy": int(np.clip(round(arousal / 10), 1, 10)), "arousal": round(arousal),
                    "valence": round(valence), "mood": mood_words(arousal, valence),
                    "loudness_vs_peak_db": round(loud_rel, 1), "notes_per_sec": round(rate, 1),
                    "brightness_hz": round(bright), "minor_chords_pct": round(100 * minor)})
    if not out:
        return {}
    w = np.array([s["end"] - s["start"] for s in out])
    A = float(np.average([s["arousal"] for s in out], weights=w))
    V = float(np.average([s["valence"] for s in out], weights=w))
    e = [s["arousal"] for s in out]
    peak = out[int(np.argmax(e))]
    if len(e) < 2 or max(e) - min(e) < 10:
        arc = "stays at about the same energy throughout"
    elif e[-1] - e[0] >= 10 and peak is out[-1]:
        arc = "builds up to the end"
    elif e[0] - e[-1] >= 10 and peak is out[0]:
        arc = "starts strongest and winds down"
    elif peak is not out[0] and peak is not out[-1]:
        arc = "rises to a peak in the middle, then comes down"
    else:
        arc = "goes up and down between sections"
    return {"arousal": round(A), "valence": round(V), "energy": int(np.clip(round(A / 10), 1, 10)),
            "mood": mood_words(A, V), "arc": arc, "contrast": round(max(e) - min(e)),
            "peak": {"label": peak["label"], "start": peak["start"], "end": peak["end"]}, "sections": out}


def refine_tempo(y, sr, coarse):
    """The beat tracker's tempo can only take values one analysis frame apart: around 150 BPM that's 152.0 or
    161.5, around 120 it's 117.5 or 123.0. Re-measure it at 4x finer time resolution, only within 6% of the coarse
    value (so a half/double-speed choice is never changed), and interpolate between frames. On synthetic tracks
    of known tempo this lands within 0.1 BPM; it returns the coarse value if there's no clear peak."""
    import librosa
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    lag0 = 60 / coarse * sr / hop
    lo, hi = int(lag0 * 0.94), int(np.ceil(lag0 * 1.06)) + 1
    ac = librosa.autocorrelate(env - env.mean(), max_size=hi + 2)
    k = lo + int(np.argmax(ac[lo:hi]))
    if k in (lo, hi - 1) or ac[k] <= 0:
        return coarse  # the peak is at the edge of the window, or there is none: keep what we had
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    k = k + 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else k
    return float(60 * sr / hop / k)


TACTUS_MIN, TACTUS_MAX = 50, 200  # the beat people tap along to sits in this range (BPM)
PREFERRED_TEMPO = 120             # and, given no other evidence, near here (listeners' preferred tempo is ~2 Hz)


def tactus_factor(y, sr, tempo):
    """A beat tracker reading below 50 BPM has counted bars or half bars - typical of soft pads or held chords that
    only change once a bar. Return how many beats to split each counted one into (2, 3 or 4): the split the
    audio's rhythm supports most (onset autocorrelation at that beat length); if nothing in the audio favours one,
    the one nearest the preferred tempo. 1 when the reading is already a tempo people tap along to."""
    import librosa
    if tempo >= TACTUS_MIN:
        return 1
    options = [k for k in (2, 3, 4) if TACTUS_MIN <= tempo * k <= TACTUS_MAX]
    if not options:
        return 1
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env = env - env.mean()
    ac = librosa.autocorrelate(env, max_size=int(60 / tempo * sr / hop) + 2)
    ac = ac / (ac[0] + 1e-9)
    evidence = {k: float(ac[int(round(60 / (tempo * k) * sr / hop))]) for k in options}
    best = max(evidence, key=evidence.get)
    if evidence[best] >= 0.1:  # a real (if weak) periodicity at that beat length; uncorrelated onsets sit near 0
        return best
    return min(options, key=lambda k: abs(np.log(tempo * k / PREFERRED_TEMPO)))


def _analyze_audio(path, light=False):
    """The expensive part: everything that depends only on the audio (cacheable)."""
    import librosa

    y, sr = load_audio(path)
    if len(y) < 2 * SR:
        raise ValueError("audio is shorter than 2 seconds")
    duration = len(y) / sr
    # Split off the drums so they don't blur the harmony, and analyse each part on its own.
    y_perc, y_harm, separation = instruments.separate(y, sr, fast=light)  # drums, everything else

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
    raised = 1
    if clear_beat:
        tempo = refine_tempo(y, sr, tempo)
        raised = tactus_factor(y, sr, tempo)
        if raised > 1:  # the tracker counted bars (or half bars): split each counted beat into `raised` beats
            beat_frames = np.concatenate([np.linspace(a, b, raised, endpoint=False) for a, b in
                                          zip(beat_frames[:-1], beat_frames[1:])] + [beat_frames[-1:]]).round().astype(int)
            tempo *= raised
    # Drums? (noisy hits, not a guitar's pick attacks) - and if so, make sure the beats sit on the kicks.
    are_drums, hit_noise = instruments.hits_are_drums(y_perc, sr)
    on_kicks = instruments.onto_kicks(y_perc, sr, beat_frames, HOP) if clear_beat and are_drums else None
    if on_kicks is not None:
        beat_frames = on_kicks
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
    # Only *say* it's sharp or flat when the whole recording agrees: bends, slides and vibrato (blues guitar, a
    # singer) pull the estimate around, and one 20 s stretch can read 10 cents flat while the next reads in tune.
    chunk = 20 * sr
    parts = [float(librosa.estimate_tuning(y=y_harm[i:i + chunk], sr=sr)) for i in range(0, len(y_harm) - chunk // 2, chunk)]
    # agree = the stretches spread less than half the offset they claim (and never more than 15 cents), and
    # none of them points the other way
    spread = np.percentile(parts, 75) - np.percentile(parts, 25) if len(parts) >= 2 else 0.0
    tuning_steady = len(parts) < 2 or (spread <= min(0.15, max(0.06, abs(tuning) / 2))
                                       and all(np.sign(p_) == np.sign(tuning) or abs(p_) < 0.05 for p_ in parts))
    # Notes are read from C2 (65 Hz) up: below that, kicks and sub-bass (electronic 808s, a pitched kick) swamp the
    # pitch picture. Bass notes lower than C2 still count through their overtones.
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP, tuning=tuning,
                                        fmin=librosa.note_to_hz("C2"), n_octaves=6)
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

    # With almost no percussive sound (legato strings, pads, soft piano) the beat grid can lock onto the wrong
    # speed, and a wrong grid can make chord changes look like 5/4 or 7/4. So in that case the tempo is marked
    # approximate and odd meters aren't claimed.
    weak_pulse = clear_beat and (float(np.sum(y_perc ** 2)) < 0.06 * float(np.sum(y ** 2)) or raised > 1)
    meter = detect_meter(rms_db, chroma,
                         librosa.util.fix_frames(tracked_frames[tracked_frames < n], x_min=0, x_max=n), clear_beat,
                         allow_odd=not weak_pulse)
    if light:  # tag mode: the bass's presence is enough; skip the (slow) note-by-note transcription
        bass, bass_line = [None] * max(0, len(btimes) - 1), []
    else:
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
    if clear_beat:
        m = min(len(drum_rms), len(low_band), len(mid_band), len(rms_db))
        sections = refine_sections(sections, [drum_rms[:m] ** 2, low_band[:m], mid_band[:m]], sr / HOP, btimes,
                                   duration, rms_db, drums=are_drums
                                   and float(np.sum(y_perc ** 2)) >= 0.03 * float(np.sum(y ** 2)),
                                   key_changes=[e for _, e, _ in ksec[:-1]])
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
        # loudness as power (a mean of dB values lets the gaps between kicks drag a beat-only intro far down),
        # and how much harmony (chords, pads, leads: 200 Hz-2 kHz of the drumless audio) plays
        sec["loudness_db"] = round(float(10 * np.log10(np.mean(10 ** (rms_db[a:b] / 10)) + 1e-12)), 1)
        sec["harmony_db"] = round(float(10 * np.log10(np.mean(mid_band[a:b]) + 1e-12)), 1)
    loudest_drums = max((sec["drums_level"] for sec in sections), default=0.0)
    # Drums = enough percussive sound AND noisy hits (a guitar's pick attacks are percussive but pitched).
    drums_anywhere = are_drums and float(np.sum(y_perc ** 2)) >= 0.03 * float(np.sum(y ** 2))
    if not drums_anywhere:
        loudest_drums = 0.0  # no real drums anywhere (same rule as instruments.drum_analysis)
    # A separate bass instrument? On a phone recording a bass guitar and a guitar's low strings look alike
    # (phones cut everything below ~100 Hz). So: report one only in a band (drums) or when there's real
    # energy below 100 Hz, where a guitar can't go. Otherwise the low notes are the main instrument's own
    # lowest part (e.g. the thumb line of fingerstyle guitar), which is still transcribed and written out.
    below_100 = float(spec[(spec_f >= 30) & (spec_f < 100)].sum() / (spec.sum() + 1e-12))
    separate_bass = drums_anywhere or below_100 >= 0.03
    loudest_low = max((sec["bass_low_db"] for sec in sections), default=0.0)
    for sec in sections:
        sec["drums"] = loudest_drums > 0 and sec.pop("drums_level") >= 0.1 * loudest_drums
        # Bass plays here if the low end is within 12 dB of its loudest section and isn't swamped by mids.
        sec["bass"] = sec.pop("bass_low_db") >= loudest_low - 12 and sec.pop("bass_ratio_db") >= -9

    result = {
        "file": Path(path).name,
        "duration_sec": round(duration, 1),
        "duration": fmt_time(duration),
        # what we report (and bend the MIDI by): 0 unless the whole recording agrees it's off A440
        "tuning_cents": int(round(tuning * 100)) if tuning_steady else 0,
        "tuning_spread_cents": [int(round(100 * p_)) for p_ in parts],
        "tempo_bpm": round(tempo, 1) if clear_beat else None,
        "tempo_approximate": bool(weak_pulse),
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
        "chord_timeline": [{"time": fmt_time(c["start"]), "start": round(float(c["start"]), 2),
                            "chord": c["name"], "roman": c["roman"],
                            "with_bass": c.get("slash", c["name"]), "beats": c["beats"],
                            "clarity": c.get("clarity")} for c in chords],
        "loops": loops,
        "sections": sections,
        "rhythm": rhythm_features(y, y_perc, y_harm, beat_times, clear_beat, duration, sr),
        "sound": sound_features(y, sr, rms_db, duration),
        "separation": separation,
        "drums": instruments.drum_analysis(y_perc, sr, 256, beat_times, float(np.sum(y ** 2)),
                                           beats_per_bar=meter["beats_per_bar"], are_drums=are_drums)
        if clear_beat else {"present": False},
        "bass": bass_summary(bass, btimes, chords, main_key, duration),
    }
    if result["drums"].get("present") and hit_noise < instruments.DRUMS_SURE:
        # Only slightly noise-like hits: light percussion (shaker, brushes) or a hard-picked instrument. Say so;
        # the musician's tick ("drums" or not) settles it.
        result["drums"]["unsure"] = True
    if light:
        result["bass"] = {"present": any(sec["bass"] for sec in sections) and separate_bass, "light": True}
        if not separate_bass:
            for sec in sections:
                sec["low_end"], sec["bass"] = sec["bass"], False
    elif not any(sec["bass"] for sec in sections):  # no low end anywhere: the "bass notes" were chord tones
        result["bass"] = {"present": False}
        bass_line = []
    elif not separate_bass:
        # The low notes belong to the main instrument: describe them as its lowest part, not as a bass.
        result["bass"] = {"present": False, "low_part_only": True}
        for sec in sections:
            sec["low_end"], sec["bass"] = sec["bass"], False
        result["low_part"] = bass_summary(bass, btimes, chords, main_key, duration)
    # Nothing below 80 Hz? Then for a guitar a "note" below E2 is the pitch tracker an octave too low. Whether to
    # correct that depends on the instrument (a piano's or cello's low notes are real), which the musician may
    # tell us, so it's decided after the cache (finish_low_part); here we keep both readings.
    if bass_line and float(spec[(spec_f >= 30) & (spec_f < 80)].sum() / (spec.sum() + 1e-12)) < 0.005:
        up = lambda n: n + 12 * ((40 - n + 11) // 12) if n is not None and n < 40 else n
        result["low_end_missing"] = True
        result["low_part_raised"] = bass_summary([up(n) for n in bass], btimes, chords, main_key, duration)
    loudest = max((n[3] for n in bass_line), default=0.0)
    result["bass_transcription"] = [  # note-by-note, for the bass MIDI file
        {"start": round(a, 3), "end": round(e, 3), "midi": n, "velocity": int(np.clip(110 + 2.5 * (db - loudest), 35, 120))}
        for a, e, n, db in bass_line]
    if clear_beat:
        result["beat_times"] = [round(float(t), 3) for t in beat_times]
        down = meter.get("downbeat_frame")
        result["downbeat_sec"] = round(float(librosa.frames_to_time(down, sr=sr, hop_length=HOP)), 3) \
            if down is not None else result["beat_times"][0]
    result["feel"] = feel_analysis(y, sr, rms_db, sections, chords, result)

    # --- best-effort top line (the musician's typed melody is added later, in analyze())
    if light:
        result["top_line"] = {"present": False}
    else:
        top, grid = instruments.top_line(y_harm, sr, btimes, tuning)
        result["top_line"] = top_line_summary(top, grid, chords, main_key)
    result["light"] = light
    if clear_beat and not light:  # sidechain pumping, judged on the beats actually heard
        result["pump"] = electronic.pump(y_harm, sr, librosa.frames_to_time(tracked_frames, sr=sr, hop_length=HOP))
    if not light:  # loudness as streaming services measure it (all channels, the file's own sample rate)
        try:
            result["sound"]["streaming"] = streaming_loudness(path)
        except Exception as e:  # a bonus measurement: never lose the analysis over it
            result["sound"]["streaming"] = {"error": str(e)[:200]}
    return result, {"chroma": chroma.astype(np.float32), "bounds": np.asarray(bounds)}


# Bump when the analysis changes, so cached results from older code aren't reused.
ANALYSIS_VERSION = "2026-10-21a"


def _cache_key(path, light=False):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return f"{h.hexdigest()[:32]}-{ANALYSIS_VERSION}-{instruments.separation_method()}" + ("-light" if light else "")


# What the musician can tell us when uploading (all optional): which instruments play, and the genre.
# The analysis reasons about eight families (drums, bass, guitar, keys, synth, voice, strings, winds); each
# instrument the musician can pick belongs to one, and adds what's specific to it.
# key -> (label, family, General MIDI program for its MIDI files, lowest usual note as MIDI; None = no pitch)
INSTRUMENT_INFO = {
    "drums": ("drum kit", "drums", None, None),
    "drum_machine": ("drum machine / 808 kit", "drums", None, None),
    "percussion": ("hand percussion", "drums", None, None),
    "bass": ("bass guitar", "bass", 33, 23),                     # B0, a 5-string
    "upright_bass": ("double bass", "bass", 32, 28),             # E1
    "synth_bass": ("synth bass / 808", "bass", 38, 0),           # no limit
    "acoustic_guitar": ("acoustic guitar", "guitar", 25, 40),    # E2 (drop D reaches D2)
    "electric_guitar": ("electric guitar", "guitar", 27, 40),
    "classical_guitar": ("classical guitar", "guitar", 24, 40),
    "ukulele": ("ukulele", "guitar", 24, 60),                    # C4 (re-entrant gCEA)
    "banjo": ("banjo", "guitar", 105, 50),                       # D3 (open G)
    "mandolin": ("mandolin", "guitar", 25, 55),                  # G3
    "guitar": ("guitar (other)", "guitar", 25, 40),
    "piano": ("piano", "keys", 0, 21),                           # A0
    "electric_piano": ("electric piano / Rhodes", "keys", 4, 28),  # E1 (73-key Rhodes)
    "organ": ("organ", "keys", 16, 24),                          # C1 (pedals)
    "accordion": ("accordion", "keys", 21, 28),
    "mallets": ("vibraphone / mallets", "keys", 11, 53),         # F3
    "keys": ("keys (other)", "keys", 0, 21),
    "synth_pad": ("synth pads", "synth", 89, 0),
    "synth_lead": ("lead synth", "synth", 81, 0),
    "sampler": ("samples / vocal chops", "synth", 54, 0),
    "synth": ("synth (other)", "synth", 81, 0),
    "voice": ("lead vocals", "voice", 53, 38),                   # D2, a low bass voice
    "backing_vocals": ("backing vocals / choir", "voice", 52, 36),
    "violin": ("violin", "strings", 40, 55),                     # G3
    "viola": ("viola", "strings", 41, 48),                       # C3
    "cello": ("cello", "strings", 42, 36),                       # C2
    "strings": ("string section", "strings", 48, 28),            # with double basses
    "harp": ("harp", "strings", 46, 23),
    "flute": ("flute", "winds", 73, 60),                         # C4
    "clarinet": ("clarinet", "winds", 71, 50),                   # D3 (sounding)
    "saxophone": ("saxophone", "winds", 65, 44),                 # Ab2, a tenor's low Bb
    "trumpet": ("trumpet", "winds", 56, 54),                     # F#3
    "trombone": ("trombone", "winds", 57, 40),                   # E2
    "harmonica": ("harmonica", "winds", 22, 60),
    "winds": ("wind / brass (other)", "winds", 71, 34),
}
INSTRUMENTS = {k: v[0] for k, v in INSTRUMENT_INFO.items()}
FAMILY_LABEL = {"drums": "drums / percussion", "bass": "bass", "guitar": "guitar", "keys": "piano / keys",
                "voice": "voice", "strings": "strings", "winds": "wind / brass", "synth": "synth / electronic"}
INSTRUMENT_GROUPS = [  # how the upload form lays them out
    ("Drums & percussion", ["drums", "drum_machine", "percussion"]),
    ("Bass", ["bass", "upright_bass", "synth_bass"]),
    ("Guitars & plucked", ["acoustic_guitar", "electric_guitar", "classical_guitar", "ukulele", "banjo", "mandolin",
                           "guitar"]),
    ("Keys", ["piano", "electric_piano", "organ", "accordion", "mallets", "keys"]),
    ("Synths & samples", ["synth_pad", "synth_lead", "sampler", "synth"]),
    ("Voice", ["voice", "backing_vocals"]),
    ("Strings", ["violin", "viola", "cello", "strings", "harp"]),
    ("Wind & brass", ["flute", "clarinet", "saxophone", "trumpet", "trombone", "harmonica", "winds"]),
]
GUITARS = {"acoustic_guitar", "electric_guitar", "classical_guitar", "guitar"}  # six strings: capo shapes, guitar clef


# What the trained instrument model (tagger.py: MTG-Jamendo's 40 instrument classes) can name -> our instruments.
HEARD_MAP = {
    "accordion": "accordion", "acousticbassguitar": "bass", "acousticguitar": "acoustic_guitar", "bass": "bass",
    "beat": "drum_machine", "bell": "mallets", "bongo": "percussion", "brass": "winds", "cello": "cello",
    "clarinet": "clarinet", "classicalguitar": "classical_guitar", "computer": "synth", "doublebass": "upright_bass",
    "drummachine": "drum_machine", "drums": "drums", "electricguitar": "electric_guitar",
    "electricpiano": "electric_piano", "flute": "flute", "guitar": "guitar", "harmonica": "harmonica", "harp": "harp",
    "horn": "winds", "keyboard": "keys", "oboe": "winds", "orchestra": "strings", "organ": "organ", "pad": "synth_pad",
    "percussion": "percussion", "piano": "piano", "pipeorgan": "organ", "rhodes": "electric_piano",
    "sampler": "sampler", "saxophone": "saxophone", "strings": "strings", "synthesizer": "synth",
    "trombone": "trombone", "trumpet": "trumpet", "viola": "viola", "violin": "violin", "voice": "voice",
}
HEARD_NAME = {"brass": "brass", "horn": "French horn", "oboe": "oboe", "orchestra": "orchestra", "pipeorgan": "pipe organ",
              "computer": "electronics", "synthesizer": "synthesizer", "keyboard": "keyboard", "beat": "programmed beat",
              "bell": "bells", "bongo": "bongos", "guitar": "guitar", "acousticbassguitar": "acoustic bass",
              "rhodes": "Rhodes", "voice": "voice", "drums": "drums", "bass": "bass"}
HEARD_SHOW = 0.15     # the model's score above which we list an instrument as heard
HEARD_SURE = 0.5      # ...and above which it's sure enough to mention one the musician didn't tick
HEARD_FEATURE = 0.4   # a reference "features" an instrument at this score; yours "hardly has it" below HEARD_SHOW
HEARD_MAX = 6


def hear_instruments(paths, progress=lambda m: None):
    """Which instruments a trained model hears in each file: {path: {"instruments": [...], "voice": p, ...}}, or
    {} when the model isn't installed here (it's optional) or fails. Its scores are a separate opinion from our
    own measurements: they're shown and compared, never used to overrule what the analysis or the musician says."""
    import tagger
    if os.environ.get("MUSIC_HEAR", "1") == "0" or not tagger.available() or not paths:
        return {}
    progress("Listening for which instruments play…")
    try:
        found = tagger.predict([str(p) for p in paths], load_audio)
    except Exception as e:  # a bonus: never lose the analysis over it
        progress(f"  ! instrument model skipped ({str(e)[:120]})")
        return {}
    out = {}
    for path, summary in found.items():
        scores = summary.get("instrument_scores") or dict(summary.get("instruments") or [])
        best = {}
        for cls, p in scores.items():
            key = HEARD_MAP.get(cls)
            if key and p >= HEARD_SHOW and p > best.get(key, (None, 0))[1]:
                best[key] = (HEARD_NAME.get(cls) or INSTRUMENTS[key].split(" / ")[0], p)
        heard = sorted(({"key": k, "name": n, "p": round(p, 2)} for k, (n, p) in best.items()), key=lambda x: -x["p"])
        out[path] = {"instruments": heard[:HEARD_MAX], "model": tagger.backend(),
                     **{k: summary.get(k) for k in ("voice", "electronic", "acoustic") if summary.get(k) is not None}}
    return out


def heard_notes(r):
    """What the model hears that the musician didn't tick (sure ones only) - a suggestion, never a change."""
    h, ctx = r.get("heard") or {}, r.get("context") or {}
    told = set(ctx.get("keys") or [])
    if not h.get("instruments") or not told:
        return []
    return [x for x in h["instruments"] if x["p"] >= HEARD_SURE and family(x["key"]) not in told]


def family(key):
    return INSTRUMENT_INFO[key][1] if key in INSTRUMENT_INFO else key


def named_instrument(result, fam):
    """What to call the family's instrument: the one the musician picked, if they picked exactly one in it."""
    picked = [k for k in (result.get("context") or {}).get("chosen") or [] if family(k) == fam]
    return INSTRUMENTS[picked[0]] if len(picked) == 1 else FAMILY_LABEL.get(fam, fam)
# label, usual tempo range (BPM) or None when the genre spans everything, one genre-specific idea
GENRES = {
    "pop": ("Pop", (90, 130), "Pop listeners expect the hook early and a lift into the chorus: check that the chorus "
                              "is the energy peak (see Energy & mood) and that the hook arrives within ~30 s."),
    "rock": ("Rock", None, "Rock gets its impact from contrast: try dropping to one guitar (and voice) before the last "
                           "chorus, then bring everything back in."),
    "blues": ("Blues", None, "Blues is built on 12-bar choruses and call-and-response: leave space after each phrase "
                             "for an answering lick, and try a turnaround (I–IV–I–V) in the last two bars."),
    "jazz": ("Jazz", None, "Try reharmonising: approach each chord with its own ii–V (e.g. Em7–A7 into D), or swap a V7 "
                           "for the chord a tritone away."),
    "folk": ("Folk / acoustic", None, "Let playing technique make the dynamics: fingerpick the verses, strum the choruses, "
                                      "and keep the arrangement sparse."),
    "rnb": ("R&B / soul", (60, 110), "R&B leans on extended chords (maj7, m9, 11) and a laid-back groove that sits "
                                     "slightly behind the beat."),
    "hiphop": ("Hip-hop", (70, 100), "Beats usually loop 2–4 bars: find the strongest 4-bar stretch (see the chord loops) "
                                     "and build the beat around it."),
    "electronic": ("Electronic / dance", (115, 135), "Dance tracks move in 8- and 16-bar phrases: plan a breakdown and "
                                                     "a build (riser, snare roll) into the drop."),
    "house": ("House", (118, 130), "House lives on the kick on every beat and the off-beat hi-hat: keep the groove "
                                   "constant and make the changes with filters, a breakdown and sidechained pads."),
    "techno": ("Techno", (125, 150), "Techno develops by subtraction and slow change: let one loop run for 16-32 bars "
                                     "and evolve it with a filter, a delay send or one new percussion part at a time."),
    "trance": ("Trance", (130, 145), "Trance builds on a long breakdown with the lead melody alone, then a riser and "
                                     "snare roll into a drop that brings back the full chord progression."),
    "dubstep": ("Dubstep / bass music", (135, 150), "Bass music runs at about 140 BPM with the snare on beat 3 "
                                                    "(half-time); the drop is carried by the bass sound itself, so "
                                                    "give it one phrase to answer another."),
    "ukg": ("UK garage / 2-step", (128, 138), "2-step skips the kick on beats 2 and 4 and swings the hats: try "
                                              "shuffling your 16ths and chopping the vocal or chords into stabs."),
    "dnb": ("Drum & bass", (160, 180), "Drum & bass sits around 170–175 BPM; a half-time section is a classic way to "
                                       "create contrast."),
    "trap": ("Trap", (130, 160), "Trap is counted at about 140 with a half-time snare on beat 3; the energy comes "
                                 "from hi-hat rolls (triplets, 32nds) and 808 slides between notes."),
    "synthwave": ("Synthwave / synth-pop", (80, 118), "Synthwave leans on gated-reverb snares, arpeggiated synths "
                                                      "on 16ths and a I–bVI–bVII move in a minor key."),
    "lofi": ("Lo-fi / chillhop", (60, 95), "Lo-fi's charm is imperfection: swing the drums, detune or wobble the keys "
                                           "slightly and roll off the highs."),
    "metal": ("Metal", None, "Power chords and the Aeolian/Phrygian modes are the language: try palm-muted root notes "
                             "locked to the kick, and a flat-2nd note for menace."),
    "classical": ("Classical / film", None, "Develop the main motif instead of repeating it: sequence it, invert it, "
                                            "or hand it to another instrument (see Melody variations)."),
    "latin": ("Latin", None, "Try organising the rhythm around a clave (3-2 son clave) in the percussion or the comping."),
    "country": ("Country", None, "Country often rides I–IV–V over a train beat; try sliding into chord tones the way a "
                                 "pedal steel does."),
    "ambient": ("Ambient", None, "Let chords sustain and change slowly; a drone on the home note can anchor long, "
                                 "evolving textures."),
}


def clean_context(instruments=None, genre=None):
    """Keep only known choices (they come from a web form)."""
    inst = [i for i in (instruments or []) if i in INSTRUMENTS]
    genre = genre if genre in GENRES else None
    return {"instruments": list(dict.fromkeys(inst)), "genre": genre} if inst or genre else None


def apply_context(result, ctx):
    """Use what the musician told us. The audio analysis can't tell a bass guitar from a guitar's low strings on a
    phone recording, or count a beat twice; the musician can. Their answers decide those questions."""
    if not ctx:
        return
    chosen = [i for i in ctx.get("instruments") or [] if i in INSTRUMENT_INFO]
    fams = list(dict.fromkeys(family(i) for i in chosen))
    inst, genre = set(fams), ctx.get("genre")
    result["context"] = {"instruments": [INSTRUMENTS[i] for i in chosen],
                         "keys": fams,      # the families: what the analysis reasons with
                         "chosen": chosen,  # the instruments themselves

                         "genre": GENRES[genre][0] if genre else None, "genre_key": genre, "notes": []}
    notes = result["context"]["notes"]
    if inst:
        drums = result.get("drums") or {}
        if "drums" not in inst and drums.get("present"):
            result["drums"] = {"present": False, "energy_share_pct": drums.get("energy_share_pct"), "user_said_none": True}
            for sec in result["sections"]:
                sec["drums"] = False
        elif "drums" in inst and not drums.get("present"):
            notes.append("You said there are drums, but they're too quiet or too blended into the mix for us to analyse here.")
        bass = result.get("bass") or {}
        if "bass" in inst and bass.get("low_part_only") and result.get("low_part"):
            # The low notes are a real bass after all (a phone hides the difference).
            result["bass"] = dict(result["low_part"], present=True, from_user=True)
            for sec in result["sections"]:
                sec["bass"] = sec.get("low_end", True)
        elif "bass" in inst and not bass.get("present"):
            notes.append("You said there's a bass, but we couldn't hear any low notes. Phone microphones often lose them.")
        elif "bass" not in inst and bass.get("present"):
            # No bass instrument: the low notes are the main instrument's own lowest part.
            result["low_part"] = bass
            result["bass"] = {"present": False, "low_part_only": True, "user_said_none": True}
            for sec in result["sections"]:
                sec["bass"] = False
        if "drums" not in inst and not result["drums"].get("present"):
            result["drums"]["user_said_none"] = True
        if "bass" not in inst and result["bass"].get("low_part_only"):
            result["bass"]["user_said_none"] = True
        if "drums" in inst and result["drums"].get("present"):
            result["drums"].pop("unsure", None)
        if result["bass"].get("low_part_only"):
            result["low_part_instrument"] = next((i for i in ("guitar", "keys", "strings", "synth", "winds", "voice")
                                                  if i in inst), None)
            picked = [k for k in chosen if family(k) == result["low_part_instrument"]]
            if len(picked) == 1:  # the instrument itself: its own lowest note, MIDI sound and name
                result["low_part_specific"] = picked[0]
    if genre and GENRES[genre][1] and result.get("tempo_bpm"):
        lo, hi = GENRES[genre][1]
        bpm = result["tempo_bpm"]
        alt = next((x for x in (bpm * 2, bpm / 2) if lo <= x <= hi), None)
        if not lo <= bpm <= hi and alt:
            result["context"]["felt_tempo"] = round(alt)
            notes.append(f"{GENRES[genre][0]} usually sits at {lo}–{hi} BPM, so you probably feel this at about "
                         f"{round(alt)} BPM. We counted {'every other beat' if alt > bpm else 'twice per beat'}.")


# Each instrument's lowest usual note (MIDI) and how to name its low part in the report. A transcribed note
# below the instrument's range is a pitch-tracker octave error. None = not told: assume the most common case
# for a solo phone recording, a guitar, and say so in the report.
INSTRUMENT_LOW = {
    "guitar": (40, "the guitar's low strings"),            # E2 (standard tuning; drop D reaches D2 = 38)
    "keys": (21, "the left hand of the piano/keys"),        # A0
    "strings": (36, "the lowest strings (e.g. a cello)"),   # C2
    "winds": (34, "the low notes of the wind/brass"),       # Bb1 (tuba, bass clarinet go lower; rare solo)
    "synth": (0, "the synth's lowest notes"),               # no limit
    "voice": (38, "the lowest sung notes"),                 # D2, a low bass voice
    "bass": (23, "the bass"),                               # B0, a 5-string bass
    None: (40, "the main instrument's own low notes"),
}


def low_part_range(result):
    """(lowest usual note, how to name the low part) for the instrument playing it: the one the musician picked,
    else its family, else (not told) a guitar."""
    spec = result.get("low_part_specific")
    if spec in INSTRUMENT_INFO and INSTRUMENT_INFO[spec][3] is not None:
        return INSTRUMENT_INFO[spec][3], f"the {INSTRUMENTS[spec].split(' / ')[0]}'s lowest notes"
    return INSTRUMENT_LOW.get(result.get("low_part_instrument"), INSTRUMENT_LOW[None])


def finish_low_part(result):
    """When the low notes are the main instrument's own (no separate bass) and nothing sounds below 80 Hz, notes
    below that instrument's range are octave errors and are moved up. Notes it can really play stay as heard."""
    if not (result["bass"].get("low_part_only") and result.get("low_end_missing")):
        return
    lowest = low_part_range(result)[0]
    up = lambda n: n + 12 * ((lowest - n + 11) // 12) if n < lowest else n
    notes = result.get("bass_transcription", [])
    result["octave_raised"] = sum(n["midi"] < lowest for n in notes)
    result["bass_transcription"] = [{**n, "midi": up(n["midi"])} for n in notes]
    if result["octave_raised"] and lowest == 40 and result.get("low_part_raised"):
        result["low_part"] = result["low_part_raised"]  # summary made with the same E2 rule


def lead_instrument(result):
    """The instrument the melody line most likely belongs to, when the musician named exactly one lead."""
    inst = [i for i in ((result.get("context") or {}).get("keys") or []) if i not in ("drums", "bass")]
    if "voice" in inst:
        return "voice"
    return inst[0] if len(inst) == 1 else None


def analyze(path, melody_text=None, cache_dir=None, context=None, light=False):
    """Analyse one track. The audio analysis (the slow part) is cached by file content, so
    re-running the same recording - e.g. to add a melody or retry the AI - is near-instant.
    light=True (tag mode): skip the musician-only work - bass transcription, melody line, sheet music, ideas."""
    import pickle
    result = extras = None
    cache_file = None
    if cache_dir:
        # a full analysis serves tag mode too; a light one only serves tag mode
        for key in ([_cache_key(path), _cache_key(path, True)] if light else [_cache_key(path)]):
            try:
                with open(Path(cache_dir) / f"{key}.pkl", "rb") as f:
                    result, extras = pickle.load(f)
                result = {**result, "file": Path(path).name, "cached": True}
                break
            except (OSError, pickle.PickleError, EOFError, ValueError):
                result = None
        cache_file = Path(cache_dir) / f"{_cache_key(path, light)}.pkl"
    if result is None:
        result, extras = _analyze_audio(path, light)
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
    apply_context(result, context)
    finish_low_part(result)
    if light:
        result["blues_form"] = detect_blues_form(result)
        result["song_form"] = detect_song_form(result)
        return result
    # Sheet music and its rhythm are cheap, so they're made after the cache: improving them needs no
    # re-analysis of the audio.
    result["bass_sheet"] = bass_sheet(result, [(c["start"], c["chord"]) for c in result["chord_timeline"]])
    result["bass_rhythm"] = notation.rhythm_from_score(result["bass_sheet"]) if result.get("bass_sheet") else {}
    result["blues_form"] = detect_blues_form(result)
    result["song_form"] = detect_song_form(result)
    if melody_text:
        result["melody"] = inspiration.analyse_melody(melody_text, key_tonic(main_key), result["mode"], MODES,
                                                      lambda pc: spell(pc, main_key))
        if "error" not in result["melody"]:
            pcs = [n % 12 for n in result["melody"]["variations_midi"]["original"]]
            result["melody"]["check"] = check_melody(extras["chroma"], extras["bounds"], pcs, main_key)
    result["references"] = shared_dna(result)
    result["ideas"] = rule_based_ideas(result)
    result["guitar"] = guitar_capo(result)
    if result.get("blues_form"):
        result["blues_variations"] = blues_variations(result)
    genre = (context or {}).get("genre")
    if genre:
        result["ideas"].append({"area": GENRES[genre][0].lower(), "idea": GENRES[genre][2]})
    # The producer's view (electronic.py) is worked out after the cache too: it depends on what the musician told us
    result["producer"] = electronic.producer_view(result)
    result["ideas"] += electronic.producer_ideas(result, result["producer"])
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
    out["groove"] = inspiration.groove_reference((r.get("drums") or {}).get("traits", []), r.get("tempo_bpm"))
    if r.get("blues_form"):
        out["form"] = {"form": r["blues_form"]["name"], "examples": inspiration.TWELVE_BAR_EXAMPLES}
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

def flat5(fifth):
    """The blue note a half step below the 5th, spelled as a flattened 5th (Ab in D, not G#)."""
    return fifth[:-1] if fifth.endswith("#") else fifth + "b"


# Standard open-position guitar shapes (root pitch class, quality) -> fingering, low E string first (x = not played).
OPEN_SHAPES = {
    (0, ""): ("C", "x32010"), (9, ""): ("A", "x02220"), (7, ""): ("G", "320003"), (4, ""): ("E", "022100"),
    (2, ""): ("D", "xx0232"), (9, "m"): ("Am", "x02210"), (4, "m"): ("Em", "022000"), (2, "m"): ("Dm", "xx0231"),
    (9, "7"): ("A7", "x02020"), (11, "7"): ("B7", "x21202"), (0, "7"): ("C7", "x32310"), (2, "7"): ("D7", "xx0212"),
    (4, "7"): ("E7", "020100"), (7, "7"): ("G7", "320001"), (9, "m7"): ("Am7", "x02010"), (4, "m7"): ("Em7", "022030"),
    (2, "m7"): ("Dm7", "xx0211"),
}
MAX_CAPO = 7  # beyond the 7th fret the neck is too cramped for open shapes to be the easy option


def guitar_capo(r):
    """Where to put a capo so the song's chords become open shapes (as Moises' and musicanalyzer.net's capo modes
    do). For each capo fret, the share of chord time (power chords left out: they're movable shapes that work
    anywhere) whose shape is an open chord. A capo is suggested only if it beats playing without one by a
    quarter of the song and covers at least three quarters of it. Returns None when there's nothing to say."""
    parsed = [(parse_chord(c["chord"]), c["seconds"]) for c in r.get("chords_used") or []]
    parsed = [((p[0], p[1]), t) for p, t in parsed if p]
    chords = [(rq, t) for rq, t in parsed if rq[1] != "5"]
    total = sum(t for _, t in chords)
    # A song played mostly in power chords (blues and rock riffs) gains nothing from a capo: say nothing.
    if not chords or total <= 0 or total < 0.5 * sum(t for _, t in parsed):
        return None

    def at(capo):
        shapes = [(OPEN_SHAPES.get(((root - capo) % 12, q)), root, q, t) for (root, q), t in chords]
        return sum(x[3] for x in shapes if x[0]) / total, shapes
    cover = {k: at(k) for k in range(MAX_CAPO + 1)}
    best = max(range(MAX_CAPO + 1), key=lambda k: (round(cover[k][0], 2), -k))
    capo = best if best and cover[best][0] >= max(0.75, cover[0][0] + 0.25) else 0
    share, shapes = cover[capo]
    if share < 0.75:
        return None  # mostly barre chords whatever you do: nothing helpful to suggest
    k = r["key_index"]
    return {"capo": capo, "open_share_pct": round(100 * share),
            "shapes": [{"sounds": spell(root, k) + q, "shape": s[0], "fingering": s[1]} for s, root, q, _ in shapes if s]}


# How big a difference has to be before a listener notices it - the bar for "they do X, you do Y".
TEMPO_JND = 0.04          # tempo changes of about 4% are noticed (discrimination studies put it at 2-6%)
LOUDNESS_JND_LU = 1.0     # about 1 dB/LU is the smallest loudness change most people hear
BAND_DIFF_DB = 3.0        # a 3 dB shift in one frequency band is a clear tonal difference
BRIGHTNESS_RATIO = 2 ** (1 / 3)  # a third of an octave brighter or darker
WIDTH_DIFF = 0.2          # stereo correlation (-1..1)
SYNC_DIFF_PCT = 15        # share of off-beat hits, percentage points
CONTRAST_DIFF = 15        # energy contrast between sections, on the 0-100 arousal scale (a mood band is ~20)
LRA_DIFF_LU = 3 * LOUDNESS_JND_LU  # loudness range: three just-noticeable steps
HARMONIC_RATIO = 1.5      # chords changing half as often again (or more) is a different harmonic pace


PUMP_DIFF_DB = electronic.PUMP_DB  # a difference in ducking depth you can hear
PLR_DIFF_DB = 3.0          # peak-to-loudness ratio: 3 dB more or less limiting is a clearly different density
DROP_DIFF_DB = electronic.DROP_DB
MAX_REFERENCES = 3
AREA_GROUPS = {"tempo": "tempo & groove", "rhythm": "tempo & groove", "key": "key & harmony",
               "harmony": "key & harmony", "arrangement": "arrangement & energy", "sound": "sound & mix",
               "production": "sound & mix"}
# What it means when every song you admire differs from yours the same way (keys of compare_tracks' notes).
CONSENSUS = {
    "tempo_faster": "move faster than yours: pace may be part of what draws you to them.",
    "tempo_slower": "move slower than yours: they take more time, and give each beat more room.",
    "peak_sooner": "reach full energy sooner than yours: they get to the point faster.",
    "peak_later": "take longer to reach full energy: a slower build.",
    "contrast_more": "move between quiet and loud far more than yours: a real breakdown and a bigger return.",
    "contrast_less": "hold one level more than yours does.",
    "sync_more": "put more of their hits between the beats: more syncopation.",
    "sync_less": "sit more squarely on the beat than yours.",
    "drums_have": "have drums, and yours doesn't.",
    "bass_have": "have a separate bass line, and yours doesn't.",
    "chords_faster": "change chords more often than yours.",
    "chords_slower": "hold their chords longer than yours.",
    "louder": "are mastered louder than yours.",
    "more_even": "keep a more even level (less loudness range) than yours.",
    "wider": "are noticeably wider in stereo than yours.",
    "brighter": "are clearly brighter than yours.",
    "darker": "are darker and warmer than yours.",
    "pump_more": "pump with the kick (sidechain) more than yours.",
    "pump_less": "pump less than yours: their pads and synths hold steadier.",
    "drop_harder": "land their drop harder than yours: more contrast going into it.",
    "more_limited": "are more heavily limited than yours: denser and louder, with flatter peaks.",
    "more_punch": "keep more punch in their peaks than yours: less limiting.",
    "low_mono": "keep the low end in mono, and yours is wide.",
    "four_floor": "put the kick on every beat (four-on-the-floor), and yours doesn't.",
    "breakdown_have": "have a breakdown (the drums drop out), and yours doesn't.",
}
for _b in ("sub (<60 Hz)", "bass (60–250)", "low-mid (250–2k)", "high-mid (2k–6k)", "air (>6k)"):
    CONSENSUS[f"band_more_{_b}"] = f"put more energy into the {_b} range than yours."
    CONSENSUS[f"band_less_{_b}"] = f"put less energy into the {_b} range than yours."


def _first_peak(r):
    """When the track first gets within 10 points of its highest section energy (arousal 0-100)."""
    secs = (r.get("feel") or {}).get("sections") or []
    if not secs:
        return None
    top = max(s["arousal"] for s in secs)
    return next(s["start"] for s in secs if s["arousal"] >= top - 10)


def _key_relation(a, b):
    if a == b:
        return "the same key"
    nb = camelot_neighbours(a)
    if nb["relative"] == b:
        return "relative keys (the same notes)"
    if b in (nb["up a fifth"], nb["down a fifth"]):
        return "neighbours on the Camelot wheel (a fifth apart)"
    if key_tonic(a) == key_tonic(b):
        return "the same home note, one major and one minor"
    return f"{min((key_tonic(a) - key_tonic(b)) % 12, (key_tonic(b) - key_tonic(a)) % 12)} semitones apart"


def _producer(r):
    """The producer's view for a result, worked out on the spot for results saved before it existed."""
    return r.get("producer") if "producer" in r else electronic.producer_view(r)


def compare_tracks(mine, ref):
    """Your track next to a reference you admire: the same measurements side by side, and - only where the gap is
    big enough to hear (the constants above) - what they do differently and what you could try. Each note has a
    key (what differs, which way) so notes from several references can be matched up. Comparing a track with
    itself reports no differences."""
    rows, notes = [], []
    phone = lambda r: (r["sound"]["frequency_balance_pct"]["sub (<60 Hz)"] < 1
                       and r["sound"]["frequency_balance_pct"]["air (>6k)"] < 1)

    # Tempo (a half- or double-time reading is the same pulse, not a different tempo)
    tm, tr = mine.get("tempo_bpm"), ref.get("tempo_bpm")
    show = lambda r, t: (("~" if r.get("tempo_approximate") else "") + f"{t:.0f} BPM") if t else "free"
    rows.append(("Tempo", show(mine, tm), show(ref, tr)))
    # an approximate tempo (weak pulse) can't support a "you're 6% slower" claim
    if tm and tr and not mine.get("tempo_approximate") and not ref.get("tempo_approximate"):
        ratio, factor = min(((tm * f / tr, f) for f in (1, 2, 0.5)), key=lambda x: abs(np.log(x[0])))
        if abs(ratio - 1) > TEMPO_JND:
            faster = "faster" if ratio < 1 else "slower"
            felt = (f" (your {tm:.0f} BPM felt at {'double' if factor == 2 else 'half'} speed is {tm * factor:.0f})"
                    if factor != 1 else "")
            notes.append(("tempo", f"Theirs moves about {abs(1 / ratio - 1):.0%} {faster}: {tr:.0f} BPM against your "
                          f"{tm * factor:.0f}{felt}. Try yours {'faster' if ratio < 1 else 'slower'}, around "
                          f"{tr / factor:.0f} BPM as you count it, and see whether the feel you're after is partly "
                          "its pace.", "tempo_faster" if ratio < 1 else "tempo_slower"))
        elif factor != 1:
            notes.append(("tempo", f"Same pulse, but theirs is felt at {'double' if factor == 2 else 'half'} your "
                          "counted speed: the drums or strumming mark different beats.", "tempo_felt"))
    # Key, mode, metre
    rows.append(("Key", f"{mine['key']} ({mine['camelot']})", f"{ref['key']} ({ref['camelot']})"))
    rows.append(("Mode", mine["mode"], ref["mode"]))
    rel = _key_relation(mine["key_index"], ref["key_index"])
    if rel != "the same key":
        notes.append(("key", f"Your keys are {rel}. To borrow their sound directly, or to play the two back to back, "
                      f"try yours in {ref['key']}.", "key_other"))
    if mine["mode"] != ref["mode"] and ref.get("mode_confidence") != "low":
        notes.append(("harmony", f"They lean on {ref['mode']}, you on {mine['mode']}: {MODE_INFO.get(ref['mode'], '')} "
                      "One borrowed colour note can bring that flavour into a section of yours.", f"mode_{ref['mode']}"))
    mm, mr = (mine.get("meter") or {}).get("label"), (ref.get("meter") or {}).get("label")
    rows.append(("Time signature", mm or "-", mr or "-"))
    # Shape: length, sections, how soon it gets going, how much it moves
    rows.append(("Length", mine["duration"], ref["duration"]))
    rows.append(("Sections", " ".join(s["label"] for s in mine["sections"]), " ".join(s["label"] for s in ref["sections"])))
    pm, pr = _first_peak(mine), _first_peak(ref)
    if pm is not None and pr is not None:
        rows.append(("First reaches full energy", fmt_time(pm), fmt_time(pr)))
        if pm - pr > 20 and pm > 1.5 * max(pr, 1):
            notes.append(("arrangement", f"They reach full energy by {fmt_time(pr)}, you by {fmt_time(pm)}. A shorter "
                          "intro, or bringing your loudest section's idea in earlier, gets the listener there sooner.",
                          "peak_sooner"))
        elif pr - pm > 20 and pr > 1.5 * max(pm, 1):
            notes.append(("arrangement", f"They take until {fmt_time(pr)} to reach full energy, you get there by "
                          f"{fmt_time(pm)}: a slower build leaves you more room to grow later.", "peak_later"))
    fm, fr = mine.get("feel") or {}, ref.get("feel") or {}
    if fm and fr:
        rows.append(("Energy · mood", f"{fm['energy']}/10 · {fm['mood']}", f"{fr['energy']}/10 · {fr['mood']}"))
        if fr["contrast"] - fm["contrast"] > CONTRAST_DIFF:
            notes.append(("arrangement", f"Their sections differ much more in energy (theirs {fr['arc']}; yours "
                          f"{fm['arc']}). A real breakdown, or a fuller last section, would give yours that shape.",
                          "contrast_more"))
        elif fm["contrast"] - fr["contrast"] > CONTRAST_DIFF:
            notes.append(("arrangement", "Yours moves between quiet and loud far more than theirs: theirs holds one "
                          "level, which is part of its steady, hypnotic feel.", "contrast_less"))
    # Dance arrangement (when either is a dance track): where the first drop lands, how hard, a breakdown
    dm_, dr_ = (_producer(mine) or {}).get("map"), (_producer(ref) or {}).get("map")
    if dm_ or dr_:
        drop = lambda m: (f"{fmt_time(m['first_drop_sec'])} (bar {m['first_drop_bar']})"
                          if m and m.get("first_drop_sec") is not None else "from the start" if m else "-")
        imp = lambda m: f"+{m['impact_db']} dB" if m and m.get("impact_db") is not None else "-"
        rows += [("First drop", drop(dm_), drop(dr_)), ("Drop lands", imp(dm_), imp(dr_))]
        if dm_ and dr_ and dm_.get("impact_db") is not None and dr_.get("impact_db") is not None \
                and dr_["impact_db"] - dm_["impact_db"] >= DROP_DIFF_DB:
            notes.append(("arrangement", f"Their drop lands {dr_['impact_db']} dB above what leads into it, yours "
                          f"{dm_['impact_db']} dB. Thin out your breakdown, or cut the kick and low end in the bar "
                          "before the drop, so it has further to jump.", "drop_harder"))
        if dm_ and dr_ and dr_["has_breakdown"] and not dm_["has_breakdown"]:
            notes.append(("arrangement", "Theirs has a breakdown (the drums drop out, then build back) and yours keeps "
                          "the drums going throughout. Even 8 bars without the kick resets the listener's ear.",
                          "breakdown_have"))
    # Groove
    sm, sr_ = mine["rhythm"].get("syncopation_pct"), ref["rhythm"].get("syncopation_pct")
    if sm is not None and sr_ is not None:
        rows.append(("Off-beat hits", f"{sm}%", f"{sr_}%"))
        if sr_ - sm > SYNC_DIFF_PCT:
            notes.append(("rhythm", f"{sr_}% of their strong hits fall between the beats against your {sm}%. Anticipate a "
                          "few chord changes by an 8th note, or push the kick or strum off the beat.", "sync_more"))
        elif sm - sr_ > SYNC_DIFF_PCT:
            notes.append(("rhythm", f"Yours is more syncopated ({sm}% off-beat against their {sr_}%); theirs gets its "
                          "drive from sitting squarely on the beat.", "sync_less"))
    dm, dr = bool((mine.get("drums") or {}).get("present")), bool((ref.get("drums") or {}).get("present"))
    bm, br = bool((mine.get("bass") or {}).get("present")), bool((ref.get("bass") or {}).get("present"))
    rows.append(("Drums · bass", f"{'yes' if dm else 'no'} · {'yes' if bm else 'no'}", f"{'yes' if dr else 'no'} · {'yes' if br else 'no'}"))
    if dr and not dm:
        notes.append(("arrangement", "Theirs has drums and yours doesn't. Even a simple kick-and-snare pattern, or a "
                      "percussive strum, would bring in some of that push.", "drums_have"))
    if br and not bm:
        notes.append(("arrangement", "Theirs has a separate bass line. A bass following your chord roots would add "
                      "the weight theirs has.", "bass_have"))
    if dm and dr:
        km, kr = (mine["drums"].get("traits") or []), (ref["drums"].get("traits") or [])
        rows.append(("Drum pattern", ", ".join(km) or "-", ", ".join(kr) or "-"))
        if "four-on-the-floor" in kr and "four-on-the-floor" not in km:
            notes.append(("rhythm", "Their kick lands on every beat (four-on-the-floor), the engine of house and "
                          "techno; yours doesn't. Try a kick on all four beats for a section and hear how it drives.",
                          "four_floor"))
    pmp, prp = (mine.get("pump") or {}), (ref.get("pump") or {})
    if pmp and prp:
        pump_txt = lambda p: f"{p['depth_db']} dB ({p['kind']})" if p.get("kind") else "none"
        rows.append(("Sidechain pump", pump_txt(pmp), pump_txt(prp)))
        if prp["depth_db"] - pmp["depth_db"] >= PUMP_DIFF_DB:
            yours = "barely move" if not pmp.get("kind") else f"duck {pmp['depth_db']} dB"
            notes.append(("production", f"Their pads and synths duck about {prp['depth_db']} dB after every kick and "
                          f"swell back (sidechain pumping); yours {yours}. A compressor on the pads keyed from the "
                          "kick, with the release timed to the beat, gets you that breathing groove.", "pump_more"))
        elif pmp["depth_db"] - prp["depth_db"] >= PUMP_DIFF_DB:
            notes.append(("production", f"Yours pumps harder ({pmp['depth_db']} dB against their "
                          f"{prp['depth_db'] or 'none'}): theirs holds steadier under the kick.", "pump_less"))
    # Instruments (the model's opinion, when it's installed): what they feature that yours hardly has
    him, hir = (mine.get("heard") or {}).get("instruments"), (ref.get("heard") or {}).get("instruments")
    if him is not None and hir is not None and (him or hir):
        top3 = lambda xs: ", ".join(x["name"] for x in xs[:3]) or "nothing clear"
        rows.append(("Instruments heard", top3(him), top3(hir)))
        have = {x["key"] for x in him}  # everything in yours at HEARD_SHOW or more
        # drums and bass: our own measurement compares those (above), more reliably than the model
        for x in [x for x in hir if x["p"] >= HEARD_FEATURE and x["key"] not in have
                  and family(x["key"]) not in ("drums", "bass")][:2]:
            notes.append(("arrangement", f"Theirs features {x['name']} (the instrument model is {round(100 * x['p'])}% "
                          "sure); in yours it hardly registers. A part on it, or a sound in the same register and "
                          "colour, could bring in some of what you like about theirs.", f"inst_{x['key']}"))
    # Harmony: how often the chords change
    hm, hr = mine.get("harmonic_rhythm_beats"), ref.get("harmonic_rhythm_beats")
    if hm and hr:
        rows.append(("A chord change every", f"{hm} beats", f"{hr} beats"))
        if hm / hr >= HARMONIC_RATIO:
            notes.append(("harmony", f"Their chords change about every {hr} beats, yours every {hm}: more movement "
                          "under the melody. Try splitting some of your long chords in two.", "chords_faster"))
        elif hr / hm >= HARMONIC_RATIO:
            notes.append(("harmony", f"Their chords sit longer (every {hr} beats against your {hm}), which gives the "
                          "melody room. Holding one of your chords twice as long is worth a try.", "chords_slower"))
    # Sound: loudness, punch, tone, width, frequency balance
    lm, lr = (mine["sound"].get("streaming") or {}), (ref["sound"].get("streaming") or {})
    if lm.get("lufs") is not None and lr.get("lufs") is not None:
        rows.append(("Loudness", f"{lm['lufs']} LUFS", f"{lr['lufs']} LUFS"))
        if lr["lufs"] - lm["lufs"] > LOUDNESS_JND_LU:
            notes.append(("sound", f"Theirs is mastered {lr['lufs'] - lm['lufs']:.1f} LU louder. On Spotify both are "
                          f"levelled to about {SPOTIFY_TARGET_LUFS:g} LUFS, so match the mix first; loudness comes last, "
                          "from a limiter on the master.", "louder"))
        if lm.get("plr_db") is not None and lr.get("plr_db") is not None:
            rows.append(("Peak-to-loudness (punch)", f"{lm['plr_db']} dB", f"{lr['plr_db']} dB"))
            if lm["plr_db"] - lr["plr_db"] >= PLR_DIFF_DB:
                notes.append(("production", f"Theirs is more heavily limited (peak-to-loudness {lr['plr_db']} dB against "
                              f"your {lm['plr_db']}): a denser, flatter, louder master. If that's the sound you want, "
                              "a compressor on the mix bus and a limiter on the master get you there; the cost is punch.",
                              "more_limited"))
            elif lr["plr_db"] - lm["plr_db"] >= PLR_DIFF_DB:
                notes.append(("production", f"Theirs keeps more of its peaks (peak-to-loudness {lr['plr_db']} dB against "
                              f"your {lm['plr_db']}): it hits harder at the same loudness. Ease off the limiter, or "
                              "compress less.", "more_punch"))
        if lm.get("loudness_range_lu") is not None and lr.get("loudness_range_lu") is not None:
            rows.append(("Loudness range", f"{lm['loudness_range_lu']} LU", f"{lr['loudness_range_lu']} LU"))
            if lm["loudness_range_lu"] - lr["loudness_range_lu"] > LRA_DIFF_LU:
                notes.append(("sound", f"Your level moves more ({lm['loudness_range_lu']} LU against their "
                              f"{lr['loudness_range_lu']}): theirs is more even, usually from compression.", "more_even"))
        cm, cr = lm.get("stereo_correlation"), lr.get("stereo_correlation")
        if cm is not None and cr is not None:
            rows.append(("Stereo correlation", f"{cm}", f"{cr}"))
            if cm - cr > WIDTH_DIFF:
                notes.append(("sound", "Theirs is noticeably wider: instruments panned apart, or stereo reverb and "
                              "doubled parts on the sides.", "wider"))
        wm, wr = lm.get("band_widths") or {}, lr.get("band_widths") or {}
        if wm.get("low") is not None and wr.get("low") is not None:
            wide = lambda st: "wide" if electronic.low_end_wide(st) else "mono"
            rows.append((f"Low end (below {electronic.LOW_END_HZ} Hz)", wide(lm), wide(lr)))
            if electronic.low_end_wide(lm) and not electronic.low_end_wide(lr):
                notes.append(("production", f"Their low end is mono and yours is wide (correlation {wm['low']} against "
                              f"their {wr['low']}). Mono the bass and kick below about {electronic.LOW_END_HZ} Hz and "
                              "keep the width higher up.", "low_mono"))
    bm_, br_ = mine["sound"]["brightness_hz"], ref["sound"]["brightness_hz"]
    rows.append(("Brightness", f"{bm_} Hz", f"{br_} Hz"))
    if br_ / max(bm_, 1) >= BRIGHTNESS_RATIO:
        notes.append(("sound", "Theirs is clearly brighter: more cymbals, string noise or upper harmonics. A high "
                      "shelf, a brighter instrument or a high part would close the gap.", "brighter"))
    elif bm_ / max(br_, 1) >= BRIGHTNESS_RATIO:
        notes.append(("sound", "Yours is clearly brighter than theirs; theirs sits darker and warmer.", "darker"))
    fbm, fbr = mine["sound"]["frequency_balance_pct"], ref["sound"]["frequency_balance_pct"]
    gaps = [(band, 10 * np.log10((fbr[band] + 0.1) / (fbm[band] + 0.1))) for band in fbm]
    big = [(band, d) for band, d in gaps if abs(d) >= BAND_DIFF_DB]
    if big:
        band, d = max(big, key=lambda x: abs(x[1]))
        more = "more" if d > 0 else "less"
        caveat = (" Part of this is the recording: a phone microphone loses the lowest and highest ranges."
                  if phone(mine) and not phone(ref) and band in ("sub (<60 Hz)", "air (>6k)") else "")
        notes.append(("sound", f"Theirs puts {abs(d):.0f} dB {more} energy into the {band} range than yours."
                      + caveat, f"band_{more}_{band}"))
    return {"reference": ref["file"], "rows": [{"aspect": a, "yours": y, "theirs": t} for a, y, t in rows],
            "ideas": [{"area": a, "idea": i, "key": k} for a, i, k in notes]}


def compare_all(mine, refs):
    """Your track next to every reference: one comparison each, then what they have in common - a difference that
    every song you admire shares is a stronger lead than one song's quirk - and where you're already close."""
    refs = [x for x in refs if x and "error" not in x]
    each = [compare_tracks(mine, x) for x in refs]
    if not each:
        return None
    shared = []
    if len(each) >= 2:
        common = set.intersection(*[{i["key"] for i in c["ideas"]} for c in each])
        for key in dict.fromkeys(i["key"] for i in each[0]["ideas"]):  # in the order the first comparison lists
            phrase = CONSENSUS.get(key) or (f"feature {INSTRUMENTS.get(key[5:], key[5:])}, which hardly registers in "
                                            "yours." if key.startswith("inst_") else None)
            if key in common and phrase:
                area = next(i["area"] for i in each[0]["ideas"] if i["key"] == key)
                who = "Both songs you admire" if len(each) == 2 else f"All {len(each)} songs you admire"
                shared.append({"area": area, "key": key, "idea": f"{who} {phrase}"})
    groups = list(dict.fromkeys(AREA_GROUPS.values()))
    differs = {g: sum(1 for c in each if any(AREA_GROUPS.get(i["area"]) == g for i in c["ideas"])) for g in groups}
    curves = [{"file": x["file"], "duration_sec": x.get("duration_sec"),
               "sound": {k: (x.get("sound") or {}).get(k) for k in ("energy_line", "octave_db")}} for x in refs]
    return {"references": [c["reference"] for c in each], "each": each, "shared": shared, "curves": curves,
            "close": [g for g in groups if not differs[g]],
            "different": [{"area": g, "references": n} for g, n in differs.items() if n]}


def rule_based_ideas(r):
    k = r["key_index"]
    t = key_tonic(k)
    band = bool((r.get("drums") or {}).get("present") or (r.get("bass") or {}).get("present"))
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
                      f"either retune to A440 or detune those by {cents:+d} cents so everything sits together."})

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
    if r.get("blues_form") and key_is_minor(k):
        ideas.append({"area": "melody/harmony", "idea":
                      f"It's a minor blues: the blues scale ({note(0)} {note(3)} {note(5)} {flat5(note(7))} {note(7)} "
                      f"{note(10)}) fits the chords directly, and its b5 ({flat5(note(7))}) is the colour note. On the V7 "
                      f"chord ({chord(7, '7')}), play its major 3rd ({note(11)}) for a strong pull back to {note(0)}."})
    elif r.get("blues_form"):
        ideas.append({"area": "melody/harmony", "idea":
                      f"It's a blues, so the melody lives on the blues scale — "
                      f"{' '.join([note(0), note(3), note(5), flat5(note(7)), note(7), note(10)])} "
                      f"(minor pentatonic plus the b5) — over major, dominant-7th chords. On the {chord(0)} chord, bend "
                      f"the minor 3rd ({note(3)}) up towards the major 3rd ({note(4)}); end phrases on {note(0)} or the "
                      f"b7 ({note(10)}). Over the {chord(5)} chord, {note(10)} and {note(3)} sound bluesiest."})
    elif r.get("mode_confidence") == "low":
        ideas.append({"area": "melody/harmony", "idea":
                      f"The scale could be {r['mode']} or {r['mode_runner_up']}: the notes that would settle it "
                      f"hardly appear. That's an opportunity, because one melody note can decide it for you. "
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
        ideas.append({"area": "bass", "idea": "The bass mostly plays chord roots. Try approach notes: on the last 8th "
                      "before each chord change, play a note a half-step above or below the next root. It pulls the "
                      "ear into the new chord."})

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
                      f"The track never visits the {numeral} chord ({chord(off, q)}), even though it fits the scale. "
                      f"That makes it a fresh place to go for a bridge or a new section."})

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
            variants.append(f"a half-time groove (drums feel like ~{bpm / 2:.0f} BPM) for a heavier, spacious section"
                            if band else f"a half-time feel (strum or pick on every other beat, ~{bpm / 2:.0f} BPM) "
                                         f"for a slower, spacious section")
        if bpm < 100:
            variants.append(f"a double-time feel at ~{bpm * 2:.0f} BPM for an energetic remix")
        ideas.append({"area": "remix", "idea": f"Same chords, new feel: try {' or '.join(variants)}."})
    else:
        ideas.append({"area": "rhythm", "idea": "There's no clear pulse. Try a slow, soft pulse (around 70 BPM) under "
                      "one section, so the free-time parts around it stand out."})

    # Energy arc
    s = r["sound"]
    if s["energy_flat"]:
        mid = fmt_time(r["duration_sec"] / 2)
        ideas.append({"area": "arrangement", "idea": f"The energy stays almost flat. Create a breakdown around {mid} "
                      + ("(drop the drums and bass for 4–8 bars) and bring everything back for a bigger return."
                         if band else "(play softer and sparser for 4–8 bars: single notes, or just the low strings), "
                                      "then come back at full strength.")})
    build, quiet = s["biggest_build"], s["quietest_moment"]
    to_sec = lambda ts: int(ts.split(":")[0]) * 60 + int(ts.split(":")[1])
    if build and quiet and abs(to_sec(build["time"]) - to_sec(quiet)) <= 10:
        # The quietest moment and the biggest build are one event: a dip and a comeback.
        ideas.append({"area": "arrangement", "idea": f"The biggest contrast is the dip around {quiet} that comes back "
                      f"+{build['db']} dB by {build['time']}. "
                      + ("Make it a real breakdown: strip it to one instrument for 4–8 bars, then hit the return with "
                         "everything (a riser or a bar of silence just before helps)." if band else
                         "Make the contrast deliberate: play that passage quieter and sparser for 4–8 bars, then hit "
                         "the return hard (a beat of silence just before helps).")})
        quiet = None
    elif build and not s["energy_flat"]:
        ideas.append({"area": "arrangement", "idea": f"The strongest moment is the build into {build['time']} "
                      f"(+{build['db']} dB). Set it up harder: a riser, a drum fill, or a bar of silence "
                      f"right before it."})
    if quiet:
        ideas.append({"area": "arrangement", "idea": f"The quietest stretch is around {s['quietest_moment']}. That's a "
                      f"natural spot for a solo instrument, a field recording, or a new melodic idea."})

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
                      "by an 8th note, or " + ("add a syncopated percussion layer (shaker/rim) to add swing."
                      if band else "add muted percussive 'chops' on the off-beats to make it swing.")})
    elif g == "heavily syncopated":
        ideas.append({"area": "rhythm", "idea": "The groove is already very syncopated. A section of straight, "
                      "on-beat hits would give you contrast, and make the syncopation land harder when it comes back."})

    # Sound balance
    bands = s["frequency_balance_pct"]
    if bands["sub (<60 Hz)"] < 1 and bands["air (>6k)"] < 1:
        ideas.append({"area": "sound", "idea": "There's almost nothing below 60 Hz or above 6 kHz, which is normal for a "
                      "phone recording, so judge the mix from a proper one. When you produce it, those empty ranges are "
                      "room to fill: a sub-bass under the roots, and something airy on top (shimmer, hi-hats, a high "
                      "counter-melody)."})
    elif bands["air (>6k)"] < 2:
        ideas.append({"area": "sound", "idea": "There's very little high-end 'air'. A bright layer (hi-hats, shimmer "
                      "reverb, an airy pad or a high counter-melody) would open up the mix."})
    if bands["sub (<60 Hz)"] + bands["bass (60–250)"] < 15:
        ideas.append({"area": "sound", "idea": "The low end is light. A sub-bass, or a bass line following the chord "
                      "roots, would give it more weight."})
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


def low_part(r):
    """What to call the transcribed low line: a bass instrument, or the main instrument's own lowest part
    (no separate bass detected, e.g. solo guitar). Returns (word for text, word for file names)."""
    return ("lowest part", "low_part") if (r.get("bass") or {}).get("low_part_only") else ("bass", "bass")


def export_sheets(r, out_dir):
    """Bass sheet music: MusicXML (opens in MuseScore, Sibelius, Dorico, most DAWs) and, when Verovio is
    installed, an engraved page to view or print to PDF."""
    sheet = r.get("bass_sheet")
    if not sheet or not sheet["notes"]:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w.\-]+", "_", Path(r["file"]).stem)
    k = r["key_index"]
    word, tag = low_part(r)
    # six-string guitars (and the banjo) are written an octave above how they sound, in treble clef
    guitar = r.get("low_part_instrument") == "guitar" and r.get("low_part_specific", "guitar") in GUITARS | {"banjo"}
    # Bass and guitar are both written an octave above how they sound; a guitar in treble clef.
    xml = notation.to_musicxml(sheet, f"{Path(r['file']).stem} — {word}",
                               "Guitar (low part)" if guitar else word.capitalize(), key_fifths(k), r["tempo_bpm"],
                               lambda pc: spell(pc, k), parse_chord, clef=("G", 2) if guitar else ("F", 4),
                               octave_up=tag == "bass" or guitar)
    files = [{"file": f"{stem}_{tag}.musicxml", "label": f"{word} sheet music (MusicXML: open in MuseScore, Sibelius, "
                                                         "Dorico or a DAW to edit and play)"}]
    (out_dir / files[0]["file"]).write_text(xml, encoding="utf-8")
    pages = notation.render_svg(xml)
    if pages:
        note = (f" · {r['key']} · {r['meter']['label']} · {r['tempo_bpm']:.0f} BPM · bar 1 starts at "
                f"{fmt_time(max(0, sheet['bar1_time']))} in the recording · automatic transcription: check it by ear")
        (out_dir / f"{stem}_{tag}.html").write_text(
            notation.sheet_html(f"{Path(r['file']).stem} — {word}", pages, html.escape(note)), encoding="utf-8")
        files.append({"file": f"{stem}_{tag}.html", "label": f"{word} sheet music, engraved ({len(pages)} page(s); "
                                                            "open in a browser, print to PDF)"})
    return files


def export_midis(r, out_dir):
    """Write MIDI for each detected loop, a relative-substitution variation, and AI progressions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w.\-]+", "_", Path(r["file"]).stem)
    bpm = r["tempo_bpm"] or 90
    cents = r.get("tuning_cents", 0) if abs(r.get("tuning_cents", 0)) >= 10 else 0
    bar = (r.get("meter") or {}).get("beats_per_bar", 4)  # one chord = one full bar (3 beats in 3/4)
    files = []
    line = r.get("bass_transcription") or []
    word, tag = low_part(r)
    # General MIDI sound: bass (finger) for a bass; otherwise the instrument the musician named
    spec = r.get("low_part_specific")
    basses = [k for k in (r.get("context") or {}).get("chosen") or [] if family(k) == "bass"]
    if not spec and (r.get("bass") or {}).get("present") and len(basses) == 1:
        spec = basses[0]  # a double bass or a synth bass sounds like one in the MIDI file
    program = (INSTRUMENT_INFO[spec][2] if spec in INSTRUMENT_INFO else
               {"guitar": 25, "keys": 0, "strings": 42, "synth": 81, "winds": 71, "voice": 53}.get(
                   r.get("low_part_instrument"), 33))
    if line:
        fname = f"{stem}_{tag}_transcription.mid"
        write_notes_midi(out_dir / fname, [(n["start"], n["end"], n["midi"], n["velocity"]) for n in line],
                         bpm, cents=cents, beats_per_bar=bar, program=program)
        files.append({"file": fname, "label": f"{word}, note by note ({len(line)} notes at their real times)",
                      "chords": []})
    sheet = r.get("bass_sheet")
    if sheet and sheet["notes"]:
        # The same notes as written on the sheet: on the grid, steady tempo, bar 1 at the start.
        sec = lambda slot: slot / notation.SLOTS_PER_BEAT * 60 / bpm
        fname = f"{stem}_{tag}_from_sheet.mid"
        write_notes_midi(out_dir / fname, [(sec(s), sec(s + n), m, v) for s, n, m, v in sheet["notes"]],
                         bpm, cents=cents, beats_per_bar=bar, program=program)
        files.append({"file": fname, "label": f"{word}, from the sheet music (on the beat grid, {sheet['bars']} bars, "
                                              f"bar 1 = {fmt_time(max(0, sheet['bar1_time']))} in the recording)",
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
    for v in r.get("blues_variations") or []:
        save(f"blues {v['name']}", v["chords"])
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
If a blues Form is given (8-, 12- or 16-bar), the alt_progressions must keep that form: exactly one chord per bar,
as many chords as the form has bars (e.g. a quick change, dominant 7ths throughout, a ii-V before the turnaround).
Chord names ending in 5 (D5, C#5) are power chords (root and fifth), not single notes; call single notes "the note C#".
If the musician named the instruments, write ideas for that line-up; suggesting an added instrument is fine if you say so.
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
    ctx = r.get("context") or {}
    lines = [f"The musician says: instruments {', '.join(ctx['instruments']) or 'not given'}; genre "
             f"{ctx.get('genre') or 'not given'}." + (f" Felt tempo ~{ctx['felt_tempo']} BPM." if ctx.get("felt_tempo") else "")
             ] if ctx else []
    lines += [
        f"Length {r['duration']}, {r['tempo_bpm'] or 'free'} BPM{' (approximate, weak beat)' if r.get('tempo_approximate') else ''} "
        f"({rh['tempo_feel']}), {s['tone']} tone.",
        f"Key {r['key']} (confidence {r['key_confidence']}{', uncertain' if r['key_confidence'] < 0.6 else ''}); "
        f"mode {r['mode']} ({r['mode_confidence']} confidence); scale {' '.join(r['scale_notes'])}."
        + (f" Tuned {r['tuning_cents']:+d} cents." if abs(r.get('tuning_cents', 0)) >= 10 else ""),
        "Sections: " + "; ".join(f"{x['label']} {fmt_time(x['start'])}-{fmt_time(x['end'])} {x['energy']} energy "
                                 f"[{', '.join(x['main_chords'][:3])}]" for x in r["sections"]) + ".",
        "Most used chords: " + ", ".join(c["chord"] for c in r["chords_used"][:6]) + ".",
        "Chord timeline: " + ", ".join(f"{c['time']} {c['chord']}" for c in r["chord_timeline"][:28])
        + (" ..." if len(r["chord_timeline"]) > 28 else "") + ".",
    ]
    if r.get("blues_form"):
        b = r["blues_form"]
        lines.append(f"Form: {b['name']}, choruses of {b['chorus_sec']} s starting at "
                     + ", ".join(fmt_time(t) for t in b["choruses"]) + ".")
    if r.get("song_form"):
        f_ = r["song_form"]
        lines.append(f"Song form: AABA, 8-bar sections starting at " + ", ".join(fmt_time(t) for t in f_["starts"])
                     + " (the third is the bridge).")
    if r["loops"]:
        lines.append("Chord loops: " + "; ".join(f"{' - '.join(l['chords'])} ({' - '.join(l['roman'])})"
                                                 for l in r["loops"]) + ".")
    if d.get("grid"):
        lines.append(f"Drums: {', '.join(d.get('traits') or ['groove'])}; {d['repetition']} repetition; "
                     f"16-step grid {d['grid']} (K kick, S snare, X both)"
                     + (f"; fills at {', '.join(fmt_time(t) for t in d['fills'])}" if d.get("fills") else "; no fills")
                     + ".")
    if b.get("low_part_only"):
        lines.append("No drums and no separate bass instrument: one instrument (e.g. solo guitar) playing its own "
                     "bass notes.")
        b = r.get("low_part") or {}
    elif d.get("string_attacks"):
        lines.append("No drums (the percussive sound is picking/strumming).")
    if b.get("present"):
        brh = r.get("bass_rhythm") or {}
        lines.append(f"{'Lowest part' if (r.get('bass') or {}).get('low_part_only') else 'Bass'}: {b['style']}; main notes " + ", ".join(f"{n['note']} {n['pct']}%" for n in b["main_notes"][:3])
                     + (f"; usual note positions in the bar {brh.get('usual_positions')} (16ths; X = most bars, "
                        f"x = some), {brh['syncopated_pct']}% syncopated" if brh else "") + ".")
    f = r.get("feel") or {}
    if f:
        lines.append(f"Feel: energy {f['energy']}/10, {f['mood']}; the track {f['arc']}; per section "
                     + ", ".join(f"{x['label']} {x['energy']}/10 {x['mood']}" for x in f["sections"]) + ".")
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
    c = _norm_comparison(r.get("comparison"))
    for e in (c or {}).get("each", []):
        lines.append(f"The musician admires {e['reference']}. Side by side, yours vs theirs: "
                     + "; ".join(f"{x['aspect']} {x['yours']} vs {x['theirs']}" for x in e["rows"]) + ".")
        if e["ideas"]:
            lines.append("Differences big enough to hear: " + " | ".join(i["idea"][:110] for i in e["ideas"]) + ".")
    if (c or {}).get("shared"):
        lines.append("What every song they admire shares that theirs doesn't: " + " ".join(i["idea"] for i in c["shared"])
                     + " Help them get closer to this, in their own way.")
    h = r.get("heard") or {}
    if h.get("instruments"):
        lines.append("A trained model hears: " + ", ".join(f"{x['name']} ({x['p']:.0%})" for x in h["instruments"])
                     + ". Suggest ideas for these instruments, and for ones that would contrast with them.")
    v = r.get("producer") or {}
    if v:
        bits = [f"sidechain pump {v['pump']['depth_db']} dB" if (v.get("pump") or {}).get("kind") else "no sidechain pump"]
        if v.get("plr_db") is not None:
            bits.append(f"peak-to-loudness {v['plr_db']} dB")
        if v.get("low_end_wide"):
            bits.append("wide (not mono) low end")
        m = v.get("map")
        if m:
            bits.append("dance arrangement " + ", ".join(f"{x['role']} {fmt_time(x['start'])} {x['bars']} bars"
                                                         for x in m["sections"]))
        lines.append("Producer's view: " + "; ".join(bits) + ".")
    if r.get("memory_profile"):
        lines.append(r["memory_profile"])
    if r.get("past"):
        lines.append("Compared with their own saved tracks: " + " ".join(r["past"]))
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
    # Up to five describing words may sit between the verb and the chord ("introduce a secondary dominant A7",
    # "add a bright, open Dsus4"), but no preposition: after one, the chord is what's already there.
    suggest = re.compile(r"\b(?:insert|add|play|use|try|layer|introduce|swap|put|with|substitute|borrow)\s+"
                         r"(?:(?!(?:before|after|over|under|on|into|to|than|from|of|at|in|the)\b)[\w#',-]+\s+){0,5}$", re.I)
    tl = sorted(r["chord_timeline"], key=lambda c: c.get("start", _to_sec(c["time"])))
    starts = [c.get("start", _to_sec(c["time"])) for c in tl]
    ends = starts[1:] + [r.get("duration_sec", 1e9)]

    def sounding(t):  # chords heard during the second m:ss (half a second of slack before it)
        return [c for c, a, b in zip(tl, starts, ends) if a < t + 1 and b > t - 0.5]

    claims = [(m_.start(), m_.group(1), m_.group(2)) for m_ in
              re.finditer(r"\b([A-G][b#]?(?:maj7|m7|m|7|5)?)(?:\s+chord)?\s+at\s+(\d{1,2}:\d{2})\b", text)]
    # "the 0:05 D5" names the chord that's there now, even inside a suggestion ("replace the 0:05 D5 with Bm")
    claims += [(None, m_.group(2), m_.group(1)) for m_ in
               re.finditer(r"\bthe\s+(\d{1,2}:\d{2})\s+([A-G][b#]?(?:maj7|m7|m|7|5)?)\b(?!\s+section)", text)]
    for pos, name, t in claims:
        # look back to the start of the chord's own clause, however long the description is
        clause = re.split(r"[.;:!?()]", text[:pos])[-1] if pos is not None else ""
        if pos is not None and suggest.search(clause):
            continue  # a chord the AI suggests adding, not a claim about the recording
        want = parse_chord(name)
        if not want:
            continue
        near = sounding(_to_sec(t))
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
    out = {
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
    if r.get("blues_form"):
        # The AI is asked to keep the 12-bar form but doesn't always (it once padded a pop loop to 12 chords and
        # called it a blues), so check each progression and drop the ones that don't follow the form.
        form = r["blues_form"]
        tpl = form.get("template") or BLUES_FORMS["12-bar blues"]
        kept = [p for p in out["alt_progressions"] if fits_form(p["chords"], tpl, key_tonic(r["key_index"]))]
        if len(kept) < len(out["alt_progressions"]):
            out["dropped_progressions"] = len(out["alt_progressions"]) - len(kept)
        out["alt_progressions"] = kept
    return out


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

# Plain-English explanations, shown as hover tooltips and, grouped, in the "How to read this report" guide.
GUIDE = {
    "The basics": {
        "Length": "How long the track runs, in minutes and seconds.",
        "Tempo": "How fast the beat goes, in beats per minute (BPM). 60 is one beat a second. Ballads tend to sit "
                 "around 60-80, pop around 90-130 and dance music around 120-130. Now and then we'll read half or "
                 "double the speed you feel.",
        "Meter": "How the beats group into bars. 4/4 means four beats a bar (most pop, rock and blues), 3/4 means "
                 "three (that waltz sway), and 5/4 or 7/4 are odd meters you'll hear in prog, jazz and film music. "
                 "'Assumed' means the evidence was thin, so we went with the most common one, 4/4.",
        "Bar": "One full cycle of the meter, like four beats in 4/4. We count chord changes, sections and riffs in bars.",
        "Key": "The home note and scale the music keeps coming back to, like E major. Major usually sounds brighter, "
               "minor darker.",
        "Camelot": "A code DJs and producers use for keys: numbers 1-12 round a wheel, A for minor and B for major. "
                   "Tracks with the same code, a number either side (same letter), or the same number with the other "
                   "letter mix together smoothly.",
        "Mode": "The flavour of the scale: which notes surround the home note. Hover over a mode's name to see what "
                "it feels like.",
        "Note names": "A letter plus an octave number. C4 is middle C and A4 is 440 Hz. A bass guitar's lowest string "
                      "is E1 and a guitar's is E2. # means a semitone up (sharp), b a semitone down (flat).",
    },
    "Harmony": {
        "Confidence": "How well the notes fit the key, from 0 to 1. Above 0.75 is clear. Below 0.6 the harmony is "
                      "ambiguous, which is common with modal music, drones and power chords.",
        "Runner-up": "The next most likely key. It's often the 'relative' key (C major and A minor share the same "
                     "notes), which is why the two are easy to mix up.",
        "Mode confidence": "How clearly one mode beats the others. 'Low' means another mode fits almost as well, "
                           "usually because the note that would settle it hardly gets played.",
        "Colour note": "The note that gives this mode its character. Lean on it in a melody and the mode comes "
                       "through clearly.",
        "Scale": "The seven notes of the key's mode, starting from the home note.",
        "Chord names": "C is C major and Cm is C minor. C7 is a dominant 7th (bluesy, and it wants to move to the "
                       "chord a 4th up), Cm7 a minor 7th (soft and jazzy), Cmaj7 a major 7th (dreamy). C5 is a power "
                       "chord: just the root and 5th, neither major nor minor, the bread and butter of rock and "
                       "blues guitar.",
        "Slash chord": "A/E means an A chord with E as the lowest (bass) note.",
        "Capo": "A clamp across the guitar neck that raises every string by the fret it sits on, so easy open "
                "chord shapes sound higher. Fingerings are written from the low E string to the high E string: "
                "a number is the fret to press, 0 an open string, x a string you don't play. Chords played as "
                "power chords (C5, D5) aren't counted: that shape moves anywhere on the neck.",
        "Chord root": "The note a chord is named after, like D in D7. Bass lines often play it.",
        "Roman numerals": "Chords named by where they sit in the key. I is the home chord, and IV and V are built on "
                          "the 4th and 5th notes. Upper case is major, lower case minor, so 'i - iv - V' describes "
                          "the same progression in any key.",
        "AABA": "The 32-bar song form behind jazz standards and classic pop: an 8-bar A section played twice, a "
                "contrasting 8-bar B section (the bridge, or 'middle eight'), then A once more. We spot it by "
                "comparing the chords of 8-bar stretches.",
        "Chord loop": "A short run of chords that repeats, like the four chords under a verse. We look for loops "
                      "within each key and show them starting on the home chord.",
        "Chord timeline": "Every chord change in order, with the time it starts. Names like A/E show the bass note "
                          "when it isn't the chord's root. A ? means that chord's notes were no louder than the other "
                          "notes sounding at the time, as in a fast run or a lick, so the name is a best guess.",
        "Harmonic rhythm": "How often the chord changes.",
        "Time in each key": "How long the music stays in each key. Songs often move to a new key (modulate) for a "
                            "chorus or a bridge.",
        "Key timeline": "The keys in the order they come, with their start and end times.",
        "Drone / pedal": "A note that keeps sounding while the chords change above it.",
        "Tuning": "How far the instrument sits from standard pitch (A = 440 Hz), in cents. 100 cents is one semitone.",
        "Major/minor blur": "The major and the minor 3rd both stand out. You hear this a lot in blues and rock, where "
                            "the minor 3rd is a 'blue note' played over major chords.",
    },
    "Form": {
        "Form": "The order of the song's sections. Each letter is a section, and a letter that comes back sounds "
                "like the earlier one (A B A C could be verse, chorus, verse, bridge). We work this out "
                "automatically, so treat it as a rough map.",
        "Section energy": "Low, medium or high: how loud the section is compared with the track as a whole.",
        "Instrument columns": "✓ means that instrument clearly plays in the section. — means it doesn't, or it's too "
                              "quiet for us to tell.",
    },
    "Energy & mood": {
        "Energy": "How loud the track is from start (left) to end (right), in 24 slices. Taller bars are louder, so "
                  "you can spot builds, drops and breakdowns at a glance.",
        "Energy level": "How intense a section feels, from 1 to 10. We work it out from how loud it is (compared with "
                        "the loudest section), how many notes or hits there are each second, how bright it sounds, "
                        "and the tempo.",
        "Mood": "Where the music sits between dark and bright, as positivity from 0 to 100. We base it on the mode, "
                "how much of the time minor chords play, brightness and tempo. Put together with energy, that gives "
                "words like 'calm', 'driving' or 'melancholic'. It's a rule of thumb, not a verdict: how you play "
                "shapes the feel too.",
        "Positivity": "0 is dark or sad, 100 bright or happy. We estimate it from the mode, minor chords, brightness "
                      "and tempo.",
        "Loudness vs peak": "How much quieter the section is than the loudest one, in dB. 0 is the loudest, and -6 dB "
                            "sounds about half as loud.",
        "Notes per second": "How many notes or hits start each second. It's a measure of how busy the playing is.",
        "Brightness": "The 'centre of gravity' of the sound, in Hz. Higher sounds brighter or harsher (cymbals, "
                      "distortion), lower sounds darker or warmer.",
        "Minor chords": "How much of the section's time minor chords take up. More minor usually feels darker.",
    },
    "Instruments": {
        "Separation": "We pull the drums apart from everything else before we analyse anything. Demucs, an AI model, "
                      "does this well. Without it we use a simpler split by sound type: sharp hits versus held notes.",
        "Drum grid": "One bar cut into 16 sixteenth-notes (1 e & a 2 e & a 3 e & a 4 e & a). K is a kick-heavy hit, "
                     "S a snare-heavy hit, X a strong hit (kick and snare), x a lighter hit, and . means there's "
                     "usually nothing. h is a hi-hat or cymbal.",
        "Groove repetition": "How closely each bar sticks to the main drum groove. Tight feels like a loop, loose "
                             "means plenty of variation.",
        "Groove traits": "A backbeat puts the snare on beats 2 and 4 (rock and pop). Four-on-the-floor puts the kick on "
                         "every beat (disco and house). Half-time puts the snare only on beat 3, so it feels slower.",
        "Fill": "A short break from the main groove, usually leading into a new section.",
        "Bass style": "Roots means mostly the chords' root notes. Pedal holds one note under changing chords. Riff is "
                      "a repeating figure. Walking is a note on every beat, moving step by step, as in jazz.",
        "Clear notes": "The share of beats where we could hear a clear bass pitch. Rests, quiet notes and muddy "
                       "notes don't count.",
        "Lowest part": "When there's no separate bass instrument, we take the lowest notes of the main instrument "
                       "(say, a guitarist's thumb on the low strings) and analyse them like a bass line.",
        "Sheet music": "The bass notes we found in the recording, placed on a 16th-note grid in bars of the detected "
                       "meter, with the chords written above. It's automatic, so check it by ear. Swing and triplets "
                       "get rounded to the nearest 16th.",
        "Note lengths": "A whole note is 4 beats, a half note 2, a quarter note 1, an 8th half a beat and a 16th a "
                        "quarter of a beat. A dot makes a note half as long again.",
        "Rhythm grid": "One bar in 16ths (1 e & a 2 e & a ...). x or X marks where a note starts: X in most bars, x in "
                       "some, . rarely.",
    },
    "Rhythm & sound": {
        "Pulse": "How steady the tempo is: locked to a click, played by feel, loose, or free with no fixed beat.",
        "Syncopation": "Hits that land between the beats rather than on them. It's what makes a groove feel pushed "
                       "or funky.",
        "Percussive": "How much of the sound's energy comes from sharp attacks (drums, but also picked or strummed "
                      "notes) rather than held notes.",
        "Tone": "Dark and warm, balanced, or bright, going by the brightness in Hz. Clean or noisy tells you how "
                "pitched or how noise-like the sound is.",
        "dBFS": "Loudness measured against the digital maximum (0 dBFS). The more negative, the quieter.",
        "LUFS": "Loudness as ears hear it (ITU-R BS.1770, the standard streaming services use): low bass counts for "
                "less, the 2-4 kHz range for more, and silent moments are left out. Spotify plays music at about "
                "-14 LUFS, turning louder tracks down.",
        "True peak": "The highest level the waveform reaches, including between the samples, where a converter or an "
                     "mp3/AAC encoder can overshoot. Above 0 dBTP the sound clips; Spotify asks for -1 dBTP or lower.",
        "Loudness range": "How much the level moves between the quieter and louder passages, in LU (EBU Tech 3342: "
                          "the spread from the 10th to the 95th percentile of 3-second loudness). Small means even and "
                          "compressed, large means dynamic.",
        "Stereo correlation": "How similar the left and right channels are: +1 is mono, values between 0 and +1 are "
                              "normal for a stereo mix (lower = wider), and below 0 means parts are out of phase and "
                              "cancel on a mono speaker.",
        "Dynamic range": "The gap between the loud and quiet parts, in dB. Bigger means more contrast.",
        "Frequency balance": "How the energy spreads across the range: sub (felt more than heard), bass, low-mids "
                             "(body), high-mids (presence) and air (sparkle).",
    },
    "Instruments we hear": {
        "Instruments we hear": "A trained model's opinion of which instruments play (Essentia's instrument model, "
                               "MTG-UPF, 40 instruments), with how sure it is. It's a second opinion next to our own "
                               "measurements and what you tick: shown and compared, never used to change a result.",
    },
    "Producer's view": {
        "Sidechain": "Pumping: the pads, chords and synths duck right after each kick and swell back before the next, "
                     "usually from a compressor triggered by the kick. Gentle is 3-6 dB, heavy (the obvious French-house "
                     "'breathing') more than 6.",
        "Drop": "In dance music, the moment the full beat and bass come back in after a breakdown or build. We map each "
                "section's role (intro, build, drop, breakdown, outro), its length in bars, and how far the drop jumps "
                "in level above what leads into it.",
        "Mono low end": "Club systems and phone speakers play the bass in mono, so a wide or out-of-phase low end "
                        "partly cancels on them. Correlation below 120 Hz near +1 is mono and safe; below 0.6 is wide.",
        "PLR": "Peak-to-loudness ratio: true peak minus loudness (LUFS). Below about 8 dB a master is heavily limited "
               "(dense, loud, little punch); 8-11 is typical of modern club and pop masters; above that it's dynamic.",
    },
    "Melody": {
        "Scale degree": "A note's place in the key: 1 is the home note, 5 the fifth above it, b6 a flattened sixth, "
                        "and so on. Degrees describe a melody whatever key it's in.",
        "Melody shape": "The overall contour (rising, falling, arch or wave), plus its range and how much it moves by "
                        "step to the next note versus leaping.",
        "Intervals": "The distance from each note to the next: half-step, whole-step, 3rds, 4ths, 5ths and so on. "
                     "Small steps sound smooth and singable, big leaps dramatic.",
        "Top line": "The loudest pitched part, picked out automatically. On a full-band phone recording it's often "
                    "the chord instrument rather than the melody.",
        "Variations": "Classic ways to develop a short idea. Inversion flips its intervals upside down, retrograde "
                      "plays it backwards, a sequence repeats it a step higher, and an answer phrase brings it home "
                      "to the home note.",
    },
    "Ideas & files": {
        "Shared DNA": "Building blocks (chord loops, scale, groove, form, melodic tricks) your track has in common "
                      "with well-known recordings. They're there for study and inspiration, not to say it 'sounds like' "
                      "anything.",
        "AI brainstorm": "Written by an AI language model that reads the analysis numbers, never your audio. Treat it "
                         "like a sparring partner: we check any times and chords it names, but its opinions are its own.",
        "MIDI": "A file of notes rather than sound. Drag it into a DAW (GarageBand, Logic, Ableton, FL Studio and so "
                "on) or MuseScore, pick any instrument, then edit, transpose or loop it.",
        "MusicXML": "The standard file format for sheet music. It opens in MuseScore (free), Sibelius, Dorico, Finale "
                    "and many DAWs, where you can read it, play it back and edit it.",
        "Report file": "This whole report as a text file (Markdown, .md). It opens in any text editor or notes app and "
                       "shows up formatted on GitHub, Notion or Obsidian.",
        "Data file": "Every number behind the report, in a file programs can read (JSON). Handy for spreadsheets, "
                     "your own scripts or another tool.",
    },
}
GLOSSARY = {term: text for group in GUIDE.values() for term, text in group.items()}
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


BLUES_TIPS = {
    8: "An 8-bar blues, e.g. 'Key to the Highway': I–V–IV–IV–I–V–I–V. Shorter and quicker-turning than the 12-bar; "
       "one pass is a 'chorus'.",
    16: "A 16-bar blues: the 12-bar idea stretched, either with 8 bars on the home chord before the change (e.g. "
        "'Hoochie Coochie Man') or with an extended V–IV turnaround. One pass is a 'chorus'.",
}
BLUES_TIP = ("The 12-bar blues: 4 bars on the home chord (I), 2 on the IV chord, 2 on I, then V, IV, I, I "
             "(often V in the last bar to lead back round). One pass is a 'chorus'; solos and verses follow it.")


def tip(text, term=None, explanation=None):
    """Wrap text in a hover tooltip (an <abbr>). Works in the web app and most Markdown viewers."""
    title = explanation or GLOSSARY[term or text]
    return f'<abbr title="{html.escape(title, quote=True)}">{text}</abbr>'


def mode_tip(mode):
    return tip(mode, explanation=MODE_INFO.get(mode, mode))


def guide():
    parts = []
    for group, terms in list(GUIDE.items()) + [("Modes", MODE_INFO)]:
        items = "".join(f"<li><b>{html.escape(k)}</b> — {html.escape(v)}</li>" for k, v in terms.items())
        parts.append(f"<p><b>{html.escape(group)}</b></p><ul>{items}</ul>")
    return ["<details><summary>📖 How to read this report</summary>", "",
            "<p>Hover over any underlined word in the report (or tap it on a phone) and it'll explain itself. "
            "Here they all are in one place.</p>" + "".join(parts), "", "</details>", ""]


def sparkline(values):
    return "".join(BARS[min(len(BARS) - 1, int(v * (len(BARS) - 1) + 0.5))] for v in values)


def bar(pct, width=20):
    return "█" * int(round(pct / 100 * width))


def track_report(r):
    L = [f"## 🎵 {r['file']}", ""]
    if "error" in r:
        return L + [f"⚠️ We couldn't analyse this file ({r['error']}). If it plays on your computer, try "
                    "exporting it again as mp3 or wav and uploading that.", ""]
    rh, s = r["rhythm"], r["sound"]
    tempo = f"{r['tempo_bpm']:.0f} BPM" if r["tempo_bpm"] else "no clear beat"
    if r["tempo_bpm"] and r.get("tempo_approximate"):
        tempo = f"~{tempo} (approximate: there are hardly any clear attacks to measure the beat from)"
    meter = r.get("meter") or {}
    if r["tempo_bpm"] and meter.get("label"):
        tempo += f" in {meter['label']}" + (" (waltz feel)" if meter["beats_per_bar"] == 3 else "") + \
            (" (assumed)" if meter.get("confidence") == "assumed" else "")
    ctx = r.get("context")
    if ctx:
        told = [x for x in (", ".join(ctx["instruments"]), ctx.get("genre")) if x]
        L += [f"_You told us: {' · '.join(told)}._"] + [f"- _{n}_" for n in ctx.get("notes", [])] + [""]
    L += [f"**{tip(r['duration'], 'Length')}** · **{tip(tempo, explanation=GLOSSARY['Tempo'] + ' ' + GLOSSARY['Meter'])}** · "
          f"**{tip(r['key'], 'Key')}** ({tip(r['camelot'], 'Camelot')}) · **{mode_tip(r['mode'])}** · "
          f"{tip(rh['tempo_feel'], 'Pulse')} · {tip(s['tone'], 'Tone')}", ""]
    h = r.get("heard") or {}
    if h.get("instruments"):
        heard = " · ".join(f"{x['name']} {round(100 * x['p'])}%" for x in h["instruments"])
        L += [f"_{tip('Instruments we hear', 'Instruments we hear')}: {heard}._"]
        extra = heard_notes(r)
        if extra:
            L += [f"- _We also hear {', '.join(x['name'] for x in extra)}, which you didn't tick. If "
                  f"{'it is' if len(extra) == 1 else 'they are'} there, tick {'it' if len(extra) == 1 else 'them'} next "
                  "time and we'll use it (nothing here was changed on the model's say-so)._"]
        elif not ctx:
            L += ["- _Tick the right ones when you upload and we'll use them: they settle the drums, the bass and "
                  "whose melody it is._"]
        L.append("")
    L += comparison_report(r)  # what the musician came for, when they brought songs they admire

    # Harmony
    conf = r["key_confidence"]
    conf_note = "clear" if conf >= 0.75 else "fairly clear" if conf >= 0.6 else "low, so the harmony is ambiguous"
    mode_note = ("" if r["mode_confidence"] == "high" else
                 f"; could also be {r['mode_runner_up']}" if r["mode_confidence"] == "medium" else
                 f"; not sure, {r['mode_runner_up']} fits almost as well")
    L += ["### Harmony", "", "_What the music is built from: its key, scale and chords._", "",
          f"- **{tip('Key')}:** {r['key']} ({tip('confidence', 'Confidence')} {conf}: {conf_note}; "
          f"{tip('runner-up', 'Runner-up')} {r['runner_up_key']})",
          f"- **{tip('Mode')}:** {mode_tip(r['mode'])} ({tip(r['mode_confidence'] + ' confidence', 'Mode confidence')}"
          f"{mode_note}) — {tip('colour note', 'Colour note')} **{r['color_note']}**",
          f"- **{tip('Scale')}:** {' '.join(r['scale_notes'])}",
          ]
    if r.get("drone"):
        who = " (the bass holds it)" if (r.get("bass") or {}).get("pedal_note") == r["scale_notes"][0] else ""
        L.append(f"- **{tip('Drone / pedal')}:** {r['scale_notes'][0]} sounds through {r['drone_pct']}% of the "
                 f"track{who}")
    if r.get("third_mix"):
        L.append(f"- **{tip('Major/minor blur')}:** both thirds ({r['thirds']}) are prominent")
    cents = r.get("tuning_cents", 0)
    if abs(cents) >= 10:
        L.append(f"- **{tip('Tuning')}:** about {abs(cents)} cents {'sharp' if cents > 0 else 'flat'} of A440 "
                 f"(A ≈ {440 * 2 ** (cents / 1200):.0f} Hz). We've allowed for that in everything below")
    L.append(f"- **{tip('Chords used most', 'Chord names')}:** " + ", ".join(f"{c['chord']} ({fmt_time(c['seconds'])})" for c in r["chords_used"]))
    g = r.get("guitar")
    chosen = (r.get("context") or {}).get("chosen") or (r.get("context") or {}).get("keys") or []
    if g and (not chosen or any(c in GUITARS for c in chosen)):  # capo shapes are a six-string guitar's
        shapes = " · ".join(f"{s['shape']} `{s['fingering']}`" + (f" (sounds {s['sounds']})" if g["capo"] else "")
                            for s in g["shapes"])
        where = f"capo on fret {g['capo']}, then play these open shapes" if g["capo"] else "all open shapes, no capo needed"
        L.append(f"- **{tip('For guitar', 'Capo')}:** {where} ({g['open_share_pct']}% of the chords): {shapes}")
    if r["harmonic_rhythm_beats"]:
        L.append(f"- **{tip('Harmonic rhythm')}:** a chord change every ~{r['harmonic_rhythm_beats']} beats")
    elif r.get("harmonic_rhythm_sec"):
        L.append(f"- **{tip('Harmonic rhythm')}:** a chord change every ~{r['harmonic_rhythm_sec']} seconds")
    L += ["", f"**{tip('Time in each key')}**", "", "| Key | Time | Share |", "|-----|------|-------|"]
    for k, v in r["time_in_key"].items():
        L.append(f"| {k} | {fmt_time(v['seconds'])} | {v['percent']}% |")
    if r["key_changes"]:
        L += ["", f"**{tip('Key timeline')}:** " + " → ".join(f"{s_['key']} ({s_['start']}–{s_['end']})" for s_ in r["key_sections"])]
    b = r.get("blues_form")
    if b:
        L += ["", f"**{tip('Form', explanation=BLUES_TIPS.get(b.get('bars', 12), BLUES_TIP))}: {b['name']}** — choruses of "
              f"~{b['chorus_sec']} s ({b.get('bars', 12)} bars of {b['bar_sec']} s), starting at "
              + ", ".join(fmt_time(t) for t in b["choruses"])
              + f" · {b['match_pct']}% of the bars follow the pattern"
              + (" · there's an intro or pickup before the first full chorus" if b["choruses"][0] > b["bar_sec"] * 2 else "")]
        k = r["key_index"]
        home = spell(key_tonic(k), k)
        for v in r.get("blues_variations") or []:
            L.append(f"- **Try:** {v['name']} — `{' | '.join(v['chords'])}` — {v['why']}")
        four, five = spell((key_tonic(k) + 5) % 12, k), spell((key_tonic(k) + 7) % 12, k)
        if key_is_minor(k):
            L.append(f"_This is a minor blues, so you'd expect minor 7th chords on the home and IV chords "
                     f"({home}m7, {four}m7), a dominant {five}7, and the blues scale's b5 ({flat5(five)}) as its "
                     "colour note._")
        else:
            L.append(f"_In a blues, “{r['key']}” just means {home} is home. You'd expect dominant-7th chords "
                     f"({home}7, {four}7, {five}7) and the blue notes {spell((key_tonic(k) + 3) % 12, k)} (flat 3rd) "
                     f"and {spell((key_tonic(k) + 10) % 12, k)} (flat 7th) rubbing against the major scale._")
    sf = r.get("song_form")
    if sf:
        names = ["A", "A", "B (bridge)", "A"]
        L += ["", f"**{tip('Song form', 'AABA')}: {sf['name']}** — " + ", ".join(
              f"{n} at {fmt_time(t)}" for n, t in zip(names, sf["starts"])) + f" (8 bars each, {sf['section_sec']} s)"
              + (f" · the whole AABA comes round {sf['cycles']} times" if sf["cycles"] > 1 else "")]
    if r["loops"]:
        L += ["", f"**{tip('Chord loops', 'Chord loop')}**", ""]
        for lp in r["loops"]:
            L.append(f"- `{' – '.join(lp['chords'])}`  ({tip(' – '.join(lp['roman']), 'Roman numerals')} "
                     f"in {lp['key']}) "
                     f"· {', '.join(lp['where'])} · ×{lp['repeats']}")
    if r["chord_timeline"]:
        L += ["", "<details markdown=\"1\"><summary>Full chord timeline (with bass notes as slash chords)</summary>", "",
              f"_{GLOSSARY['Chord timeline']} {GLOSSARY['Chord names']}_", "",
              " · ".join(f"{c['time']} {c.get('with_bass', c['chord'])}" + ("?" if (c.get("clarity") or 9) < UNCLEAR_CHORD else "")
                         for c in r["chord_timeline"]), "",
              "</details>"]

    # Structure
    L += ["", "### Structure", "", "_The sections in order, and what changes from one to the next._", "",
          *([timeline_svg(r), ""] if timeline_svg(r) else []),
          f"| {tip('Section', 'Form')} | Time | Length | {tip('Energy', 'Section energy')} | Key | "
          f"{tip('Main chords', 'Chord names')} | {tip('Drums', 'Instrument columns')} | {tip('Bass', 'Instrument columns')} |",
          "|---------|------|--------|--------|-----|-------------|-------|------|"]
    for sec in r["sections"]:
        L.append(f"| **{sec['label']}** | {fmt_time(sec['start'])}–{fmt_time(sec['end'])} | "
                 f"{fmt_time(sec['end'] - sec['start'])} | {sec['energy']} | {sec['key']} | "
                 f"{', '.join(sec['main_chords']) or '—'} | {'✓' if sec.get('drums') else '—'} | "
                 f"{'✓' if sec.get('bass') else '—'} |")
    L += ["", f"{tip('Form')}: **{' '.join(sec['label'] for sec in r['sections'])}**"]
    L += feel_report(r)
    L += instruments_report(r)
    L += melody_report(r)
    L += dna_report(r)

    # Rhythm & sound
    L += ["", "### Groove & sound", "", "_How the track moves, and what it sounds like._", "",
          f"- **{tip('Pulse')}:** {rh['tempo_feel']}"
          + (f" (beat-to-beat variation {rh['tempo_variation_pct']}%)" if "tempo_variation_pct" in rh else ""),
          f"- **Rhythm:** {rh['busyness']} ({tip(str(rh['onsets_per_sec']) + ' notes/hits per sec', 'Notes per second')})"
          + (f", {rh['groove']} ({rh['syncopation_pct']}% {tip('off-beat', 'Syncopation')})" if "groove" in rh else ""),
          f"- **{tip('Percussive', 'Percussive')} vs tonal:** {rh['percussive_pct']}% percussive energy",
          f"- **{tip('Tone')}:** {s['tone']} ({tip(str(s['brightness_hz']) + ' Hz', 'Brightness')} brightness), {s['texture']}",
          *loudness_lines(s),
          f"- **{tip('Energy arc', 'Energy')}:** `{sparkline(s['energy_curve'])}`"
          + (f" — biggest build into **{s['biggest_build']['time']}**" if s["biggest_build"] else "")
          + (f", quietest around **{s['quietest_moment']}**" if s["quietest_moment"] else ""),
          "", f"**{tip('Frequency balance')}**", "", "```"]
    for name, pct in s["frequency_balance_pct"].items():
        L.append(f"{name:<18} {bar(pct):<20} {pct:>5}%")
    L.append("```")
    L += producer_report(r)

    # Ideas
    L += past_report(r)
    L += ["", "### 💡 Ideas to try", "", "_We worked these out from the analysis itself, no AI involved. Try "
          "them out and keep whatever works for you._", ""]
    for i in r["ideas"]:
        L.append(f"- **{i['area']}:** {i['idea']}")

    ai = r.get("ai")
    if ai:
        L += ["", "### 🤖 AI brainstorm", "", f"_{GLOSSARY['AI brainstorm']}_", ""] + ([f"_Suggestions by {ai['model']}"
                                                + (" (its reply got cut off, so a few parts may be missing)" if ai.get("partial") else "")
                                                + "._", ""] if ai.get("model") else [])
        if ai.get("pending"):
            L.append("_⏳ The AI is still writing ideas for this track. They'll show up here in a moment._")
        elif ai.get("error"):
            L.append(f"_{ai['error']}_")
        else:
            if ai["vibe"]:
                L += [f"> {ai['vibe']}", ""]
            for i in ai["ideas"]:
                L.append(f"- **{i.get('area', 'idea')}:** {i['idea']}")
            if ai.get("dropped_progressions"):
                L += ["", f"_The AI also suggested {ai['dropped_progressions']} progression(s) that broke the "
                      f"{(r.get('blues_form') or {}).get('name', 'blues')} form, so we left them out. The variations "
                      "under Harmony keep it._"]
            if ai["alt_progressions"]:
                L += ["", "**Alternative progressions**", ""]
                for p in ai["alt_progressions"]:
                    L.append(f"- *{p.get('name', '')}*: `{' – '.join(map(str, p['chords']))}` — {p.get('why', '')}")
            if ai["title_ideas"]:
                L += ["", "**Title ideas:** " + " · ".join(f"_{t}_" for t in ai["title_ideas"])]

    if r.get("sheets"):
        L += ["", "### 🎼 Sheet music", "", f"_{GLOSSARY['Sheet music']} {GLOSSARY['MusicXML']}_", ""]
        for m in r["sheets"]:
            L.append(f"- `sheets/{m['file']}` — {m['label']}")
    if r.get("midi"):
        L += ["", "### 🎹 MIDI sketches (drag into your DAW)", "", f"_{GLOSSARY['MIDI']}_", ""]
        for m in r["midi"]:
            L.append(f"- `midi/{m['file']}` — {m['label']}" + (f": {' – '.join(m['chords'])}" if m["chords"] else ""))
    L.append("")
    return L


def melody_report(r):
    L = ["", "### 🎶 Melody", "", "_The tune: its notes, its shape and ways to develop it. For the full "
         "picture, type the melody you hear when you upload._", ""]
    m = r.get("melody")
    if m and m.get("error"):
        L += [f"_{m['error']}_", ""]
    elif m:
        chk = m.get("check") or {}
        L += [f"**Your melody:** `{' '.join(m['notes'])}`", ""]
        if chk:
            strengths = ", ".join(f"{n['note']} (#{n['rank']} of 12)" for n in chk["notes"])
            L += [f"> **Checked against the recording:** {chk['verdict']} "
                  f"_How strongly each note sounds in the track (1 = strongest of 12): {strengths}._", ""]
        L += [
              f"- **{tip('Scale degrees', 'Scale degree')}:** {' '.join(m['degrees'])} (1 = {r['scale_notes'][0]}, the home note)",
              f"- **Fits:** {', '.join(mode_tip(x) for x in m['fits_modes']) or 'no single mode — it mixes notes from the major and minor scales on ' + r['scale_notes'][0]}"
              + (f" · outside the detected scale: {', '.join(m['outside_detected_scale'])}" if m["outside_detected_scale"] else ""),
              f"- **{tip('Shape', 'Melody shape')}:** {m['shape']} · range {m['range']} · {m['steps_pct']}% steps, {m['leaps_pct']}% leaps",
              f"- **{tip('Moves', 'Intervals')}:** {', '.join(m['intervals'])}"]
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
        verdict = ("it's {0}% chord notes, so it's probably following the chords rather than playing a melody "
                   "of its own".format(t["chord_tone_pct"]) if t["follows_chords"] else
                   f"{t['chord_tone_pct']}% chord notes, so it looks like a real melodic line")
        lead = lead_instrument(r)
        if lead == "voice":
            verdict += ". With a singer in the track, this is most likely the vocal melody"
        elif lead and not t["follows_chords"]:
            verdict += (f". It's most likely the {named_instrument(r, lead)} line, since that's the only lead "
                        "instrument you named")
        L += [f"**{tip('Top line', 'Top line')} (automatic):** range {t['range']} · main notes {main} · "
              f"repeating 4-note figures: {motif} · {verdict}."]
        if not m:
            L += ["", "_Want a proper melody analysis? Type the melody or riff when you upload "
                  "(for example `A E F E A E D E`)._"]
        L.append("")
    return L


def dna_report(r):
    d = r.get("references") or {}
    ai_refs = (r.get("ai") or {}).get("references") or []
    if not (d.get("progressions") or d.get("mode") or d.get("groove") or d.get("melody") or d.get("form") or ai_refs):
        return []
    L = ["", "### 🎧 Shared DNA with well-known music", "",
         f"_{GLOSSARY['Shared DNA']}_", ""]
    for p in d.get("progressions", []):
        how = "uses the same loop as" if p["match"] == "same loop" else "contains"
        L.append(f"- **Chords:** your `{' – '.join(p['your_chords'])}` {how} **{p['progression']}** — "
                 + ", ".join(p["examples"]))
    if d.get("mode"):
        L.append(f"- **Scale:** {mode_tip(d['mode']['mode'])} is the sound of " + ", ".join(d["mode"]["examples"]))
    if d.get("form"):
        L.append(f"- **Form:** {d['form']['form']}, the backbone of blues, rock'n'roll and jazz — "
                 + ", ".join(d["form"]["examples"]))
    if d.get("groove"):
        L.append(f"- **Groove:** {d['groove']}")
    for m in d.get("melody", []):
        L.append(f"- **Melody:** {m['device']} — {m['reference']}")
    if ai_refs:
        L += ["", "_Suggested by the AI, so give them a listen before you rely on them:_", ""]
        L += [f"- **{x.get('artist', '?')} — “{x.get('track', '?')}”:** {x.get('shared', '')}" for x in ai_refs]
    L.append("")
    return L


def instruments_report(r):
    d, b = r.get("drums") or {}, r.get("bass") or {}
    solo = b.get("low_part_only")
    ctx = r.get("context") or {}
    chosen = ctx.get("chosen") or ctx.get("keys") or []  # results saved before single instruments only had families
    melodic = [i for i in chosen if family(i) not in ("drums", "bass")]
    named = [i for i in melodic  # the solo instrument's low part is written out below, so it isn't "other"
             if not (solo and family(i) == r.get("low_part_instrument") and len(melodic) == 1)]
    if not d.get("present") and not b.get("present") and not solo and not d.get("string_attacks") and not named:
        return []
    how = ("with Demucs, an AI model built for the job" if r.get("separation") == "demucs" else
           "with a lighter split that tells sharp hits from held notes")
    L = ["", "### Instruments", "", "_Who's playing, and what the drums and bass are up to._", "",
         f"_We separated the drums from everything else {tip(how, 'Separation')}._", ""]
    if d.get("present") and d.get("grid"):
        where = [sec["label"] for sec in r["sections"] if sec.get("drums")]
        parts = [f"plays in sections {' '.join(where)}" if where else "plays throughout",
                 f"{tip('groove repetition', 'Groove repetition')}: {d['repetition']}"]
        parts += [tip(t, 'Groove traits') for t in d.get("traits", [])]
        if d.get("fills"):
            parts.append(f"likely {tip('fills', 'Fill')} at " + ", ".join(fmt_time(t) for t in d["fills"]))
        g = d["grid"]
        counts = " ".join(f"{b + 1} e & a" for b in range(len(g) // 4))  # 4 beats in 4/4, 3 in 3/4
        rows = ["         " + counts, "Drums    " + " ".join(g)]
        if d.get("hats"):
            rows.append("Cymbals  " + " ".join(d["hats"]))
        L += [f"**🥁 Drums** — " + " · ".join(parts), "", f"Main groove ({tip('how to read', 'Drum grid')}):", "",
              "```", *rows, "```", ""]
    elif d.get("present"):
        L += ["**🥁 Drums** — they're there, but we couldn't find a clear repeating pattern.", ""]
    if d.get("unsure"):
        L += ["_We're not sure about these. The hits are only slightly noisy, which fits light percussion "
              "(shaker, brushes) or a hard-picked instrument. If there aren't any drums, tick the instruments you "
              "hear next time you upload and leave “drums” unticked._", ""]
    elif d.get("user_said_none"):
        L += ["**🥁 Drums** — none, as you said. Any percussive sound is just the attack of the notes.", ""]
    elif d.get("string_attacks"):
        L += ["**🥁 Drums** — none. The percussive sound you can hear is plucked or strummed strings, not a drum kit.", ""]
    if solo and b.get("user_said_none"):
        L += ["**🎸 Bass** — no bass instrument, as you said. The low notes belong to the main instrument, and "
              "we've analysed them below as its lowest part.", ""]
    elif solo:
        L += ["**🎸 Bass** — we don't think there's a separate bass. There are no drums, and nothing below "
              "100 Hz, where a bass guitar sits. The low notes are "
              f"{low_part_range(r)[1]} (like a guitarist's "
              "thumb or a pianist's left hand), and we've analysed them below as the lowest part. If there is a "
              "bass, tick it when you upload.", ""]
    if solo:
        b = dict(r.get("low_part") or {}, present=True)
        if r.get("octave_raised"):
            inst = r.get("low_part_instrument")
            L += [f"_Nothing below 80 Hz reached the microphone, so we moved {r['octave_raised']} note(s) below the "
                  + (f"{named_instrument(r, inst)}'s range" if inst else "range of a guitar (the most likely instrument)")
                  + " up an octave. Notes that low are almost always a pitch-tracking slip."
                  + ("" if inst else " If your instrument really goes that low (piano, cello), tick it when you upload.") + "_", ""]
        elif r.get("low_end_missing"):
            L += ["_Nothing below 80 Hz reached the microphone, so we've shown the lowest notes as we heard them. "
                  "Check their octave by ear._", ""]
    if named:
        L += [f"**Other instruments you named** — {', '.join(INSTRUMENTS.get(i, i) for i in named)}. We don't write these "
              "out one by one. We transcribe the drums (as a grid) and the lowest part (the bass, or the main "
              "instrument's own low notes), and what everyone else plays together shows up under Harmony (the "
              "chords) and Melody (the top line).", ""]
    if b.get("present"):
        notes = ", ".join(f"{n['note']} {n['pct']}%" for n in b["main_notes"])
        where = [sec["label"] for sec in r["sections"] if sec.get("bass")]
        parts = [f"plays in sections {' '.join(where)}" if where else "plays throughout" if solo else "barely audible",
                 f"{tip('clear notes', 'Clear notes')} on {b['tracked_pct']}% of beats",
                 f"range {tip(b['range'], 'Note names')}", f"{tip('style', 'Bass style')}: {b['style']}",
                 f"main notes: {notes}", f"on the {tip('chord root', 'Chord root')} {b['follows_roots_pct']}% of the time"]
        if b.get("riff"):
            parts.append(f"repeating riff: `{' '.join(b['riff'])}`")
        if b.get("from_user"):
            parts.append("you said there's a bass (on this recording it sounds just like low strings, so we went with you)")
        L += [f"**{tip('Lowest part') if solo else '🎸 Bass'}** — " + " · ".join(parts), ""]
        L += bass_rhythm_report(r)
    else:
        L += ["**🎸 Bass** — we couldn't hear a clear bass line.", ""]
    return L


def bass_rhythm_report(r):
    """Rhythm of the bass, read from its sheet music."""
    rh, sheet = r.get("bass_rhythm") or {}, r.get("bass_sheet") or {}
    if not rh:
        return []
    bpb = sheet["beats_per_bar"]
    counts = " ".join(f"{b + 1} e & a" for b in range(bpb))
    values = ", ".join(f"{name} {pct}%" for name, pct in rh["note_values"])
    word = low_part(r)[0]
    L = [f"**{word.capitalize()} rhythm** (read from the {tip('sheet music', 'Sheet music')}, {sheet['bars']} bars) — "
         f"{len(sheet['notes'])} notes, ~{rh['notes_per_bar']} per bar · {tip('note lengths', 'Note lengths')}: {values} · "
         f"{rh['offbeat_pct']}% start between beats ({rh['sixteenth_offbeat_pct']}% on a 16th) · "
         f"{rh['syncopated_pct']}% {tip('syncopated', 'Syncopation')} (held across the beat) · "
         f"{rh['distinct_bar_rhythms']} different bar rhythms", ""]
    if rh["main_bar_rhythm_pct"] >= 25:  # a real repeating figure
        L += [f"Most common bar rhythm (in {rh['main_bar_rhythm_pct']}% of the bars with {word}; {tip('how to read', 'Rhythm grid')}):", "",
              "```", "         " + counts, ("Low part " if word != "bass" else "Bass     ") + " ".join(rh["main_bar_rhythm"]), "```", ""]
    else:  # played freely: show where notes usually land instead
        L += [f"The rhythm changes from bar to bar, so here's where the {word} usually plays "
              f"(X = a note starts there in at least half the bars, x = in a quarter to half of them; "
              f"{tip('how to read', 'Rhythm grid')}):", "",
              "```", "         " + counts, ("Low part " if word != "bass" else "Bass     ") + " ".join(rh.get("usual_positions", "")), "```", ""]
    return L


SECTION_HUES = [210, 35, 140, 330, 265, 90, 5, 180, 50, 300]  # one colour per section letter


def timeline_svg(r):
    """The whole track at a glance: its loudness over time, with the sections as coloured blocks underneath
    (a repeated letter keeps its colour) and a time axis. Plain SVG, so it shows on the web page and in most
    Markdown viewers; text uses the page's own colour, so it reads in light and dark mode."""
    line, secs, dur = (r.get("sound") or {}).get("energy_line"), r.get("sections") or [], r.get("duration_sec") or 0
    if not line or not secs or dur <= 0:
        return ""
    W, H, top, band = 720, 112, 6, 22
    curve_h = H - band - 30
    x = lambda t: 4 + (W - 8) * t / dur
    pts = [(4 + (W - 8) * (i + 0.5) / len(line), top + curve_h * (1 - v)) for i, v in enumerate(line)]
    area = f"M4,{top + curve_h} " + " ".join(f"L{a:.1f},{b:.1f}" for a, b in pts) + f" L{W - 4},{top + curve_h} Z"
    hue = {}
    for s in secs:
        hue.setdefault(s["label"], SECTION_HUES[len(hue) % len(SECTION_HUES)])
    y0 = top + curve_h + 4
    blocks = "".join(
        f'<rect x="{x(s["start"]):.1f}" y="{y0}" width="{max(1.0, x(s["end"]) - x(s["start"]) - 1):.1f}" height="{band}" '
        f'fill="hsl({hue[s["label"]]},70%,55%)" fill-opacity="0.45"><title>{html.escape(s["label"])} '
        f'{fmt_time(s["start"])}-{fmt_time(s["end"])}, {s["energy"]} energy</title></rect>'
        + (f'<text x="{(x(s["start"]) + x(s["end"])) / 2:.1f}" y="{y0 + 15}" text-anchor="middle" font-size="12" '
           f'font-weight="700" fill="currentColor">{html.escape(s["label"])}</text>' if x(s["end"]) - x(s["start"]) > 14 else "")
        for s in secs)
    step = next(st for st in (10, 15, 30, 60, 120, 300, 600) if dur / st <= 10)
    ticks = "".join(f'<text x="{x(t):.1f}" y="{H - 4}" text-anchor="middle" font-size="10" fill="currentColor" '
                    f'fill-opacity="0.7">{fmt_time(t)}</text>' for t in range(0, int(dur) + 1, step))
    return (f'<svg class="timeline" viewBox="0 0 {W} {H}" width="100%" role="img" '
            f'aria-label="Loudness over time with the sections {" ".join(s["label"] for s in secs)}" '
            f'style="max-width:{W}px;display:block;margin:4px 0 12px">'
            f'<path d="{area}" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-opacity="0.55" '
            f'stroke-width="1"/>{blocks}{ticks}</svg>')


def loudness_lines(s):
    """Loudness as streaming services measure it, and what that means on Spotify (its published rule)."""
    st = s.get("streaming") or {}
    if not st or "error" in st:
        return [f"- **Loudness:** average {s['avg_loudness_db']} {tip('dBFS')}, {tip('dynamic range', 'Dynamic range')} "
                f"{s['dynamic_range_db']} dB"]
    parts = [f"{tip(str(st['lufs']) + ' LUFS', 'LUFS')} integrated", f"{tip('true peak', 'True peak')} {st['true_peak_db']} dBTP"]
    if st.get("loudness_range_lu") is not None:
        parts.append(f"{tip('loudness range', 'Loudness range')} {st['loudness_range_lu']} LU")
    if st.get("stereo_correlation") is not None:
        parts.append(f"{tip('stereo correlation', 'Stereo correlation')} {st['stereo_correlation']}")
    elif st.get("mono_file"):
        parts.append("a mono file (measured as it plays, on both speakers)")
    L = [f"- **Loudness:** " + " · ".join(parts)]
    gain, tp, lufs = st["spotify_gain_db"], st["true_peak_db"], st["lufs"]
    if gain < -0.4:
        spot = (f"Spotify turns it down {abs(gain)} dB to its usual {SPOTIFY_TARGET_LUFS:g} LUFS, so making it any "
                "louder gains nothing there.")
    elif lufs < SPOTIFY_TARGET_LUFS - 0.4 and gain < SPOTIFY_TARGET_LUFS - lufs - 0.4:
        lift = (f"Spotify would lift it only {gain} dB" if gain >= 0.1 else "Spotify won't turn it up at all")
        spot = (f"{lift}: it keeps the true peak at or below -1 dBTP, and this track's peaks reach {tp} dBTP. So it "
                f"plays at {st['spotify_plays_at']} LUFS, quieter than most tracks around it. A limiter on the master "
                "would let it come up further.")
    elif gain > 0.4:
        spot = f"Spotify turns it up {gain} dB to its usual {SPOTIFY_TARGET_LUFS:g} LUFS."
    else:
        spot = "That's Spotify's own playback level, so it plays as it is."
    L.append(f"- **On Spotify:** {spot}")
    if tp > 0:
        L.append(f"- **Clipping:** the peaks go {tp} dB above digital full scale, so this file distorts once it's "
                 "converted or played back. Record a little further from the source or with lower input gain, or put "
                 "a limiter on the master.")
    elif tp > -1 or (lufs > SPOTIFY_TARGET_LUFS and tp > -2):
        limit = -2 if lufs > SPOTIFY_TARGET_LUFS else -1
        why = (f"the {limit} dBTP Spotify recommends for masters louder than {SPOTIFY_TARGET_LUFS:g} LUFS" if limit == -2
               else f"the {limit} dBTP Spotify recommends")
        L.append(f"- **Peaks:** at {tp} dBTP they're above {why}, so converting to streaming formats can add a little "
                 f"distortion. A true-peak limiter set to {limit} dBTP fixes it.")
    corr = st.get("stereo_correlation")
    if corr is not None and corr < 0:
        L.append(f"- **Phase:** left and right are partly out of phase (correlation {corr}), so parts of the mix "
                 "cancel out on a mono speaker such as a phone. Check for a flipped channel or a very wide stereo effect.")
    return L


def feel_report(r):
    f = r.get("feel") or {}
    if not f:
        return []
    L = ["", f"### ⚡ Energy & mood", "",
         f"_We measured these from the audio. {tip('Energy', 'Energy level')} comes from loudness, notes per "
         f"second, brightness and tempo, and {tip('mood', 'Mood')} from the mode, how much minor chords play, "
         f"brightness and tempo._", "",
         f"- **Overall:** energy **{f['energy']}/10**, mood **{f['mood']}** "
         f"(positivity {f['valence']}/100) · the track {f['arc']}"
         + (f" · peak: section **{f['peak']['label']}** ({fmt_time(f['peak']['start'])}–{fmt_time(f['peak']['end'])})"
            if f["contrast"] >= 10 else "")]
    if f["contrast"] < 10 and len(f["sections"]) > 1:
        band = (r.get("drums") or {}).get("present") or (r.get("bass") or {}).get("present")
        L.append("- **Contrast:** the sections all feel much the same. "
                 + ("A breakdown (drop the drums and bass) or a fuller last section would give the track more shape."
                    if band else "A quieter, sparser passage, or a bigger, fuller last section, would give the track "
                                 "more shape."))
    L += ["", f"| Section | Time | {tip('Energy', 'Energy level')} | {tip('Mood')} | {tip('Positivity')} | "
          f"{tip('Loudness vs peak')} | {tip('Notes/sec', 'Notes per second')} | {tip('Brightness')} | {tip('Minor chords')} |",
          "|---------|------|--------|------|------------|------------------|-----------|------------|--------------|"]
    for s in f["sections"]:
        L.append(f"| **{s['label']}** | {fmt_time(s['start'])}–{fmt_time(s['end'])} | "
                 f"`{'█' * s['energy']}{'░' * (10 - s['energy'])}` {s['energy']} | {s['mood']} | {s['valence']} | "
                 f"{s['loudness_vs_peak_db']} dB | {s['notes_per_sec']} | {s['brightness_hz']} Hz | "
                 f"{s['minor_chords_pct']}% |")
    L.append("")
    return L


def reference_report(r):
    """The reference song, briefly: the facts the comparison rests on, its shape and its loudness."""
    L = [f"## 🎯 Reference: {r['file']}", ""]
    if "error" in r:
        return L + [f"⚠️ We couldn't analyse the reference ({r['error']}), so there's no comparison this time. "
                    "Try exporting it again as mp3 or wav.", ""]
    tempo = f"{r['tempo_bpm']:.0f} BPM" if r["tempo_bpm"] else "no clear beat"
    L += ["_A song you're measuring yourself against. We analysed it the same way as yours; the comparison is at "
          "the top of each of your tracks._", "",
          f"**{r['duration']}** · **{tempo}** · **{r['key']}** · {mode_tip(r['mode'])} · form "
          f"**{' '.join(s['label'] for s in r['sections'])}**", ""]
    svg = timeline_svg(r)
    if svg:
        L += [svg, ""]
    L += loudness_lines(r["sound"]) + [""]
    return L


def past_report(r):
    """What the musician's memory adds: this track next to their own saved ones, and close calls it settles."""
    if "memory_tracks" not in r:
        return []
    L = ["", "### 🪞 Next to your earlier tracks", ""]
    if r["memory_tracks"] < 2:
        return L + [f"_Your memory is on. Once you've saved two or more analyses, we'll compare each new track with "
                    "your own earlier ones here._"]
    L += [f"_Compared with the {r['memory_tracks']} tracks in your history (see “What we remember” on your profile)._",
          ""]
    L += [f"- {x}" for x in r.get("past") or []] or ["- Nothing stands out: this one sits right among your others."]
    L += [f"- **Close call:** {x}" for x in r.get("close_calls") or []]
    L += ["", "_The ideas below are ordered by what you've found useful before; ones you've already seen come last._"]
    return L


REF_HUES = [28, 200, 290]  # one colour per reference in the overlays (yours is drawn in the text colour)


def _overlay_svg(series, xs_label, y_label, aria, W=720, H=150, x_ticks=None):
    """Lines laid over each other: series = [(name, [(x 0-1, y 0-1)], style)], style "mine" or a hue."""
    top, bottom, left = 14 * len(series) + 10, 26, 18  # room for the legend above, and for the edge labels
    ph = H - top - bottom
    paths = []
    for name, pts, style in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{left + (W - 2 * left) * x:.1f},{top + ph * (1 - y):.1f}"
                     for i, (x, y) in enumerate(pts))
        stroke = ('stroke="currentColor" stroke-width="2.2"' if style == "mine" else
                  f'stroke="hsl({style},75%,50%)" stroke-width="1.6" stroke-dasharray="5 3"')
        paths.append(f'<path d="{d}" fill="none" {stroke}><title>{html.escape(name)}</title></path>')
    ticks = "".join(f'<text x="{left + (W - 2 * left) * x:.1f}" y="{H - 8}" text-anchor="middle" font-size="10" '
                    f'fill="currentColor" fill-opacity="0.7">{html.escape(t)}</text>' for x, t in (x_ticks or []))
    legend = "".join(
        f'<text x="{W - 6}" y="{12 + 14 * i}" text-anchor="end" font-size="11" '
        + ('font-weight="700" fill="currentColor"' if style == "mine" else f'fill="hsl({style},75%,45%)"')
        + f'>{"━" if style == "mine" else "╍"} {html.escape(name[:40])}</text>'
        for i, (name, _, style) in enumerate(series))
    return (f'<svg class="overlay" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(aria)}" '
            f'style="max-width:{W}px;display:block;margin:4px 0 4px">{"".join(paths)}{ticks}{legend}</svg>'
            f'\n\n<span class="muted small">{html.escape(y_label)}</span>')


def energy_overlay_svg(mine, refs):
    """The shape of each song: loudness over time (0-1 over 40 dB), on one time axis."""
    tracks = [(f"yours: {mine['file']}", mine, "mine")] + [(r["file"], r, REF_HUES[i % len(REF_HUES)])
                                                         for i, r in enumerate(refs)]
    tracks = [t for t in tracks if (t[1].get("sound") or {}).get("energy_line") and t[1].get("duration_sec")]
    if len(tracks) < 2:
        return ""
    longest = max(t[1]["duration_sec"] for t in tracks)
    series = []
    for name, r, style in tracks:
        line, dur = r["sound"]["energy_line"], r["duration_sec"]
        series.append((name, [((i + 0.5) / len(line) * dur / longest, v) for i, v in enumerate(line)], style))
    step = next(st for st in (10, 15, 30, 60, 120, 300, 600) if longest / st <= 10)
    ticks = [(t / longest, fmt_time(t)) for t in range(0, int(longest) + 1, step)]
    return _overlay_svg(series, "", "Loudness over time on one clock: higher is louder, and the length shows too.",
                        "Loudness over time of your track and the songs you admire", x_ticks=ticks)


def tonal_overlay_svg(mine, refs):
    """Tonal balance in octaves, each song's own share per octave (so loudness doesn't matter)."""
    tracks = [(f"yours: {mine['file']}", mine, "mine")] + [(r["file"], r, REF_HUES[i % len(REF_HUES)])
                                                         for i, r in enumerate(refs)]
    tracks = [t for t in tracks if (t[1].get("sound") or {}).get("octave_db")]
    if len(tracks) < 2:
        return ""
    bands = list(tracks[0][1]["sound"]["octave_db"])
    if any(list(t[1]["sound"]["octave_db"]) != bands for t in tracks):
        return ""
    lo = min(min(t[1]["sound"]["octave_db"].values()) for t in tracks)
    lo = max(lo, -60.0)
    span = -lo or 1.0
    xs = [i / (len(bands) - 1) for i in range(len(bands))]
    series = [(name, [(x, max(0.0, (r["sound"]["octave_db"][b] - lo) / span)) for x, b in zip(xs, bands)], style)
              for name, r, style in tracks]
    names = {b: (f"{float(b) / 1000:g}k" if float(b) >= 1000 else b) for b in bands}
    return _overlay_svg(series, "", "Tonal balance by octave (Hz): each song's own share of its energy, so how loud "
                        "it is doesn't matter. Where the lines part, one puts more into that range.",
                        "Tonal balance of your track and the songs you admire",
                        x_ticks=[(x, names[b]) for x, b in zip(xs, bands)])


def _norm_comparison(c):
    """Results saved before several references were possible held one comparison; read both the same way."""
    if c and "each" not in c:
        c = {"references": [c["reference"]], "each": [c], "shared": [], "close": [], "different": []}
    return c


def comparison_report(r):
    """Your track next to the songs you admire: where you stand, what they all share, the overlays, and each
    reference in detail. Comes first in the track's report: it's what you came to find out."""
    c = _norm_comparison(r.get("comparison"))
    if not c:
        return []
    n = len(c["each"])
    names = ", ".join(f"**{x}**" for x in c["references"])
    L = [f"### 🎯 Next to the {'songs' if n > 1 else 'song'} you admire", "",
         f"_Your track beside {names}, measured the same way. We only point out differences big enough to hear; a song "
         "you admire is a guide, not a target._", ""]
    if c.get("close") or c.get("different"):
        stand = []
        if c["close"]:
            stand.append("**Already close:** " + ", ".join(c["close"]))
        if c["different"]:
            stand.append("**Where you differ:** " + ", ".join(
                d["area"] + (f" ({d['references']} of {n})" if n > 1 else "") for d in c["different"]))
        L += [" · ".join(stand), ""]
    if c.get("shared"):
        L += [f"**What {'both' if n == 2 else f'all {n}'} have in common that yours doesn't**, the strongest leads:", ""]
        L += [f"- **{i['area']}:** {i['idea']}" for i in c["shared"]] + [""]
    elif n > 1:
        L += [f"_No difference is shared by {'both' if n == 2 else f'all {n}'}; each one's notes below are its own._", ""]
    curves = c.get("curves") or []
    for svg in (energy_overlay_svg(r, curves), tonal_overlay_svg(r, curves)):
        if svg:
            L += [svg, ""]
    for k, e in enumerate(c["each"]):
        L += [f"<details markdown=\"1\"{' open' if k == 0 else ''}><summary><b>Side by side with "
              f"{html.escape(e['reference'])}</b></summary>", "", "| | Yours | Theirs |", "|---|---|---|"]
        L += [f"| {x['aspect']} | {x['yours']} | {x['theirs']} |" for x in e["rows"]]
        L.append("")
        if e["ideas"]:
            L += ["**What they do differently, and what you could try**", ""]
            L += [f"- **{i['area']}:** {i['idea']}" for i in e["ideas"]]
        else:
            L.append("Nothing we measure differs enough to hear: in tempo, shape, groove and sound you're already close.")
        L += ["", "</details>", ""]
    return L


def producer_report(r):
    """The producer's view (electronic.py): groove engine, sidechain, the dance arrangement, the mix for clubs."""
    v = r.get("producer")
    if not v:
        return []
    L = ["", "### 🎛️ Producer's view", "", "_What electronic and other produced music is judged by: the kick, the "
         "pumping, the arrangement in bars, and how the mix holds up on a club system or a phone._", ""]
    if v.get("kick"):
        L.append(f"- **{tip('Drum pattern', 'Drum grid')}:** " + ", ".join(v["kick"]))
    p = v.get("pump")
    if p:
        L.append(f"- **{tip('Sidechain pumping', 'Sidechain')}:** " + (
            f"{p['kind']}, the sustained sounds duck about {p['depth_db']} dB after each beat and swell back"
            if p.get("kind") else "none: the pads and synths hold steady under the kick"))
    m = v.get("map")
    if m:
        roles = {"intro": "Intro", "build": "Build", "drop": "Drop", "breakdown": "Breakdown", "outro": "Outro",
                 "groove": "Groove"}
        L += ["", f"**{tip('Dance arrangement', 'Drop')}** (bars at {r['tempo_bpm']:.0f} BPM)", "",
              "| Section | Role | Time | Bars | Level | |", "|---|---|---|---|---|---|"]
        for x in m["sections"]:
            extra = []
            if x["role"] == "intro" and x["beat_only"]:
                extra.append("drums only: easy to mix in")
            if x.get("impact_db") is not None:
                extra.append(f"lands +{x['impact_db']} dB")
            if x["role"] == "build":
                extra.append(f"climbs {x['rise_db']} dB")
            L.append(f"| {x['label']} | {roles[x['role']]} | {fmt_time(x['start'])}–{fmt_time(x['end'])} | "
                     f"{x['bars']} | {x['level_db']} dB | {'; '.join(extra)} |")
        facts = []
        if m.get("first_drop_sec") is not None:
            facts.append(f"first drop at **{fmt_time(m['first_drop_sec'])}** (bar {m['first_drop_bar']})")
        if m.get("on_phrase_pct") is not None:
            facts.append(f"{m['on_phrase_pct']}% of the sections last a whole number of {electronic.PHRASE_BARS}-bar "
                         "phrases (give or take a bar)")
        if facts:
            L += ["", "_" + "; ".join(facts).capitalize() + "._"]
        L.append("")
    w = v.get("widths")
    if w and w.get("low") is not None:
        state = ("wide: part of it cancels on mono club systems and phones" if v["low_end_wide"] else
                 "mono enough for club systems and phones" if w.get("low_share", 0) >= electronic.LOW_END_SHARE
                 else "hardly any low end to judge")
        L.append(f"- **{tip('Low end', 'Mono low end')}:** correlation {w['low']} below {electronic.LOW_END_HZ} Hz, "
                 f"{state} · mids {w['mid']} · highs {w['high']}")
    if v.get("plr_db") is not None:
        plr = v["plr_db"]
        how = ("heavily limited: dense and loud, little punch left" if plr < 8 else
               "typical of a modern club or pop master" if plr < 11 else "dynamic: plenty of punch in the peaks")
        L.append(f"- **{tip('Peak-to-loudness', 'PLR')}:** {plr} dB, {how}")
    if v.get("dj"):
        k = r["key_index"]
        nb = camelot_neighbours(k)
        mixes = ", ".join(dict.fromkeys(camelot(x) for x in (k, nb["relative"], nb["up a fifth"], nb["down a fifth"])))
        L.append(f"- **{tip('Mixes with', 'Camelot')}:** tracks at {v['dj']['bpm_low']}–{v['dj']['bpm_high']} BPM "
                 f"(within the pitch fader's ~6%) in {mixes}")
    return L


def build_report(results):
    refs = [r for r in results if r.get("role") == "reference"]
    results = [r for r in results if r.get("role") != "reference"]
    ok = [r for r in results if "error" not in r]
    L = ["# 🎵 Music Inspiration Report", "",
         f"_{len(results)} track{'' if len(results) == 1 else 's'}, analysed and ready for ideas"
         + (f", next to {'your reference ' if len(refs) == 1 else 'the songs you admire: '}"
            + ", ".join(x["file"] for x in refs) if refs else "") + "._", ""]
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
              "_These pairs sit well together on the Camelot wheel, so they're good for medleys, mashups, DJ "
              "transitions or a set list._", ""]
        pairs = pairings(ok)
        if pairs:
            for p in pairs:
                L.append(f"- **{p['a']}** + **{p['b']}**: {p['keys']} — {p['relation']}, {p['tempo']}")
        else:
            L.append("- None of these share a compatible key, so together they'd contrast rather than blend.")
        L.append("")
    for r in refs:
        L += reference_report(r)

    L += ["---",
          "_A quick word on accuracy: all of this is automatic, so treat it as a well-informed first listen. "
          "We get the key right about 70–80% of the time on real music, and when we miss it's usually the "
          "relative key (B minor instead of D major). Chords come out best when the harmony is clear and "
          "sustained, and we simplify them to major, minor and 7ths. Tempo can land at half or double what you "
          "feel, and section letters are a rough map. It's a starting point for ideas, not a transcription._",
          "", f"_Analyzer version {CODE_VERSION}._"]
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
    for name in ("musicanalyze.py", "instruments.py", "inspiration.py", "notation.py", "electronic.py"):
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
                 cache_dir=DEFAULT_CACHE_DIR, context=None, reference=None, memory=None):
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
    refs = []
    ref_paths = [Path(x) for x in ([reference] if isinstance(reference, (str, Path)) else list(reference or []))
                 ][:MAX_REFERENCES]
    heard = hear_instruments([*map(str, ref_paths), *(str(Path(f)) for f in files)], progress)
    # songs the musician admires: analysed first, so each comparison can reach the AI too
    for reference in ref_paths:
        reference = Path(reference)
        progress(f"Listening to a song you admire, {reference.name}…")
        try:
            ref = analyze(reference, cache_dir=cache_dir)
            ref.update(role="reference", analyzer_version=CODE_VERSION, heard=heard.get(str(reference)))
        except Exception as e:
            progress(f"  ! {reference.name}: {e}")
            ref = {"file": reference.name, "role": "reference", "error": "unreadable or unsupported audio"
                   if "Error opening" in str(e) or "Invalid data" in str(e) else str(e)[:200]}
        refs.append(ref)
    try:
        for i, f in enumerate(files, 1):
            f = Path(f)
            progress(f"[{i}/{len(files)}] Listening to {f.name}…")
            try:
                r = analyze(f, inspiration.melody_for(f.name, melody_text), cache_dir=cache_dir, context=context)
                r["analyzer_version"] = CODE_VERSION
                if heard.get(str(f)):
                    r["heard"] = heard[str(f)]
                if refs:
                    r["comparison"] = compare_all(r, refs)
                if memory:  # a signed-in musician who switched their memory on (memory.py)
                    import memory as mem
                    r["past"] = mem.compare_with_past(r, memory, fmt_time)
                    r["close_calls"] = mem.close_call_notes(r, memory)
                    r["ideas"] = mem.rank_ideas(r["ideas"], memory)
                    r["memory_profile"] = mem.ai_profile(memory)
                    r["memory_tracks"] = memory.get("tracks", 0)
                r["midi"] = export_midis(r, out / "midi")
                try:
                    r["sheets"] = export_sheets(r, out / "sheets")
                except Exception as e:  # the sheet is a bonus; never lose the analysis over it
                    progress(f"  ! {f.name}: sheet music skipped ({e})")
                    r["sheets"] = []
                if ai_pool:
                    pending.append((r, ai_pool.submit(_timed_brainstorm, r, llm)))  # overlaps the next track
            except Exception as e:  # unreadable / corrupt file — keep going
                progress(f"  ! {f.name}: {e}")
                r = {"file": f.name, "error": "unreadable or unsupported audio" if "Error opening" in str(e)
                     or "Invalid data" in str(e) else str(e)[:200]}
            results.append(r)
        results += refs

        if pending:
            for r, _ in pending:
                r["ai"] = {"pending": True}
            report = _write_outputs(results, out)
            if on_analysis_done:
                on_analysis_done(report)
            for n, (r, fut) in enumerate(pending, 1):
                progress(f"Your analysis is ready. Waiting on the AI's ideas ({n}/{len(pending)})…")
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
    ap.add_argument("--instruments", help="which instruments play, comma-separated: " + ", ".join(INSTRUMENTS))
    ap.add_argument("--genre", choices=list(GENRES), help="the genre")
    ap.add_argument("--reference", action="append", help="a song you admire, compared side by side with each of "
                    "your tracks (up to 3: repeat the option)")
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
                             melody_text=args.melody, cache_dir=None if args.no_cache else DEFAULT_CACHE_DIR,
                             context=clean_context((args.instruments or "").replace(" ", "").split(","), args.genre),
                             reference=args.reference)
    print(report)
    print(f"\nWrote {out / 'music_report.md'}, {out / 'music_results.json'} and MIDI files in {out / 'midi'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
