"""Links from an analysis to other music apps: Hooktheory Trends for the chords you play (what usually comes next,
and songs with the same progression). The theory library's listening examples link to YouTube, Spotify and
Songsterr (lexicon.listen).

Everything here is a plain link, built from data the page already shows. Nothing is sent anywhere until someone
clicks, and nothing is ever logged in on their behalf.

Hooktheory Trends link format (checked against the live page): https://www.hooktheory.com/trends#key=D&scale=major&path=1.5.6.4
opens D → A → Bm → G. `path` counts scale degrees of `scale` (major, minor, dorian, phrygian, lydian, mixolydian,
locrian, harmonicMinor, phrygianDominant), so each chord must be the triad that scale builds on that degree. In a major
key, `b3`, `b6` and `b7` are the major chords borrowed from the parallel minor (C: E♭, A♭, B♭). Any chord that fits
neither gets no link at all: a link to the wrong chords would teach the wrong thing.
"""

from urllib.parse import quote

import lexicon
import musicanalyze as ma

TRENDS = lexicon.TRENDS
SCALES = {  # our mode names -> (Hooktheory's scale name, its steps above the tonic)
    "Ionian (major)": ("major", [0, 2, 4, 5, 7, 9, 11]),
    "Aeolian (natural minor)": ("minor", [0, 2, 3, 5, 7, 8, 10]),
    "Dorian": ("dorian", [0, 2, 3, 5, 7, 9, 10]),
    "Phrygian": ("phrygian", [0, 1, 3, 5, 7, 8, 10]),
    "Lydian": ("lydian", [0, 2, 4, 6, 7, 9, 11]),
    "Mixolydian": ("mixolydian", [0, 2, 4, 5, 7, 9, 10]),
    "Locrian": ("locrian", [0, 1, 3, 5, 6, 8, 10]),
    "Harmonic minor": ("harmonicMinor", [0, 2, 3, 5, 7, 8, 11]),
    "Phrygian dominant": ("phrygianDominant", [0, 1, 4, 5, 7, 8, 10]),
}
BORROWED = {3: "b3", 8: "b6", 10: "b7"}  # major chords from the parallel minor, written as Hooktheory writes them
MAX_CHORDS = 4  # Trends follows a path chord by chord; past four, hardly any songs share the exact path


def _shape(intervals):
    """(third, fifth) of a chord's notes above its root; None for a note it doesn't have (C5, Csus4: no third)."""
    third = 4 if 4 in intervals else 3 if 3 in intervals else None
    fifth = 7 if 7 in intervals else 6 if 6 in intervals else 8 if 8 in intervals else None
    return third, fifth


def trends_path(names, tonic, mode):
    """(Hooktheory scale, "1.5.6.4") for chord names in a key, or None if any chord isn't the scale's own triad (or,
    in a major key, a chord borrowed from the parallel minor)."""
    scale, steps = SCALES[mode]
    out = []
    for name in names:
        p = ma.parse_chord(name)
        if not p:
            return None
        off = (p[0] - tonic) % 12
        third, fifth = _shape(p[2])
        if off in steps:
            d = steps.index(off)
            want = ((steps[(d + 2) % 7] - off) % 12, (steps[(d + 4) % 7] - off) % 12)
            if (third is None or third == want[0]) and (fifth is None or fifth == want[1]):
                out.append(str(d + 1))
                continue
        if scale == "major" and off in BORROWED and (third, fifth) in ((4, 7), (None, 7)):
            out.append(BORROWED[off])
            continue
        return None
    return scale, ".".join(out)


def trends_url(tonic_name, scale, path):
    return f"{TRENDS}key={quote(tonic_name)}&scale={scale}&path={path}"


TRIAD = {(4, 7): "", (3, 7): "m", (3, 6): "dim", (4, 8): "aug"}


def as_shown(path, k, mode):
    """The chords as Hooktheory will show them: the triad on each degree (a power chord or a 7th becomes the plain
    triad, since Trends lists triads)."""
    scale, steps = SCALES[mode]
    tonic, out = ma.key_tonic(k), []
    for tok in path.split("."):
        if tok in BORROWED.values():  # a major chord from the parallel minor
            off = next(o for o, t in BORROWED.items() if t == tok)
            out.append(ma.spell((tonic + off) % 12, k))
            continue
        d = int(tok) - 1
        off = steps[d]
        q = TRIAD.get(((steps[(d + 2) % 7] - off) % 12, (steps[(d + 4) % 7] - off) % 12), "")
        out.append(ma.spell((tonic + off) % 12, k) + q)
    return out


def _mode_for(r, key_name):
    """The scale to read a loop in: the track's mode when the loop is in the track's key and the mode is clear,
    else plain major or minor."""
    if key_name == r.get("key") and r.get("mode_confidence") != "low" and r.get("mode") in SCALES:
        return r["mode"]
    return "Aeolian (natural minor)" if key_name.endswith("minor") else "Ionian (major)"


def _dedupe(names):
    out = []
    for n in names:
        if not out or out[-1] != n:
            out.append(n)
    return out


def chord_links(r):
    """[{chords, numerals, key, url}] for the progressions in a track: each chord loop the analysis found (in its own
    key), else the first chords as they're played. Only progressions Hooktheory can show exactly."""
    if not r or "error" in r:
        return []
    found = [(l["chords"], l.get("key") or r["key"], "loop") for l in r.get("loops") or []
             if len(l.get("chords", [])) >= 2]
    if not found:
        timeline = _dedupe([c["chord"] for c in r.get("chord_timeline") or []])
        if len(timeline) >= 2:
            found = [(timeline, r["key"], "opening")]
    out, seen = [], set()
    for names, loop_key, kind in found:
        names = _dedupe(names)[:MAX_CHORDS]
        best = None  # (chords covered, key, mode): the loop's own key or the track's, whichever explains more of it
        for key_name in dict.fromkeys([loop_key, r["key"]]):
            try:
                k = ma.KEY_NAMES.index(key_name)
            except ValueError:
                continue
            mode = _mode_for(r, key_name)
            # the longest opening stretch Hooktheory can show exactly (at least two chords)
            n = next((n for n in range(len(names), 1, -1) if trends_path(names[:n], ma.key_tonic(k), mode)), 0)
            if n and (best is None or n > best[0]):
                best = (n, key_name, k, mode)
        if not best:
            continue
        n, key_name, k, mode = best
        tp = trends_path(names[:n], ma.key_tonic(k), mode)
        if tp in seen:
            continue
        seen.add(tp)
        tonic = key_name.split()[0]
        out.append({"kind": kind, "chords": names[:n], "cut": names[n:],
                    "key": key_name if mode in ("Ionian (major)", "Aeolian (natural minor)")
                    else f"{tonic} {mode.split(' (')[0]}",
                    "shown": as_shown(tp[1], k, mode), "url": trends_url(tonic, *tp)})
    return out
