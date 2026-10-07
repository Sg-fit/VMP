"""Melody analysis and "shared DNA" comparisons with well-known music, for musicanalyze.py.

- A melody the musician types ("A E F E A E D E") is analysed against the track's key: scale
  degrees, which modes it fits, shape, melodic devices, chords that harmonise it, and generated
  variations (inversion, retrograde, sequence, answer phrase).
- Chord loops, modes, drum grooves and melodic devices are matched against a small library of
  well-known examples. Only widely documented examples are listed here; the optional AI
  brainstorm can add more, clearly labelled as suggestions to double-check.
"""

import re

NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DEGREE = ["1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"]
INTERVAL_NAME = {0: "repeat", 1: "half-step", 2: "whole-step", 3: "minor 3rd", 4: "major 3rd", 5: "4th",
                 6: "tritone", 7: "5th", 8: "minor 6th", 9: "major 6th", 10: "minor 7th", 11: "major 7th"}

# ---------------------------------------------------------------- reference library
# Chord loops as (semitones above the first chord's root, "M"/"m"). Matching ignores key and
# rotation, so vi-IV-I-V matches I-V-vi-IV and E minor loops match A minor ones.
PROGRESSIONS = [
    ("I–V–vi–IV (the 'four chords' of pop)", [(0, "M"), (7, "M"), (9, "m"), (5, "M")],
     ["“Let It Be” — The Beatles", "“No Woman, No Cry” — Bob Marley", "“With or Without You” — U2",
      "“Zombie” — The Cranberries (as vi–IV–I–V)"]),
    ("I–vi–IV–V (the '50s / doo-wop progression)", [(0, "M"), (9, "m"), (5, "M"), (7, "M")],
     ["“Stand by Me” — Ben E. King", "“Every Breath You Take” — The Police"]),
    ("i–bVII–bVI–V (Andalusian cadence)", [(0, "m"), (10, "M"), (8, "M"), (7, "M")],
     ["“Hit the Road Jack” — Ray Charles", "“Runaway” — Del Shannon", "“Stray Cat Strut” — Stray Cats"]),
    ("I–bVII–IV (Mixolydian rock)", [(0, "M"), (10, "M"), (5, "M")],
     ["“Sweet Child o' Mine” — Guns N' Roses", "“Sympathy for the Devil” — The Rolling Stones"]),
    ("i–bVII–bVI (descending minor)", [(0, "m"), (10, "M"), (8, "M")],
     ["“All Along the Watchtower” — Bob Dylan / Jimi Hendrix"]),
    ("I–IV–V (three-chord rock'n'roll)", [(0, "M"), (5, "M"), (7, "M")],
     ["“La Bamba” — Ritchie Valens", "“Twist and Shout” — The Isley Brothers / The Beatles"]),
    ("i–IV (Dorian vamp)", [(0, "m"), (5, "M")],
     ["“Oye Como Va” — Santana", "“Evil Ways” — Santana"]),
    ("ii–V–I (the jazz cadence)", [(0, "m"), (5, "M"), (10, "M")],
     ["“Autumn Leaves” (jazz standard)", "“Fly Me to the Moon” (jazz standard)"]),
]

MODE_REFERENCES = {
    "Dorian": ["“Oye Como Va” — Santana", "“Scarborough Fair” — Simon & Garfunkel", "“So What” — Miles Davis"],
    "Mixolydian": ["“Norwegian Wood” — The Beatles", "“Sweet Home Alabama” — Lynyrd Skynyrd"],
    "Lydian": ["“The Simpsons” theme — Danny Elfman", "“Flying in a Blue Dream” — Joe Satriani"],
    "Phrygian dominant": ["“Misirlou” — Dick Dale", "“Hava Nagila” (traditional)", "flamenco"],
    "Harmonic minor": ["neoclassical metal (Yngwie Malmsteen)", "Baroque and Classical minor-key music"],
    "Phrygian": ["flamenco", "a lot of metal riffing and film-score tension"],
    "Aeolian (natural minor)": ["“Stairway to Heaven” — Led Zeppelin (A minor intro)",
                                "“Losing My Religion” — R.E.M."],
}

GROOVE_REFERENCES = {
    "backbeat": "the standard rock/pop backbeat — the same skeleton as “Billie Jean” (Michael Jackson)",
    "four-on-the-floor": "four-on-the-floor, the disco/house pulse — “Stayin' Alive” (Bee Gees), "
                         "“One More Time” (Daft Punk)",
    "half-time": "a half-time feel, as in hip-hop/trap beats and half-time rock choruses",
    "3+3+2": "the 3+3+2 tresillo — the rhythm under reggaeton/dancehall and pop such as “Shape of You” "
             "(Ed Sheeran)",
}


def _quality(q):
    if q == "5":
        return None  # power chord: no third, matches either
    return "m" if q.startswith("m") and not q.startswith("maj") else "M"


def _signature(chords):
    """[(root, q), ...] -> [(interval from first root, M/m/None), ...]"""
    r0 = chords[0][0]
    return [((r - r0) % 12, _quality(q)) for r, q in chords]


def _same(sig, ref):
    # A power chord (no audible third) can't confirm a progression whose identity depends on major vs
    # minor, so it never matches (e.g. "E5 - A" is not evidence of the i-IV Dorian vamp).
    return len(sig) == len(ref) and all(a[0] == b[0] and a[1] == b[1] for a, b in zip(sig, ref))


def match_progression(chords, cyclic=True):
    """chords: [(root_pc, quality)]. Returns library matches (exact or contained).
    cyclic=True for a detected loop (it repeats, so it may wrap from last chord to first);
    cyclic=False for chords that were just played in order, which must match without wrapping."""
    found = []
    n = len(chords)
    for name, ref, examples in PROGRESSIONS:
        rots = [chords[i:] + chords[:i] for i in range(n)] if cyclic else             [chords[i:] for i in range(n - len(ref) + 1)]
        if any(_same(_signature(r), ref) for r in rots):
            found.append({"progression": name, "match": "same loop", "examples": examples})
        elif len(ref) >= 3 and n > len(ref) and any(_same(_signature(r[:len(ref)]), ref) for r in rots):
            found.append({"progression": name, "match": "contains it", "examples": examples})
    return found


def groove_reference(traits):
    for key, text in GROOVE_REFERENCES.items():
        if any(key in t for t in traits):
            return text
    return None


# ---------------------------------------------------------------- typed melody

def parse_notes(text):
    """'A E F E | A E D E' or 'A3 E4 F4' -> list of (pc, octave or None). Unknown tokens are ignored."""
    out = []
    for tok in re.split(r"[\s,|/;\-–—]+", text.strip()):
        m = re.fullmatch(r"([A-Ga-g])(#|b|♯|♭)?(\d)?", tok)
        if m:
            pc = (NOTE_PC[m.group(1).upper()] + {None: 0, "#": 1, "♯": 1, "b": -1, "♭": -1}[m.group(2)]) % 12
            out.append((pc, int(m.group(3)) if m.group(3) else None))
    return out


def to_midi(parsed):
    """Give notes octaves: as typed, else the nearest move from the previous note (start near middle C)."""
    midi, prev = [], None
    for pc, octave in parsed:
        if octave is not None:
            n = 12 * (octave + 1) + pc
        elif prev is None:
            n = 60 + pc  # first note lands between C4 and B4, where most melodies sit
        else:
            n = min((12 * o + pc for o in range(2, 9)), key=lambda c: abs(c - prev))
        midi.append(n)
        prev = n
    return midi


def analyse_melody(text, tonic, mode, modes, spell):
    """Analyse a typed melody against the track's home note (tonic pc) and detected mode.

    modes: {name: (intervals, colour degree)}; spell(pc) -> note name in the track's key."""
    parsed = parse_notes(text)
    if len(parsed) < 3:
        return {"error": f"Couldn't read enough notes from “{text}” (use letters like A E F# Bb)."}
    midi = to_midi(parsed)
    pcs = [n % 12 for n in midi]
    rel = [(p - tonic) % 12 for p in pcs]
    used = set(rel)
    steps = [b - a for a, b in zip(midi, midi[1:])]
    sizes = [abs(s) for s in steps]

    scale = set(modes[mode][0])
    outside = sorted({spell(p) for p, r in zip(pcs, rel) if r not in scale})
    fits = [name for name, (ivs, _) in modes.items() if used <= set(ivs)]

    # Shape
    ups, downs = sum(s > 0 for s in steps), sum(s < 0 for s in steps)
    lo, hi = min(midi), max(midi)
    anchor_pc, anchor_n = max(((p, pcs.count(p)) for p in set(pcs)), key=lambda x: x[1])
    alternating = sum(pcs[i] == anchor_pc for i in range(len(pcs)) if i % 2 == 1) >= 0.6 * (len(pcs) // 2) or \
        sum(pcs[i] == anchor_pc for i in range(len(pcs)) if i % 2 == 0) >= 0.6 * ((len(pcs) + 1) // 2)
    shape = ("circles around " + spell(anchor_pc) if anchor_n >= 0.4 * len(pcs) else
             "rises" if ups > 2 * downs else "falls" if downs > 2 * ups else "moves up and down")

    devices = []
    if alternating and anchor_n >= 3:
        devices.append({"device": f"pedal-tone line: every other note returns to {spell(anchor_pc)}",
                        "reference": "the classic pedal-point riff idea — “Thunderstruck” (AC/DC) alternates "
                                     "melody notes with one repeated note; Baroque pedal-point figures do the same"})
    pairs = list(zip(rel, rel[1:]))
    if (8, 7) in pairs:
        devices.append({"device": f"b6→5 ‘sigh’ ({spell((tonic + 8) % 12)}→{spell((tonic + 7) % 12)})",
                        "reference": "the minor-key lament half-step — the same F→E fall that ends the "
                                     "“Hit the Road Jack” bass line (A–G–F–E), and a staple of flamenco"})
    if (7, 8) in pairs and (8, 7) in pairs:
        devices.append({"device": f"half-step neighbour above the 5th ({spell((tonic + 7) % 12)}–"
                                  f"{spell((tonic + 8) % 12)}–{spell((tonic + 7) % 12)})",
                        "reference": "a Spanish/Phrygian colour — it pulls toward the 5th like a dominant"})
    if (5, 7) in pairs or (7, 5) in pairs:
        devices.append({"device": f"whole-step neighbour 4–5 ({spell((tonic + 5) % 12)}/{spell((tonic + 7) % 12)})",
                        "reference": "a folk/blues-style move that keeps the line hovering on the 5th"})
    major_key = 4 in scale
    if major_key and 3 in used:
        devices.append({"device": f"borrowed minor third ({spell((tonic + 3) % 12)}) in a major key",
                        "reference": "modal mixture — a note borrowed from the parallel minor for a sudden "
                                     "bittersweet shade; a staple of pop, film and classical writing"})
    elif not major_key and 4 in used:
        devices.append({"device": f"borrowed major third ({spell((tonic + 4) % 12)}) in a minor key",
                        "reference": "modal mixture — a brief brightening borrowed from the parallel major"})
    if all(r in {0, 3, 5, 7, 10} for r in used):
        devices.append({"device": "uses only minor-pentatonic notes",
                        "reference": "the core of blues and rock riffing"})
    elif all(r in {0, 2, 4, 7, 9} for r in used):
        devices.append({"device": "uses only major-pentatonic notes",
                        "reference": "folk, country and a lot of East-Asian melody"})

    # Harmonise: chords from the scale that contain each melody note.
    def triad(off):
        ivs = sorted(scale)
        i = ivs.index(off)
        third, fifth = ivs[(i + 2) % 7], ivs[(i + 4) % 7]
        q = "m" if (third - off) % 12 == 3 else ""
        if (fifth - off) % 12 == 6:
            q = "dim"
        return off, q, {(off) % 12, third % 12, fifth % 12}
    triads = [triad(o) for o in sorted(scale)] if len(scale) == 7 else []
    harmony = {}
    for p, r in zip(pcs, rel):
        name = spell(p)
        if name not in harmony:
            harmony[name] = [spell((tonic + o) % 12) + q for o, q, tones in triads if r in tones and q != "dim"][:3]

    # Variations (stay in the detected scale)
    scale_list = sorted(scale)

    def snap(n):
        return min((n + d for d in range(-2, 3)), key=lambda c: (((c - tonic) % 12) not in scale, abs(c - n)))

    def scale_step(n, k):
        """Move n by k scale steps."""
        out = n
        step = 1 if k > 0 else -1
        for _ in range(abs(k)):
            out += step
            while (out - tonic) % 12 not in scale:
                out += step
        return out
    inversion = [midi[0]]
    for s in steps:
        inversion.append(snap(inversion[-1] - s))
    variations = {
        "original": midi,
        "inversion (upside down)": inversion,
        "retrograde (backwards)": midi[::-1],
        "sequence (one scale step higher)": [scale_step(n, 1) for n in midi],
        "answer phrase (ends on the home note)": midi[:-1] + [min((12 * o + tonic for o in range(2, 9)),
                                                                   key=lambda c: abs(c - midi[-1]))],
    }
    name_of = lambda n: f"{spell(n % 12)}{n // 12 - 1}"
    return {
        "input": text,
        "notes": [spell(p) for p in pcs],
        "degrees": [DEGREE[r] for r in rel],
        "fits_modes": fits,
        "outside_detected_scale": outside,
        "range": f"{name_of(lo)}–{name_of(hi)} ({hi - lo} semitones)",
        "steps_pct": round(100 * sum(1 <= x <= 2 for x in sizes) / max(1, len(sizes))),
        "leaps_pct": round(100 * sum(x > 2 for x in sizes) / max(1, len(sizes))),
        "intervals": [("up " if s > 0 else "down " if s < 0 else "") + INTERVAL_NAME[abs(s) % 12] for s in steps],
        "shape": shape,
        "devices": devices,
        "harmonize": harmony,
        "variations": {k: [name_of(n) for n in v] for k, v in variations.items()},
        "variations_midi": variations,
    }


def melody_for(file_name, text):
    """Pick the melody that applies to this file. Lines like 'copy 4: A E F E' target a file whose
    name contains 'copy 4'; a line without a name applies to every file."""
    if not text or not text.strip():
        return None
    generic = None
    for line in text.strip().splitlines():
        if ":" in line:
            who, notes = line.split(":", 1)
            if who.strip() and who.strip().lower().replace(" ", "_") in file_name.lower().replace(" ", "_"):
                return notes.strip()
        elif line.strip():
            generic = generic or line.strip()
    return generic
