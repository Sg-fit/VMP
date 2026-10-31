"""A track's portrait: the whole track as one round, colourful picture, drawn from what the analysis measured.

Time runs clockwise from the top, like a clock. Every layer is a measurement, drawn by one fixed rule, so two
tracks' pictures can be compared at a glance and nothing in them is decoration:

- the outer ring is the chords, each coloured by its root's place on the circle of fifths, so closely related
  chords get neighbouring colours (minor chords darker, power chords greyer);
- the thin rings inside it are who plays when, one ring per part, brighter where it's louder;
- the petal is how loud each moment is, tinted by its section's mood, one colour for each mood the analysis names
  (cool for calm moods, warm for lively ones, violet and crimson for dark ones);
- the dots are the notes, coloured on the same wheel: the tune's further out (higher notes further out), the
  bass's further in;
- the centre is the key's home note in its colour, with the tempo.

reading() says in words what the picture shows. It's shown under the picture and given to the AI, which reads
the picture from the same values that draw it (it never sees the audio), and the page draws it as SVG with
<title> tooltips and data-t0/data-t1 spans like the other charts, so it lights up and plays along with them.
"""

import colorsys
import math
import re

import charts

esc = charts.esc
NOTES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
PC = {**{n: i for i, n in enumerate(NOTES)}, "C#": 1, "D#": 3, "Gb": 6, "G#": 8, "A#": 10, "Cb": 11, "Fb": 4,
      "E#": 5, "B#": 0}
SIZE = 420
C = SIZE / 2
DISC = 200            # the picture's background disc
RING = (176, 198)     # the chords
LINEUP = (155, 172)   # who plays when
PETAL = (34, 150)     # loudness: from the centre's edge to the lineup rings
TUNE = (96, 146)      # the top line's notes
LOW = (44, 88)        # the bass notes
HUB = 30              # the key
MAX_DOTS = 420        # per line: enough to show its shape; more only adds weight to the page
VISIBLE = .03         # a chord colour the eye picks out: 3% of the ring is an 11° arc, a clear patch at any size
RINGS = 6             # who-plays-when rings: every row the line-up can have (six separated stems)
FAR = 3               # steps round the circle of fifths past which a chord is "far": I, IV, V and ii sit within two
EDGE_SEC = 8          # a swell in the first or last seconds is the track starting or stopping (as for builds)
NEAR_PEAK = .75       # a swell: within a quarter of the loudness scale of the loudest moment
BG = "#151b26"
# One colour per mood the analysis names (musicanalyze.mood_words: three levels of energy by three of positivity),
# so a tint always agrees with the words in the report. Calm moods are cool, lively ones warm, darker moods bluer
# or redder; all bright enough to read on the dark disc.
MOOD_COLOURS = {
    "calm, peaceful": ("#2ec4b6", "teal"), "reflective, wistful": ("#8a9cf0", "periwinkle"),
    "sad, melancholic": ("#4f7cff", "blue"), "warm, upbeat": ("#f4a259", "amber"),
    "steady, bittersweet": ("#c48fb8", "mauve"), "brooding, tense": ("#9b6ae0", "violet"),
    "uplifting, energetic": ("#ffd23f", "gold"), "driving, intense": ("#f2603a", "orange-red"),
    "dark, dramatic": ("#e0304a", "crimson"),
}


def fifths_steps(pc, home):
    """How many steps apart two notes are on the circle of fifths (0-6)."""
    d = ((pc - home) * 7) % 12
    return min(d, 12 - d)


def hue(pc):
    """A note's colour: its place on the circle of fifths (C red, G orange, D yellow ... F pink)."""
    return ((pc * 7) % 12) * 30


def hsl(h, s, l, a=None):
    r, g, b = colorsys.hls_to_rgb(h / 360, l, s)
    rgb = f"{round(r * 255)},{round(g * 255)},{round(b * 255)}"
    return f"rgba({rgb},{a})" if a is not None else f"rgb({rgb})"


def parse_chord(name):
    """(root pitch class, quality) of a chord name like F#m7, Bb, D5, Bdim; (None, None) for no chord."""
    m = re.match(r"([A-G][#b]?)(.*)", str(name or ""))
    if not m or m.group(1) not in PC:
        return None, None
    rest = m.group(2)
    q = ("dim" if rest.startswith(("dim", "°")) else "power" if rest == "5" else
         "minor" if rest.startswith("m") and not rest.startswith("maj") else "major")
    return PC[m.group(1)], q


def chord_colour(name, a=None):
    pc, q = parse_chord(name)
    if pc is None:
        return "rgba(255,255,255,.12)" if a is None else f"rgba(255,255,255,{a})"
    s, l = {"major": (.72, .60), "minor": (.62, .42), "dim": (.45, .32), "power": (.22, .58)}[q]
    return hsl(hue(pc), s, l, a)


def _mood(s):
    """A section's mood words, as the analysis named them (or would, from its energy and positivity)."""
    if s.get("mood") in MOOD_COLOURS:
        return s["mood"]
    import musicanalyze as ma
    return ma.mood_words(s.get("arousal") or 0, s.get("valence") or 0)


def mood_colour(s):
    return MOOD_COLOURS[_mood(s)][0]


def mood_word(s):
    """The tint's name and the mood it stands for, e.g. "amber (warm, upbeat)"."""
    return f"{MOOD_COLOURS[_mood(s)][1]} ({_mood(s)})"


def _home(r):
    """The key's home note (pitch class) and whether the key is minor."""
    pc, q = parse_chord((r.get("key") or "").replace(" major", "").replace(" minor", "m"))
    return (pc if pc is not None else r.get("key_index", 0) % 12), q == "minor"


def _chords(r, dur):
    tl = r.get("chord_timeline") or []
    out = []
    for i, c in enumerate(tl):
        t0 = float(c.get("start", 0))
        t1 = float(tl[i + 1]["start"]) if i + 1 < len(tl) else dur
        if t1 > t0:
            out.append({"t0": t0, "t1": min(t1, dur), "chord": c.get("chord"), "roman": c.get("roman", "")})
    return out


def _energy(r):
    """Loudness over the track, smoothed over three points and scaled 0-1: the timeline's fine line (in dB, as the
    ear hears it) when there is one, else the coarser 24-step energy curve."""
    snd = r.get("sound") or {}
    e = [float(x) for x in (snd.get("energy_line") or snd.get("energy_curve") or [])]
    if not e:
        return []
    sm = [sum(e[max(0, i - 1):i + 2]) / len(e[max(0, i - 1):i + 2]) for i in range(len(e))]
    top = max(sm) or 1.0
    return [x / top for x in sm]


def _top_notes(r):
    return [(float(t), int(n)) for t, n in ((r.get("top_line") or {}).get("events") or [])]


def _bass_notes(r):
    return [(float(n["start"]), int(n["midi"])) for n in (r.get("bass_transcription") or []) if n.get("midi") is not None]


def _thin(notes, k=MAX_DOTS):
    return notes if len(notes) <= k else [notes[round(i * (len(notes) - 1) / (k - 1))] for i in range(k)]


class Polar:
    def __init__(self, dur):
        self.dur = max(float(dur or 0), 1e-6)

    def ang(self, t):
        return -math.pi / 2 + 2 * math.pi * min(max(t, 0), self.dur) / self.dur

    def xy(self, rad, t):
        a = self.ang(t)
        return C + rad * math.cos(a), C + rad * math.sin(a)

    def band(self, r1, r2, t0, t1):
        """An arc band between radii r1 < r2 from t0 to t1 (split so no piece goes past half way round)."""
        parts, n = [], max(1, math.ceil((t1 - t0) / (0.45 * self.dur)))
        for k in range(n):
            a, b = t0 + (t1 - t0) * k / n, t0 + (t1 - t0) * (k + 1) / n
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self.xy(r2, a), self.xy(r2, b), self.xy(r1, b), self.xy(r1, a)
            parts.append(f"M{x1:.1f},{y1:.1f}A{r2},{r2} 0 0 1 {x2:.1f},{y2:.1f}L{x3:.1f},{y3:.1f}"
                         f"A{r1},{r1} 0 0 0 {x4:.1f},{y4:.1f}Z")
        return "".join(parts)


def svg(r, small=False, width=None):
    """The portrait as inline SVG. small=True: a decorative thumbnail (no dots, no tooltips) for summaries."""
    dur = float(r.get("duration_sec") or 0)
    if dur <= 0 or "error" in r:
        return ""
    P = Polar(dur)
    span = (lambda t0, t1: "") if small else charts.span
    tip = (lambda s: "") if small else (lambda s: f"<title>{esc(s)}</title>")
    fmt = charts._fmt_time
    home, minor = _home(r)
    body = [f'<circle cx="{C}" cy="{C}" r="{DISC}" fill="{BG}"/>']

    # the petal: loudness, tinted by each section's mood
    e = _energy(r)
    secs = (r.get("feel") or {}).get("sections") or r.get("sections") or []
    if e:
        def at(t):  # each value sits at the middle of its stretch of the track; in between, a straight line
            x = min(max(t / dur * len(e) - .5, 0), len(e) - 1)
            i = min(int(x), len(e) - 2) if len(e) > 1 else 0
            return e[i] if len(e) == 1 else e[i] + (e[i + 1] - e[i]) * (x - i)
        for s in secs:
            t0, t1 = float(s["start"]), float(s["end"])
            steps = max(2, int((t1 - t0) / dur * 240))
            pts = [P.xy(PETAL[0] + (PETAL[1] - PETAL[0]) * at(t0 + (t1 - t0) * i / steps), t0 + (t1 - t0) * i / steps)
                   for i in range(steps + 1)]
            ix0, iy0 = P.xy(PETAL[0], t0)
            ix1, iy1 = P.xy(PETAL[0], t1)
            large = 1 if (t1 - t0) / dur > 0.5 else 0
            d = (f"M{ix0:.1f},{iy0:.1f}" + "".join(f"L{x:.1f},{y:.1f}" for x, y in pts)
                 + f"L{ix1:.1f},{iy1:.1f}A{PETAL[0]},{PETAL[0]} 0 {large} 0 {ix0:.1f},{iy0:.1f}Z")
            words = f"Section {s.get('label', '')} {fmt(t0)}–{fmt(t1)}: {s.get('mood') or ''}"
            if isinstance(s.get("energy"), int):
                words += f", energy {s['energy']}/10"
            body.append(f'<path d="{d}" fill="{mood_colour(s)}" fill-opacity=".9"'
                        f'{span(t0, t1)}>{tip(words)}</path>')

    # the chords
    for c in _chords(r, dur):
        words = f"{fmt(c['t0'])}–{fmt(c['t1'])} · {c['chord']}" + (f" ({c['roman']})" if c["roman"] else "")
        body.append(f'<path d="{P.band(RING[0], RING[1], c["t0"], c["t1"])}" fill="{chord_colour(c["chord"])}" '
                    f'stroke="{BG}" stroke-width=".6"{span(c["t0"], c["t1"])}>{tip(words)}</path>')

    # who plays when
    if not small:
        import musicanalyze as ma  # the lineup lives with the analysis; imported here so charts stay standalone
        rows = ma.lineup(r)[:RINGS]
        if rows:
            h = (LINEUP[1] - LINEUP[0]) / len(rows)
            for k, row in enumerate(rows):
                r1, r2 = LINEUP[0] + k * h + .5, LINEUP[0] + (k + 1) * h - .5
                for cell in row["cells"]:
                    if not cell["on"]:
                        continue
                    words = f"{row['name']} in section {cell['label']} ({fmt(cell['start'])}–{fmt(cell['end'])})"
                    if cell.get("note"):
                        words += f", {cell['note']}"
                    body.append(f'<path d="{P.band(r1, r2, cell["start"], cell["end"])}" fill="#fff" '
                                f'fill-opacity="{.15 + .45 * float(cell.get("p") or .6):.2f}"'
                                f'{span(cell["start"], cell["end"])}>{tip(words)}</path>')

    # the notes: the tune further out (higher notes further out), the bass further in
    if not small:
        for notes, (r1, r2), rad, what in ((_thin(_top_notes(r)), TUNE, 1.7, "The tune's notes"),
                                           (_thin(_bass_notes(r)), LOW, 2.1, "The bass notes")):
            if not notes:
                continue
            lo, hi = min(n for _, n in notes), max(n for _, n in notes)
            dots = []
            for t, n in notes:
                x, y = P.xy(r1 + (r2 - r1) * ((n - lo) / (hi - lo) if hi > lo else .5), t)
                dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad}" fill="{hsl(hue(n % 12), .85, .66)}"/>')
            body.append(f'<g>{tip(what + ": higher notes sit further out, coloured like the chords")}{"".join(dots)}</g>')

    # section edges, letters round the outside
    for s in r.get("sections") or []:
        t0 = float(s["start"])
        x1, y1 = P.xy(PETAL[0], t0)
        x2, y2 = P.xy(RING[1], t0)
        body.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#fff" stroke-opacity=".35" '
                    f'stroke-width=".8"/>')
        if not small:
            tx, ty = P.xy(RING[1] + 10, (t0 + float(s["end"])) / 2)
            body.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="11" text-anchor="middle" dominant-baseline="middle" '
                        f'fill="currentColor" fill-opacity=".7"{span(t0, float(s["end"]))}>{esc(s["label"])}'
                        f'<title>Section {esc(s["label"])}: {fmt(t0)}–{fmt(float(s["end"]))}</title></text>')

    # the centre: home note and tempo
    key_short = (r.get("key") or "").replace(" major", "").replace(" minor", "m")
    words = f"Key {r.get('key', '')}: its home note, in its colour on the wheel"
    body.append(f'<circle cx="{C}" cy="{C}" r="{HUB}" fill="{hsl(hue(home), .6 if minor else .7, .4 if minor else .58)}">'
                f'{tip(words)}</circle>')
    if not small:
        body.append(f'<text x="{C}" y="{C - 2}" font-size="15" font-weight="600" text-anchor="middle" fill="#fff">'
                    f'{esc(key_short)}</text>')
        if r.get("tempo_bpm"):
            body.append(f'<text x="{C}" y="{C + 13}" font-size="9" text-anchor="middle" fill="#fff" fill-opacity=".85">'
                        f'{round(r["tempo_bpm"])} BPM</text>')

    label = f"Picture of the whole track: {summary(r)}"
    extra = "" if small else f' data-pdur="{dur:.2f}" data-c="{C}"'
    return charts._svg(SIZE, SIZE, "".join(body), label, cls="portrait", max_w=width or (150 if small else SIZE),
                       decorative=small, extra=extra)


def summary(r):
    """One line for the picture's accessible label."""
    return " ".join(line for _, line in reading(r)[:2])


def swells(r):
    """Where the petal is widest: up to two moments near the loudest (within NEAR_PEAK), at least a tenth of the
    track apart and away from its first and last seconds, in time order. Also key moments the AI may cite."""
    dur, e = float(r.get("duration_sec") or 0), _energy(r)
    peaks = []
    for i in sorted(range(len(e)), key=lambda i: -e[i]):
        t = (i + .5) / len(e) * dur
        if e[i] < NEAR_PEAK or len(peaks) == 2:
            break
        if EDGE_SEC <= t <= dur - EDGE_SEC and all(abs(t - p) > .1 * dur for p in peaks):
            peaks.append(t)
    return sorted(peaks)


def reading(r):
    """What the picture shows, layer by layer, in plain sentences: [(layer, sentence)]. Every sentence is read off
    the same values that draw the picture."""
    dur = float(r.get("duration_sec") or 0)
    if dur <= 0 or "error" in r:
        return []
    fmt = charts._fmt_time
    out = []
    home, _ = _home(r)
    chords = [c for c in _chords(r, dur) if parse_chord(c["chord"])[0] is not None]
    total = sum(c["t1"] - c["t0"] for c in chords)
    if total > 0:
        root = lambda c: parse_chord(c["chord"])[0]
        share = {}
        for c in chords:
            share[root(c)] = share.get(root(c), 0) + (c["t1"] - c["t0"]) / total
        seen = {pc for pc, v in share.items() if v >= VISIBLE}  # the colours the eye picks out
        near = sum(v for pc, v in share.items() if fifths_steps(pc, home) <= 1)
        line = (f"The ring stays close to the centre's colour: {near:.0%} of the time the chords' roots sit within one "
                f"step of home on the circle of fifths" if near >= .7 else
                f"The ring travels round the colour wheel: only {near:.0%} of the time are the chords' roots within one "
                f"step of home")
        line += f", in {len(seen)} main colour{'s' if len(seen) != 1 else ''}"
        far = max((c for c in chords if root(c) in seen), key=lambda c: (fifths_steps(root(c), home), -c["t0"]))
        steps = fifths_steps(root(far), home)
        if steps >= FAR:
            scale = {PC.get(n) for n in r.get("scale_notes") or []}
            line += (f"; the furthest is {far['chord']} (first at {fmt(far['t0'])}), {steps} steps away"
                     + (", on a root from outside the key's scale" if scale and root(far) not in scale else ""))
        out.append(("Chords", line + "."))

    snd, feel = r.get("sound") or {}, r.get("feel") or {}
    if _energy(r):
        peaks = swells(r)
        line = "The petal swells most at " + " and again at ".join(fmt(t) for t in peaks) if peaks else \
            "The petal keeps much the same size"
        if snd.get("quietest_moment"):
            line += f"; it's thinnest around {snd['quietest_moment']}"
        out.append(("Loudness", line + "."))

    secs = feel.get("sections") or []
    if secs:
        tints = {}
        for s in secs:
            labels = tints.setdefault(mood_word(s), [])
            if s["label"] not in labels:
                labels.append(s["label"])
        if len(tints) == 1:
            out.append(("Mood", f"Every section has the same tint, {next(iter(tints))}."))
        else:
            out.append(("Mood", "The tints: " + "; ".join(f"{w} in {' and '.join(labs)}" for w, labs in tints.items())
                        + "."))

    top = _top_notes(r)
    tl = r.get("top_line") or {}
    if top:
        line = f"The tune's dots spread over {tl.get('range', 'its range')}"
        if tl.get("chord_tone_pct") is not None:
            line += f"; {tl['chord_tone_pct']}% of its notes belong to the chord under them"
        empty = [s for s in r.get("sections") or []
                 if s["end"] - s["start"] >= 8 and not any(s["start"] <= t < s["end"] for t, _ in top)]
        if empty:
            line += "; no tune dots in section " + ", ".join(f"{s['label']} ({fmt(s['start'])}–{fmt(s['end'])})" for s in empty)
        out.append(("Notes", line + "."))
    if _bass_notes(r) and (r.get("bass") or {}).get("style"):
        out.append(("Bass", f"The inner dots are the bass, which {r['bass']['style']}."))

    import musicanalyze as ma
    rows = ma.lineup(r)[:RINGS]
    if rows and r.get("sections"):
        count = {i: sum(1 for row in rows if row["cells"][i]["on"]) for i in range(len(r["sections"]))}
        if len(set(count.values())) == 1:
            out.append(("Who plays", f"The same {count[0]} part{'s' if count[0] != 1 else ''} play all the way round."))
        else:
            full, thin = max(count, key=count.get), min(count, key=count.get)
            S = r["sections"]
            out.append(("Who plays", f"The rings are fullest in section {S[full]['label']} ({count[full]} parts, "
                                     f"{fmt(S[full]['start'])}) and emptiest in {S[thin]['label']} ({count[thin]}, "
                                     f"{fmt(S[thin]['start'])})."))
    return out


LEGEND = [
    ("Time", "runs clockwise from the top, like a clock; the letters round the edge are the sections."),
    ("Outer ring", "the chords, coloured by their root on the circle of fifths: neighbouring colours are closely "
                   "related chords. Darker means minor, greyer a power chord."),
    ("Thin rings", "who plays when, one ring per part, brighter where it's louder."),
    ("Petal", "how loud each moment is, tinted by its section's mood: cool colours for calm moods (teal peaceful, "
              "blue sad), warm ones for lively moods (amber warm, gold uplifting), violet and crimson for dark ones."),
    ("Dots", "the notes, coloured on the same wheel: the tune's further out (higher notes further out), the bass's "
             "further in."),
    ("Centre", "the key's home note in its colour, and the tempo."),
]
