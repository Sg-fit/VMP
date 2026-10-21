"""Pictures of the analysis: small inline SVG charts for the report and the website.

Every chart is plain SVG text, so it shows on the web page and in the downloaded report alike. Colours come from
the page's own tokens (--c1 blue, --c2 orange, --c3 aqua: a palette checked for colour-blind separation in light
and dark mode; text in the page's ink), with fallbacks for viewers that don't define them. Your own track is
always drawn in the ink colour, so it never competes with a series colour. Marks carry <title> tooltips (hover),
and every chart sits next to the numbers it shows, so nothing is told by colour alone.
"""

import html
import math

C1, C2, C3 = "var(--c1, #2a78d6)", "var(--c2, #eb6834)", "var(--c3, #1baf7a)"
SERIES = [C1, C2, C3]
GRID = "currentColor"  # drawn at low opacity: a recessive hairline in either theme
ACCENT = "var(--accent, #ff5a1f)"
esc = lambda s: html.escape(str(s), quote=True)


def _svg(w, h, body, label, cls="chart", max_w=None, decorative=False):
    role = 'aria-hidden="true"' if decorative else f'role="img" aria-label="{esc(label)}"'
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" width="100%" {role} '
            f'style="max-width:{max_w or w}px;display:block;overflow:visible" '
            f'font-family="inherit">{body}</svg>')


def _text(x, y, s, size=11, anchor="start", weight=None, opacity=None, extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    o = f' fill-opacity="{opacity}"' if opacity is not None else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="currentColor"{w}{o} '
            f'{extra}>{esc(s)}</text>')


def _hbar(x, y, length, h, fill, title=""):
    """A horizontal bar: square at the baseline, a 4px rounded data end."""
    length = max(0.0, length)
    r = min(4.0, h / 2, length)
    d = (f"M{x:.1f},{y:.1f} h{max(0, length - r):.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} "
         f"a{r},{r} 0 0 1 {-r},{r} h{-max(0, length - r):.1f} z")
    t = f"<title>{esc(title)}</title>" if title else ""
    return f'<path d="{d}" style="fill:{fill}">{t}</path>'


def _fmt_time(sec):
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def _time_ticks(dur, x):
    step = next((st for st in (10, 15, 30, 60, 120, 300, 600) if dur / st <= 10), 600)
    return [(x(t), _fmt_time(t)) for t in range(0, int(dur) + 1, step)]


# ---------------------------------------------------------------- key: the Camelot wheel
def camelot_wheel(codes, highlight=None, counts=None, size=180, label="Camelot wheel"):
    """codes: {code like '8A': key name}. highlight: the track's code (filled) - its mixable neighbours are
    tinted; counts: {code: n} instead (a profile: how often each key is used, darker = more)."""
    cx = cy = size / 2
    r_out, r_mid, r_in = size / 2 - 2, size / 2 * 0.70, size / 2 * 0.40
    near = set()
    if highlight:
        n, letter = int(highlight[:-1]), highlight[-1]
        near = {f"{n % 12 + 1}{letter}", f"{(n - 2) % 12 + 1}{letter}", f"{n}{'A' if letter == 'B' else 'B'}"}
    top = max(counts.values()) if counts else 1
    parts = []
    for ring, (r1, r0) in (("B", (r_out, r_mid)), ("A", (r_mid, r_in))):
        for n in range(1, 13):
            code = f"{n}{ring}"
            a0, a1 = math.radians(n * 30 - 15 - 90), math.radians(n * 30 + 15 - 90)
            pts = [(cx + r1 * math.cos(a0), cy + r1 * math.sin(a0)), (cx + r1 * math.cos(a1), cy + r1 * math.sin(a1)),
                   (cx + r0 * math.cos(a1), cy + r0 * math.sin(a1)), (cx + r0 * math.cos(a0), cy + r0 * math.sin(a0))]
            d = (f"M{pts[0][0]:.1f},{pts[0][1]:.1f} A{r1},{r1} 0 0 1 {pts[1][0]:.1f},{pts[1][1]:.1f} "
                 f"L{pts[2][0]:.1f},{pts[2][1]:.1f} A{r0},{r0} 0 0 0 {pts[3][0]:.1f},{pts[3][1]:.1f} Z")
            if counts is not None:
                c = counts.get(code, 0)
                style = (f"fill:{C1};fill-opacity:{0.18 + 0.82 * c / top:.2f}" if c else
                         "fill:currentColor;fill-opacity:0.05")
                tip = f"{code} {codes.get(code, '')}: {c} track{'s' if c != 1 else ''}"
            elif code == highlight:
                style, tip = f"fill:{ACCENT}", f"{code} {codes.get(code, '')}: this track's key"
            elif code in near:
                style, tip = f"fill:{ACCENT};fill-opacity:0.28", f"{code} {codes.get(code, '')}: mixes smoothly with it"
            else:
                style, tip = "fill:currentColor;fill-opacity:0.05", f"{code} {codes.get(code, '')}"
            parts.append(f'<path d="{d}" style="{style}" stroke="var(--surface, #fff)" stroke-width="2">'
                         f'<title>{esc(tip)}</title></path>')
            rm = (r1 + r0) / 2
            am = math.radians(n * 30 - 90)
            bold = code == highlight or (counts and counts.get(code))
            if size < 130:  # too small to read the codes: the centre names the key
                continue
            parts.append(_text(cx + rm * math.cos(am), cy + rm * math.sin(am) + 3.5, code, size=max(8, size / 20),
                               anchor="middle", weight=700 if bold else None, opacity=None if bold else 0.55))
    if counts:
        total = sum(counts.values())
        parts.append(_text(cx, cy + 2, total, size=size / 8, anchor="middle", weight=700))
        parts.append(_text(cx, cy + size / 12 + 2, "tracks" if total != 1 else "track", size=size / 16,
                           anchor="middle", opacity=0.7))
    if highlight:
        parts.append(_text(cx, cy - 2, highlight, size=size / 8, anchor="middle", weight=700))
        parts.append(_text(cx, cy + size / 12, codes.get(highlight, "").replace(" major", " maj").replace(" minor", " min"),
                           size=size / 16, anchor="middle", opacity=0.7))
    return _svg(size, size, "".join(parts), label, max_w=size)


# ---------------------------------------------------------------- tempo
def tempo_ruler(bpm, approximate=False, usual=None, width=300, lo=40, hi=200, label=None):
    """Where the tempo sits on a 40-200 BPM ruler, with half and double time as hollow ghosts (the speed you may
    feel instead) and, when the genre is known, its usual range as a band."""
    H, pad = 70, 10
    x = lambda b: pad + (width - 2 * pad) * (min(hi, max(lo, b)) - lo) / (hi - lo)
    base = 40
    parts = [f'<line x1="{pad}" y1="{base}" x2="{width - pad}" y2="{base}" stroke="currentColor" stroke-opacity="0.35"/>']
    if usual:
        a, b = usual
        parts.append(f'<rect x="{x(a):.1f}" y="{base - 9}" width="{x(b) - x(a):.1f}" height="18" rx="3" '
                     f'style="fill:{C1};fill-opacity:0.16"><title>usual for the genre: {a}-{b} BPM</title></rect>')
    for t in range(60, hi + 1, 30):
        parts.append(f'<line x1="{x(t):.1f}" y1="{base - 3}" x2="{x(t):.1f}" y2="{base + 3}" stroke="currentColor" '
                     f'stroke-opacity="0.35"/>')
        parts.append(_text(x(t), H - 4, t, size=10, anchor="middle", opacity=0.6))
    if bpm:
        for f, word in ((0.5, "half"), (2, "double")):
            g = bpm * f
            if lo <= g <= hi:
                parts.append(f'<circle cx="{x(g):.1f}" cy="{base}" r="5" fill="none" stroke="currentColor" '
                             f'stroke-opacity="0.5" stroke-dasharray="2 2"><title>{word} time: {g:.0f} BPM</title></circle>')
        parts.append(f'<circle cx="{x(bpm):.1f}" cy="{base}" r="7" style="fill:{ACCENT}" stroke="var(--surface, #fff)" '
                     f'stroke-width="2"><title>{bpm:.0f} BPM{" (approximate)" if approximate else ""}</title></circle>')
        parts.append(_text(x(bpm), base - 14, f"{'~' if approximate else ''}{bpm:.0f}", size=14, anchor="middle",
                           weight=700))
    return _svg(width, H, "".join(parts), label or (f"Tempo {bpm:.0f} BPM" if bpm else "No clear tempo"), max_w=width)


# ---------------------------------------------------------------- mood
MOOD_CORNERS = [("tense", 0.02, 0.06, "start"), ("excited", 0.98, 0.06, "end"),
                ("sad", 0.02, 0.97, "start"), ("calm", 0.98, 0.97, "end")]


def mood_map(points, size=200, path=True, label="Energy and mood map"):
    """points: [(label, valence 0-100, arousal 0-100, tooltip)]. Across: darker to brighter mood; up: calmer to
    more energetic. With path=True the points are a journey (the sections in order)."""
    pad = 14
    span = size - 2 * pad
    px = lambda v: pad + span * v / 100
    py = lambda a: pad + span * (1 - a / 100)
    parts = [f'<rect x="{pad}" y="{pad}" width="{span}" height="{span}" fill="currentColor" fill-opacity="0.03" '
             f'stroke="currentColor" stroke-opacity="0.15"/>',
             f'<line x1="{px(50)}" y1="{pad}" x2="{px(50)}" y2="{pad + span}" stroke="currentColor" stroke-opacity="0.15"/>',
             f'<line x1="{pad}" y1="{py(50)}" x2="{pad + span}" y2="{py(50)}" stroke="currentColor" stroke-opacity="0.15"/>']
    for word, fx, fy, anchor in MOOD_CORNERS:
        parts.append(_text(pad + 4 + (span - 8) * fx if anchor == "start" else pad + span * fx - 2,
                           pad + 10 + (span - 12) * fy, word, size=9, anchor=anchor, opacity=0.55))
    parts.append(_text(size / 2, size - 1, "mood →", size=9, anchor="middle", opacity=0.55))
    parts.append(_text(3, size / 2, "energy →", size=9, anchor="middle", opacity=0.55,
                       extra=f'transform="rotate(-90 3 {size / 2})"'))
    pts = [(lab, px(v), py(a), tip) for lab, v, a, tip in points if v is not None and a is not None]
    if path and len(pts) > 1:
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (_, x, y, _) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="currentColor" stroke-opacity="0.45" stroke-width="1.5" '
                     f'stroke-linejoin="round"/>')
    for i, (lab, x, y, tip) in enumerate(pts):
        r = 8 if path else 4.5
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" style="fill:{ACCENT if path else C1}" '
                     f'stroke="var(--surface, #fff)" stroke-width="2"><title>{esc(tip)}</title></circle>')
        if path and lab:
            parts.append(_text(x, y + 3.5, lab, size=9, anchor="middle", weight=700,
                               extra='style="fill:var(--on-accent, #111)"'))
    return _svg(size, size, "".join(parts), label, max_w=size)


# ---------------------------------------------------------------- loudness
def loudness_meter(lufs, true_peak=None, plays_at=None, width=300, lo=-30, hi=-4, target=-14):
    """Integrated loudness on a -30..-4 LUFS scale, with Spotify's -14 and where Spotify would play it."""
    H, pad, base = 64, 10, 30
    x = lambda v: pad + (width - 2 * pad) * (min(hi, max(lo, v)) - lo) / (hi - lo)
    parts = [f'<rect x="{pad}" y="{base - 5}" width="{width - 2 * pad}" height="10" rx="5" fill="currentColor" '
             f'fill-opacity="0.08"/>',
             f'<rect x="{pad}" y="{base - 5}" width="{x(lufs) - pad:.1f}" height="10" rx="5" style="fill:{C1}">'
             f'<title>{lufs} LUFS integrated</title></rect>',
             f'<line x1="{x(target):.1f}" y1="{base - 11}" x2="{x(target):.1f}" y2="{base + 11}" stroke="currentColor" '
             f'stroke-width="2"><title>Spotify plays at {target} LUFS</title></line>',
             _text(x(target), base + 23, f"Spotify {target}", size=10, anchor="middle", opacity=0.8)]
    for t in range(lo, hi + 1, 6):
        if abs(t - target) > 3:
            parts.append(_text(x(t), base + 23, t, size=10, anchor="middle", opacity=0.5))
    parts.append(_text(min(x(lufs), width - 40), base - 12, f"{lufs} LUFS", size=12, anchor="middle", weight=700))
    if plays_at is not None and abs(plays_at - lufs) >= 0.4:
        parts.append(f'<circle cx="{x(plays_at):.1f}" cy="{base}" r="5" fill="none" stroke="currentColor" '
                     f'stroke-width="1.5"><title>plays on Spotify at {plays_at} LUFS</title></circle>')
    return _svg(width, H, "".join(parts), f"Loudness {lufs} LUFS against Spotify's {target}", max_w=width)


# ---------------------------------------------------------------- bars (one series)
def hbars(rows, width=420, unit="", label="Bar chart", fill=C1, bar_h=14, gap=10, label_w=130, top=None):
    """rows: [(label, value, tooltip)]. One series: no legend; values printed at the bar ends."""
    if not rows:
        return ""
    top = top or max(v for _, v, _ in rows) or 1
    H = len(rows) * (bar_h + gap)
    room = width - label_w - 56
    parts = []
    for i, (lab, v, tip) in enumerate(rows):
        y = i * (bar_h + gap)
        parts.append(_text(label_w - 8, y + bar_h - 3, lab, size=11, anchor="end", opacity=0.85))
        parts.append(_hbar(label_w, y, room * v / top, bar_h, fill, tip or f"{lab}: {v}{unit}"))
        parts.append(_text(label_w + room * v / top + 6, y + bar_h - 3, f"{v:g}{unit}", size=11, opacity=0.75))
    return _svg(width, H, "".join(parts), label, max_w=width)


# ---------------------------------------------------------------- chords over time
FUNCTION_STYLE = {"home": (C1, "home chord (I)"), "away": (C3, "moving away (IV, ii)"),
                  "tension": (C2, "tension (V, vii)"), "other": ("currentColor", "other / borrowed")}


def chord_function(roman):
    """Roughly what a chord does in the key, from its Roman numeral: home (I, vi, iii), away (IV, ii),
    tension (V, vii) - borrowed chords (a flat or sharp degree) are 'other'."""
    r = (roman or "").strip()
    if not r or r[0] in "b#":
        return "other"
    num = "".join(ch for ch in r.upper() if ch in "IV")
    return {"I": "home", "VI": "home", "III": "home", "IV": "away", "II": "away", "V": "tension",
            "VII": "tension"}.get(num, "other")


def chord_strip(timeline, duration, width=720):
    """The chords along the time axis, coloured by what they do (with a legend), named where they fit."""
    if not timeline or not duration:
        return ""
    H, top, band = 58, 4, 26
    x = lambda t: 4 + (width - 8) * t / duration
    parts = []
    for i, c in enumerate(timeline):
        s = c.get("start", 0.0)
        e = timeline[i + 1]["start"] if i + 1 < len(timeline) else duration
        fn = chord_function(c.get("roman"))
        col, _ = FUNCTION_STYLE[fn]
        w = max(1.0, x(e) - x(s) - 2)  # a 2px surface gap between neighbouring chords
        op = "0.25" if fn == "other" else "0.85"
        unclear = (c.get("clarity") or 9) < 1.0
        parts.append(f'<rect x="{x(s) + 1:.1f}" y="{top}" width="{w:.1f}" height="{band}" rx="2" '
                     f'style="fill:{col};fill-opacity:{"0.35" if unclear else op}">'
                     f'<title>{esc(c["time"])} {esc(c["chord"])} ({esc(c.get("roman", ""))})'
                     f'{" — unclear" if unclear else ""}</title></rect>')
        name = c["chord"] + ("?" if unclear else "")
        if w >= 7 * len(name) + 6:
            parts.append(_text(x(s) + 1 + w / 2, top + band / 2 + 4, name, size=11, anchor="middle", weight=600,
                               extra='style="fill:#fff"' if fn != "other" else ""))
    for tx, lab in _time_ticks(duration, x):
        parts.append(_text(tx, top + band + 13, lab, size=9, anchor="middle", opacity=0.6))
    lx = 4
    for fn in ("home", "away", "tension", "other"):
        col, word = FUNCTION_STYLE[fn]
        parts.append(f'<rect x="{lx}" y="{H - 10}" width="10" height="10" rx="2" '
                     f'style="fill:{col};fill-opacity:{0.25 if fn == "other" else 0.85}"/>')
        parts.append(_text(lx + 14, H - 1, word, size=10, opacity=0.8))
        lx += 22 + 6.2 * len(word)
    return _svg(width, H, "".join(parts), "Chords over time, coloured by what they do in the key", max_w=width)


# ---------------------------------------------------------------- drums
def drum_grid(grid, hats=None, beats_per_bar=4, width=520):
    """The main groove as a picture: one row per drum, one column per 16th note, a dot for each hit."""
    if not grid:
        return ""
    n = len(grid)
    rows = [("Kick", lambda ch: ch in "KX", C1, 7), ("Snare", lambda ch: ch in "SX", C2, 7)]
    if "x" in grid:
        rows.append(("Other hit", lambda ch: ch == "x", "currentColor", 5))
    if hats:
        rows.append(("Hi-hat", None, C3, 5))
    left, top, cell, rh = 70, 18, (width - 74) / n, 22
    H = top + rh * len(rows) + 4
    parts = []
    for b in range(beats_per_bar):
        if b % 2 == 0:
            parts.append(f'<rect x="{left + b * 4 * cell:.1f}" y="{top - 2}" width="{4 * cell:.1f}" '
                         f'height="{rh * len(rows) + 2}" fill="currentColor" fill-opacity="0.04"/>')
    for i in range(n):
        lab = str(i // 4 + 1) if i % 4 == 0 else ("&" if i % 4 == 2 else "")
        parts.append(_text(left + (i + 0.5) * cell, 11, lab, size=10, anchor="middle",
                           weight=700 if i % 4 == 0 else None, opacity=0.8 if i % 4 == 0 else 0.5))
    for r, (name, hit, col, rad) in enumerate(rows):
        y = top + r * rh + rh / 2
        parts.append(_text(left - 8, y + 4, name, size=11, anchor="end", opacity=0.85))
        for i in range(n):
            cxp = left + (i + 0.5) * cell
            on = (hats[i] == "h") if hit is None and hats and i < len(hats) else (hit(grid[i]) if hit else False)
            if on:
                dim = ' fill-opacity="0.6"' if col == "currentColor" else ""
                parts.append(f'<circle cx="{cxp:.1f}" cy="{y:.1f}" r="{rad}" style="fill:{col}"{dim}>'
                             f'<title>{name}, step {i + 1}</title></circle>')
            else:
                parts.append(f'<circle cx="{cxp:.1f}" cy="{y:.1f}" r="1.5" fill="currentColor" fill-opacity="0.2"/>')
    return _svg(width, H, "".join(parts), f"Drum groove: {grid}" + (f" / hats {hats}" if hats else ""), max_w=width)


# ---------------------------------------------------------------- the low part as a piano roll
SHARPS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def piano_roll(notes, duration, note_name=None, width=720, height=130):
    """notes: [{start, end, midi}] - each note a bar at its pitch and time; C notes labelled."""
    notes = [n for n in notes if n.get("midi") is not None]
    note_name = note_name or (lambda m: f"{SHARPS[m % 12]}{m // 12 - 1}")
    if not notes or not duration:
        return ""
    lo, hi = min(n["midi"] for n in notes) - 1, max(n["midi"] for n in notes) + 1
    left, top, bottom = 34, 4, 16
    ph = height - top - bottom
    x = lambda t: left + (width - left - 4) * t / duration
    y = lambda m: top + ph * (1 - (m - lo) / max(1, hi - lo))
    row = ph / max(1, hi - lo)
    parts = []
    used = sorted({n["midi"] for n in notes})
    each = len(used) <= 10 and ph / max(1, hi - lo) >= 9  # room to name every pitch, else just the Cs
    named = set(used) if each else {m for m in range(lo, hi + 1) if m % 12 == 0}
    for m in range(lo, hi + 1):
        if m in named:
            parts.append(f'<line x1="{left}" y1="{y(m):.1f}" x2="{width - 4}" y2="{y(m):.1f}" stroke="currentColor" '
                         f'stroke-opacity="0.12"/>')
            parts.append(_text(left - 4, y(m) + 3, note_name(m) if each else f"C{m // 12 - 1}", size=9,
                               anchor="end", opacity=0.6))
    for n in notes:
        w = max(1.5, x(n["end"]) - x(n["start"]) - 1)
        parts.append(f'<rect x="{x(n["start"]):.1f}" y="{y(n["midi"]) - row / 2 + 0.5:.1f}" width="{w:.1f}" '
                     f'height="{max(2.0, row - 1):.1f}" rx="1.5" style="fill:{C1}">'
                     f'<title>{esc(note_name(n["midi"]))} at {_fmt_time(n["start"])}</title></rect>')
    for tx, lab in _time_ticks(duration, x):
        parts.append(_text(tx, height - 3, lab, size=9, anchor="middle", opacity=0.6))
    return _svg(width, height, "".join(parts), "The low part as a piano roll: pitch against time", max_w=width)


# ---------------------------------------------------------------- dance arrangement and pumping
ROLE_STYLE = {"drop": (C2, 0.9), "build": (C3, 0.85), "breakdown": (C1, 0.85), "intro": ("currentColor", 0.18),
              "outro": ("currentColor", 0.18), "groove": ("currentColor", 0.35)}


def dance_map_svg(m, duration, width=720):
    """The arrangement as role blocks along time, each named and counted in bars where it fits."""
    if not m or not duration:
        return ""
    H, top, band = 52, 4, 30
    x = lambda t: 4 + (width - 8) * t / duration
    parts = []
    for s in m["sections"]:
        col, op = ROLE_STYLE[s["role"]]
        w = max(1.0, x(s["end"]) - x(s["start"]) - 2)
        parts.append(f'<rect x="{x(s["start"]) + 1:.1f}" y="{top}" width="{w:.1f}" height="{band}" rx="2" '
                     f'style="fill:{col};fill-opacity:{op}"><title>{esc(s["role"])} {_fmt_time(s["start"])}-'
                     f'{_fmt_time(s["end"])}, {s["bars"]} bars</title></rect>')
        name = f"{s['role']} · {s['bars']}"
        dark = col != "currentColor"
        if w >= 6.5 * len(name) + 8:
            parts.append(_text(x(s["start"]) + 1 + w / 2, top + band / 2 + 4, name, size=11, anchor="middle",
                               weight=600, extra='style="fill:#fff"' if dark else ""))
        elif w >= 6.5 * len(s["role"]) + 6:
            parts.append(_text(x(s["start"]) + 1 + w / 2, top + band / 2 + 4, s["role"], size=10, anchor="middle",
                               weight=600, extra='style="fill:#fff"' if dark else ""))
    for tx, lab in _time_ticks(duration, x):
        parts.append(_text(tx, H - 4, lab, size=9, anchor="middle", opacity=0.6))
    return _svg(width, H, "".join(parts), "Dance arrangement: " + ", ".join(
        f"{s['role']} {s['bars']} bars" for s in m["sections"]), max_w=width)


def pump_shape(shape, width=200, height=70):
    """One beat's level, averaged over the track: a sidechained pad ducks at the kick and swells back."""
    if not shape:
        return ""
    lo = min(min(shape), -12.0)
    left, top, bottom = 26, 6, 16
    x = lambda i: left + (width - left - 4) * i / (len(shape) - 1)
    y = lambda v: top + (height - top - bottom) * (v / lo)
    d = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for i, v in enumerate(shape))
    area = d + f" L{x(len(shape) - 1):.1f},{y(lo):.1f} L{x(0):.1f},{y(lo):.1f} Z"
    parts = [f'<path d="{area}" style="fill:{C1};fill-opacity:0.06"/>',
             f'<path d="{d}" fill="none" style="stroke:{C1}" stroke-width="2" stroke-linejoin="round"/>',
             _text(left - 4, y(0) + 3, "0", size=9, anchor="end", opacity=0.6),
             _text(left - 4, y(lo) + 3, f"{lo:.0f}", size=9, anchor="end", opacity=0.6),
             _text(left, height - 3, "kick", size=9, opacity=0.6),
             _text(width - 4, height - 3, "next kick", size=9, anchor="end", opacity=0.6)]
    return _svg(width, height, "".join(parts), "One beat's level (dB), averaged over the track", max_w=width)


# ---------------------------------------------------------------- comparison: yours against theirs, per measure
def dumbbell(rows, names, width=560):
    """rows: [(measure, unit, [value per track: yours first])]. Each row gets its own scale (the measures differ);
    yours is the ink dot, each song you admire a coloured ring-dot; a line spans the gap. Legend on top."""
    rows = [r for r in rows if sum(v is not None for v in r[2]) >= 2]
    if not rows:
        return ""
    left, right, top, rh = 150, 60, 24, 28
    H = top + rh * len(rows)
    span = width - left - right
    parts, lx = [], left
    for i, nm in enumerate(names):
        col = "currentColor" if i == 0 else SERIES[(i - 1) % 3]
        parts.append(f'<circle cx="{lx + 5}" cy="8" r="5" style="fill:{col}"/>')
        word = "yours" if i == 0 else nm[:24]
        parts.append(_text(lx + 14, 12, word, size=10, opacity=0.85))
        lx += 26 + 6.0 * len(word)
    for k, (measure, unit, vals) in enumerate(rows):
        y = top + k * rh + rh / 2
        have = [v for v in vals if v is not None]
        lo, hi = min(have), max(have)
        pad = (hi - lo) * 0.15 or max(1.0, abs(hi) * 0.05)
        lo, hi = lo - pad, hi + pad
        x = lambda v: left + span * (v - lo) / (hi - lo)
        parts.append(_text(left - 10, y + 4, measure, size=11, anchor="end", opacity=0.85))
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + span}" y2="{y}" stroke="currentColor" '
                     f'stroke-opacity="0.1"/>')
        parts.append(f'<line x1="{x(lo + pad):.1f}" y1="{y}" x2="{x(hi - pad):.1f}" y2="{y}" stroke="currentColor" '
                     f'stroke-opacity="0.35" stroke-width="2"/>')
        for i, v in sorted(enumerate(vals), key=lambda iv: iv[0] == 0):  # yours drawn last, on top
            if v is None:
                continue
            col = "currentColor" if i == 0 else SERIES[(i - 1) % 3]
            who = "yours" if i == 0 else names[i]
            tip = f"<title>{esc(who)}: {v:g}{esc(unit)}</title>"
            if i and vals[0] is not None and abs(x(v) - x(vals[0])) < 6:  # same as yours: a ring around your dot
                parts.append(f'<circle cx="{x(v):.1f}" cy="{y}" r="{8 + 3 * (i - 1)}" fill="none" '
                             f'style="stroke:{col}" stroke-width="2">{tip}</circle>')
            else:
                parts.append(f'<circle cx="{x(v):.1f}" cy="{y}" r="{6 if i == 0 else 5.5}" style="fill:{col}" '
                             f'stroke="var(--surface, #fff)" stroke-width="2">{tip}</circle>')
        if vals[0] is not None:
            parts.append(_text(width - right + 8, y + 4, f"{vals[0]:g}{unit}", size=10, opacity=0.75))
    return _svg(width, H, "".join(parts), "Yours against the songs you admire, measure by measure", max_w=width)


# ---------------------------------------------------------------- small things for lists
def sparkline(values, width=120, height=28, label="Energy over time"):
    """A tiny area of the loudness over a track (0-1 values)."""
    if not values or len(values) < 2:
        return ""
    x = lambda i: 1 + (width - 2) * i / (len(values) - 1)
    y = lambda v: 2 + (height - 4) * (1 - v)
    d = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    return _svg(width, height, f'<path d="{d} L{x(len(values) - 1):.1f},{height} L1,{height} Z" '
                f'style="fill:{C1};fill-opacity:0.12"/><path d="{d}" fill="none" style="stroke:{C1}" '
                f'stroke-width="1.5" stroke-linejoin="round"/>', label, cls="spark", max_w=width)


def mini_meter(value, top=10, width=64, height=8):
    """A small horizontal meter for a table cell (e.g. energy 7/10)."""
    w = (width - 2) * max(0, min(value, top)) / top
    return (f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
            f'aria-label="{value} of {top}" style="vertical-align:middle;margin-right:6px">'
            f'<rect x="0" y="0" width="{width}" height="{height}" rx="4" fill="currentColor" fill-opacity="0.1"/>'
            f'<rect x="0" y="0" width="{w:.1f}" height="{height}" rx="4" style="fill:{C1}"/></svg>')


def tempo_dots(tempos, width=300, lo=40, hi=200):
    """Every track's tempo as a dot on the ruler (stacked where they meet): where your music usually sits."""
    if not tempos:
        return ""
    pad, base = 10, 50
    x = lambda b: pad + (width - 2 * pad) * (min(hi, max(lo, b)) - lo) / (hi - lo)
    seen, parts = {}, [f'<line x1="{pad}" y1="{base}" x2="{width - pad}" y2="{base}" stroke="currentColor" '
                       f'stroke-opacity="0.35"/>']
    for t in sorted(tempos):
        slot = round(x(t) / 9)
        k = seen.get(slot, 0)
        seen[slot] = k + 1
        parts.append(f'<circle cx="{x(t):.1f}" cy="{base - 7 - 9 * min(k, 4)}" r="4" style="fill:{C1}" '
                     f'stroke="var(--surface, #fff)" stroke-width="1.5"><title>{t:.0f} BPM</title></circle>')
    for t in range(60, hi + 1, 30):
        parts.append(_text(x(t), base + 14, t, size=9, anchor="middle", opacity=0.6))
    return _svg(width, base + 18, "".join(parts), "Tempos of your tracks", max_w=width)


# ---------------------------------------------------------------- the home page's example pictures
def example_roll(width=260, height=80):
    """An illustration: a short bass line as a piano roll (made-up notes)."""
    line = [40, 40, 47, 45, 40, 43, 45, 47, 40, 40, 52, 50, 47, 45, 43, 42]
    return piano_roll([{"start": i * 0.5, "end": i * 0.5 + 0.4, "midi": m} for i, m in enumerate(line)], 8,
                      width=width, height=height)


def example_overlay(width=520, height=150):
    """An illustration for the home page: your track's energy (solid) against a song you admire (dashed).
    Made-up curves, labelled as an example."""
    def curve(points):
        return [(i / (len(points) - 1), v) for i, v in enumerate(points)]
    yours = curve([.2, .25, .3, .55, .6, .6, .62, .58, .6, .65, .6, .55, .5, .3, .2])
    theirs = curve([.15, .2, .3, .45, .9, .92, .9, .35, .3, .6, .95, .95, .92, .4, .15])
    pad, top = 6, 26
    ph = height - top - 18
    xy = lambda p: (pad + (width - 2 * pad) * p[0], top + ph * (1 - p[1]))
    path = lambda pts: " ".join(f"{'M' if i == 0 else 'L'}{xy(p)[0]:.1f},{xy(p)[1]:.1f}" for i, p in enumerate(pts))
    parts = [f'<path d="{path(theirs)}" fill="none" style="stroke:{C1}" stroke-width="2" stroke-dasharray="6 4"/>',
             f'<path d="{path(yours)}" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linejoin="round"/>',
             f'<line x1="{pad}" y1="{top + ph}" x2="{width - pad}" y2="{top + ph}" stroke="currentColor" stroke-opacity="0.2"/>',
             f'<line x1="{pad}" y1="9" x2="{pad + 18}" y2="9" stroke="currentColor" stroke-width="2.5"/>',
             _text(pad + 24, 13, "your track", size=11, opacity=0.85),
             f'<line x1="{pad + 110}" y1="9" x2="{pad + 128}" y2="9" style="stroke:{C1}" stroke-width="2" '
             f'stroke-dasharray="6 4"/>',
             _text(pad + 134, 13, "a song you admire", size=11, opacity=0.85),
             _text(xy(theirs[4])[0], xy(theirs[4])[1] - 6, "their drop", size=10, anchor="middle", opacity=0.75),
             _text(width - pad, height - 3, "example · energy over time", size=9, anchor="end", opacity=0.55)]
    return _svg(width, height, "".join(parts), "Example: your track's energy against a song you admire",
                max_w=width)


# ---------------------------------------------------------------- over time (the history page)
def _date_ticks(t0, t1, x, n=5):
    from datetime import datetime
    span = max(1.0, t1 - t0)
    fmt = "%d %b" if span < 200 * 86400 else "%b %Y"
    return [(x(t0 + span * k / (n - 1)), datetime.fromtimestamp(t0 + span * k / (n - 1)).strftime(fmt)) for k in range(n)]


def _running_median(points, k=5):
    """The median of each point with its neighbours (k at a time): the trend without one odd track pulling it."""
    vals = [v for _, v, _ in points]
    out = []
    for i in range(len(points)):
        win = sorted(vals[max(0, i - k // 2):i + k // 2 + 1])
        out.append((points[i][0], win[len(win) // 2]))
    return out


def time_series(points, t0, t1, unit="", label="Over time", lo=None, hi=None, width=460, height=170, decimals=0):
    """Each track as a dot at the date it was saved; a line through the running median shows the trend.
    One series: no legend (the title names it)."""
    if not points:
        return ""
    points = sorted(points)
    vals = [v for _, v, _ in points]
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    left, right, top, bottom = 40, 10, 8, 22
    t1 = max(t1, points[-1][0])
    t0 = min(t0, points[0][0])
    x = lambda t: left + (width - left - right) * (t - t0) / max(1.0, t1 - t0)
    y = lambda v: top + (height - top - bottom) * (1 - (v - lo) / (hi - lo))
    parts = []
    for k in range(3):
        v = lo + pad + (hi - lo - 2 * pad) * k / 2
        parts.append(f'<line x1="{left}" y1="{y(v):.1f}" x2="{width - right}" y2="{y(v):.1f}" stroke="currentColor" '
                     f'stroke-opacity="0.1"/>')
        parts.append(_text(left - 6, y(v) + 3, f"{v:.{decimals}f}", size=9, anchor="end", opacity=0.6))
    if len(points) >= 3:
        trend = _running_median(points)
        d = " ".join(f"{'M' if i == 0 else 'L'}{x(t):.1f},{y(v):.1f}" for i, (t, v) in enumerate(trend))
        parts.append(f'<path d="{d}" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"><title>trend (running median)</title></path>')
    for t, v, tip in points:
        parts.append(f'<circle cx="{x(t):.1f}" cy="{y(v):.1f}" r="4.5" style="fill:{C1}" stroke="var(--surface, #fff)" '
                     f'stroke-width="2"><title>{esc(tip)}</title></circle>')
    for tx, lab in _date_ticks(t0, t1, x):
        parts.append(_text(tx, height - 4, lab, size=9, anchor="middle", opacity=0.6))
    return _svg(width, height, "".join(parts), label, max_w=width)


def activity_columns(buckets, width=460, height=150, label="Tracks analysed over time"):
    """Tracks analysed per period, as columns (empty periods stay visible as gaps)."""
    if not buckets:
        return ""
    left, right, top, bottom = 26, 6, 14, 22
    top_n = max(1, max(b[2] for b in buckets))
    n = len(buckets)
    slot = (width - left - right) / n
    bw = min(24.0, max(2.0, slot - 2))
    ph = height - top - bottom
    parts = [f'<line x1="{left}" y1="{top + ph}" x2="{width - right}" y2="{top + ph}" stroke="currentColor" '
             f'stroke-opacity="0.3"/>', _text(left - 6, top + 4, top_n, size=9, anchor="end", opacity=0.6),
             _text(left - 6, top + ph + 3, 0, size=9, anchor="end", opacity=0.6)]
    for i, (_, lab, c, *_) in enumerate(buckets):
        if not c:
            continue
        h = ph * c / top_n
        cx = left + slot * i + (slot - bw) / 2
        r = min(4.0, bw / 2, h)
        d = (f"M{cx:.1f},{top + ph:.1f} v{-(h - r):.1f} a{r},{r} 0 0 1 {r},{-r} h{bw - 2 * r:.1f} "
             f"a{r},{r} 0 0 1 {r},{r} v{h - r:.1f} z")
        parts.append(f'<path d="{d}" style="fill:{C1}"><title>{esc(lab)}: {c} track{"s" if c != 1 else ""}</title></path>')
        if bw >= 14:
            parts.append(_text(cx + bw / 2, top + ph - h - 3, c, size=9, anchor="middle", opacity=0.75))
    step = max(1, n // 5)
    for i in range(0, n, step):
        short = buckets[i][3] if len(buckets[i]) > 3 else buckets[i][1]
        parts.append(_text(left + slot * (i + 0.5), height - 4, short, size=9, anchor="middle", opacity=0.6))
    return _svg(width, height, "".join(parts), label, max_w=width)


def key_timeline(points, t0, t1, width=460, height=170):
    """Each track's key over time: up the side, its place on the Camelot wheel (1-12; neighbours are a step apart),
    minor (A) and major (B) in two colours with a legend."""
    if not points:
        return ""
    left, right, top, bottom = 30, 10, 22, 22
    t1 = max(t1, max(p[0] for p in points))
    t0 = min(t0, min(p[0] for p in points))
    x = lambda t: left + (width - left - right) * (t - t0) / max(1.0, t1 - t0)
    y = lambda n: top + (height - top - bottom) * (n - 1) / 11
    parts = []
    for n in (1, 4, 7, 10, 12):
        parts.append(f'<line x1="{left}" y1="{y(n):.1f}" x2="{width - right}" y2="{y(n):.1f}" stroke="currentColor" '
                     f'stroke-opacity="0.08"/>')
        parts.append(_text(left - 6, y(n) + 3, n, size=9, anchor="end", opacity=0.6))
    cols = {"A": C2, "B": C1}
    for t, n, letter, tip in sorted(points):
        parts.append(f'<circle cx="{x(t):.1f}" cy="{y(n):.1f}" r="4.5" style="fill:{cols[letter]}" '
                     f'stroke="var(--surface, #fff)" stroke-width="2"><title>{esc(tip)}</title></circle>')
    for i, (letter, word) in enumerate((("A", "minor (A)"), ("B", "major (B)"))):
        parts.append(f'<circle cx="{left + 6 + 100 * i}" cy="8" r="4.5" style="fill:{cols[letter]}"/>')
        parts.append(_text(left + 15 + 100 * i, 12, word, size=10, opacity=0.85))
    for tx, lab in _date_ticks(t0, t1, x):
        parts.append(_text(tx, height - 4, lab, size=9, anchor="middle", opacity=0.6))
    return _svg(width, height, "".join(parts), "Keys over time on the Camelot wheel", max_w=width)


# ---------------------------------------------------------------- scale degrees: what the chords and melody use
def degree_chart(rows, melody_source=None, width=520, height=190):
    """For each degree of the scale, two columns: the share of chord time built on it (blue) and its share of the
    melody (orange, when we have a melody). The mode's colour note is marked. Unused degrees show as an empty slot,
    so what's missing is visible at a glance."""
    if not rows:
        return ""
    has_mel = any(r_["melody"] is not None for r_ in rows)
    left, right, top, bottom = 34, 6, 26, 40
    ph = height - top - bottom
    top_v = max(0.25, max(max(r_["harmony"], r_["melody"] or 0) for r_ in rows))
    slot = (width - left - right) / len(rows)
    bw = min(22.0, (slot - 10) / (2 if has_mel else 1))
    y = lambda v: top + ph * (1 - v / top_v)
    parts = []
    for k in (0, 0.5, 1):
        v = top_v * k
        parts.append(f'<line x1="{left}" y1="{y(v):.1f}" x2="{width - right}" y2="{y(v):.1f}" stroke="currentColor" '
                     f'stroke-opacity="{0.3 if k == 0 else 0.08}"/>')
        parts.append(_text(left - 6, y(v) + 3, f"{v:.0%}", size=9, anchor="end", opacity=0.6))
    mel_word = {"typed": "in your typed melody", "top line": "in the top line"}.get(melody_source, "in the melody")
    series = [("harmony", C1, "chords built on it")] + ([("melody", C2, mel_word)]
                                                         if has_mel else [])
    lx = left
    for _, col, word in series:
        parts.append(f'<rect x="{lx}" y="4" width="10" height="10" rx="2" style="fill:{col}"/>')
        parts.append(_text(lx + 14, 13, word, size=10, opacity=0.85))
        lx += 30 + 6.2 * len(word)
    for i, r_ in enumerate(rows):
        x0 = left + slot * i + (slot - bw * len(series) - 2 * (len(series) - 1)) / 2
        if r_["colour"]:
            parts.append(f'<rect x="{left + slot * i + 2:.1f}" y="{top - 4}" width="{slot - 4:.1f}" height="{ph + 4}" '
                         f'rx="3" style="fill:{ACCENT}" fill-opacity="0.08"/>')
        for j, (key, col, word) in enumerate(series):
            v = r_[key] or 0.0
            bx = x0 + j * (bw + 2)
            if v <= 0:
                parts.append(f'<line x1="{bx:.1f}" y1="{top + ph - 1:.1f}" x2="{bx + bw:.1f}" y2="{top + ph - 1:.1f}" '
                             f'style="stroke:{col}" stroke-width="2" stroke-dasharray="3 2">'
                             f'<title>{esc(r_["degree"])} ({esc(r_["note"])}): not {esc(word)}</title></line>')
                continue
            h = ph * v / top_v
            rr = min(4.0, bw / 2, h)
            d = (f"M{bx:.1f},{top + ph:.1f} v{-(h - rr):.1f} a{rr},{rr} 0 0 1 {rr},{-rr} h{bw - 2 * rr:.1f} "
                 f"a{rr},{rr} 0 0 1 {rr},{rr} v{h - rr:.1f} z")
            parts.append(f'<path d="{d}" style="fill:{col}"><title>{esc(r_["degree"])} ({esc(r_["note"])}): '
                         f'{v:.0%} {esc(word)}</title></path>')
        cx = left + slot * (i + 0.5)
        parts.append(_text(cx, top + ph + 14, r_["degree"], size=11, anchor="middle", weight=700))
        parts.append(_text(cx, top + ph + 26, r_["note"], size=10, anchor="middle", opacity=0.7))
        if r_["colour"]:
            parts.append(_text(cx, top + ph + 38, "★ colour", size=9, anchor="middle", weight=700))
    return _svg(width, height, "".join(parts), "Scale degrees used by the chords and the melody", max_w=width)
