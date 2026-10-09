"""Sheet music from transcribed notes, for musicanalyze.py.

Notes found at real times are placed on a 16th-note grid that follows the detected beats (so tempo
drift is followed), split into bars of the detected meter, written as MusicXML (opens in MuseScore,
Sibelius, Dorico, Finale and most DAWs), engraved to SVG with Verovio when it's installed, and turned
into a grid-aligned MIDI file. The rhythm analysis is computed from the same notation.
"""

import html
from collections import Counter

import numpy as np

SLOTS_PER_BEAT = 4  # 16th notes
# Notatable lengths in 16ths, longest first: whole, dotted half, half, dotted quarter, quarter,
# dotted eighth, eighth, 16th.
VALUES = [(16, "whole", 0), (12, "half", 1), (8, "half", 0), (6, "quarter", 1), (4, "quarter", 0),
          (3, "eighth", 1), (2, "eighth", 0), (1, "16th", 0)]
KIND = {"": "major", "m": "minor", "7": "dominant", "m7": "minor-seventh", "maj7": "major-seventh",
        "5": "power", "dim": "diminished", "sus2": "suspended-second", "sus4": "suspended-fourth"}
STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def time_to_slot(t, beat_times):
    """Seconds -> position in 16ths along the detected beats (extrapolating past the ends)."""
    bt = np.asarray(beat_times, dtype=float)
    period = float(np.median(np.diff(bt))) if len(bt) > 1 else 0.5
    if t <= bt[0]:
        beat = (t - bt[0]) / period
    elif t >= bt[-1]:
        beat = len(bt) - 1 + (t - bt[-1]) / period
    else:
        beat = float(np.interp(t, bt, np.arange(len(bt))))
    return beat * SLOTS_PER_BEAT


def slot_to_time(slot, beat_times):
    bt = np.asarray(beat_times, dtype=float)
    period = float(np.median(np.diff(bt))) if len(bt) > 1 else 0.5
    beat = slot / SLOTS_PER_BEAT
    if beat <= 0:
        return float(bt[0] + beat * period)
    if beat >= len(bt) - 1:
        return float(bt[-1] + (beat - len(bt) + 1) * period)
    return float(np.interp(beat, np.arange(len(bt)), bt))


NICE = [1, 2, 3, 4, 6, 8, 12, 16]  # note lengths a bassist would write, in 16ths


def quantize(notes, chords, beat_times, beats_per_bar, downbeat_time, duration=None):
    """notes: [(start s, end s, midi, velocity)]; chords: [(start s, name)]; downbeat_time: a beat that
    starts a bar. Returns notes as (slot, length, midi, velocity) and chords as (slot, name), counted in
    16ths from the start of bar 1."""
    bar_slots = beats_per_bar * SLOTS_PER_BEAT
    # Beat trackers place beats a little late compared with attacks (and players push or lay back), so
    # find the attacks' average offset from the 8th-note grid and remove it before rounding.
    starts = np.array([time_to_slot(a, beat_times) for a, _, _, _ in notes])
    lag = 0.0
    if len(starts) >= 8:
        lag = float(np.angle(np.mean(np.exp(1j * np.pi * starts))) / np.pi)  # in 16ths, within +-1
        lag = float(np.clip(lag, -0.5, 0.5))
    raw = []
    for a, e, n, v in notes:
        s = int(round(time_to_slot(a, beat_times) - lag))
        f = int(round(time_to_slot(e, beat_times) - lag))
        raw.append([s, max(1, f - s), n, v])
    raw.sort()
    clean = []
    for s, length, n, v in raw:  # monophonic: one note at a time, the first attack on a slot wins
        if clean and clean[-1][0] == s:
            continue
        clean.append([s, length, n, v])
    for i, c in enumerate(clean):
        gap = clean[i + 1][0] - c[0] if i + 1 < len(clean) else None
        length = c[1]
        if gap is not None and gap - length <= 1:  # a hair short of the next note: it's legato
            length = gap
        fits = [x for x in NICE if gap is None or x <= gap]
        c[1] = min(fits, key=lambda x: (abs(x - length), -x)) if fits else gap
    # Bar 1 starts on the downbeat at or before the first event (notes, or chords if they come first).
    events = [c[0] for c in clean] + [int(round(time_to_slot(t, beat_times))) for t, _ in chords]
    first = min(events) if events else 0
    down = int(round(time_to_slot(downbeat_time, beat_times)))
    origin = down + ((first - down) // bar_slots) * bar_slots
    notes_q = [(s - origin, length, n, v) for s, length, n, v in clean if length > 0]
    chords_q = []
    for t, name in chords:
        slot = max(0, int(round(time_to_slot(t, beat_times))) - origin)
        if chords_q and chords_q[-1][1] == name:
            continue
        if chords_q and chords_q[-1][0] == slot:
            chords_q[-1] = (slot, name)
        else:
            chords_q.append((slot, name))
    end = max([s + length for s, length, _, _ in notes_q] + [0])
    if duration:
        end = max(end, int(round(time_to_slot(duration, beat_times))) - origin)
    bars = max(1, -(-end // bar_slots))
    chords_q = [(s, name) for s, name in chords_q if s < bars * bar_slots]
    return {"bar_slots": bar_slots, "beats_per_bar": beats_per_bar, "bars": bars, "notes": notes_q,
            "chords": chords_q, "bar1_time": round(slot_to_time(origin, beat_times), 2)}


def _split(length, pos, bar_slots):
    """Break a length (in 16ths) at bar lines and into notatable values -> [(len, type, dots)]."""
    out = []
    while length > 0:
        room = bar_slots - pos % bar_slots
        chunk = min(length, room)
        while chunk > 0:
            for v, typ, dots in VALUES:
                if v <= chunk and (v < 8 or pos % 4 == 0):  # half/whole notes start on a beat
                    out.append((v, typ, dots))
                    chunk -= v
                    length -= v
                    pos += v
                    break
    return out


def _pitch_xml(midi, spell):
    name = spell(midi % 12)
    step, acc = name[0], name[1:]
    alter = acc.count("#") - acc.count("b")
    octave = (midi - (STEP_PC[step] + alter)) // 12 - 1
    return (f"<pitch><step>{step}</step>" + (f"<alter>{alter}</alter>" if alter else "") +
            f"<octave>{octave}</octave></pitch>")


def _harmony_xml(name, parse_chord, spell):
    p = parse_chord(name)
    if not p:
        return ""
    root, q = spell(p[0]), p[1]
    alter = root[1:].count("#") - root[1:].count("b")
    kind = KIND.get(q, "major")
    return (f"<harmony><root><root-step>{root[0]}</root-step>" + (f"<root-alter>{alter}</root-alter>" if alter else "") +
            f"</root><kind text=\"{html.escape(q)}\">{kind}</kind></harmony>")


def to_musicxml(score, title, part_name, key_fifths, bpm, spell, parse_chord, clef=("F", 4), octave_up=True):
    """octave_up: written an octave above how it sounds, the standard for bass guitar."""
    bar_slots, bpb = score["bar_slots"], score["beats_per_bar"]
    written_up = 12 if octave_up else 0
    starts = {s: (length, n) for s, length, n, _ in score["notes"]}
    chords_at = dict(score["chords"])
    timeline = sorted(set([s for s, _, _, _ in score["notes"]]))
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">',
           '<score-partwise version="4.0">',
           f"<work><work-title>{html.escape(title)}</work-title></work>",
           '<identification><encoding><software>Music Inspiration Analyzer</software></encoding></identification>',
           f'<part-list><score-part id="P1"><part-name>{html.escape(part_name)}</part-name></score-part></part-list>',
           '<part id="P1">']
    # Build one event list over the whole piece: notes and the rests between them.
    events, pos = [], 0
    for s in timeline:
        if s > pos:
            events.append((pos, s - pos, None))
        length, n = starts[s]
        events.append((s, length, n))
        pos = s + length
    total = score["bars"] * bar_slots
    if pos < total:
        events.append((pos, total - pos, None))
    # Rests are split where a chord changes, so every chord symbol has a note or rest to sit on.
    cuts = sorted(chords_at)
    split = []
    for s, length, n in events:
        if n is None:
            inner = [c for c in cuts if s < c < s + length]
            for a, b in zip([s] + inner, inner + [s + length]):
                split.append((a, b - a, None))
        else:
            split.append((s, length, n))
    events = split
    measures = [[] for _ in range(score["bars"])]
    for s, length, n in events:
        pieces = _split(length, s, bar_slots)
        p = s
        for i, (v, typ, dots) in enumerate(pieces):
            measures[p // bar_slots].append((p % bar_slots, v, typ, dots, n, i > 0, i < len(pieces) - 1))
            p += v
    for m_i, items in enumerate(measures):
        out.append(f'<measure number="{m_i + 1}">')
        if m_i == 0:
            out.append(f"<attributes><divisions>{SLOTS_PER_BEAT}</divisions><key><fifths>{key_fifths}</fifths></key>"
                       f"<time><beats>{bpb}</beats><beat-type>4</beat-type></time>"
                       f"<clef><sign>{clef[0]}</sign><line>{clef[1]}</line></clef>"
                       + ("<transpose><diatonic>0</diatonic><chromatic>0</chromatic><octave-change>-1</octave-change>"
                          "</transpose>" if octave_up else "") + "</attributes>")
            out.append(f'<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                       f"<per-minute>{round(bpm or 90)}</per-minute></metronome></direction-type>"
                       f'<sound tempo="{round(bpm or 90)}"/></direction>')
        bar_start = m_i * bar_slots
        in_bar = [slot for slot in chords_at if bar_start <= slot < bar_start + bar_slots]
        if all(n is None for *_, n, _, _ in items) and in_bar in ([], [bar_start]):
            if in_bar:
                out.append(_harmony_xml(chords_at[bar_start], parse_chord, spell))
            out.append(f'<note><rest measure="yes"/><duration>{bar_slots}</duration></note></measure>')
            continue
        beams = _beams(items)
        for i, (off, v, typ, dots, n, tie_stop, tie_start) in enumerate(items):
            for slot, name in chords_at.items():  # chord symbols at or just before this event
                if bar_start + off <= slot < bar_start + off + v:
                    out.append(_harmony_xml(name, parse_chord, spell))
            note = "<note>" + ("<rest/>" if n is None else _pitch_xml(n + written_up, spell)) + \
                f"<duration>{v}</duration>"
            if n is not None and tie_stop:
                note += '<tie type="stop"/>'
            if n is not None and tie_start:
                note += '<tie type="start"/>'
            note += f"<type>{typ}</type>" + "<dot/>" * dots
            note += "".join(f'<beam number="{lvl}">{kind}</beam>' for lvl, kind in beams.get(i, []))
            if n is not None and (tie_stop or tie_start):
                note += "<notations>" + ('<tied type="stop"/>' if tie_stop else "") + \
                        ('<tied type="start"/>' if tie_start else "") + "</notations>"
            out.append(note + "</note>")
        out.append("</measure>")
    out.append("</part></score-partwise>")
    return "\n".join(out)


def _beams(items):
    """Beam 8ths and 16ths that share a beat (rests break a beam). Returns {item index: [(level, kind)]}."""
    out, run = {}, []

    def close():
        if len(run) >= 2:
            for j, i in enumerate(run):
                out[i] = [(1, "begin" if j == 0 else "end" if j == len(run) - 1 else "continue")]
            for j, i in enumerate(run):  # second beam joins neighbouring 16ths; a lone 16th gets a hook
                if items[i][2] != "16th":
                    continue
                prev = j > 0 and items[run[j - 1]][2] == "16th"
                nxt = j < len(run) - 1 and items[run[j + 1]][2] == "16th"
                kind = ("continue" if prev and nxt else "end" if prev else "begin" if nxt else
                        "forward hook" if j == 0 else "backward hook")
                out[i].append((2, kind))
        run.clear()

    beat = None
    for i, (off, v, typ, dots, n, _, _) in enumerate(items):
        b = off // SLOTS_PER_BEAT
        if n is None or typ not in ("eighth", "16th") or (off + v - 1) // SLOTS_PER_BEAT != b or b != beat:
            close()
        if n is not None and typ in ("eighth", "16th") and (off + v - 1) // SLOTS_PER_BEAT == b:
            run.append(i)
        beat = b
    close()
    return out


def render_svg(musicxml):
    """Engrave MusicXML to SVG pages with Verovio. Returns [] if Verovio isn't installed."""
    try:
        import verovio
    except ImportError:
        return []
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 40, "adjustPageHeight": True,
                   "footer": "none", "header": "auto", "breaks": "auto"})
    if not tk.loadData(musicxml):
        return []
    return [tk.renderToSVG(i + 1) for i in range(tk.getPageCount())]


def sheet_html(title, pages, notes_html=""):
    """One printable page holding every engraved SVG page (print it to PDF from the browser)."""
    body = "\n".join(f'<div class="page">{svg}</div>' for svg in pages)
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<style>body{margin:0;background:#fff;color:#111;font-family:system-ui,sans-serif}"
            ".top{padding:12px 16px}.page svg{width:100%;height:auto;display:block}"
            "@media print{.top{display:none}.page{page-break-after:always}}</style></head><body>"
            f"<div class='top'><b>{html.escape(title)}</b>{notes_html}</div>{body}</body></html>")


def rhythm_from_score(score):
    """Rhythm analysis computed from the notation."""
    bar_slots = score["bar_slots"]
    notes = score["notes"]
    if not notes:
        return {}
    lengths = Counter(length for _, length, _, _ in notes)
    name = {1: "16ths", 2: "8ths", 3: "dotted 8ths", 4: "quarters", 6: "dotted quarters", 8: "halves",
            12: "dotted halves", 16: "whole notes"}
    values = [(name.get(k, f"{k}/16"), round(100 * c / len(notes))) for k, c in lengths.most_common(4)]
    off_beat = [s for s, _, _, _ in notes if s % SLOTS_PER_BEAT]
    on_16 = [s for s in off_beat if s % 2]
    across = [s for s, length, _, _ in notes if s % SLOTS_PER_BEAT and (s % SLOTS_PER_BEAT) + length > SLOTS_PER_BEAT]
    per_bar = Counter(s // bar_slots for s, _, _, _ in notes)
    bars_with_notes = [b for b in range(score["bars"]) if per_bar[b]]
    patterns = Counter(tuple(sorted(s % bar_slots for s, _, _, _ in notes if s // bar_slots == b)) for b in bars_with_notes)
    top, count = patterns.most_common(1)[0] if patterns else ((), 0)
    grid = "".join("x" if i in top else "." for i in range(bar_slots))
    # Live playing rarely repeats a bar exactly, so also: how often a note starts on each 16th of the bar.
    hits = Counter(s % bar_slots for s, _, _, _ in notes)
    share = [hits[i] / max(1, len(bars_with_notes)) for i in range(bar_slots)]
    profile = "".join("X" if p >= 0.5 else "x" if p >= 0.25 else "." for p in share)
    return {
        "notes": len(notes),
        "note_values": values,
        "notes_per_bar": round(len(notes) / max(1, len(bars_with_notes)), 1),
        "offbeat_pct": round(100 * len(off_beat) / len(notes)),
        "sixteenth_offbeat_pct": round(100 * len(on_16) / len(notes)),
        "syncopated_pct": round(100 * len(across) / len(notes)),
        "main_bar_rhythm": grid,  # one character per 16th: x = a note starts here
        "main_bar_rhythm_pct": round(100 * count / max(1, len(bars_with_notes))),
        "distinct_bar_rhythms": len(patterns),
        "usual_positions": profile,  # X = a note starts here in >= 50% of bars, x = 25-50%
        "position_pct": [round(100 * p) for p in share],
    }
