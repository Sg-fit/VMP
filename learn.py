"""The educational view of an analysis: the patterns the analysis found in a track, each turned into a lesson.

Nothing here measures anything new. Every pattern comes from a field the analysis already wrote (the blues form,
the AABA match, the famous progressions, the mode, the meter, the drum traits, the chords, the riffs), so the Learn
view and the professional report always agree. Each pattern gets: what we saw in YOUR track (with its times and
chords), how sure the analysis is, the library entry (lexicon.py) with well-known examples and YouTube searches,
and something to try. Where the analysis wasn't sure, the lesson says so and asks the student to check by ear.
"""

import charts
import lexicon
import musicanalyze as ma


def _t(sec):
    return ma.fmt_time(sec)


def _numerals(names, key):
    out = []
    for n in names:
        p = ma.parse_chord(n)
        out.append(ma.roman(p[0], p[1], key) if p else "?")
    return out


def _card(name, seen, how_sure=None, visual=None, main=True):
    e = lexicon.find(name)
    if not e:
        raise KeyError(f"no library entry for {name!r}")
    return {"entry": e, "seen": [s for s in seen if s], "how_sure": how_sure, "visual": visual, "main": main}


def _blues_grid(b, key):
    """The form as a row of bars: the numeral and the chord in this key, three lines of four for a 12-bar."""
    tonic = ma.key_tonic(key)
    names = {0: "I", 5: "IV", 7: "V", 2: "ii", 8: "♭VI"}
    cells = []
    for i, step in enumerate(b["template"]):
        cells.append(f'<div class="bar"><span class="n">{i + 1}</span><b>{names.get(step, "?")}</b>'
                     f'<span>{ma.spell((tonic + step) % 12, key)}</span></div>')
    per = 4 if b["bars"] % 4 == 0 else b["bars"]
    rows = ["".join(cells[i:i + per]) for i in range(0, len(cells), per)]
    return '<div class="formgrid">' + "".join(f'<div class="line">{r}</div>' for r in rows) + "</div>"


def _chord_row(names, numerals):
    return '<div class="chordrow">' + "".join(f'<div><b>{n}</b><span>{r}</span></div>'
                                              for n, r in zip(names, numerals)) + "</div>"


def _scale_row(notes, colour):
    return '<div class="chordrow scale">' + "".join(
        f'<div class="{"hl" if n == colour else ""}"><b>{n}</b><span>{i + 1}</span></div>'
        for i, n in enumerate(notes)) + "</div>"


def patterns(r):
    """[card] for one analysed track, most worth learning first. A card is main (a pattern to study) or a basic
    (a building block that's there too)."""
    if not r or "error" in r:
        return []
    key = r["key_index"]
    cards = []
    # ---- form
    b = r.get("blues_form")
    if b:
        starts = ", ".join(_t(x) for x in b.get("choruses", [])[:8])
        quick = b["template"][1] == 5
        base, _, variant = b["name"].partition(" (")
        cards.append(_card(base, [
            f"Your track follows the {base} form" + (f" ({variant}" if variant else "")
            + f": {len(b.get('choruses', []))} choruses of about {b['chorus_sec']:.0f} seconds ({b['bar_sec']:.1f} s a bar), starting at {starts}.",
            f"In {r['key']} the three chords are {ma.spell(ma.key_tonic(key), key)} (I), "
            f"{ma.spell((ma.key_tonic(key) + 5) % 12, key)} (IV) and {ma.spell((ma.key_tonic(key) + 7) % 12, key)} (V).",
            "It visits IV in bar 2 and comes straight back: that's the 'quick change'." if quick else None,
            "Listen from the start of any chorus and count the bars on your fingers: the change to IV comes on bar "
            + ("2." if quick else "5."),
        ], how_sure=f"the chords match the form in {b['match_pct']}% of the bars", visual=_blues_grid(b, key)))
    sf = r.get("song_form")
    if sf:
        starts = sf.get("starts", [])
        cards.append(_card("AABA", [
            f"We found four sections of about {sf['section_sec']:.0f} seconds starting at "
            + ", ".join(_t(x) for x in starts[:4]) + ": the chords of the first, second and fourth match, and the "
            "third is different (the bridge).",
            "Check it by ear: does the part at " + (_t(starts[2]) if len(starts) > 2 else "the third section")
            + " sound new, and does the part after it sound like the opening again? If not, it isn't AABA.",
        ], how_sure=f"a {sf['match_pct']}% chord match; the section map in the full analysis may disagree, so "
                    "trust your ears"))
    # ---- harmony: famous progressions
    for p in (r.get("references") or {}).get("progressions", []):
        try:
            card = _card(p["progression"], [
                f"Your chords {' – '.join(p['your_chords'])} "
                + ("are the same loop as" if p["match"] == "same loop" else "contain")
                + f" {p['progression'].split(' (')[0]}.",
                f"In {r['key']}, your chords as numerals: {' – '.join(_numerals(p['your_chords'], key))}.",
            ], visual=_chord_row(p["your_chords"], _numerals(p["your_chords"], key)))
        except KeyError:
            continue
        cards.append(card)
    # ---- the mode (the plain major and minor go with the key, as a basic)
    mode, conf = r.get("mode"), r.get("mode_confidence")
    if mode and conf != "low" and mode not in ("Ionian (major)", "Aeolian (natural minor)") and lexicon.find(mode):
        cards.append(_card(mode, [
            f"The notes of your track fit {mode} best: {' '.join(r['scale_notes'])}.",
            f"Its colour note is {r['color_note']}: listen for it in the melody.",
        ], how_sure=f"{conf} confidence" + (f" (it could also be {r['mode_runner_up']})" if conf == "medium" else ""),
            visual=_scale_row(r["scale_notes"], r["color_note"])))
    if r.get("third_mix") and b:
        cards.append(_card("Major/minor blur", [
            f"Both 3rds of the key sound in your track ({r.get('thirds')}): the blue note against the major chords.",
        ]))
    # ---- rhythm
    m = r.get("meter") or {}
    if m.get("confidence") == "detected" and m.get("beats_per_bar") == 3:
        cards.append(_card("Waltz time (3/4)", [f"The beats group in threes: {m['label']}"
                                                + (f" at about {r['tempo_bpm']:.0f} BPM." if r.get("tempo_bpm") else ".")]))
    elif m.get("confidence") == "detected" and m.get("beats_per_bar") in (5, 7):
        cards.append(_card("Odd meters (5/4, 7/4)", [f"The beats group in {m['beats_per_bar']}s: {m['label']}."]))
    d = r.get("drums") or {}
    traits = " ".join(d.get("traits", [])) if d.get("present") else ""
    grid = charts.drum_grid(d["grid"], d.get("hats"), len(d["grid"]) // 4, bpm=r.get("tempo_bpm")) \
        if d.get("present") and d.get("grid") else None
    for word, name in (("backbeat", "Backbeat"), ("four-on-the-floor", "Four-on-the-floor"),
                       ("half-time", "Half-time feel"), ("3+3+2", "Tresillo (3+3+2)")):
        if word in traits:
            cards.append(_card(name, [f"Your drums: “{t}”." for t in d["traits"] if word in t]
                               + ["The grid below is your main groove: one column per 16th note."], visual=grid))
            grid = None  # one picture of the groove is enough
    rh = r.get("rhythm") or {}
    if rh.get("groove") in ("some syncopation", "heavily syncopated"):
        cards.append(_card("Syncopation", [
            f"About {rh.get('syncopation_pct')}% of the strong hits land between the beats ({rh['groove']})."]))
    # ---- melody and the low part
    bs = r.get("bass") or {}
    if bs.get("present") and bs.get("riff"):
        cards.append(_card("Riff", [f"The bass repeats a one-bar riff: {' '.join(bs['riff'])}."],
                           visual=_chord_row(bs["riff"], [f"beat {i + 1}" for i in range(len(bs["riff"]))])))
    t = r.get("top_line") or {}
    if t.get("present") and t.get("motifs"):
        x = t["motifs"][0]
        cards.append(_card("Motif", [f"A 4-note figure comes back {x['times']} times in the top line: "
                                     f"{' '.join(x['notes'])}."], main=False))
    if r.get("drone"):
        held = (bs.get("pedal_note") or (r.get("scale_notes") or ["?"])[0]) if bs.get("present") else None
        cards.append(_card("Pedal / drone", [
            f"One note keeps sounding under the chords for about {r.get('drone_pct', 0)}% of the track"
            + (f" (the bass holds {held})." if held else ".")], main=False))
    if r.get("key_changes"):
        secs = r.get("key_sections") or []
        cards.append(_card("Modulation", [" → ".join(f"{s['start']} {s['key']}" for s in secs) + "."]
                           + [_relative_note(secs)], main=False))
    # ---- production
    pump = (r.get("producer") or {}).get("pump") or {}  # as the producer's view reports it
    if pump.get("kind"):
        cards.append(_card("Sidechain pumping", [f"{pump['kind'].capitalize()} pumping: the sustained sounds duck about "
                                                 f"{pump['depth_db']} dB after each beat."]))
    dance = (r.get("producer") or {}).get("map") or []
    drops = [s for s in dance if s.get("role") == "drop"]
    if drops:
        cards.append(_card("Drop", ["Drops at " + ", ".join(_t(s["start"]) for s in drops[:4]) + "."]))
    # ---- building blocks: the key, and the chord types it uses
    tonic = ma.spell(ma.key_tonic(key), key)
    cards.append(_card("Key", [f"Home is {tonic}: the track is in {r['key']}, and its scale is "
                               f"{' '.join(r.get('scale_notes') or [])}.",
                               f"Its relative {'minor' if not ma.key_is_minor(key) else 'major'} is "
                               f"{ma.KEY_NAMES[ma.camelot_neighbours(key)['relative']]}: the same notes, another home."],
                       how_sure=f"key confidence {r.get('key_confidence')}", main=False))
    used = [c["chord"] for c in r.get("chords_used") or []]
    kinds = {"Power chord": "5", "Dominant 7th chord": "7", "Major 7th chord": "maj7", "Minor 7th chord": "m7",
             "Major chord": "", "Minor chord": "m"}
    for name, q in kinds.items():
        these = [c for c in used if (ma.parse_chord(c) or (0, None))[1] == q]
        if these:
            cards.append(_card(name, [f"In your track: {', '.join(these[:6])}."], main=False))
    seen, out = set(), []
    for c in cards:  # one card per entry (a progression can match two ways)
        if c["entry"]["slug"] not in seen:
            seen.add(c["entry"]["slug"])
            out.append(c)
    return out


def _relative_note(secs):
    keys = [s["key"] for s in secs]
    pairs = {tuple(sorted((a, b))) for a, b in zip(keys, keys[1:])}
    rel = [f"{a} and {b}" for a, b in pairs if _relative(a, b)]
    return ("Relative keys (same notes, another home): " + "; ".join(sorted(rel)) + ".") if rel else None


def _relative(a, b):
    try:
        ka, kb = ma.KEY_NAMES.index(a), ma.KEY_NAMES.index(b)
    except ValueError:
        return False
    return ma.camelot_neighbours(ka)["relative"] == kb


def facts(r):
    """The headline numbers, each linked to the library."""
    if not r or "error" in r:
        return []
    m = r.get("meter") or {}
    out = [("Key", r["key"], "key"), ("Mode", r.get("mode"), (lexicon.find(r.get("mode")) or {}).get("slug"))]
    if r.get("tempo_bpm"):
        out.append(("Tempo", f"{r['tempo_bpm']:.0f} BPM", "tempo"))
    if m.get("label"):
        out.append(("Meter", m["label"], "meter"))
    out.append(("Length", r.get("duration"), "length"))
    return [(k, v, s) for k, v, s in out if v]


def lesson(results):
    """[{name, facts, main, basics}] for every own track of an analysis (failed files left out)."""
    out = []
    for r in results:
        if r.get("role") == "reference" or "error" in r:
            continue
        cards = patterns(r)
        out.append({"name": r["file"], "facts": facts(r), "main": [c for c in cards if c["main"]],
                    "basics": [c for c in cards if not c["main"]]})
    return out
