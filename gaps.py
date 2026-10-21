"""What a track uses and what it leaves out, in analytical terms - the map the ideas (ours and the AI's) are built on.

For each degree of the detected scale: how strongly it sounds overall, how much of the chord time is built on it,
and how much of the melody uses it. From that and the rest of the analysis, the gaps:
- the mode's colour note (the one note that gives Dorian, Lydian, minor... its character) barely heard;
- scale degrees the melody never touches; a melody of chord tones only (no tension); a narrow range; all steps or
  all leaps;
- scale degrees the harmony never builds on; no dominant (no tension and release) in a major/minor key; one or
  two chords only; every chord the same length;
- mood: every section in the same corner of the energy/mood map, and the corners the track never visits - with
  the levers our mood measure is built from (mode, minor chords, brightness, tempo, density, loudness).

Each gap carries the evidence (numbers) and a concrete rule-based suggestion; the AI is asked to fill the same gaps
in its own words. A gap is only called when a stated threshold says so.
"""

DEGREE = ["1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"]
ROMAN_SEMITONE = {"I": 0, "II": 2, "III": 4, "IV": 5, "V": 7, "VI": 9, "VII": 11}
COLOUR_WEAK = 0.75       # the colour note sounds at under 75% of a typical scale note: its flavour hardly comes through
DEGREE_USED = 0.03       # a degree carrying under 3% of the chord time or the melody counts as unused
CHORD_TONES_ONLY = 85    # a melody that's 85%+ chord tones has no tension notes to speak of
NARROW_RANGE = 7         # a typed melody spanning under a fifth (7 semitones)
MOOD_SPREAD = 10         # sections closer than 10 points (0-100) on both axes share one mood (feel_report's rule)
SAME_LENGTH = 0.15       # chord lengths varying by under 15% (coefficient of variation): one harmonic pace throughout
MIN_CHORDS_FOR_PACE = 8


def mode_word(mode):
    """'Ionian (major)' -> 'major', 'Dorian' -> 'Dorian': what a musician calls the sound."""
    return mode.split("(")[1].rstrip(")") if "(" in mode else mode


def _chord_spans(r):
    tl, dur = r.get("chord_timeline") or [], r.get("duration_sec") or 0
    out = []
    for i, c in enumerate(tl):
        end = tl[i + 1]["start"] if i + 1 < len(tl) else dur
        out.append((c, max(0.0, end - c.get("start", 0.0))))
    return out


def _roman_root(roman):
    """Semitones above the tonic of a Roman numeral's root (bVII -> 10), or None."""
    r = (roman or "").strip()
    acc = 0
    while r[:1] in ("b", "#"):
        acc += -1 if r[0] == "b" else 1
        r = r[1:]
    num = ""
    for ch in r:
        if ch.upper() in "IV":
            num += ch.upper()
        else:
            break
    return (ROMAN_SEMITONE[num] + acc) % 12 if num in ROMAN_SEMITONE else None


def degree_table(r, modes):
    """One row per scale degree: its note, how strongly it sounds, its share of chord time (as a root), its share
    of the melody, and whether it's the mode's colour note."""
    ivs, colour = modes.get(r.get("mode"), (None, None))
    if not ivs:
        return []
    notes = r.get("scale_notes") or []
    strength = r.get("note_strength") or {}
    spans = _chord_spans(r)
    total = sum(d for _, d in spans) or 1.0
    roots = {}
    for c, d in spans:
        semi = _roman_root(c.get("roman"))
        if semi is not None:
            roots[semi] = roots.get(semi, 0.0) + d
    mel, source = melody_degrees(r)
    rows = []
    for iv, note in zip(ivs, notes):
        rows.append({"degree": DEGREE[iv], "semitones": iv, "note": note, "strength": strength.get(note),
                     "harmony": round(roots.get(iv, 0.0) / total, 3), "melody": None if mel is None else mel.get(iv, 0.0),
                     "colour": iv == colour})
    return rows, source


def melody_degrees(r):
    """{semitones above the tonic: share of the melody}, and where it came from: the melody the musician typed
    (preferred), else the automatic top line when it's a real line of its own; (None, None) otherwise."""
    m = r.get("melody") or {}
    if m.get("degrees"):
        counts = {}
        for d in m["degrees"]:
            if d in DEGREE:
                counts[DEGREE.index(d)] = counts.get(DEGREE.index(d), 0) + 1
        n = sum(counts.values()) or 1
        return {k: v / n for k, v in counts.items()}, "typed"
    t = r.get("top_line") or {}
    pct = t.get("note_pct")
    if t.get("present") and pct and not t.get("follows_chords"):
        tonic_note = (r.get("scale_notes") or [None])[0]
        names = [n for n in (r.get("scale_notes") or [])]
        out = {}
        # note names -> semitones above the tonic, via the scale (out-of-scale notes are left out)
        ivs = r.get("_scale_ivs") or []
        for note, share in pct.items():
            if note in names:
                out[ivs[names.index(note)]] = share / 100
        return (out, "top line") if out and tonic_note else (None, None)
    return None, None


def analyse(r, modes, triads, note_of=None):
    """The gaps, as [{id, area, title, have, missing, evidence, idea}], and the degree table for the chart.
    triads(semitones, quality) -> chord name, for naming the chords that would bring a missing note in;
    note_of(semitones) -> the note's name in the key (for notes outside the scale, like a blues's blue notes)."""
    if "error" in r or not r.get("mode"):
        return {"degrees": [], "gaps": []}
    ivs, colour = modes.get(r["mode"], ([], None))
    r = dict(r, _scale_ivs=ivs)
    table, source = degree_table(r, modes) if ivs else ([], None)
    gaps = []
    by_iv = {row["semitones"]: row for row in table}
    # quality of the triad on each scale degree, from the scale itself
    pcs = set(ivs)

    def triad_on(iv):
        third = next(((iv + s) % 12 for s in (3, 4) if (iv + s) % 12 in pcs), None)
        fifth = next(((iv + s) % 12 for s in (7, 6, 8) if (iv + s) % 12 in pcs), None)
        if third is None or fifth is None:
            return None
        q = ("m" if (third - iv) % 12 == 3 else "") + ("dim" if (fifth - iv) % 12 == 6 else "")
        q = "dim" if q == "mdim" else q
        return triads(iv, q), {iv, third, fifth}

    blues = bool(r.get("blues_form"))
    others = sorted(x["strength"] for x in table if x["strength"] is not None and not x["colour"] and x["semitones"])
    typical = others[len(others) // 2] if others else None
    # --- in a blues the colour is the blue notes (b3, b7), and the form decides the chords
    if blues and typical:
        tonic = (r.get("scale_notes") or [None])[0]
        strength = r.get("note_strength") or {}
        minor = 3 in ivs  # a minor blues already has b3 and b7: its blue note is the b5
        blue_ivs = (6,) if minor else (3, 10)
        blue = [(note_of(iv), strength.get(note_of(iv))) for iv in blue_ivs
                if note_of and strength.get(note_of(iv)) is not None]
        weak = [(n, s) for n, s in blue if s < COLOUR_WEAK * typical]
        if blue and len(weak) == len(blue):
            gaps.append({"id": "blue_notes", "area": "melody", "title": "Blue notes",
                         "have": f"a {r['blues_form']['name']} on {tonic}",
                         "missing": "the blue notes " + " and ".join(n for n, _ in blue) + " are barely heard ("
                                    + ", ".join(f"{s:.2f}" for _, s in blue) + f" against a typical {typical:.2f})",
                         "evidence": {"blue": dict(blue), "typical": typical},
                         "idea": f"Bend into {blue[0][0]} (a quarter tone to a semitone) over the home chord, and "
                                 f"answer each phrase with a lick that lands on {tonic}."})
    # --- the colour note
    if colour is not None and colour in by_iv and not blues:
        row = by_iv[colour]
        if row["strength"] is not None and typical and row["strength"] < COLOUR_WEAK * typical:
            carriers = [c for iv in ivs if (t := triad_on(iv)) and colour in t[1] for c in [t[0]]]
            gaps.append({"id": "colour", "area": "harmony", "title": "The key's colour note",
                         "have": f"a {mode_word(r['mode'])} key, whose colour note is {row['note']} "
                                 f"(degree {row['degree']})",
                         "missing": f"{row['note']} is barely heard (strength {row['strength']:.2f} against a typical "
                                    f"{typical:.2f}), so the {mode_word(r['mode'])} sound hardly comes through",
                         "evidence": {"strength": row["strength"], "typical": typical},
                         "idea": f"Let {row['note']} be heard: hold it on a strong beat in the melody, or use a chord "
                                 f"that contains it ({', '.join(carriers[:3])})." if carriers else
                                 f"Let {row['note']} be heard: hold it on a strong beat in the melody."})
    # --- the melody's scale degrees
    if source:
        used = {iv for iv, share in (melody_degrees(r)[0] or {}).items() if share >= DEGREE_USED}
        missing = [row for row in table if row["semitones"] not in used]
        if missing:
            names = ", ".join(f"{x['degree']} ({x['note']})" for x in missing)
            colour_missing = any(x["colour"] for x in missing)
            gaps.append({"id": "melody_degrees", "area": "melody", "title": "Scale notes the melody doesn't use",
                         "have": f"the {source} melody uses {len(table) - len(missing)} of the {len(table)} scale notes",
                         "missing": f"it never touches {names}" + (" - including the colour note" if colour_missing else ""),
                         "evidence": {"missing": [x["degree"] for x in missing], "source": source},
                         "idea": "Bring in " + ", ".join(x["note"] for x in missing[:2]) + ": as a passing note between "
                                 "two you already use, or as a held note that resolves by step (4 falls to 3, 7 rises to 1)."})
    m, t = r.get("melody") or {}, r.get("top_line") or {}
    if m.get("range"):
        try:
            span = int(m["range"].split("(")[1].split()[0])
        except (IndexError, ValueError):
            span = None
        if span is not None and span < NARROW_RANGE:
            gaps.append({"id": "melody_range", "area": "melody", "title": "Melodic range",
                         "have": f"the melody spans {m['range']}", "missing": "a high point to build towards (it stays "
                         "within a fifth)", "evidence": {"semitones": span},
                         "idea": "Give one phrase a peak: lift it a third or more above the rest, once, near the "
                                 "loudest section."})
        if m.get("leaps_pct", 0) < 10 and len(m.get("notes") or []) >= 6:
            gaps.append({"id": "melody_leaps", "area": "melody", "title": "Leaps", "have": "a melody that moves by step",
                         "missing": "any leap (over a whole tone)", "evidence": {"leaps_pct": m.get("leaps_pct")},
                         "idea": "Add one leap of a 4th or 6th at the start of a phrase, then step back the other way."})
        elif m.get("steps_pct", 100) < 30 and len(m.get("notes") or []) >= 6:
            gaps.append({"id": "melody_steps", "area": "melody", "title": "Stepwise motion",
                         "have": "a melody that mostly leaps", "missing": "stepwise lines that join the leaps up",
                         "evidence": {"steps_pct": m.get("steps_pct")},
                         "idea": "Fill one of the leaps with the scale notes in between, as a run."})
    if t.get("present") and t.get("chord_tone_pct", 0) >= CHORD_TONES_ONLY and not m.get("degrees"):
        gaps.append({"id": "tension_notes", "area": "melody", "title": "Tension notes",
                     "have": f"a top line that's {t['chord_tone_pct']}% chord tones",
                     "missing": "notes outside the chord (suspensions, 9ths, passing notes) that pull and resolve",
                     "evidence": {"chord_tone_pct": t["chord_tone_pct"]},
                     "idea": "Delay a chord tone: hold the note from the previous chord over the change (a suspension) "
                             "and let it fall by step."})
    # --- the harmony's scale degrees
    if table:
        used = [x for x in table if x["harmony"] >= DEGREE_USED]
        unused = [x for x in table if x["harmony"] < DEGREE_USED and x["semitones"]]
        spans = _chord_spans(r)
        distinct = {c["chord"] for c, _ in spans}
        if unused and len(used) <= 4 and not blues:  # a blues form decides its own chords
            # the plain major and minor chords first: a diminished one is a passing colour, not a first try
            options = [t_[0] for x in unused if (t_ := triad_on(x["semitones"])) and not t_[0].endswith("dim")]
            gaps.append({"id": "harmony_degrees", "area": "harmony", "title": "Chords the key offers, unused",
                         "have": "chords built on " + ", ".join(x["degree"] for x in used),
                         "missing": "anything on " + ", ".join(f"{x['degree']} ({x['note']})" for x in unused),
                         "evidence": {"used": [x["degree"] for x in used], "unused": [x["degree"] for x in unused]},
                         "idea": "Try " + " or ".join(options[:2]) + " in place of a repeat of one of your chords."
                                 if options else "Try a chord on one of the unused scale degrees."})
        dominant = any(_roman_root(c.get("roman")) == 7 for c, _ in spans)
        plain = r["mode"] in ("Ionian (major)", "Aeolian (natural minor)", "Harmonic minor")
        if plain and spans and not dominant and len(distinct) >= 2:
            five = triads(7, "" if r["mode"] != "Aeolian (natural minor)" else "")
            gaps.append({"id": "dominant", "area": "harmony", "title": "Tension and release",
                         "have": "no chord on the 5th degree",
                         "missing": "a dominant (V) that pulls back home, the strongest tension/release in a key",
                         "evidence": {"dominant": False},
                         "idea": f"Put {five}7 (the V7, major even in a minor key) before a return to the home chord."})
        if len(distinct) <= 2 and len(spans) >= 4:
            gaps.append({"id": "static_harmony", "area": "harmony", "title": "Harmonic movement",
                         "have": f"{len(distinct)} chord{'s' if len(distinct) != 1 else ''} throughout",
                         "missing": "harmonic movement: a second progression for contrast",
                         "evidence": {"chords": sorted(distinct)},
                         "idea": "Keep your loop for the verse and write a contrasting section that starts on a "
                                 "different chord of the key."})
        lengths = [d for _, d in spans[:-1] if d > 0]
        if len(lengths) >= MIN_CHORDS_FOR_PACE:
            mean = sum(lengths) / len(lengths)
            sd = (sum((x - mean) ** 2 for x in lengths) / len(lengths)) ** 0.5
            if mean and sd / mean < SAME_LENGTH:
                gaps.append({"id": "harmonic_pace", "area": "rhythm", "title": "Harmonic pace",
                             "have": f"every chord lasts about {mean:.1f} s", "missing": "changes of pace: chords that "
                             "arrive early, or two quick chords where one usually sits",
                             "evidence": {"cv": round(sd / mean, 2)},
                             "idea": "Speed the chords up into the end of a section (two per bar instead of one), or "
                                     "anticipate a change by an 8th note."})
    # --- mood
    secs = (r.get("feel") or {}).get("sections") or []
    if secs:
        quad = lambda s: ("excited" if s["valence"] >= 50 else "tense") if s["arousal"] >= 50 else \
            ("calm" if s["valence"] >= 50 else "sad")
        visited = {quad(s) for s in secs}
        va = [s["valence"] for s in secs]
        ar = [s["arousal"] for s in secs]
        same = max(va) - min(va) < MOOD_SPREAD and max(ar) - min(ar) < MOOD_SPREAD
        missing_q = [q for q in ("calm", "excited", "tense", "sad") if q not in visited]
        f = r.get("feel") or {}
        levers = {"mode": r["mode"], "minor_chords_pct": round(sum(s["minor_chords_pct"] for s in secs) / len(secs)),
                  "brightness_hz": (r.get("sound") or {}).get("brightness_hz"), "tempo_bpm": r.get("tempo_bpm"),
                  "notes_per_sec": round(sum(s["notes_per_sec"] for s in secs) / len(secs), 1),
                  "energy": f.get("energy"), "valence": f.get("valence")}
        if same or len(visited) == 1:
            gaps.append({"id": "mood", "area": "arrangement", "title": "Mood contrast",
                         "have": f"every section feels {secs[0]['mood']} (" + ", ".join(sorted(visited)) + " corner)",
                         "missing": "a section in another mood: " + " or ".join(missing_q[:2]),
                         "evidence": {"visited": sorted(visited), "levers": levers},
                         "idea": _mood_idea(missing_q[0] if missing_q else "tense", levers)})
        else:
            gaps.append({"id": "mood", "area": "arrangement", "title": "Moods not yet visited",
                         "have": "sections that move between " + " and ".join(sorted(visited)),
                         "missing": "never " + " or ".join(missing_q) if missing_q else "nothing: it covers the map",
                         "evidence": {"visited": sorted(visited), "levers": levers},
                         "idea": _mood_idea(missing_q[0], levers) if missing_q else ""})
            if not missing_q:
                gaps.pop()
    return {"degrees": table, "melody_source": source, "gaps": gaps}


MOOD_LEVERS = {
    # what our energy and positivity measures are made of (musicanalyze.feel_analysis), and so what moves them
    "excited": "more energy and a brighter mood: denser rhythm, a faster feel, more high end, major chords in place "
               "of minor ones",
    "calm": "less energy, bright but gentle: fewer notes per second, softer and warmer, held major chords",
    "tense": "more energy but darker: minor chords, a b2 or b6, a driving rhythm and louder low end",
    "sad": "less energy and darker: minor chords, sparser rhythm, a slower feel, the highs rolled off",
}


def _mood_idea(target, levers):
    return f"For {'an' if target[0] in 'aeiou' else 'a'} {target} section: {MOOD_LEVERS[target]}."


def ai_lines(g):
    """The gaps as lines for the AI prompt."""
    out = []
    for x in g["gaps"]:
        out.append(f"- [{x['id']}] {x['title']}: the track has {x['have']}; missing: {x['missing']}.")
    return out
