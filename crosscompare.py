"""Comparing tracks with each other: your own (two takes, versions of one song, the tracks of an EP) or yours with a
result another musician chose to share with you.

The wording is neutral ("Take 2 is 8% faster than Take 1"), and a difference is only named when it's big enough to
hear - the same just-noticeable steps as the comparison with a song you admire (musicanalyze.TEMPO_JND etc.).
Nothing here reads audio: it works on saved results only.
"""

import musicanalyze as ma

MAX_TRACKS = 4           # the chart palette (ink + three colours) stays readable and colour-blind safe up to four
LENGTH_RATIO = 1.25      # a quarter longer or shorter
ENERGY_STEP = 1          # one step on the 1-10 energy scale
MOOD_STEP = 10           # positivity 0-100: half a mood band
CHORD_SECONDS = 4        # a chord held this long in total counts as used


def _chords(r):
    return {c["chord"] for c in r.get("chords_used") or [] if c.get("seconds", 0) >= CHORD_SECONDS}


def _form(r):
    return (r.get("blues_form") or r.get("song_form") or {}).get("name") or " ".join(s["label"] for s in r["sections"])


def table(tracks):
    """[(measure, [shown value per track])] - the same measurements side by side."""
    def stream(r, k):
        return ((r.get("sound") or {}).get("streaming") or {}).get(k)
    rows = [
        ("Tempo", lambda r: (("~" if r.get("tempo_approximate") else "") + f"{r['tempo_bpm']:.0f} BPM")
         if r.get("tempo_bpm") else "free"),
        ("Key", lambda r: f"{r['key']} ({r['camelot']})"),
        ("Mode", lambda r: r["mode"]),
        ("Time signature", lambda r: (r.get("meter") or {}).get("label") or "-"),
        ("Length", lambda r: r["duration"]),
        ("Form", _form),
        ("Energy · mood", lambda r: f"{(r.get('feel') or {}).get('energy', '-')}/10 · {(r.get('feel') or {}).get('mood', '-')}"),
        ("Drums · bass", lambda r: f"{'yes' if (r.get('drums') or {}).get('present') else 'no'} · "
                                   f"{'yes' if (r.get('bass') or {}).get('present') else 'no'}"),
        ("Main chords", lambda r: ", ".join(c["chord"] for c in (r.get("chords_used") or [])[:5])),
        ("Loudness", lambda r: f"{stream(r, 'lufs')} LUFS" if stream(r, "lufs") is not None else "-"),
        ("Brightness", lambda r: f"{(r.get('sound') or {}).get('brightness_hz', '-')} Hz"),
    ]
    return [(name, [f(r) for r in tracks]) for name, f in rows]


def differences(a, b, na, nb):
    """What's audibly different from a to b, in plain words: [{area, text, key}]. Empty when they're alike."""
    out = []
    add = lambda area, key, text: out.append({"area": area, "key": key, "text": text})
    ta, tb = a.get("tempo_bpm"), b.get("tempo_bpm")
    if ta and tb and not a.get("tempo_approximate") and not b.get("tempo_approximate"):
        ratio = tb / ta
        felt = min((ratio * f for f in (1, 2, 0.5)), key=lambda x: abs(x - 1))
        if abs(felt - 1) > ma.TEMPO_JND:
            add("tempo", "tempo", f"{nb} is {abs(felt - 1):.0%} {'faster' if felt > 1 else 'slower'} than {na} "
                                  f"({tb:.0f} against {ta:.0f} BPM).")
    if a["key_index"] != b["key_index"]:
        rel = ma._key_relation(a["key_index"], b["key_index"])
        nb_ = ma.camelot_neighbours(a["key_index"])
        mix = b["key_index"] in (nb_["relative"], nb_["up a fifth"], nb_["down a fifth"])
        add("key", "key", f"{na} is in {a['key']}, {nb} in {b['key']}: {rel}"
                          + (", so they mix smoothly one into the other." if mix else "."))
    if a["mode"] != b["mode"] and b.get("mode_confidence") != "low" and a.get("mode_confidence") != "low":
        add("harmony", "mode", f"{nb} leans on {b['mode']}, {na} on {a['mode']}.")
    ma_, mb_ = (a.get("meter") or {}).get("label"), (b.get("meter") or {}).get("label")
    if ma_ and mb_ and ma_ != mb_:
        add("rhythm", "meter", f"{na} is in {ma_}, {nb} in {mb_}.")
    da, db = a.get("duration_sec") or 0, b.get("duration_sec") or 0
    if da and db and max(da, db) / min(da, db) >= LENGTH_RATIO:
        add("arrangement", "length", f"{nb} is {'longer' if db > da else 'shorter'}: {b['duration']} against {a['duration']}.")
    fa, fb = a.get("feel") or {}, b.get("feel") or {}
    if fa and fb:
        if abs(fb["energy"] - fa["energy"]) >= ENERGY_STEP:
            add("arrangement", "energy", f"{nb} has {'more' if fb['energy'] > fa['energy'] else 'less'} energy "
                                         f"({fb['energy']}/10 against {fa['energy']}/10).")
        if abs(fb["valence"] - fa["valence"]) >= MOOD_STEP:
            add("arrangement", "mood", f"{nb} is {'brighter' if fb['valence'] > fa['valence'] else 'darker'} in mood "
                                       f"({fb['mood']}, against {fa['mood']}).")
    for part in ("drums", "bass"):
        pa, pb = bool((a.get(part) or {}).get("present")), bool((b.get(part) or {}).get("present"))
        if pa != pb:
            add("arrangement", part, f"{nb} {'has' if pb else 'has no'} {part}; {na} {'has' if pa else 'has no'}.")
    ca, cb = _chords(a), _chords(b)
    if ca and cb:
        new, gone = sorted(cb - ca), sorted(ca - cb)
        if new:
            add("harmony", "chords_new", f"Chords in {nb} that {na} doesn't use: {', '.join(new[:5])}.")
        if gone:
            add("harmony", "chords_gone", f"Chords in {na} that {nb} doesn't use: {', '.join(gone[:5])}.")
    ha, hb = a.get("harmonic_rhythm_beats"), b.get("harmonic_rhythm_beats")
    if ha and hb and max(ha, hb) / min(ha, hb) >= ma.HARMONIC_RATIO:
        add("harmony", "harmonic_pace", f"{nb}'s chords change {'less' if hb > ha else 'more'} often "
                                        f"(every {hb} beats against {ha}).")
    sa, sb = (a.get("rhythm") or {}).get("syncopation_pct"), (b.get("rhythm") or {}).get("syncopation_pct")
    if sa is not None and sb is not None and abs(sb - sa) > ma.SYNC_DIFF_PCT:
        add("rhythm", "syncopation", f"{nb} is {'more' if sb > sa else 'less'} syncopated ({sb}% of hits off the beat "
                                     f"against {sa}%).")
    la = ((a.get("sound") or {}).get("streaming") or {}).get("lufs")
    lb = ((b.get("sound") or {}).get("streaming") or {}).get("lufs")
    if la is not None and lb is not None and abs(lb - la) > ma.LOUDNESS_JND_LU:
        add("sound", "loudness", f"{nb} is {abs(lb - la):.1f} LU {'louder' if lb > la else 'quieter'} ({lb} against {la} LUFS).")
    ba, bb = (a.get("sound") or {}).get("brightness_hz"), (b.get("sound") or {}).get("brightness_hz")
    if ba and bb and max(ba, bb) / min(ba, bb) >= ma.BRIGHTNESS_RATIO:
        add("sound", "brightness", f"{nb} sounds {'brighter' if bb > ba else 'darker'} ({bb} against {ba} Hz).")
    if _form(a) != _form(b) and (a.get("blues_form") or b.get("blues_form") or a.get("song_form") or b.get("song_form")):
        add("arrangement", "form", f"{na}'s form is {_form(a)}, {nb}'s is {_form(b)}.")
    return out


def shared(tracks, names):
    """What every one of the tracks has in common: a sound of your own, or the thread through an EP."""
    out = []
    keys = {r["key_index"] for r in tracks}
    if len(keys) == 1:
        out.append(f"All in {tracks[0]['key']}.")
    else:
        first = tracks[0]["key_index"]
        nb = ma.camelot_neighbours(first)
        if all(k in (first, nb["relative"], nb["up a fifth"], nb["down a fifth"]) for k in keys):
            out.append("Their keys are neighbours on the Camelot wheel, so they flow into each other.")
    tempos = [r["tempo_bpm"] for r in tracks if r.get("tempo_bpm") and not r.get("tempo_approximate")]
    if len(tempos) == len(tracks) and max(tempos) / min(tempos) - 1 <= ma.TEMPO_JND:
        out.append(f"The same pace: {min(tempos):.0f}-{max(tempos):.0f} BPM.")
    meters = {(r.get("meter") or {}).get("label") for r in tracks}
    if len(meters) == 1 and None not in meters:
        out.append(f"All in {meters.pop()}.")
    modes = {r["mode"] for r in tracks}
    if len(modes) == 1:
        out.append(f"All in the {modes.pop()} colour.")
    common = set.intersection(*[_chords(r) for r in tracks]) if all(_chords(r) for r in tracks) else set()
    if common:
        out.append("Chords they all use: " + ", ".join(sorted(common)[:6]) + ".")
    energies = [(r.get("feel") or {}).get("energy") for r in tracks]
    if all(e is not None for e in energies) and max(energies) - min(energies) < ENERGY_STEP:
        out.append(f"The same energy, {energies[0]}/10.")
    for part in ("drums", "bass"):
        have = [bool((r.get(part) or {}).get("present")) for r in tracks]
        if all(have):
            out.append(f"{part.capitalize()} in every one.")
    return out


def versions(chain, names):
    """What changed from each version to the next: [(from name, to name, [differences])]."""
    return [(names[i], names[i + 1], differences(chain[i], chain[i + 1], names[i], names[i + 1]))
            for i in range(len(chain) - 1)]


def report(tracks, names, are_versions=False):
    """Everything the compare page shows, as data."""
    tracks, names = tracks[:MAX_TRACKS], names[:MAX_TRACKS]
    base = tracks[0]
    pairs = [] if are_versions else [(names[0], names[i], differences(base, tracks[i], names[0], names[i]))
                                     for i in range(1, len(tracks))]
    return {"names": names, "table": table(tracks), "shared": shared(tracks, names) if len(tracks) > 1 else [],
            "pairs": pairs, "versions": versions(tracks, names) if are_versions else [],
            "blend": ma.pairings([dict(r, file=n) for r, n in zip(tracks, names)])}
