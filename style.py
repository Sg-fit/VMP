"""Your style: what stays the same across your tracks and upload sessions, where you range, and what you never do.

It works on saved results only (no audio): each track's measurements become one row of features, and the rows are
read together.
- Signatures: a measurement whose middle half of tracks sits within one audible step (the same just-noticeable
  differences the comparisons use: 4% tempo, 1 LU, a third of an octave...), or a choice two thirds of the tracks
  make (minor keys, 4/4, drums). Fewer than MIN_TRACKS tracks can't show a habit, so nothing is claimed then.
- Harmony without the key: chords as Roman numerals, so "i - VI - III - VII" in A minor and in E minor count as the
  same move. A loop that turns up in two or more tracks is a signature progression.
- Sessions: tracks uploaded together form a session; the first and latest sessions are compared for audible drift.
- The most typical track (fewest audible differences from your usual) and the outlier (most).
- Unexplored ground: options no track takes (no major key, no 3/4, no key change...), as ideas to stretch.
"""

import re
import statistics
from collections import Counter

import charts
import musicanalyze as ma

MIN_TRACKS = 3          # a habit needs at least three examples: with two, any shared value may be a coincidence
MAJORITY = 2 / 3        # a choice two thirds of the tracks make is a habit
NUMERAL_SHARE = 0.05    # a numeral that fills at least 5% of a track's chord time counts as used in that track
CHORD_LOOP_MIN = 2      # a loop in two or more tracks is a signature progression

# (key, name, unit, how to get it, the smallest difference you'd hear, log scale?)
NUMERIC = [
    ("tempo", "Tempo", " BPM", lambda r: None if r.get("tempo_approximate") else r.get("tempo_bpm"), ma.TEMPO_JND, True),
    ("energy", "Energy", "/10", lambda r: (r.get("feel") or {}).get("energy"), 1, False),
    ("mood", "Positivity", "/100", lambda r: (r.get("feel") or {}).get("valence"), 10, False),
    ("length", "Length", " min", lambda r: (r.get("duration_sec") or 0) / 60 or None, 0.25, True),
    ("pace", "Chord change every", " beats", lambda r: r.get("harmonic_rhythm_beats"), ma.HARMONIC_RATIO - 1, True),
    ("offbeat", "Off-beat hits", "%", lambda r: (r.get("rhythm") or {}).get("syncopation_pct"), ma.SYNC_DIFF_PCT, False),
    ("busy", "Notes per second", "", lambda r: (r.get("rhythm") or {}).get("onsets_per_sec"), 0.25, True),
    ("bright", "Brightness", " Hz", lambda r: (r.get("sound") or {}).get("brightness_hz"), ma.BRIGHTNESS_RATIO - 1, True),
    ("lufs", "Loudness", " LUFS", lambda r: ((r.get("sound") or {}).get("streaming") or {}).get("lufs"),
     ma.LOUDNESS_JND_LU, False),
    ("lra", "Loudness range", " LU", lambda r: ((r.get("sound") or {}).get("streaming") or {}).get("loudness_range_lu"),
     ma.LRA_DIFF_LU, False),
]
# How the words read: low -> high
WORDS = {"tempo": ("slow", "fast"), "energy": ("calm", "intense"), "mood": ("dark", "bright"),
         "length": ("short", "long"), "pace": ("quick chord changes", "chords that linger"),
         "offbeat": ("on the beat", "syncopated"), "busy": ("sparse", "busy"), "bright": ("dark-toned", "bright-toned"),
         "lufs": ("quiet", "loud"), "lra": ("even", "dynamic")}


# ...and as comparisons, for change over time
MORE_WORDS = {"tempo": ("slower", "faster"), "energy": ("calmer", "more intense"), "mood": ("darker", "brighter"),
              "length": ("shorter", "longer"), "pace": ("quicker chord changes", "chords that linger longer"),
              "offbeat": ("more on the beat", "more syncopated"), "busy": ("sparser", "busier"),
              "bright": ("darker-toned", "brighter-toned"), "lufs": ("quieter", "louder"), "lra": ("more even", "more dynamic")}
# the energy arc (musicanalyze.feel) said of many tracks
ARC_PLURAL = {"stays at about the same energy throughout": "stay at about the same energy throughout",
              "builds up to the end": "build up to the end", "starts strongest and winds down": "start strongest and wind down",
              "rises to a peak in the middle, then comes down": "rise to a peak in the middle, then come down",
              "goes up and down between sections": "go up and down between sections"}


def _num_fmt(key, v):
    return f"{v:.1f}" if key in ("length", "busy", "pace") else f"{v:.0f}"


def features(r):
    """One track's measurements as style features."""
    f = {k: get(r) for k, _, _, get, _, _ in NUMERIC}
    feel = r.get("feel") or {}
    d, b = r.get("drums") or {}, r.get("bass") or {}
    f.update({
        "minor": r["key"].endswith("minor"),
        "mode": r.get("mode"),
        "meter": (r.get("meter") or {}).get("label") or "free",
        "key": r.get("key"), "camelot": r.get("camelot"),
        "drums": bool(d.get("present")),
        "bass": bool(b.get("present")),
        "modulates": bool(r.get("key_changes")),
        "arc": feel.get("arc"),
        "contrast": feel.get("contrast"),
        "arousal": feel.get("arousal"),
        "form": (r.get("blues_form") or r.get("song_form") or {}).get("name"),
        "numerals": numerals(r),
        "loops": [" – ".join(lp["roman"]) for lp in r.get("loops") or []],
        "colours": colours(r),
    })
    return f


def numerals(r):
    """{Roman numeral: share of the track's chord time}, key-independent."""
    time = Counter()
    tl = r.get("chord_timeline") or []
    for i, c in enumerate(tl):
        end = tl[i + 1]["start"] if i + 1 < len(tl) else (r.get("duration_sec") or c.get("start", 0))
        if c.get("roman"):  # the chord's place in the key; its colour (a 7th, a power chord) is counted in colours()
            time[re.sub(r"(maj)?\d+$", "", c["roman"]) or c["roman"]] += max(0.0, end - c.get("start", 0))
    total = sum(time.values()) or 1
    return {k: v / total for k, v in time.items()}


# In a minor key these are the key's own chords (natural minor, plus the harmonic minor's major V and the
# raised VII), though the numerals are written against the major scale
MINOR_OWN = {"i", "ii°", "bIII", "iv", "v", "V", "bVI", "bVII", "vii°"}


def colours(r):
    """Shares of chord time: power chords, 7ths, minor chords, chords from outside the key."""
    tl = r.get("chord_timeline") or []
    time = Counter()
    total = 0.0
    for i, c in enumerate(tl):
        end = tl[i + 1]["start"] if i + 1 < len(tl) else (r.get("duration_sec") or c.get("start", 0))
        t = max(0.0, end - c.get("start", 0))
        total += t
        name = c.get("chord", "")
        if name.endswith("5"):
            time["power"] += t
        elif "7" in name:
            time["seventh"] += t
        if "m" in name.replace("maj", "") and not name.endswith("5"):
            time["minor"] += t
        roman = re.sub(r"(maj)?\d+$", "", c.get("roman") or "")
        if charts.chord_function(c.get("roman")) == "other" and not (r["key"].endswith("minor") and roman in MINOR_OWN):
            time["outside"] += t
    return {k: time[k] / total for k in ("power", "seventh", "minor", "outside")} if total else {}


def _quartiles(xs):
    xs = sorted(xs)
    if len(xs) < 2:
        return xs[0], xs[0], xs[0]
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[0], statistics.median(xs), q[2]


def _gap(key, a, b):
    """How many audible steps apart two values are."""
    step = next(s for k, _, _, _, s, _ in NUMERIC if k == key)
    log = next(lg for k, _, _, _, _, lg in NUMERIC if k == key)
    if log:
        lo, hi = sorted((a, b))
        return (hi / lo - 1) / step if lo > 0 else 0.0
    return abs(a - b) / step


AXIS_STEPS = 4  # a chart row spans at least four audible steps, so a difference you couldn't hear doesn't look big


def _axis(key, lo, hi, med):
    step = next(s for k, _, _, _, s, _ in NUMERIC if k == key)
    log = next(lg for k, _, _, _, _, lg in NUMERIC if k == key)
    if log and med > 0:
        need = (1 + step) ** (AXIS_STEPS / 2)
        return min(lo, med / need), max(hi, med * need)
    return min(lo, med - step * AXIS_STEPS / 2), max(hi, med + step * AXIS_STEPS / 2)


def ranges(rows):
    """For each numeric feature: the values per track, the middle half, the median, and whether it's a habit."""
    out = []
    for key, name, unit, _, _, _ in NUMERIC:
        vals = [(t["name"], t["f"][key]) for t in rows if t["f"].get(key) is not None]
        if len(vals) < 2:
            continue
        q1, med, q3 = _quartiles([v for _, v in vals])
        steady = len(vals) >= MIN_TRACKS and _gap(key, q1, q3) <= 1
        lo, hi = min(v for _, v in vals), max(v for _, v in vals)
        out.append({"key": key, "name": name, "unit": unit, "values": vals, "q1": q1, "median": med, "q3": q3,
                    "lo": lo, "hi": hi, "steady": steady, "axis": _axis(key, lo, hi, med)})
    return out


def _share(rows, test):
    n = sum(1 for t in rows if test(t["f"]))
    return n, len(rows)


def signatures(rows, rng):
    """Plain sentences about what you keep doing, the clearest first."""
    if len(rows) < MIN_TRACKS:
        return []
    out = []
    n = len(rows)
    minor, _ = _share(rows, lambda f: f["minor"])
    if minor >= MAJORITY * n:
        out.append(("Keys", "Always a minor key." if minor == n else f"Minor keys: {minor} of {n} tracks."))
    elif n - minor >= MAJORITY * n:
        out.append(("Keys", "Always a major key." if minor == 0 else f"Major keys: {n - minor} of {n} tracks."))
    modes = Counter(t["f"]["mode"] for t in rows if t["f"]["mode"])
    mode, mc = modes.most_common(1)[0] if modes else (None, 0)
    if mode and mc >= MAJORITY * n and mode not in ("Ionian (major)", "Aeolian (natural minor)"):
        out.append(("Mode", f"{mode} is your colour: {mc} of {n} tracks."))
    meters = Counter(t["f"]["meter"] for t in rows)
    meter, cnt = meters.most_common(1)[0]
    if cnt >= MAJORITY * n:
        out.append(("Time", f"{meter} every time." if cnt == n else f"{meter} almost every time ({cnt} of {n})."))
    drums, _ = _share(rows, lambda f: f["drums"])
    if drums >= MAJORITY * n:
        out.append(("Line-up", f"Drums in {drums} of {n} tracks."))
    elif n - drums >= MAJORITY * n:
        out.append(("Line-up", f"Mostly without drums ({n - drums} of {n} tracks)."))
    for x in rng:
        if x["steady"]:
            lo, hi = _num_fmt(x["key"], x["q1"]), _num_fmt(x["key"], x["q3"])
            out.append((x["name"], f"Around {_num_fmt(x['key'], x['median'])}{x['unit']}, track after track"
                                   + (f" (most of them {lo}–{hi}{x['unit']})." if lo != hi else ".")))
    arcs = Counter(t["f"]["arc"] for t in rows if t["f"]["arc"])
    if arcs:
        arc, ac = arcs.most_common(1)[0]
        if ac >= MAJORITY * n:
            out.append(("Shape", f"Your tracks {ARC_PLURAL.get(arc, arc)} ({ac} of {n})."))
    for name, (k, label) in {"power": ("power", "power chords"), "seventh": ("seventh", "7th chords"),
                             "outside": ("outside", "chords from outside the key")}.items():
        users = sum(1 for t in rows if t["f"]["colours"].get(k, 0) >= 0.15)
        if users >= MAJORITY * n:
            out.append(("Chord colour", f"You lean on {label}: they fill at least a sixth of the chord time in "
                                        f"{users} of {n} tracks."))
    return out


def vocabulary(rows):
    """[(numeral, tracks using it, share of all chord time)], most used first."""
    used, time = Counter(), Counter()
    for t in rows:
        for k, v in t["f"]["numerals"].items():
            time[k] += v / len(rows)
            if v >= NUMERAL_SHARE:
                used[k] += 1
    return sorted(((k, used[k], time[k]) for k in time if used[k]), key=lambda x: (-x[1], -x[2]))


def _rotate(loop):
    """A loop read from any chord is the same loop: compare them from one fixed rotation - from the home chord
    when it has one, else the alphabetically first."""
    parts = loop.split(" – ")
    turns = [parts[i:] + parts[:i] for i in range(len(parts))]
    home = [t for t in turns if t[0] in ("I", "i", "I5", "i5")]
    return " – ".join(min(home or turns))


def progressions(rows):
    """[(loop in numerals, tracks it's in)] for loops in CHORD_LOOP_MIN or more tracks."""
    seen = Counter()
    for t in rows:
        for lp in set(_rotate(x) for x in t["f"]["loops"] if " – " in x):  # one chord on repeat isn't a loop
            seen[lp] += 1
    return [(lp, n) for lp, n in seen.most_common() if n >= CHORD_LOOP_MIN]


def typical(rows, rng):
    """(most typical track, outlier, {name: [audible differences from your usual]})."""
    if len(rows) < MIN_TRACKS:
        return None, None, {}
    diffs = {}
    for t in rows:
        d = []
        for x in rng:
            v = t["f"].get(x["key"])
            if v is not None and _gap(x["key"], v, x["median"]) > 1 and not (x["q1"] <= v <= x["q3"]):
                lo, hi = WORDS[x["key"]]
                d.append(f"{hi if v > x['median'] else lo} ({_num_fmt(x['key'], v)}{x['unit']})")
        diffs[t["name"]] = d
    order = sorted(rows, key=lambda t: len(diffs[t["name"]]))
    out = order[-1] if len(diffs[order[-1]["name"]]) >= 3 else None
    return order[0], out, diffs


def sessions(rows):
    """[{session, when, tracks}] in time order, and what changed audibly from the first to the latest."""
    by = {}
    for t in rows:
        by.setdefault(t["session"], []).append(t)
    ss = sorted(({"session": k, "when": min(t["when"] for t in v), "tracks": v, "title": v[0].get("title", "")}
                 for k, v in by.items()), key=lambda s: s["when"])
    drift = []
    if len(ss) >= 2:
        first, last = ss[0]["tracks"], ss[-1]["tracks"]
        for key, name, unit, _, _, _ in NUMERIC:
            a = [t["f"][key] for t in first if t["f"].get(key) is not None]
            b = [t["f"][key] for t in last if t["f"].get(key) is not None]
            if a and b:
                ma_, mb = statistics.median(a), statistics.median(b)
                if _gap(key, ma_, mb) > 1:
                    lo, hi = MORE_WORDS[key]
                    drift.append((name, f"{_num_fmt(key, ma_)} → {_num_fmt(key, mb)}{unit}", hi if mb > ma_ else lo))
        fm = sum(t["f"]["minor"] for t in first) / len(first)
        lm = sum(t["f"]["minor"] for t in last) / len(last)
        if abs(fm - lm) >= 0.5:
            drift.append(("Keys", f"mostly {'minor' if fm > lm else 'major'} → mostly {'major' if fm > lm else 'minor'}",
                          "brighter" if fm > lm else "darker"))
    return ss, drift


def unexplored(rows, rng, vocab):
    """Ideas from what no track does yet."""
    if len(rows) < MIN_TRACKS:
        return []
    out = []
    fs = [t["f"] for t in rows]
    if all(f["minor"] for f in fs):
        out.append(("Major key", "Every track is in a minor key. Write one in the relative major: same notes, a "
                                 "brighter home."))
    elif not any(f["minor"] for f in fs):
        out.append(("Minor key", "Every track is in a major key. Try its relative minor: same notes, a darker home."))
    if all(f["meter"] == "4/4" for f in fs):
        out.append(("Time signature", "Everything is in 4/4. A 3/4 or 6/8 sway, or one bar of 2/4 before a chorus, "
                                      "would stand out in your catalogue."))
    if not any(f["modulates"] for f in fs):
        out.append(("Key change", "No track changes key. Lift the last chorus a whole step, or move a bridge to the "
                                  "relative key."))
    if all(f["drums"] for f in fs):
        out.append(("Arrangement", "Drums play in every track. A drumless version, or a long breakdown, would show "
                                   "another side of your writing."))
    elif not any(f["drums"] for f in fs):
        out.append(("Arrangement", "No track has drums. Try a simple beat under one, even just a kick and a shaker."))
    tempo = next((x for x in rng if x["key"] == "tempo"), None)
    if tempo and tempo["hi"] / tempo["lo"] < 1.25:
        out.append(("Tempo", f"All your tempos sit between {tempo['lo']:.0f} and {tempo['hi']:.0f} BPM. Try one well "
                             f"outside: under {tempo['lo'] * 0.8:.0f} or over {tempo['hi'] * 1.2:.0f} BPM."))
    have = {k for k, _, _ in vocab}
    major = sum(not f["minor"] for f in fs) >= len(fs) / 2
    missing = [x for x in (["ii", "iii", "bVII", "iv"] if major else ["ii°", "iv", "v", "bII"]) if x not in have]
    if missing:
        out.append(("Chords", f"You don't use {', '.join(missing[:2])}. "
                              + ("A ii before the V, or a borrowed iv, adds colour without leaving the key."
                                 if major else "A minor iv, or the Phrygian bII, darkens a minor progression.")))
    return out


def in_short(rows, sig, rng):
    """Your style in two or three plain sentences."""
    n = len(rows)
    f = [t["f"] for t in rows]
    tempo = next((x for x in rng if x["key"] == "tempo"), None)
    energy = next((x for x in rng if x["key"] == "energy"), None)
    minor = sum(x["minor"] for x in f)
    keys = ("only in minor keys" if minor == n else "only in major keys" if minor == 0 else
            "mostly in minor keys" if minor >= MAJORITY * n else "mostly in major keys" if n - minor >= MAJORITY * n
            else "in major and minor keys alike")
    parts = [f"Across {n} tracks you write {keys}"
             + (f", usually around {tempo['median']:.0f} BPM" if tempo else "") + "."]
    if energy:
        lo, hi = WORDS["energy"]
        level = energy["median"]
        parts.append(f"The feel is {lo if level <= 4 else hi if level >= 7 else 'in between, neither calm nor intense'}"
                     + (f", and steady from track to track." if energy["steady"] else ", and it moves from track to track."))
    if sig:
        parts.append(f"{len(sig)} habit{'s' if len(sig) != 1 else ''} stand{'s' if len(sig) == 1 else ''} out below.")
    elif n < MIN_TRACKS:
        parts.append(f"Add {MIN_TRACKS - n} more and we can tell your habits from coincidence.")
    return " ".join(parts)


def profile(tracks):
    """tracks: [{"r": result, "name", "session", "when", "title"?}] -> everything the style page shows."""
    rows = [dict(t, f=features(t["r"])) for t in tracks if "error" not in t["r"]]
    if not rows:
        return None
    seen = Counter()
    for t in rows:  # the same file name in two sessions stays two tracks
        seen[t["name"]] += 1
        if seen[t["name"]] > 1:
            t["name"] = f'{t["name"]} ({seen[t["name"]]})'
    rows.sort(key=lambda t: t["when"])
    rng = ranges(rows)
    sig = signatures(rows, rng)
    vocab = vocabulary(rows)
    ss, drift = sessions(rows)
    typ, out, diffs = typical(rows, rng)
    return {"tracks": len(rows), "sessions": ss, "drift": drift, "ranges": rng, "signatures": sig,
            "vocabulary": vocab, "progressions": progressions(rows), "typical": typ, "outlier": out, "diffs": diffs,
            "unexplored": unexplored(rows, rng, vocab), "in_short": in_short(rows, sig, rng), "rows": rows,
            "enough": len(rows) >= MIN_TRACKS}


def pictures(p):
    """The style page's charts."""
    if not p:
        return {}
    rows = p["rows"]
    pics = {"strips": charts.style_strips(p["ranges"], order=[t["name"] for t in rows]),
            "keys": charts.camelot_wheel(dict(ma.CAMELOT_NAMES), counts=Counter(t["f"]["camelot"] for t in rows),
                                         size=200, label="Your keys on the Camelot wheel"),
            "mood": charts.mood_map([(str(i + 1), t["f"]["mood"], t["f"]["arousal"], f"{t['name']}: "
                                      f"{(t['r'].get('feel') or {}).get('mood', '')}")
                                     for i, t in enumerate(rows) if t["f"]["mood"] is not None
                                     and t["f"]["arousal"] is not None],
                                    size=220, path=False, tags=True, label="Each track on the energy/mood map"),
            "vocab": charts.hbars([(k, n, f"{k}: in {n} of {len(rows)} tracks, {100 * s:.0f}% of all chord time")
                                   for k, n, s in p["vocabulary"][:10]], width=420, unit=" tracks", top=len(rows),
                                  label="Chords you use, as numerals, by how many tracks use them")}
    if len(p["sessions"]) >= 2:
        whens = [t["when"] for t in rows]
        for key, label in (("tempo", "Tempo over time"), ("energy", "Energy over time")):
            pts = [(t["when"], t["f"][key], f"{t['name']}: {_num_fmt(key, t['f'][key])}")
                   for t in rows if t["f"].get(key) is not None]
            if len(pts) >= 2:
                pics[key + "_time"] = charts.time_series(pts, min(whens), max(whens), label=label)
    return pics
