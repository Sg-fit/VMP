"""A musician's saved work over time: a summary for a chosen time frame, how it compares with the frame before,
what changed within it, the points for the trend charts, and the timeline's groups.

Only differences big enough to matter are called changes - the same just-noticeable steps the comparison with a
song you admire uses (musicanalyze.TEMPO_JND etc.), and never from fewer than MIN_EACH_SIDE tracks on each side.
"""

import time
from collections import Counter
from datetime import datetime, timedelta

FRAMES = {"7d": ("7 days", 7), "30d": ("30 days", 30), "90d": ("3 months", 90), "365d": ("12 months", 365),
          "all": ("All time", None)}
DEFAULT_FRAME = "all"
TEMPO_CHANGE = 0.04     # a 4% tempo difference is noticeable (musicanalyze.TEMPO_JND)
ENERGY_CHANGE = 1.0     # one step on the 1-10 energy scale
MOOD_CHANGE = 10        # positivity 0-100: half a mood band (~20 points per band)
LOUDNESS_CHANGE = 1.0   # 1 LU, about the smallest loudness change people hear
MINOR_CHANGE = 0.25     # a quarter of the tracks moving from major to minor (or back)
LENGTH_CHANGE = 1.25    # a quarter longer or shorter
MIN_EACH_SIDE = 2       # a "before" and an "after" each need at least two tracks


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def tracks_of(items):
    """Every analysed track in these saved items, each with the time it was saved (oldest first)."""
    out = []
    for it in items:
        sm = it.get("summary") or {}
        if sm.get("kind") == "tags":
            continue
        for t in sm.get("tracks") or []:
            if not t.get("error"):
                out.append({**t, "when": it["created"], "job_id": it["job_id"]})
    return sorted(out, key=lambda t: t["when"])


def split_frame(items, frame, now=None):
    """(items in the frame, items in the frame of the same length just before it; None for all time)."""
    days = FRAMES.get(frame, FRAMES[DEFAULT_FRAME])[1]
    if days is None:
        return list(items), None
    now = now or time.time()
    start, before = now - days * 86400, now - 2 * days * 86400
    return ([it for it in items if it["created"] >= start],
            [it for it in items if before <= it["created"] < start])


def summarise(items):
    tracks = tracks_of(items)
    tags = [it for it in items if (it.get("summary") or {}).get("kind") == "tags"]
    reliable = [t["tempo_bpm"] for t in tracks if t.get("tempo_bpm") and not t.get("tempo_approximate")]
    keys = Counter(t["key"] for t in tracks if t.get("key"))
    minor = [("minor" in t["key"]) for t in tracks if t.get("key")]
    return {
        "analyses": len(items) - len(tags), "tag_lists": len(tags), "tracks": len(tracks),
        "minutes": round(sum(t.get("duration_sec") or 0 for t in tracks) / 60, 1),
        "tempo": _median(reliable), "energy": _mean([t.get("energy") for t in tracks]),
        "valence": _mean([t.get("valence") for t in tracks]), "lufs": _median([t.get("lufs") for t in tracks]),
        "length": _median([t.get("duration_sec") for t in tracks]),
        "minor_share": sum(minor) / len(minor) if minor else None,
        "top_key": keys.most_common(1)[0][0] if keys else None, "keys_used": len(keys),
        "drums_share": sum(bool(t.get("drums")) for t in tracks) / len(tracks) if tracks else None,
    }


def _direction(now_, before, kind):
    """'up', 'down' or 'same' for one measure, by its noticeable step; None when either side is missing."""
    if now_ is None or before is None:
        return None
    if kind == "tempo" or kind == "length":
        step = TEMPO_CHANGE if kind == "tempo" else LENGTH_CHANGE - 1
        if not before:
            return None
        ratio = now_ / before
        return "up" if ratio > 1 + step else "down" if ratio < 1 - step else "same"
    step = {"energy": ENERGY_CHANGE, "valence": MOOD_CHANGE, "lufs": LOUDNESS_CHANGE, "minor_share": MINOR_CHANGE}[kind]
    d = now_ - before
    return "up" if d >= step else "down" if d <= -step else "same"


def compare(cur, prev):
    """Each summary measure against the previous frame: {measure: direction} (only where both have data)."""
    if not prev or not prev["tracks"] or not cur["tracks"]:
        return {}
    out = {m: _direction(cur[m], prev[m], m) for m in ("tempo", "energy", "valence", "lufs", "minor_share", "length")}
    return {m: d for m, d in out.items() if d}


CHANGE_WORDS = {
    ("tempo", "up"): "Your tempos have gone up: {after:.0f} BPM typical lately, against {before:.0f} before.",
    ("tempo", "down"): "Your tempos have come down: {after:.0f} BPM typical lately, against {before:.0f} before.",
    ("energy", "up"): "More energy lately: {after:.1f}/10 on average, against {before:.1f} before.",
    ("energy", "down"): "Calmer lately: energy {after:.1f}/10 on average, against {before:.1f} before.",
    ("valence", "up"): "Brighter in mood lately (positivity {after:.0f} against {before:.0f}).",
    ("valence", "down"): "Darker in mood lately (positivity {after:.0f} against {before:.0f}).",
    ("lufs", "up"): "Your recordings are louder lately: {after:.1f} LUFS typical, against {before:.1f}.",
    ("lufs", "down"): "Your recordings are quieter lately: {after:.1f} LUFS typical, against {before:.1f}.",
    ("minor_share", "up"): "More minor keys lately: {after:.0%} of tracks, against {before:.0%} before.",
    ("minor_share", "down"): "More major keys lately: {after:.0%} of tracks are minor, against {before:.0%} before.",
    ("length", "up"): "Longer tracks lately: {after_t} typical, against {before_t}.",
    ("length", "down"): "Shorter tracks lately: {after_t} typical, against {before_t}.",
}


def _mmss(sec):
    sec = int(round(sec or 0))
    return f"{sec // 60}:{sec % 60:02d}"


def changes(items):
    """What changed within these items: the later half of the tracks (by date) against the earlier half. Only
    noticeable changes, and only with MIN_EACH_SIDE tracks or more on each side."""
    tracks = tracks_of(items)
    if len(tracks) < 2 * MIN_EACH_SIDE:
        return []
    half = len(tracks) // 2
    early, late = tracks[:half], tracks[half:]
    if early[-1]["when"] == late[0]["when"] and len({t["when"] for t in tracks}) < 2:
        return []  # all saved at once: no "before" and "after"
    pick = {
        "tempo": lambda ts: _median([t["tempo_bpm"] for t in ts if t.get("tempo_bpm") and not t.get("tempo_approximate")]),
        "energy": lambda ts: _mean([t.get("energy") for t in ts]),
        "valence": lambda ts: _mean([t.get("valence") for t in ts]),
        "lufs": lambda ts: _median([t.get("lufs") for t in ts]),
        "minor_share": lambda ts: (lambda m: sum(m) / len(m) if m else None)(
            [("minor" in t["key"]) for t in ts if t.get("key")]),
        "length": lambda ts: _median([t.get("duration_sec") for t in ts]),
    }
    count = {"tempo": lambda ts: sum(1 for t in ts if t.get("tempo_bpm") and not t.get("tempo_approximate")),
             "lufs": lambda ts: sum(1 for t in ts if t.get("lufs") is not None)}
    out = []
    for m, f in pick.items():
        n = count.get(m, len)
        if n(early) < MIN_EACH_SIDE or n(late) < MIN_EACH_SIDE:
            continue
        b, a = f(early), f(late)
        d = _direction(a, b, m)
        if d in ("up", "down"):
            out.append({"measure": m, "direction": d, "text": CHANGE_WORDS[(m, d)].format(
                after=a, before=b, after_t=_mmss(a), before_t=_mmss(b))})
    new_keys = sorted({t["key"] for t in late if t.get("key")} - {t["key"] for t in early if t.get("key")})
    if new_keys and len(early) >= MIN_EACH_SIDE:
        out.append({"measure": "keys", "direction": "new",
                    "text": "New keys lately: " + ", ".join(new_keys[:4]) + "."})
    return out


def series(items):
    """Points for the trend charts: [(time, value, tooltip)] per measure, plus keys as Camelot positions."""
    tracks = tracks_of(items)
    label = lambda t: f"{t['file']} · {datetime.fromtimestamp(t['when']).strftime('%d %b %Y')}"
    pts = lambda key, fmt: [(t["when"], t[key], f"{label(t)}: {fmt(t[key])}") for t in tracks if t.get(key) is not None]
    return {
        "tempo": [(t["when"], t["tempo_bpm"], f"{label(t)}: {t['tempo_bpm']:.0f} BPM") for t in tracks
                  if t.get("tempo_bpm") and not t.get("tempo_approximate")],
        "energy": pts("energy", lambda v: f"energy {v}/10"),
        "valence": pts("valence", lambda v: f"positivity {v}/100"),
        "lufs": pts("lufs", lambda v: f"{v} LUFS"),
        "keys": [(t["when"], int(t["camelot"][:-1]), t["camelot"][-1], f"{label(t)}: {t['key']} ({t['camelot']})")
                 for t in tracks if t.get("camelot")],
    }


def _bucket(frame, span_days):
    if frame == "7d" or (frame == "all" and span_days <= 14):
        return "day"
    if frame in ("30d", "90d") or (frame == "all" and span_days <= 120):
        return "week"
    return "month"


def _bucket_start(ts, kind):
    d = datetime.fromtimestamp(ts)
    if kind == "day":
        d = d.replace(hour=0, minute=0, second=0, microsecond=0)
    elif kind == "week":
        d = (d - timedelta(days=d.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        d = d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return d


def _bucket_label(d, kind):
    return (d.strftime("%a %d %b") if kind == "day" else f"Week of {d.day} {d.strftime('%b')}" if kind == "week"
            else d.strftime("%B %Y"))


def _next(d, kind):
    if kind == "day":
        return d + timedelta(days=1)
    if kind == "week":
        return d + timedelta(days=7)
    return (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def activity(items, frame, now=None):
    """Tracks analysed per day, week or month across the frame (empty periods included, so gaps show)."""
    if not items:
        return [], "month"
    now = now or time.time()
    days = FRAMES.get(frame, FRAMES[DEFAULT_FRAME])[1]
    first = now - days * 86400 if days else min(it["created"] for it in items)
    kind = _bucket(frame, (now - first) / 86400)
    counts = Counter()
    for it in items:
        sm = it.get("summary") or {}
        n = 0 if sm.get("kind") == "tags" else len([t for t in sm.get("tracks") or [] if not t.get("error")]) or 1
        counts[_bucket_start(it["created"], kind)] += n
    out, d, end = [], _bucket_start(first, kind), _bucket_start(now, kind)
    while d <= end and len(out) < 400:
        out.append((d.timestamp(), _bucket_label(d, kind), counts.get(d, 0),
                    d.strftime("%b %y" if kind == "month" else "%d %b")))
        d = _next(d, kind)
    return out, kind


def groups(items, frame):
    """The timeline's groups, newest first: by day, week or month depending on the frame."""
    if not items:
        return []
    span = (max(it["created"] for it in items) - min(it["created"] for it in items)) / 86400
    kind = _bucket(frame, span)
    out = []
    for it in sorted(items, key=lambda x: -x["created"]):
        start = _bucket_start(it["created"], kind)
        if not out or out[-1]["start"] != start:
            out.append({"start": start, "label": _bucket_label(start, kind), "items": []})
        out[-1]["items"].append(it)
    return out


def overview(items, frame, now=None):
    """Everything the history page shows above the timeline, for one time frame."""
    frame = frame if frame in FRAMES else DEFAULT_FRAME
    cur_items, prev_items = split_frame(items, frame, now)
    cur = summarise(cur_items)
    prev = summarise(prev_items) if prev_items is not None else None
    act, kind = activity(cur_items, frame, now)
    now = now or time.time()
    days = FRAMES[frame][1]
    t0 = now - days * 86400 if days else min((it["created"] for it in cur_items), default=now - 86400)
    return {"frame": frame, "t0": t0, "t1": now, "frames": {k: v[0] for k, v in FRAMES.items()}, "frame_label": FRAMES[frame][0], "items": cur_items, "summary": cur, "previous": prev,
            "vs_previous": compare(cur, prev), "changes": changes(cur_items), "series": series(cur_items),
            "activity": act, "bucket": kind, "groups": groups(cur_items, frame)}


def tiles(view):
    """The summary tiles: a value each, and how it compares with the previous frame of the same length (in words
    and an arrow, never colour alone)."""
    cur, prev, vs = view["summary"], view["previous"], view["vs_previous"]
    before = f"previous {FRAMES[view['frame']][0]}" if prev is not None else None

    def note(measure, fmt):
        d = vs.get(measure)
        if not d or prev.get(measure) is None:
            return ""
        return {"up": f"▲ up from {fmt(prev[measure])}", "down": f"▼ down from {fmt(prev[measure])}",
                "same": f"≈ about the same as the {before}"}[d]
    out = []
    if cur["tracks"] or cur["tag_lists"]:
        n = ""
        if prev is not None:
            d = cur["tracks"] - prev["tracks"]
            n = f"{'+' if d > 0 else '−' if d < 0 else '±'}{abs(d)} vs the {before}"
        out.append({"label": "Tracks", "value": cur["tracks"],
                    "sub": f"{cur['analyses']} saved {'analysis' if cur['analyses'] == 1 else 'analyses'}"
                    + (f", {cur['tag_lists']} tag list{'' if cur['tag_lists'] == 1 else 's'}" if cur["tag_lists"] else ""),
                    "note": n})
        out.append({"label": "Minutes of music", "value": f"{cur['minutes']:g}", "sub": "", "note": ""})
    if cur["tempo"] is not None:
        out.append({"label": "Typical tempo", "value": f"{cur['tempo']:.0f}", "sub": "BPM (the middle value)",
                    "note": note("tempo", lambda v: f"{v:.0f}")})
    if cur["energy"] is not None:
        out.append({"label": "Energy", "value": f"{cur['energy']:.1f}", "sub": "out of 10, on average",
                    "note": note("energy", lambda v: f"{v:.1f}")})
    if cur["valence"] is not None:
        out.append({"label": "Mood", "value": f"{cur['valence']:.0f}", "sub": "positivity out of 100",
                    "note": note("valence", lambda v: f"{v:.0f}")})
    if cur["lufs"] is not None:
        out.append({"label": "Loudness", "value": f"{cur['lufs']:.1f}", "sub": "LUFS (the middle value)",
                    "note": note("lufs", lambda v: f"{v:.1f}")})
    if cur["minor_share"] is not None:
        out.append({"label": "In minor keys", "value": f"{cur['minor_share']:.0%}", "sub": "of tracks",
                    "note": note("minor_share", lambda v: f"{v:.0%}")})
    if cur["top_key"]:
        out.append({"label": "Most used key", "value": cur["top_key"], "sub": f"{cur['keys_used']} keys used",
                    "note": ""})
    return out
