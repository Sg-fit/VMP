"""A musician's memory: what the app learns from a signed-in user's own saved analyses, corrections and
feedback, and how it uses that - always visible, editable and switchable off on the "What we remember" page.

What it learns from (never the audio):
- habits: the keys, modes, tempos, time signatures, forms, chords, energy and loudness of their saved tracks,
  and the instruments and genre they usually tick;
- corrections they make to a result ("it's 3/4", "there's no bass");
- thumbs up / down on ideas, and which ideas they've already been shown;
- goals they write down.

How it's used - only ever to nudge, never to overrule what the audio clearly shows:
- the upload form is pre-filled with their usual line-up and genre (they can change it);
- a new track is compared with their own past ("your first track in a minor key", "faster than anything you've
  uploaded");
- close calls get a note: when the tempo reading is uncertain and half or double it sits where they usually
  play, we say so - the measured number stays;
- ideas they disliked are dropped, liked kinds come first, ones they've already seen go last;
- the AI gets a short profile so it can push them somewhere new instead of repeating itself.
"""

import re
from collections import Counter

MIN_PAST_TRACKS = 2       # below this there's no "usual" to compare with
HABIT_SHARE = 0.6         # an instrument/genre is "usual" when it was ticked for at least 60% of saved analyses


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def idea_key(area, text):
    """The same suggestion on different tracks has different chords and times in it ('try approach notes before
    D'); its first six words and its area identify the kind of idea."""
    words = re.findall(r"[a-z']+", (text or "").lower())[:6]
    return f"{(area or '').lower()}|{' '.join(words)}"


def build(tracks, contexts, feedback, corrections, goals="", references=(), seen=()):
    """tracks: per-track summaries of the user's saved analyses (accounts.job_summary rows, own tracks only);
    contexts: the 'About the music' answers of each saved analysis; feedback: [{area, text, vote}];
    corrections: [{field, value}]. Returns the memory dict (plain JSON, stored with each new job)."""
    ok = [t for t in tracks if not t.get("error")]
    tempos = sorted(t["tempo_bpm"] for t in ok if t.get("tempo_bpm") and not t.get("tempo_approximate"))
    lufs = [t["lufs"] for t in ok if t.get("lufs") is not None]
    chords = Counter()
    for t in ok:
        chords.update(t.get("chords") or {})
    inst = Counter(i for c in contexts if c for i in (c.get("keys") or []))
    genres = Counter(c.get("genre_key") for c in contexts if c and c.get("genre_key"))
    n_ctx = max(1, len([c for c in contexts if c]))
    taste = Counter()
    for f in feedback:
        taste[(f.get("area") or "").lower()] += 1 if f.get("vote", 0) > 0 else -1
    return {
        "tracks": len(ok),
        "keys": Counter(t["key"] for t in ok if t.get("key")).most_common(),
        "modes": Counter(t["mode"] for t in ok if t.get("mode")).most_common(),
        "meters": Counter(t["meter"] for t in ok if t.get("meter")).most_common(),
        "forms": Counter(t["form_name"] for t in ok if t.get("form_name")).most_common(),
        # a range needs at least two reliable tempos (approximate ones - weak pulse - don't count)
        "tempo": {"min": tempos[0], "max": tempos[-1], "median": tempos[len(tempos) // 2]} if len(tempos) >= 2 else None,
        "lufs_median": _median(lufs),
        "duration_median": _median([t["duration_sec"] for t in ok if t.get("duration_sec")]),
        "energy_avg": round(sum(t["energy"] for t in ok if t.get("energy")) / max(1, sum(1 for t in ok if t.get("energy"))), 1)
        if any(t.get("energy") for t in ok) else None,
        "chords": [c for c, _ in chords.most_common(12)],
        "all_chords": sorted(chords),
        "usual_instruments": [i for i, c in inst.most_common() if c >= HABIT_SHARE * n_ctx],
        "usual_genre": next((g for g, c in genres.most_common(1) if c >= HABIT_SHARE * n_ctx), None),
        "liked_areas": [a for a, s in taste.most_common() if s > 0],
        "disliked_areas": [a for a, s in sorted(taste.items(), key=lambda x: x[1]) if s < 0],
        "disliked_ideas": sorted({idea_key(f.get("area"), f.get("text")) for f in feedback if f.get("vote", 0) < 0}),
        "liked_ideas": sorted({idea_key(f.get("area"), f.get("text")) for f in feedback if f.get("vote", 0) > 0}),
        "corrections": [{"field": c["field"], "value": c["value"]} for c in corrections][-30:],
        "goals": (goals or "").strip()[:500],
        "references": list(dict.fromkeys(references))[-10:],
        "seen_ideas": sorted(set(seen))[:400],  # idea kinds already shown on their saved tracks
    }


def prefill(mem):
    """What to tick on the upload form: their usual line-up and genre."""
    if not mem:
        return {"instruments": [], "genre": None}
    return {"instruments": mem.get("usual_instruments") or [], "genre": mem.get("usual_genre")}


def compare_with_past(r, mem, fmt_time=lambda s: f"{int(s // 60)}:{int(s % 60):02d}"):
    """How this track differs from the musician's own saved tracks - only real firsts and clear departures."""
    if not mem or mem.get("tracks", 0) < MIN_PAST_TRACKS or "error" in r:
        return []
    out = []
    keys, modes, meters = dict(mem["keys"]), dict(mem["modes"]), dict(mem["meters"])
    minor = lambda k: "minor" in k
    if r.get("key") and r["key"] not in keys:
        if minor(r["key"]) and not any(minor(k) for k in keys):
            out.append(f"Your first saved track in a minor key ({r['key']}).")
        elif not minor(r["key"]) and not any(not minor(k) for k in keys):
            out.append(f"Your first saved track in a major key ({r['key']}).")
        else:
            out.append(f"A key you haven't used before: {r['key']}.")
    if r.get("mode") and r["mode"] not in modes and r.get("mode_confidence") != "low":
        out.append(f"New colour: {r['mode']} - none of your saved tracks used it.")
    m = (r.get("meter") or {}).get("label")
    if m and meters and m not in meters and (r.get("meter") or {}).get("confidence") == "detected":
        out.append(f"Your first track in {m}.")
    t, tm = r.get("tempo_bpm"), mem.get("tempo")
    if t and tm and not r.get("tempo_approximate"):
        if t > tm["max"] * 1.04:
            out.append(f"Faster than anything you've saved ({t:.0f} BPM; your range is {tm['min']:.0f}-{tm['max']:.0f}).")
        elif t < tm["min"] / 1.04:
            out.append(f"Slower than anything you've saved ({t:.0f} BPM; your range is {tm['min']:.0f}-{tm['max']:.0f}).")
    used = set(mem.get("all_chords") or [])
    new = [c["chord"] for c in r.get("chords_used") or [] if used and c["chord"] not in used and c["seconds"] >= 4]
    if new:
        out.append("Chords you haven't used in a saved track before: " + ", ".join(new[:5]) + ".")
    d, dm = r.get("duration_sec"), mem.get("duration_median")
    if d and dm and (d > 1.5 * dm or d < dm / 1.5):
        out.append(f"{'Longer' if d > dm else 'Shorter'} than usual for you: {fmt_time(d)} against a typical "
                   f"{fmt_time(dm)}.")
    st, lm = (r.get("sound") or {}).get("streaming") or {}, mem.get("lufs_median")
    if st.get("lufs") is not None and lm is not None and abs(st["lufs"] - lm) >= 3:
        out.append(f"{'Louder' if st['lufs'] > lm else 'Quieter'} than your usual ({st['lufs']} LUFS against a "
                   f"typical {lm}).")
    return out


def close_call_notes(r, mem):
    """Where the measurement was a close call and their history points one way, say so. Never changes results."""
    if not mem or mem.get("tracks", 0) < MIN_PAST_TRACKS:
        return []
    notes = []
    t, tm = r.get("tempo_bpm"), mem.get("tempo")
    if t and tm and r.get("tempo_approximate"):
        inside = lambda x: tm["min"] / 1.04 <= x <= tm["max"] * 1.04
        if not inside(t):
            alt = next((x for x in (t * 2, t / 2) if inside(x)), None)
            if alt:
                notes.append(f"The beat was hard to measure here, and you usually play between {tm['min']:.0f} and "
                             f"{tm['max']:.0f} BPM, so you may feel this at about {alt:.0f} BPM.")
    told = [c for c in mem.get("corrections") or [] if c["field"] == "drums" and c["value"] == "no"]
    if (r.get("drums") or {}).get("unsure") and len(told) >= 2:
        notes.append("We weren't sure about drums here. You've told us before that your recordings have none, so "
                     "it's probably the attack of the instrument.")
    return notes


def rank_ideas(ideas, mem, seen=None):
    """Drop ideas they disliked, put kinds they liked first and ones they've already been shown last."""
    if not mem:
        return ideas
    disliked, liked = set(mem.get("disliked_ideas") or []), set(mem.get("liked_ideas") or [])
    liked_areas, bad_areas = mem.get("liked_areas") or [], mem.get("disliked_areas") or []
    seen = set(mem.get("seen_ideas") or []) if seen is None else set(seen)
    kept = [i for i in ideas if idea_key(i.get("area"), i.get("idea")) not in disliked]

    def order(i):
        k, area = idea_key(i.get("area"), i.get("idea")), (i.get("area") or "").lower()
        return (k in seen and k not in liked,                      # already shown: last
                -(k in liked) - (area in liked_areas) + (area in bad_areas))
    return sorted(kept, key=order)


def ai_profile(mem):
    """A few lines for the AI: who this musician is and what has (not) helped them."""
    if not mem or mem.get("tracks", 0) < 1:
        return ""
    bits = []
    if mem.get("keys"):
        bits.append("usually in " + ", ".join(k for k, _ in mem["keys"][:3]))
    if mem.get("tempo"):
        bits.append(f"{mem['tempo']['min']:.0f}-{mem['tempo']['max']:.0f} BPM")
    if mem.get("meters"):
        bits.append("in " + ", ".join(m for m, _ in mem["meters"][:2]))
    if mem.get("chords"):
        bits.append("leaning on " + ", ".join(mem["chords"][:6]))
    lines = [f"About this musician (from {mem['tracks']} saved track(s)): " + "; ".join(bits) + "."]
    if mem.get("liked_areas"):
        lines.append("Ideas they found useful before: " + ", ".join(mem["liked_areas"][:5]) + ".")
    if mem.get("disliked_areas"):
        lines.append("Kinds of ideas that didn't help them: " + ", ".join(mem["disliked_areas"][:5]) + ".")
    if mem.get("goals"):
        lines.append(f"Their goals, in their words: {mem['goals']}")
    lines.append("Push them somewhere new relative to these habits rather than repeating what they already do.")
    return "\n".join(lines)
