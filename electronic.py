"""The producer's view: what electronic and other produced music is judged by, beyond key, chords and form.

- Sidechain pumping: sustained sounds ducking after every beat and swelling back. Measured on the mid band of the
  drumless audio (300 Hz-4 kHz, above the kick), as the average shape of one beat across the whole track.
- The low end in mono: club systems and phone speakers play the bass in mono, so a wide or out-of-phase low end
  partly cancels. Measured as the left/right correlation below 120 Hz (plus the mids and highs, for width).
- Punch: the peak-to-loudness ratio (true peak minus LUFS), the standard measure of how hard a master is limited.
- A dance arrangement map: intro, build, drop, breakdown, outro, with each section's length in bars, how far the
  drop jumps in level, and whether sections land on the 8-bar phrases DJs count in.
- A DJ's mixing range: the tempos and Camelot keys this track blends with.

Every judgement has a stated threshold below, and each is tested with a track that has the trait and one that
doesn't (selftest.py, producer checks).
"""

import numpy as np

PUMP_BAND = (300, 4000)   # Hz: above the kick's body, where pads, chords and synths sit
PUMP_DB = 3.0             # a 3 dB duck is clearly audible as pumping (about 3 just-noticeable loudness steps)
HEAVY_PUMP_DB = 6.0       # the obvious, "breathing" French-house kind
PUMP_NEAR_DB = 1.5        # "at the top" / "at the bottom" of the beat's level: within 1.5 dB of it
PUMP_SLOW_RISE = 1.5      # pumping falls fast and swells back slowly: the rise takes at least 1.5x the fall
LOW_END_HZ = 120          # mastering rule of thumb: keep everything below ~100-150 Hz in mono
LOW_END_CORR = 0.6        # below this the low end is audibly wide: part of it cancels on a mono system
LOW_END_SHARE = 0.02      # judge the low end only when it carries at least 2% of the energy
MID_HIGH_HZ = 2000        # mids / highs split for width by band
DROP_DB = 3.0             # a drop has to be at least this much louder than what comes before it to land
LEVEL_NEAR_TOP_DB = 3.0   # a section within 3 dB of the loudest one is "full"
BUILD_RISE_DB = 3.0       # a build rises at least 3 dB from its first third to its last
MUSIC_GONE_DB = 12.0      # harmony (chords, pads) 12 dB below its fullest section: drums alone, in effect
PHRASE_BARS = 8           # dance music moves in 8-bar phrases (and 16, 32)
DJ_PITCH = 0.06           # DJs can usually nudge tempo by about 6% before it sounds wrong
GRID_STEADY_PCT = 2.5     # beat-to-beat variation below this: played to a click or programmed
CLUB_GENRES = {"electronic", "house", "techno", "trance", "dnb", "dubstep", "ukg"}
PRODUCED_GENRES = CLUB_GENRES | {"synthwave", "lofi", "trap", "hiphop"}


def pump(y_harm, sr, beat_times, hop=128):
    """Sidechain pumping: does the sustained, non-drum sound duck right after each beat and swell back before the
    next? Returns {"depth_db", "shape"} or None when there's nothing sustained to measure or too few beats."""
    import librosa
    beat_times = np.asarray(beat_times, dtype=float)
    if len(beat_times) < 17:
        return None
    S = np.abs(librosa.stft(np.asarray(y_harm, dtype=np.float32), n_fft=1024, hop_length=hop)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    band = S[(f >= PUMP_BAND[0]) & (f < PUMP_BAND[1])].sum(axis=0)
    t = librosa.frames_to_time(np.arange(len(band)), sr=sr, hop_length=hop)
    db = 10 * np.log10(band + 1e-10)
    phases = 24
    profiles, levels = [], []
    for a, b in zip(beat_times[:-1], beat_times[1:]):
        if b - a <= 0:
            continue
        p = np.interp(np.linspace(a, b, phases, endpoint=False), t, db)
        profiles.append(p - p.max())
        levels.append(p.max())
    if len(profiles) < 16:
        return None
    levels = np.array(levels)
    active = levels >= np.percentile(levels, 90) - 20  # beats where something sustained is playing
    if active.sum() < 16:
        return None
    prof = np.median(np.array(profiles)[active], axis=0)
    # The beat's level shape, read round the cycle (so it doesn't matter where the tracker put the beat): how
    # long the level takes to fall from its top to its bottom, and to swell back. Pumping ducks fast and swells
    # back slowly; a stab or a struck chord does the opposite (a sharp attack, a slow decay).
    lo, hi = float(prof.min()), float(prof.max())
    depth = hi - lo
    dip = int(np.argmin(prof))
    ring = np.concatenate([prof, prof, prof])
    rise = next((k for k in range(1, phases + 1) if ring[phases + dip + k] >= hi - PUMP_NEAR_DB), phases)
    fall = next((k for k in range(1, phases + 1) if ring[phases + dip - k] >= hi - PUMP_NEAR_DB), phases)
    pumping = depth >= PUMP_DB and rise >= PUMP_SLOW_RISE * fall
    return {"depth_db": round(depth if pumping else 0.0, 1), "measured_db": round(depth, 1),
            "fall_beats": round(fall / phases, 2), "rise_beats": round(rise / phases, 2),
            "shape": [round(float(x), 1) for x in prof],
            "kind": "heavy" if pumping and depth >= HEAVY_PUMP_DB else "gentle" if pumping else None}


def band_widths(ch, sr):
    """Left/right correlation in three bands (low < 120 Hz, mids, highs > 2 kHz) and the low end's share of the
    energy. None for a mono file."""
    import scipy.signal as ss
    if ch.shape[0] < 2:
        return None
    x = ch[:2].astype(np.float64)
    total = float(np.sum(x ** 2)) + 1e-12
    out = {}
    for name, kind, edges in (("low", "lowpass", LOW_END_HZ), ("mid", "bandpass", (LOW_END_HZ, MID_HIGH_HZ)),
                              ("high", "highpass", MID_HIGH_HZ)):
        part = ss.sosfilt(ss.butter(4, edges, btype=kind, fs=sr, output="sos"), x, axis=1)
        den = np.sqrt(np.sum(part[0] ** 2) * np.sum(part[1] ** 2))
        out[name] = round(float(np.sum(part[0] * part[1]) / den), 2) if den > 0 else None
        if name == "low":
            out["low_share"] = round(float(np.sum(part ** 2)) / total, 3)
    return out


def low_end_wide(st):
    """True when there's real low end and it isn't mono."""
    w = (st or {}).get("band_widths") or {}
    return bool(w and w.get("low") is not None and w.get("low_share", 0) >= LOW_END_SHARE and w["low"] < LOW_END_CORR)


def is_dance(r):
    """Built for a club or a DJ: the musician says so (genre, or a synth), or the kick is four-on-the-floor."""
    ctx = r.get("context") or {}
    genre_key = ctx.get("genre_key")
    traits = (r.get("drums") or {}).get("traits") or []
    return bool((r.get("drums") or {}).get("present") and r.get("tempo_bpm")
                and (genre_key in CLUB_GENRES or "four-on-the-floor" in traits
                     or ("synth" in (ctx.get("keys") or []) and not genre_key)))


def is_produced(r):
    """Programmed or played to a click, with drums: mixing measures (sidechain, mono low end, punch) apply."""
    rh = r.get("rhythm") or {}
    ctx = r.get("context") or {}
    genre_key = ctx.get("genre_key")
    return is_dance(r) or bool((r.get("drums") or {}).get("present") and (
        genre_key in PRODUCED_GENRES or "synth" in (ctx.get("keys") or [])
        or (rh.get("tempo_variation_pct") is not None and rh["tempo_variation_pct"] < GRID_STEADY_PCT)))


def _rise_db(line, dur, s, e):
    """How much the level climbs inside a section: last third against first third, in dB (energy_line is 0-1
    over 40 dB)."""
    n = len(line)
    a, b = int(n * s / dur), max(int(n * e / dur), int(n * s / dur) + 3)
    seg = np.array(line[a:b])
    if len(seg) < 3:
        return 0.0
    k = max(1, len(seg) // 3)
    return float(40 * (seg[-k:].mean() - seg[:k].mean()))


def dance_map(r):
    """Intro, build, drop, breakdown, outro: each section's role in a dance arrangement and its length in bars.
    Roles come from the drums, the bass and each section's level: a drop is a full section (drums on, within 3 dB
    of the loudest), a breakdown is a middle section without drums, a build climbs into a drop, an intro and an
    outro frame the first and last drop."""
    secs, tempo = r.get("sections") or [], r.get("tempo_bpm")
    if not secs or not tempo or not (r.get("drums") or {}).get("present"):
        return None
    if any("loudness_db" not in s for s in secs):
        return None  # analysed by an older version
    bpb = (r.get("meter") or {}).get("beats_per_bar") or 4
    bar_sec = 60 / tempo * bpb
    loudest = max(s["loudness_db"] for s in secs)
    most_harmony = max(s["harmony_db"] for s in secs)
    has_music = lambda s: s["harmony_db"] >= most_harmony - MUSIC_GONE_DB
    line, dur = (r.get("sound") or {}).get("energy_line") or [], r.get("duration_sec") or 0
    rises = [_rise_db(line, dur, s["start"], s["end"]) if line and dur else 0.0 for s in secs]
    full = [bool(s.get("drums")) and has_music(s) and s["loudness_db"] >= loudest - LEVEL_NEAR_TOP_DB for s in secs]
    # a build climbs into a full section (a drop starts at full level, so it doesn't climb inside itself)
    # (from the end backwards, so a build leads into a real drop, not into another build)
    build = [False] * len(secs)
    for i in range(len(secs) - 2, -1, -1):
        build[i] = full[i + 1] and not build[i + 1] and rises[i] >= BUILD_RISE_DB
    full = [f and not b for f, b in zip(full, build)]
    if not any(full):
        return None
    first, last = full.index(True), len(full) - 1 - full[::-1].index(True)
    out = []
    for i, s in enumerate(secs):
        rise = rises[i]
        if full[i]:
            role = "drop"
        elif build[i]:
            role = "build"
        elif i < first:
            role = "intro"
        elif i > last:
            role = "outro"
        elif not s.get("drums"):
            role = "breakdown"
        else:
            role = "groove"
        bars = (s["end"] - s["start"]) / bar_sec
        out.append({"label": s["label"], "start": s["start"], "end": s["end"], "role": role,
                    "bars": int(round(bars)), "level_db": round(float(s["loudness_db"] - loudest), 1),
                    "beat_only": bool(s.get("drums") and not has_music(s)), "rise_db": round(float(rise), 1)})
    # A drop's impact: how much louder it is than the whole stretch leading into it (the breakdown and build since
    # the last drop, or the intro) - a riser can be as loud as the drop at its very end, so the build alone
    # would undersell a drop that follows a long, quiet breakdown.
    for i, x in enumerate(out):
        if x["role"] != "drop" or i == 0 or out[i - 1]["role"] == "drop":
            continue
        j = i
        while j > 0 and out[j - 1]["role"] != "drop":
            j -= 1
        run = out[j:i]
        w = np.array([max(0.1, y["end"] - y["start"]) for y in run])
        before = 10 * np.log10(np.sum(w * 10 ** (np.array([y["level_db"] for y in run]) / 10)) / w.sum())
        x["impact_db"] = round(float(x["level_db"] - before), 1)
    drops = [x for x in out if x["role"] == "drop"]
    # bars counted from the start: the first downbeat we heard tells us where bar lines fall (it may be a few
    # bars in, if the track opens quietly)
    phase = (r.get("downbeat_sec") or 0.0) % bar_sec
    first_drop = drops[0]["start"] if drops and drops[0] is not out[0] else None
    lengths = [x["bars"] for x in out[:-1] if x["bars"] >= 4]  # the last section's length is where the file ends
    return {"sections": out, "bar_sec": round(bar_sec, 3),
            "first_drop_sec": first_drop,
            "first_drop_bar": int(round((first_drop - phase) / bar_sec)) + 1 if first_drop is not None else None,
            "impact_db": max((x.get("impact_db") for x in drops if x.get("impact_db") is not None), default=None),
            "has_breakdown": any(x["role"] == "breakdown" for x in out),
            "beat_only_intro": out[0]["role"] == "intro" and out[0]["beat_only"],
            "on_phrase_pct": round(100 * sum(1 for b in lengths if b % PHRASE_BARS in (0, 1, PHRASE_BARS - 1))
                                   / len(lengths)) if lengths else None}


def dj_range(r):
    """The tempos and keys this track mixes with: within the pitch fader's ~6%, a Camelot neighbour key."""
    t = r.get("tempo_bpm")
    if not t or r.get("tempo_approximate"):
        return None
    return {"bpm_low": int(round(t * (1 - DJ_PITCH))), "bpm_high": int(round(t * (1 + DJ_PITCH)))}


def producer_view(r):
    """Everything above, for one analysed track (None when it isn't produced music with drums)."""
    if not is_produced(r):
        return None
    st = (r.get("sound") or {}).get("streaming") or {}
    v = {"dance": is_dance(r), "pump": r.get("pump"), "dj": dj_range(r),
         "kick": [t for t in (r.get("drums") or {}).get("traits") or []
                  if t in ("four-on-the-floor", "half-time feel (snare on 3)", "syncopated kick",
                           "backbeat (snare on 2 and 4)")],
         "widths": st.get("band_widths"), "low_end_wide": low_end_wide(st),
         "plr_db": round(st["true_peak_db"] - st["lufs"], 1) if st.get("lufs") is not None else None}
    if v["dance"]:
        v["map"] = dance_map(r)
    return v


def producer_ideas(r, v):
    """Ideas that only make sense for produced music, each tied to a measurement that shows the gap."""
    ideas = []
    if not v:
        return ideas
    p = v.get("pump") or {}
    if v["dance"] and "four-on-the-floor" in v["kick"] and p and not p.get("kind"):
        ideas.append({"area": "production", "idea": "The kick lands on every beat but the pads and synths don't move "
                      "with it. Sidechain them to the kick (a compressor keyed from the kick, 3-6 dB of gain "
                      "reduction, release timed to swell back by the next beat) and the groove starts to breathe."})
    if v["low_end_wide"]:
        ideas.append({"area": "production", "idea": f"The low end is wide (left/right correlation "
                      f"{v['widths']['low']} below {LOW_END_HZ} Hz). Club systems and phones play the bass in mono, "
                      f"so part of it cancels there. Keep everything below about {LOW_END_HZ} Hz in mono (a utility "
                      "or mid/side EQ on the bass and the master) and put the width in the mids and highs."})
    m = v.get("map")
    if m:
        dur = r.get("duration_sec") or 0
        if not m["has_breakdown"] and dur > 150:
            ideas.append({"area": "arrangement", "idea": "The drums never drop out. A breakdown (8 or 16 bars with "
                          "the kick gone, the pads or the hook alone) followed by a build gives the next drop "
                          "something to land on."})
        if m.get("impact_db") is not None and m["impact_db"] < DROP_DB:
            ideas.append({"area": "arrangement", "idea": f"The drop is only {m['impact_db']} dB louder than what "
                          "comes before it, so it doesn't land as hard as it could. Take the kick and the low end out "
                          "of the bar before it, filter the build (a high-pass sweep), or thin out the breakdown."})
        if not m["beat_only_intro"] and m["sections"][0]["role"] in ("intro", "drop", "groove"):
            ideas.append({"area": "arrangement", "idea": "For DJs: 16 or 32 bars of drums alone at the start (and "
                          "the end) make a track easy to mix in and out. Yours brings the bass or the music in "
                          "straight away."})
        if m["on_phrase_pct"] is not None and m["on_phrase_pct"] < 50:
            ideas.append({"area": "arrangement", "idea": "Most of your sections don't land on 8-bar phrases. Dance "
                          "music moves in 8, 16 and 32 bars, and DJs and dancers count in them, so changes that "
                          "fall on those boundaries feel right."})
    return ideas
