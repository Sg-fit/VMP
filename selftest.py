#!/usr/bin/env python3
"""Self-test: generate music with KNOWN answers, analyse it, and check the results.

    python selftest.py            # analysis + AI checks (what deploy.sh runs on the server)
    python selftest.py --full     # also unit checks and a complete web-app run with a simulated AI
                                  # (run before shipping changes)

Nothing personal is used: all test audio is synthesised on the fly. Exit code 0 = everything passed.
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MUSIC_SEPARATION", "simple")  # same as the server default; fast

RESULTS = []  # (group, name, ok, detail)


def check(group, name, ok, detail=""):
    RESULTS.append((group, name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


# ---------------------------------------------------------------- synthetic music with known answers

SR = 22050
NOTE = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "Bb": 10, "B": 11}


def midi(name, octave):
    return 12 * (octave + 1) + NOTE[name]


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tone(freq, dur, amp, attack=0.01, decay=1.2, harmonics=(1, 0.5, 0.33)):
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1, t / attack) * np.exp(-t * decay)
    return amp * env * sum(h * np.sin(2 * np.pi * freq * (k + 1) * t) for k, h in enumerate(harmonics))


def noise(dur, amp, decay, hp=False, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(SR * dur))
    if hp:
        x = np.diff(x, prepend=0)  # crude high-pass for hi-hats
    t = np.arange(len(x)) / SR
    return amp * x * np.exp(-t * decay)


def place(buf, sig, at):
    i = int(at * SR)
    n = min(len(sig), len(buf) - i)
    if n > 0:
        buf[i:i + n] += sig[:n]


CHORD = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "5": [0, 7]}


def song(chords, bpm, beats_per_bar, bars_per_chord=1, loops=1, drums=True, waltz=False, melody=None,
         soft=False, tail=1.0):
    """chords: [(root, quality)], one per bar(s). Returns float32 audio."""
    beat = 60 / bpm
    bars = len(chords) * bars_per_chord * loops
    total = bars * beats_per_bar * beat + tail
    y = np.zeros(int(SR * total) + SR)
    attack = 0.06 if soft else 0.01
    bar = 0
    for _ in range(loops):
        for root, q in chords:
            for _ in range(bars_per_chord):
                t0 = bar * beats_per_bar * beat
                r = NOTE[root]
                bass = midi(root, 2)
                tones = [midi(root, 3) + iv for iv in CHORD[q]]
                place(y, tone(hz(bass), beats_per_bar * beat, 0.35, attack), t0)  # bass on beat 1
                if waltz:  # oom-pah-pah: chord on beats 2 and 3
                    for b in (1, 2):
                        for m_ in tones:
                            place(y, tone(hz(m_), beat * 0.9, 0.12, attack), t0 + b * beat)
                else:
                    for m_ in tones:
                        place(y, tone(hz(m_), beats_per_bar * beat, 0.12, attack), t0)
                if melody:
                    for k, n in enumerate(melody[bar % len(melody)]):
                        place(y, tone(hz(n), beat * 0.9, 0.16, attack, harmonics=(1, 0.3)), t0 + k * beat)
                if drums:
                    for b in range(beats_per_bar):
                        tb = t0 + b * beat
                        if b % 2 == 0:  # kick on 1 and 3
                            t = np.arange(int(SR * 0.15)) / SR
                            place(y, 0.9 * np.sin(2 * np.pi * 55 * t) * np.exp(-t * 25), tb)
                        else:            # snare on 2 and 4
                            place(y, noise(0.15, 0.5, 30, seed=b), tb)
                        for h in (0, 0.5):  # 8th-note hi-hats
                            place(y, noise(0.04, 0.08, 120, hp=True, seed=b + 10), tb + h * beat)
                bar += 1
    y = y / (np.abs(y).max() + 1e-9) * 0.8
    return y.astype(np.float32)


def write(path, y):
    import soundfile as sf
    sf.write(path, y, SR)
    return path


def write_m4a(path, y):
    """Encode AAC/M4A with PyAV, to test the m4a decoding path phones produce."""
    import av
    with av.open(str(path), "w") as out:
        st = out.add_stream("aac", rate=SR)
        st.layout = "mono"
        fr = av.AudioFrame.from_ndarray(y.reshape(1, -1), format="flt", layout="mono")
        fr.sample_rate = SR
        for p in st.encode(fr):
            out.mux(p)
        for p in st.encode(None):
            out.mux(p)
    return path


def _write_bytes(path, data):
    path.write_bytes(data)
    return path


def bassline_song(bpm=96, bars=8):
    """Drums + chords + a syncopated plucked bass line with repeated notes. Returns audio, true notes."""
    beat, e = 60 / bpm, 30 / bpm
    pattern = [(0, 1, 40), (1, 1, 40), (3, 1, 40), (4, 2, 43), (6, 1, 45), (7, 1, 45)]  # E E . E | G . A A
    y = song([("E", "m"), ("E", "m"), ("C", ""), ("D", "")], bpm, 4, loops=bars // 4, drums=True)
    bass, truth = np.zeros_like(y), []
    for b in range(bars):
        for s8, l8, n in pattern:
            start, dur = b * 4 * beat + s8 * e, l8 * e * 0.9
            t = np.arange(int(SR * dur)) / SR
            place(bass, 0.5 * np.minimum(1, t / 0.005) * np.exp(-t * 4) *
                  sum(h * np.sin(2 * np.pi * hz(n) * (k + 1) * t) for k, h in enumerate((1, 0.6, 0.3, 0.15))), start)
            truth.append((start, n))
    y = y + bass
    return (y / np.abs(y).max() * 0.8).astype(np.float32), truth


def pluck(freq, dur, amp=0.3):
    """A plucked string: a sharp attack and harmonics that fade faster the higher they are."""
    t = np.arange(int(SR * dur)) / SR
    sig = sum((0.7 ** k) * np.exp(-t * (2.5 + 1.8 * k)) * np.sin(2 * np.pi * freq * (k + 1) * t) for k in range(10))
    return amp * np.minimum(1, t / 0.002) * sig


def guitar_blues(bpm=140, choruses=3):
    """Solo fingerstyle guitar, 12-bar blues in A: a thumb line on the low strings (root and fifth, nothing
    below 100 Hz, like a real guitar on a phone) and chords on 2 and 4. No drums, no bass instrument."""
    beat = 60 / bpm
    form = [0, 0, 0, 0, 5, 5, 0, 0, 7, 5, 0, 7]
    root = {0: 45, 5: 50, 7: 52}  # A2, D3, E3
    y = np.zeros(int(SR * (choruses * 48 * beat + 2)))
    for c in range(choruses):
        for b, deg in enumerate(form):
            t0, r = (c * 12 + b) * 4 * beat, root[deg]
            for i, n in enumerate((r, r + 7, r, r + 7)):
                place(y, pluck(hz(n), beat * 0.95, 0.35), t0 + i * beat)
            for i in (1, 3):
                for iv in (12, 16, 19, 22):  # a dominant 7th an octave up, strummed
                    place(y, pluck(hz(r + iv), beat * 0.9, 0.12), t0 + i * beat + 0.001 * iv)
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def make_test_set(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    G4, A4, B4, D5 = midi("G", 4), midi("A", 4), midi("B", 4), midi("D", 5)
    pop_melody = [[B4, A4, G4, D5], [A4, B4, D5, A4], [G4, B4, A4, G4], [D5, B4, A4, G4]]
    return {
        "pop4": write(folder / "pop_G_100bpm_4-4.wav",
                      song([("G", ""), ("D", ""), ("E", "m"), ("C", "")], 100, 4, loops=6, melody=pop_melody)),
        "waltz3": write(folder / "waltz_A_120bpm_3-4.wav",
                        song([("A", ""), ("E", "7"), ("A", ""), ("D", "")], 120, 3, loops=5, drums=False,
                             waltz=True, soft=True)),
        "power": write(folder / "power_E_90bpm.wav",
                       song([("E", "5"), ("A", "5"), ("D", "5"), ("A", "5")], 90, 4, loops=5)),
        "modulate": write(folder / "Bminor_to_Emajor_96bpm.wav",
                          np.concatenate([song([("B", "m"), ("E", "m"), ("F#", ""), ("B", "m")], 96, 4, loops=4, tail=0),
                                          song([("E", ""), ("A", ""), ("B", ""), ("E", "")], 96, 4, loops=3)])),
        "short_m4a": write_m4a(folder / "short_phone_clip.m4a",
                               song([("A", ""), ("E", "7")], 120, 3, loops=2, drums=False, soft=True)),
        "corrupt": _write_bytes(folder / "corrupt.mp3", b"this is not audio"),
        "bassline": write(folder / "syncopated_bass_96bpm.wav", bassline_song()[0]),
        "guitar_blues": write(folder / "solo_guitar_blues_A_140bpm.wav", guitar_blues()),
    }


# ---------------------------------------------------------------- analysis checks (known answers)

def near(value, target, tol_pct):
    return value is not None and abs(value - target) <= target * tol_pct / 100


def analysis_checks(tmp):
    import musicanalyze as ma
    print("\n== Analysis on music with known answers ==")
    files = make_test_set(Path(tmp) / "audio")
    cache = Path(tmp) / "cache"
    res = {}
    for key, f in files.items():
        t0 = time.time()
        try:
            res[key] = ma.analyze(f, "G A B D" if key == "pop4" else None, cache_dir=cache)
            res[key]["_secs"] = time.time() - t0
        except Exception as e:
            res[key] = {"error": str(e)}

    r = res["pop4"]
    if check("analysis", "4/4 pop song analyses without errors", "error" not in r, r.get("error", "")):
        check("analysis", "tempo 100 BPM", near(r["tempo_bpm"], 100, 4), f"got {r['tempo_bpm']}")
        check("analysis", "meter 4/4", r["meter"]["label"] == "4/4", r["meter"]["label"])
        check("analysis", "key G major", r["key"] == "G major", r["key"])
        loop = r["loops"][0]["chords"] if r["loops"] else []
        check("analysis", "chord loop G – D – Em – C", sorted(loop) == sorted(["G", "D", "Em", "C"]), " – ".join(loop))
        progs = [p["progression"] for p in r["references"]["progressions"]]
        check("analysis", "recognises the I–V–vi–IV progression", any("I–V–vi–IV" in p for p in progs), "; ".join(progs))
        check("analysis", "drums: backbeat found", any("backbeat" in t for t in r["drums"].get("traits", [])),
              r["drums"].get("grid", ""))
        check("analysis", "bass present and on chord roots", r["bass"].get("present") and r["bass"]["follows_roots_pct"] >= 60,
              f"{r['bass'].get('follows_roots_pct')}% roots")
        chk = (r.get("melody") or {}).get("check", {})
        check("analysis", "typed melody: all notes found in the audio", not chk.get("weak_notes"), chk.get("verdict", "")[:80])

    r = res["waltz3"]
    if check("analysis", "3/4 waltz (soft strings, no drums) analyses", "error" not in r, r.get("error", "")):
        check("analysis", "tempo 120 BPM despite soft onsets", near(r["tempo_bpm"], 120, 5), f"got {r['tempo_bpm']}")
        check("analysis", "meter 3/4", r["meter"]["label"] == "3/4", f"{r['meter']['label']} {r['meter'].get('scores')}")
        check("analysis", "key A major (E7 resolves to A)", r["key"] == "A major", r["key"])
        check("analysis", "no drums reported", not r["drums"].get("present") and not any(s["drums"] for s in r["sections"]))

    r = res["power"]
    if check("analysis", "power-chord song analyses", "error" not in r, r.get("error", "")):
        names = [c["chord"] for c in r["chord_timeline"]]
        check("analysis", "chords without thirds are labelled as power chords",
              sum(n.endswith("5") for n in names) >= 0.6 * len(names), " ".join(names[:8]))
        check("analysis", "no famous progression claimed from power chords", not r["references"]["progressions"],
              "; ".join(p["progression"] for p in r["references"]["progressions"]))

    r = res["modulate"]
    if check("analysis", "key-change song analyses", "error" not in r, r.get("error", "")):
        keys = [k["key"] for k in r["key_sections"]]
        check("analysis", "key change B minor → E major found", keys[:1] == ["B minor"] and "E major" in keys[1:],
              " → ".join(keys))

    r = res["bassline"]
    if check("analysis", "syncopated bass song analyses", "error" not in r, r.get("error", "")):
        truth = bassline_song()[1]
        found = [(n["start"], n["midi"]) for n in r.get("bass_transcription", [])]
        used, hits, pitch_ok = set(), 0, 0
        for t, n in truth:
            j = next((j for j, (ft, fn) in enumerate(found) if j not in used and abs(ft - t) <= 0.05), None)
            if j is not None:
                used.add(j)
                hits += 1
                pitch_ok += found[j][1] % 12 == n % 12
        rec, prec = hits / len(truth), hits / max(1, len(found))
        check("analysis", "bass transcription: finds the notes at their real times (>= 85%)", rec >= 0.85,
              f"{rec:.0%} of {len(truth)} notes within 50 ms")
        check("analysis", "bass transcription: few made-up notes (>= 85% real)", prec >= 0.85, f"{prec:.0%} of {len(found)}")
        check("analysis", "bass transcription: right pitches (>= 90%)", hits and pitch_ok / hits >= 0.9,
              f"{pitch_ok}/{hits}")
        sheet, rh = r.get("bass_sheet") or {}, r.get("bass_rhythm") or {}
        check("analysis", "bass sheet: bar 1 starts on the first downbeat", abs(sheet.get("bar1_time", 9)) < 0.15,
              f"bar 1 at {sheet.get('bar1_time')}s")
        check("analysis", "bass sheet: main bar rhythm is the one played (x.x...x.x...x.x.)",
              rh.get("main_bar_rhythm") == "x.x...x.x...x.x.", rh.get("main_bar_rhythm", "none"))
        check("analysis", "bass sheet: drum hits on a ringing note aren't written as new notes",
              len(sheet.get("notes", [])) <= len(truth) + 4, f"{len(sheet.get('notes', []))} notes, {len(truth)} played")
    if "error" not in res["pop4"] and "error" not in r:
        happy, dark = res["pop4"]["feel"], r["feel"]
        check("analysis", "mood: major pop song reads more positive than the E minor groove",
              happy["valence"] >= dark["valence"] + 15, f"{happy['valence']} vs {dark['valence']}")
    if "error" not in res["waltz3"] and "error" not in res["pop4"]:
        check("analysis", "energy: soft strings without drums read calmer than the drum-driven pop song",
              res["waltz3"]["feel"]["arousal"] < res["pop4"]["feel"]["arousal"],
              f"{res['waltz3']['feel']['arousal']} vs {res['pop4']['feel']['arousal']}")

    r = res["guitar_blues"]
    if check("analysis", "solo-guitar 12-bar blues analyses", "error" not in r, r.get("error", "")):
        check("analysis", "solo guitar: picking isn't reported as drums",
              not r["drums"].get("present") and not any(s["drums"] for s in r["sections"]), str(r["drums"]))
        check("analysis", "solo guitar: its low strings aren't reported as a separate bass",
              r["bass"].get("low_part_only") and not any(s["bass"] for s in r["sections"]), str(r["bass"]))
        b = r.get("blues_form") or {}
        check("analysis", "12-bar blues recognised, with its choruses (every ~20.6 s from 0:00)",
              b and near(b["chorus_sec"], 20.6, 5) and b["choruses"][0] < 1.5 and len(b["choruses"]) >= 2, str(b))
        # What the musician says decides what the audio can't (same recording, analysis from the cache)
        told = ma.analyze(files["guitar_blues"], cache_dir=cache, context=ma.clean_context(["guitar"], "blues"))
        check("analysis", "musician says 'guitar, blues': guitar notation, no drums/bass, a blues idea",
              told.get("low_part_instrument") == "guitar" and told["drums"].get("user_said_none")
              and told["bass"].get("user_said_none") and any(i["area"] == "blues" for i in told["ideas"]),
              str(told.get("context")))
        with_bass = ma.analyze(files["guitar_blues"], cache_dir=cache, context=ma.clean_context(["guitar", "bass"]))
        check("analysis", "musician says there's a bass: the low line is reported as a bass",
              with_bass["bass"].get("present") and with_bass["bass"].get("from_user"), str(with_bass["bass"])[:120])
    fake = {"tempo_bpm": 87.0, "drums": {}, "bass": {}, "sections": []}
    ma.apply_context(fake, ma.clean_context([], "dnb"))
    check("analysis", "genre fixes a half-speed tempo reading (drum & bass at 87 -> felt ~174 BPM)",
          (fake.get("context") or {}).get("felt_tempo") == 174, str(fake.get("context")))
    lows = {}
    for inst in ("guitar", "keys", None):
        low = {"bass": {"low_part_only": True}, "low_end_missing": True, "low_part_instrument": inst,
               "bass_transcription": [{"start": 0, "end": 1, "midi": 33, "velocity": 90}]}
        ma.finish_low_part(low)
        lows[inst] = low["bass_transcription"][0]["midi"]
    check("analysis", "a low note below E2 is an octave error for a guitar, but real for a piano",
          lows == {"guitar": 45, "keys": 33, None: 45}, str(lows))
    check("analysis", "unknown instrument/genre values from the form are ignored",
          ma.clean_context(["drums", "kazoo<script>"], "polka") == {"instruments": ["drums"], "genre": None})
    others = [k for k in ("pop4", "waltz3", "power", "modulate", "bassline") if "error" not in res[k]]
    check("analysis", "no 12-bar blues claimed for songs that aren't one",
          not any(res[k].get("blues_form") for k in others),
          ", ".join(k for k in others if res[k].get("blues_form")))
    if "error" not in res["pop4"]:
        check("analysis", "a band with drums still has its drums and bass", res["pop4"]["drums"].get("present")
              and res["pop4"]["bass"].get("present"))

    r = res["short_m4a"]
    check("analysis", "short phone-style .m4a clip decodes and analyses", "error" not in r, r.get("error", ""))
    r = res["corrupt"]
    check("analysis", "corrupt file is reported, not crashed on", "error" in r)

    t0 = time.time()
    again = ma.analyze(files["pop4"], cache_dir=cache)
    check("analysis", "cache: re-analysing the same file is instant", again.get("cached") and time.time() - t0 < 3,
          f"{time.time() - t0:.1f}s")
    slow = max(v.get("_secs", 0) for v in res.values())
    check("analysis", "speed: each test track analysed in under 90 s", slow < 90, f"slowest {slow:.0f}s")
    return res


# ---------------------------------------------------------------- AI check (real provider, if configured)

def ai_checks(res):
    import musicanalyze as ma
    print("\n== AI brainstorm (your configured provider) ==")
    llm, label = ma.make_llm()
    if not llm:
        check("ai", "AI provider configured", True, "no API key set - AI ideas are off (that's allowed)")
        return
    r = res.get("pop4", {})
    if "error" in r:
        check("ai", "AI test needs the pop song analysis", False)
        return
    t0 = time.time()
    ai = ma.ai_brainstorm(r, llm)
    secs = time.time() - t0
    if not check("ai", f"AI answers ({label})", not ai.get("error"), ai.get("error", f"{ai.get('model')} in {secs:.0f}s")):
        return
    check("ai", "AI answer arrives within the deadline", secs <= ma.AI_DEADLINE_SECONDS, f"{secs:.0f}s")
    check("ai", "AI gives ideas and chord progressions", len(ai["ideas"]) >= 3 and ai["alt_progressions"],
          f"{len(ai['ideas'])} ideas, {len(ai['alt_progressions'])} progressions")
    flagged = [i["idea"] for i in ai["ideas"] if "⚠ check this" in i["idea"]]
    check("ai", "AI ideas match the analysis (no flagged times/chords)", len(flagged) <= 1,
          f"{len(flagged)} flagged" + (f": {flagged[0][:90]}" if flagged else ""))


# ---------------------------------------------------------------- unit checks (logic that has bitten us)

def unit_checks():
    import inspiration as ins
    import musicanalyze as ma
    print("\n== Logic checks ==")
    check("unit", "power chords don't match minor progressions", not ins.match_progression([(4, "5"), (9, "")]))
    check("unit", "a real i–IV loop matches", [m["progression"] for m in ins.match_progression([(4, "m"), (9, "")])]
          == ["i–IV (Dorian vamp)"])
    check("unit", "chords played once don't wrap around", not ins.match_progression([(9, ""), (4, "7"), (1, "m"), (11, "")],
                                                                                    cyclic=False))
    check("unit", "cut-off AI reply is salvaged", ma._parse_json_reply('{"vibe":"x","ideas":[{"area":"a","idea":"b"},{"ar')
          == {"vibe": "x", "ideas": [{"area": "a", "idea": "b"}]})
    check("unit", "chord names with extensions parse", ma.parse_chord("B7alt")[:2] == (11, "7")
          and ma.parse_chord("F#m7b5")[1] == "m7b5")
    fake = {"chord_timeline": [{"time": "0:47", "chord": "C#m"}, {"time": "0:35", "chord": "Em"}],
            "sections": [{"label": "C", "start": 131, "end": 162}], "drums": {}, "sound": {}}
    check("unit", "AI timing check flags a wrong chord time", "⚠" in ma._check_times("Add G#7 before C#m at 0:35", fake))
    check("unit", "AI timing check flags a wrong section", "⚠" in ma._check_times("A sweep at 1:45 before the C section", fake))
    check("unit", "AI timing check accepts a correct claim", "⚠" not in ma._check_times("Lift into C#m at 0:47", fake))
    ma.AI_HEDGE_SECONDS, ma.AI_DEADLINE_SECONDS = 0.3, 1.5
    t0 = time.time()
    info = {}
    out = ma._race(["slow", "fast"], lambda m: (time.sleep(5 if m == "slow" else 0.1), '{"a":1}')[1], info, lambda e: False)
    check("unit", "AI race: a slow model is overtaken by the backup", info.get("model") == "fast" and time.time() - t0 < 1.5)
    full = '{"ideas":[{"idea":"a"},{"idea":"b"},{"idea":"c"}],"alt_progressions":[{"chords":["G","C"]}]}'
    cut = '{"ideas":[{"idea":"a"},{"idea":"b"},{"idea":"c"}],"alt_prog'
    info = {}
    out = ma._race(["quick-but-cut", "slower-complete"],
                   lambda m: (time.sleep(0.05 if m.startswith("quick") else 0.6), cut if m.startswith("quick") else full)[1],
                   info, lambda e: False, complete=ma._complete_answer)
    check("unit", "AI race: waits for a complete answer instead of a cut-off one", info.get("model") == "slower-complete")
    info = {}
    out = ma._race(["quick-but-cut", "broken"],
                   lambda m: cut if m.startswith("quick") else (_ for _ in ()).throw(TimeoutError()),
                   info, lambda e: False, complete=ma._complete_answer)
    check("unit", "AI race: a cut-off answer is still used when nothing better arrives", info.get("partial") is True)
    fake2 = {"chord_timeline": [{"time": "0:12", "chord": "D"}], "sections": [], "drums": {}, "sound": {}}
    check("unit", "AI fact-check ignores chords the AI suggests adding",
          "⚠" not in ma._check_times("Insert an A7 before the D chord at 0:12 (play A7 at 0:11)", fake2))
    notation_checks(ma)


def notation_checks(ma):
    """Sheet music from known notes: 2 bars of 3/4 at 120 BPM (beats every 0.5 s), slightly off the grid
    like a real player, the first bar starting on the 2nd detected beat."""
    import xml.etree.ElementTree as ET
    import notation as nt
    beats = [i * 0.5 for i in range(12)]
    played = [(0.52, 0.98, 45, 100), (1.01, 1.24, 52, 90), (1.26, 1.49, 52, 90),   # bar 1: A | E E (8ths)
              (1.98, 2.73, 47, 100), (2.76, 2.99, 49, 90)]                         # bar 2: B (dotted quarter) C#
    sheet = nt.quantize(played, [(0.5, "A"), (2.0, "E7")], beats, 3, 0.5)
    want = [(0, 4, 45), (4, 2, 52), (6, 2, 52), (12, 6, 47), (18, 2, 49)]
    check("unit", "notation: notes land on the right 16ths and bars (3/4)",
          [(s, l, n) for s, l, n, _ in sheet["notes"]] == want, str(sheet["notes"]))
    check("unit", "notation: chord symbols on their beats", sheet["chords"] == [(0, "A"), (12, "E7")], str(sheet["chords"]))
    xml = nt.to_musicxml(sheet, "t", "Bass", ma.key_fifths(9), 120, lambda pc: ma.spell(pc, 9), ma.parse_chord)
    root = ET.fromstring(xml.split("?>", 1)[1].split(">", 1)[1])  # drop the XML declaration and DOCTYPE
    bars = root.findall(".//measure")
    durs = [sum(int(n.find("duration").text) for n in m.findall("note")) for m in bars]
    check("unit", "notation: every bar adds up to 3 beats", durs == [12] * len(bars), str(durs))
    check("unit", "notation: A major has 3 sharps; C# spelled as C#", ma.key_fifths(9) == 3 and "<step>C</step><alter>1</alter>" in xml)
    check("unit", "notation: 8ths in the same beat are beamed", xml.count('<beam number="1">begin</beam>') >= 1)
    rh = nt.rhythm_from_score(sheet)
    check("unit", "notation: rhythm stats read from the sheet", rh["notes"] == 5 and rh["offbeat_pct"] == 40, str(rh))
    try:
        import verovio  # noqa: F401
        check("unit", "notation: engraves to SVG with Verovio", nt.render_svg(xml)[0].lstrip().startswith("<"))
    except ImportError:
        check("unit", "notation: Verovio installed (needed for the engraved sheet)", False, "pip install verovio")


# ---------------------------------------------------------------- several analyses at once

PARALLEL_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[2])
import musicanalyze  # noqa: F401  (gives this process its own numba cache folder)
import numpy as np, librosa
y = np.random.default_rng(int(sys.argv[1])).standard_normal(22050 * 15).astype("float32")
librosa.beat.beat_track(y=y, sr=22050)
librosa.onset.onset_detect(y=y, sr=22050)
"""


def parallel_checks(tmp):
    """On the server the app's worker warms up while this self-test runs. When processes shared one numba
    cache folder, that overlap crashed the self-test (segfault in librosa's beat tracker); reproduced
    locally with 13 of 24 processes crashing. Start 3 analyses at once on one fresh folder."""
    print("\n== Several analyses at once (shared compiled-code cache) ==")
    cache = Path(tmp) / "numba-shared"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache)}
    procs = [subprocess.Popen([sys.executable, "-c", PARALLEL_CHILD, str(i), str(Path(__file__).parent)],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) for i in range(3)]
    codes, errs = [], []
    for p in procs:
        try:
            _, err = p.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            p.kill()
            _, err = p.communicate()
        codes.append(p.returncode)
        errs.append(err.strip().splitlines()[-1] if p.returncode and err.strip() else "")
    check("parallel", "3 analyses starting together don't crash each other", all(c == 0 for c in codes),
          f"exit codes {codes} {' | '.join(e for e in errs if e)}")


# ---------------------------------------------------------------- full web-app run (simulated AI)

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_ai_server(port):
    from flask import Flask, jsonify, request
    from werkzeug.serving import make_server
    app = Flask("fake-ai")
    answer = json.dumps({"vibe": "test", "ideas": [{"area": "harmony", "idea": f"idea {i}"} for i in range(5)],
                         "alt_progressions": [{"name": "lift", "chords": ["G", "Am", "C", "D"], "why": "test"}],
                         "title_ideas": ["Test"], "references": []})

    @app.post("/v1/chat/completions")
    def chat():
        model = request.get_json()["model"]
        if "nemotron" in model:
            return jsonify(error="gateway timeout"), 504  # a failing backup shouldn't matter
        time.sleep(2)
        return jsonify(id="x", object="chat.completion", created=0, model=model,
                       choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answer}}],
                       usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    srv = make_server("127.0.0.1", port, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def web_checks(tmp, files):
    import zipfile

    import requests
    print("\n== Web app end to end (simulated AI) ==")
    ai_port, port = free_port(), free_port()
    fake = fake_ai_server(ai_port)
    env = {**os.environ, "PORT": str(port), "DATA_DIR": str(Path(tmp) / "jobs"), "APP_PASSWORD": "selftest",
           "NVIDIA_API_KEY": "nvapi-selftest", "MUSIC_LLM_BASE_URL": f"http://127.0.0.1:{ai_port}/v1",
           "MUSIC_SEPARATION": "simple", "MUSIC_SELFTEST_CRASH_ONCE": str(Path(tmp) / "crash-flag")}
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MUSIC_MODEL", "MUSIC_FALLBACK_MODELS"):
        env.pop(k, None)
    proc = subprocess.Popen([sys.executable, str(HERE / "app.py")], env=env, cwd=HERE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base, auth = f"http://127.0.0.1:{port}", ("x", "selftest")
    try:
        for _ in range(60):
            try:
                if requests.get(base + "/health", timeout=2).ok:
                    break
            except requests.ConnectionError:
                time.sleep(1)
        h = requests.get(base + "/health", timeout=5)
        check("web", "health page answers without a password", h.ok and "version" in h.json(), h.text[:80])
        check("web", "analysing needs no account or password", requests.get(base + "/", timeout=5).status_code == 200)

        def run_job(paths, melody="", extra=None):
            up = [("tracks", (Path(p).name, open(p, "rb"))) for p in paths]
            r = requests.post(base + "/analyze", files=up, data={"ai": "1", "melody": melody, **(extra or {})},
                              auth=auth, allow_redirects=False, timeout=60)
            job = r.headers["Location"].rsplit("/", 1)[-1]
            t0, first = time.time(), None
            while time.time() - t0 < 600:
                s = requests.get(f"{base}/jobs/{job}/status", auth=auth, timeout=10).json()
                if s["state"] == "done" and first is None:
                    first = time.time() - t0
                if s["state"] == "error" or (s["state"] == "done" and not s["ai_pending"]):
                    return job, s, first, time.time() - t0
                time.sleep(1)
            return job, {"state": "timeout"}, first, time.time() - t0

        job, s, first, total = run_job([files["pop4"], files["corrupt"]], "pop: G A B D")
        check("web", "upload → analysis → AI ideas completes", s["state"] == "done", f"{s['state']} after {total:.0f}s")
        check("web", "a worker crash (segfault-style) is retried automatically",
              (Path(tmp) / "crash-flag").exists() and s["state"] == "done", "the first job crashed once and still finished")
        check("web", "results shown before the AI finished", first is not None and first < total,
              f"results {first:.0f}s, AI {total:.0f}s" if first else "")
        page = requests.get(f"{base}/jobs/{job}", auth=auth, timeout=10).text
        check("web", "report contains the AI ideas (none pending)", "idea 0" in page and "still writing" not in page)
        check("web", "typed melody was analysed for the matching track", "Your melody" in page)
        check("web", "corrupt file reported on the page", "unreadable" in page.lower())
        z = requests.get(f"{base}/jobs/{job}/download.zip", auth=auth, timeout=30)
        names = zipfile.ZipFile(__import__("io").BytesIO(z.content)).namelist() if z.ok else []
        check("web", "zip has report, data and MIDI (incl. AI progression)",
              {"music_report.md", "music_results.json"} <= set(names) and any("_ai_" in n for n in names),
              f"{len(names)} files")
        check("web", "zip has the bass sheet music (MusicXML + engraved page) and the MIDI made from it",
              any(n.startswith("sheets/") and n.endswith(".musicxml") for n in names) and
              any(n.startswith("sheets/") and n.endswith(".html") for n in names) and
              any(n.endswith("_bass_from_sheet.mid") for n in names), ", ".join(n for n in names if "sheet" in n))
        sheet_link = re.search(r'href="([^"]+/files/sheets/[^"]+\.html)"', page)
        check("web", "the engraved sheet opens from the report",
              bool(sheet_link) and "<svg" in requests.get(base + sheet_link.group(1), auth=auth, timeout=10).text)
        res = json.loads(zipfile.ZipFile(__import__("io").BytesIO(z.content)).read("music_results.json"))
        check("web", "results carry the analyzer version", all("analyzer_version" in r for r in res if "error" not in r))

        job2, s2, first2, total2 = run_job([files["pop4"]])
        check("web", "re-upload of the same track uses the cache", first2 is not None and first2 < 15,
              f"results in {first2:.0f}s" if first2 else s2.get("state"))

        # Crash recovery: kill the worker mid-job; that job fails cleanly, the next one works.
        import psutil
        up = [("tracks", (Path(files["modulate"]).name, open(files["modulate"], "rb")))]
        r = requests.post(base + "/analyze", files=up, auth=auth, allow_redirects=False, timeout=60)
        job3 = r.headers["Location"].rsplit("/", 1)[-1]
        time.sleep(3)
        for p in psutil.Process(proc.pid).children(recursive=True):
            if any(str(c).endswith("job_runner.py") for c in p.cmdline()):
                p.kill()
        t0 = time.time()
        while time.time() - t0 < 120:
            s3 = requests.get(f"{base}/jobs/{job3}/status", auth=auth, timeout=10).json()
            if s3["state"] in ("error", "done"):
                break
            time.sleep(1)
        check("web", "a killed worker is replaced and the job finishes", s3["state"] == "done", s3.get("message", ""))
        job4, s4, _, _ = run_job([files["short_m4a"]])
        check("web", "the next job works after a crash", s4["state"] == "done", s4["state"])
        account_checks(base, job4)
        job5, s5, _, _ = run_job([files["guitar_blues"]], extra={"instruments": ["guitar", "voice"], "genre": "blues"})
        r5 = requests.get(f"{base}/jobs/{job5}/files/music_results.json", timeout=10).json()[0] if s5["state"] == "done" else {}
        sheet = requests.get(f"{base}/jobs/{job5}/files/sheets/solo_guitar_blues_A_140bpm_low_part.musicxml", timeout=10)
        check("web", "the upload form's instruments and genre reach the analysis",
              (r5.get("context") or {}).get("genre") == "Blues" and "voice" in r5["context"]["instruments"]
              and "<sign>G</sign>" in sheet.text, str(r5.get("context")))
    finally:
        import psutil
        try:
            for child in psutil.Process(proc.pid).children(recursive=True):  # the analysis worker
                child.kill()
        except psutil.Error:
            pass
        proc.kill()
        proc.wait(timeout=10)
        fake.shutdown()


def account_checks(base, job):
    """Accounts: history, profile, and saved analyses private to their owner. `job` is a finished
    anonymous analysis."""
    import requests
    anon = requests.Session()
    check("web", "an anonymous result opens for anyone with the link", anon.get(f"{base}/jobs/{job}", timeout=10).ok)
    res = anon.get(f"{base}/jobs/{job}/files/music_results.json", timeout=10).json()[0]
    key, tempo = res["key"], (f"{int(res['tempo_bpm'] + 0.5)} BPM" if res.get("tempo_bpm") else res["duration"])
    alice, bob = requests.Session(), requests.Session()
    a = alice.post(base + "/signup", data={"username": "alice", "password": "selftest-pw", "password2": "selftest-pw",
                                           "next": "/"}, allow_redirects=False, timeout=10)
    dup = bob.post(base + "/signup", data={"username": "Alice", "password": "another-pw", "password2": "another-pw"},
                   allow_redirects=False, timeout=10)
    check("web", "sign-up works; a taken username (any case) is refused", a.status_code == 302 and dup.status_code == 400)
    bob.post(base + "/signup", data={"username": "bob", "password": "selftest-pw2", "password2": "selftest-pw2"}, timeout=10)
    alice.post(f"{base}/jobs/{job}/save", timeout=10)
    hist = alice.get(base + "/history", timeout=10).text
    prof = alice.get(base + "/profile", timeout=10).text
    check("web", "a saved analysis appears in History with its key and tempo",
          "short_phone_clip" in hist and key in hist and tempo in hist, hist[-300:])
    check("web", "the profile sums up the saved music", key in prof and "Your music is" in prof)
    check("web", "a saved analysis is private: others get 404",
          anon.get(f"{base}/jobs/{job}", timeout=10).status_code == 404 and
          bob.get(f"{base}/jobs/{job}/download.zip", timeout=10).status_code == 404 and
          alice.get(f"{base}/jobs/{job}", timeout=10).ok)
    check("web", "history and profile need a sign-in",
          anon.get(base + "/history", allow_redirects=False, timeout=10).status_code == 302)
    wrong = requests.post(base + "/login", data={"username": "alice", "password": "nope"}, allow_redirects=False, timeout=10)
    evil = requests.post(base + "/login", data={"username": "alice", "password": "selftest-pw", "next": "//evil.example"},
                         allow_redirects=False, timeout=10)
    check("web", "sign-in: wrong password refused; never redirects to another site",
          wrong.status_code == 401 and evil.headers.get("Location") == "/")
    shows = lambda sess: 'data-auto="1"' in sess.get(base + "/", timeout=10).text
    visitor = requests.Session()
    first = shows(visitor)
    visitor.post(base + "/intro/seen", timeout=10)
    check("web", "the limits notice pops up on a first visit, and stays closed once closed",
          first and not shows(visitor))
    carol = requests.Session()
    carol.post(base + "/signup", data={"username": "carol", "password": "selftest-pw3", "password2": "selftest-pw3"},
               timeout=10)
    carol.post(base + "/intro/seen", timeout=10)
    other_device = requests.Session()
    other_device.post(base + "/login", data={"username": "carol", "password": "selftest-pw3"}, timeout=10)
    check("web", "closing the notice is remembered on the account (not shown again on another device)",
          not shows(other_device))
    alice.post(f"{base}/jobs/{job}/delete", timeout=10)
    check("web", "deleting from History removes the analysis",
          alice.get(f"{base}/jobs/{job}", timeout=10).status_code == 404 and "short_phone_clip" not in
          alice.get(base + "/history", timeout=10).text)


# ---------------------------------------------------------------- facts you've confirmed about your own recordings

def facts_checks(facts_file, tmp):
    """Every fact a musician confirms (from a score, or by ear) becomes a permanent check, so a later
    change can't silently break it. The file stays private (it points at your own recordings)."""
    import musicanalyze as ma
    print(f"\n== Your confirmed facts ({facts_file}) ==")
    facts = json.loads(Path(facts_file).read_text(encoding="utf-8"))
    for item in facts:
        f = Path(item["file"])
        if not f.exists():
            check("facts", f"{f.name}: file available", False, "not found - skipped")
            continue
        r = ma.analyze(f, item.get("melody"), cache_dir=Path(tmp) / "facts-cache")
        e, n = item["expect"], f.name
        if "tempo_bpm" in e:
            check("facts", f"{n}: tempo ≈ {e['tempo_bpm']}", near(r["tempo_bpm"], e["tempo_bpm"], 5), f"got {r['tempo_bpm']}")
        if "meter" in e:
            check("facts", f"{n}: meter {e['meter']}", r["meter"]["label"] == e["meter"], r["meter"]["label"])
        if "key" in e:
            check("facts", f"{n}: key {e['key']}", r["key"] == e["key"], r["key"])
        if "chords_in_order" in e:
            got = [c["chord"] for c in r["chord_timeline"]]
            check("facts", f"{n}: chords {' → '.join(e['chords_in_order'])}", got[:len(e["chords_in_order"])] ==
                  e["chords_in_order"], " → ".join(got[:8]))
        if e.get("bass_in_every_section"):
            check("facts", f"{n}: bass in every section", all(x["bass"] for x in r["sections"]),
                  "".join("B" if x["bass"] else "-" for x in r["sections"]))
        if "drums" in e:
            check("facts", f"{n}: drums {'present' if e['drums'] else 'absent'}",
                  bool(r["drums"].get("present")) == e["drums"], str(r["drums"].get("present")))
        if "bass_instrument" in e:
            got = bool(r["bass"].get("present"))
            check("facts", f"{n}: separate bass {'present' if e['bass_instrument'] else 'absent'}",
                  got == e["bass_instrument"], str(r["bass"]))
        if "no_progression_claims" in e:
            check("facts", f"{n}: no famous-progression claim", not r["references"]["progressions"],
                  "; ".join(p["progression"] for p in r["references"]["progressions"]))


# ---------------------------------------------------------------- live check against the running server

def live_checks(base, tmp):
    """Upload a real test track through the RUNNING website and wait for the result. This exercises
    the real worker process on the server (e.g. how it is started under gunicorn on Linux)."""
    import base64
    import urllib.error
    import urllib.request
    import uuid
    print(f"\n== Live upload through the running app ({base}) ==")
    pw = os.environ.get("APP_PASSWORD", "")
    headers = {"Authorization": "Basic " + base64.b64encode(f"selftest:{pw}".encode()).decode()} if pw else {}

    def get(path):
        req = urllib.request.Request(base + path, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    try:
        h = json.loads(get("/health"))
    except Exception as e:
        check("live", "app answers on /health", False, str(e))
        return
    check("live", "app answers on /health", h.get("status") == "ok", json.dumps(h))
    track = write(Path(tmp) / "live_test_waltz.wav",
                  song([("A", ""), ("E", "7"), ("A", ""), ("D", "")], 120, 3, loops=3, drums=False, waltz=True, soft=True))
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"tracks\"; filename=\"{track.name}\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + track.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    req = urllib.request.Request(base + "/analyze", data=body, method="POST",
                                 headers={**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        urllib.request.build_opener(NoRedirect).open(req, timeout=60)
        check("live", "upload accepted", False, "no redirect to a job page")
        return
    except urllib.error.HTTPError as e:
        if e.code != 302:
            check("live", "upload accepted", False, f"HTTP {e.code}")
            return
        job = e.headers["Location"].rsplit("/", 1)[-1]
    check("live", "upload accepted", True, f"job {job[:8]}")
    t0, s = time.time(), {}
    while time.time() - t0 < 400:
        s = json.loads(get(f"/jobs/{job}/status"))
        if s["state"] == "error" or (s["state"] == "done" and not s.get("ai_pending")):
            break
        time.sleep(2)
    ok = check("live", "the worker analyses it (no crash)", s.get("state") == "done",
               f"{s.get('state')}: {s.get('message', '')} after {time.time() - t0:.0f}s")
    if ok:
        page = get(f"/jobs/{job}")
        check("live", "results page shows the analysis", "Harmony" in page and "3/4" in page)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also run logic checks and a full web-app run")
    ap.add_argument("--no-ai", action="store_true", help="skip the real AI check")
    ap.add_argument("--facts", help="JSON file of facts you've confirmed about your own recordings")
    ap.add_argument("--live", metavar="URL", help="also upload a test track through the running app at URL")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    import musicanalyze as ma
    print(f"Music Inspiration Analyzer self-test — version {ma.CODE_VERSION}")
    tmp = tempfile.mkdtemp(prefix="music-selftest-")
    try:
        try:
            res = analysis_checks(tmp)
        except Exception:
            check("analysis", "analysis checks ran", False, traceback.format_exc(limit=2))
            res = {}
        if not args.no_ai:
            try:
                ai_checks(res)
            except Exception:
                check("ai", "AI check ran", False, traceback.format_exc(limit=2))
        if args.live or args.full:
            try:
                parallel_checks(tmp)
            except Exception:
                check("parallel", "parallel check ran", False, traceback.format_exc(limit=2))
        if args.live:
            try:
                live_checks(args.live.rstrip("/"), tmp)
            except Exception:
                check("live", "live check ran", False, traceback.format_exc(limit=2))
        if args.facts:
            try:
                facts_checks(args.facts, tmp)
            except Exception:
                check("facts", "facts checks ran", False, traceback.format_exc(limit=2))
        if args.full:
            try:
                unit_checks()
            except Exception:
                check("unit", "logic checks ran", False, traceback.format_exc(limit=2))
            try:
                web_checks(tmp, make_test_set(Path(tmp) / "audio"))
            except Exception:
                check("web", "web checks ran", False, traceback.format_exc(limit=3))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    failed = [r for r in RESULTS if not r[2]]
    print(f"\n{'=' * 60}\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    for g, n, _, d in failed:
        print(f"  FAIL [{g}] {n}" + (f": {d}" if d else ""))
    print("RESULT:", "PASS ✅" if not failed else "FAIL ❌")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()  # a crash inside a native library prints where it happened instead of dying silently
    main()
