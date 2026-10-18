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


WARNINGS = []  # outside problems the app handles on its own: shown, but they don't fail the self-test


def warn(group, name, detail=""):
    WARNINGS.append((group, name, detail))
    print(f"  WARN  {name}" + (f"  ({detail})" if detail else ""), flush=True)


# An AI provider being slow, overloaded or briefly down is outside our control, and the app copes (the analysis
# shows; the AI section says the ideas aren't available). A wrong key or a retired model is ours to fix: still a FAIL.
AI_OUTAGE = ("in time", "timed out", "timeout", "empty reply", "502", "503", "504", "429", "rate limit",
             "overloaded", "temporarily", "connection")


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


def guitar_blues(bpm=140, choruses=3, minor=False):
    """Solo fingerstyle guitar, 12-bar blues in A: a thumb line on the low strings (root and fifth, nothing
    below 100 Hz, like a real guitar on a phone) and chords on 2 and 4. No drums, no bass instrument.
    minor=True: an A minor blues (i and iv minor 7ths, bVI7 - V7 in bars 9-10)."""
    beat = 60 / bpm
    form = [0, 0, 0, 0, 5, 5, 0, 0, 8 if minor else 7, 7 if minor else 5, 0, 7]
    root = {0: 45, 5: 50, 7: 52, 8: 53}  # A2, D3, E3, F3
    y = np.zeros(int(SR * (choruses * 48 * beat + 2)))
    for c in range(choruses):
        for b, deg in enumerate(form):
            t0, r = (c * 12 + b) * 4 * beat, root[deg]
            for i, n in enumerate((r, r + 7, r, r + 7)):
                place(y, pluck(hz(n), beat * 0.95, 0.35), t0 + i * beat)
            for i in (1, 3):
                minor7 = minor and deg in (0, 5)
                for iv in ((12, 15, 19, 22) if minor7 else (12, 16, 19, 22)):  # a 7th chord an octave up, strummed
                    place(y, pluck(hz(r + iv), beat * 0.9, 0.12), t0 + i * beat + 0.001 * iv)
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def _partials(freq, t, harmonics, vib_hz=0.0, vib_cents=0.0):
    """A pitched tone built from harmonics (amplitudes), with optional vibrato."""
    phase_mod = 0 if not vib_hz else (vib_cents / 1200 * np.log(2)) * freq / vib_hz * np.sin(2 * np.pi * vib_hz * t)
    return sum(a * np.sin(2 * np.pi * freq * (k + 1) * t + (k + 1) * phase_mod) for k, a in enumerate(harmonics))


def electronic_track(bpm=124, bars=32):
    """House: kick on every beat, clap on 2 and 4, open hats on the off-beats, an off-beat synth bass and a pad
    that ducks after every kick (sidechain pumping). A minor: Am | F | Dm | E."""
    beat = 60 / bpm
    y = np.zeros(int(SR * (bars * 4 * beat + 1)))
    pad = np.zeros_like(y)
    loop = [(57, [57, 60, 64]), (53, [53, 57, 60]), (50, [50, 53, 57]), (52, [52, 56, 59])]  # root, chord notes
    for b in range(bars):
        t0 = b * 4 * beat
        root, notes = loop[b % 4]
        t = np.arange(int(SR * 4 * beat)) / SR
        place(pad, sum(0.06 * _partials(hz(n + 12), t, (1, .5, .33, .25, .2)) for n in notes), t0)
        for i in range(4):
            tb = t0 + i * beat
            tk = np.arange(int(SR * 0.25)) / SR  # kick: a falling sine "thump"
            place(y, 0.9 * np.sin(2 * np.pi * (45 + 80 * np.exp(-tk * 40)) * tk) * np.exp(-tk * 12), tb)
            if i in (1, 3):
                place(y, noise(0.12, 0.35, 35, seed=i), tb)  # clap
            place(y, noise(0.09, 0.12, 40, hp=True, seed=i + 20), tb + beat / 2)  # open hat
            tbass = np.arange(int(SR * beat * 0.45)) / SR  # bass on the off-beat
            place(y, 0.3 * np.exp(-tbass * 6) * _partials(hz(root - 24), tbass, (1, .45, .2)), tb + beat / 2)
    # sidechain: the pad drops after each kick and swells back
    tt = np.arange(len(pad)) / SR
    pad *= 1 - 0.75 * np.exp(-((tt % beat) / 0.12))
    y = y + pad
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def orchestral_track(bpm=72):
    """Strings, D minor, no drums: legato chords with slow attacks and vibrato over a cello line, the tempo
    breathing +-8% (rubato), and a quiet - loud - quiet arc (the middle adds octave doublings)."""
    prog = [("D", "m"), ("G", "m"), ("A", "7"), ("D", "m"), ("Bb", ""), ("G", "m"), ("A", "7"), ("D", "m")]
    plan = [(prog, 0.35), (prog, 1.0), (prog, 0.35)]  # (chords, loudness) per section
    beats = [0.0]
    for i in range(sum(len(p) for p, _ in plan) * 4):  # rubato: each beat's length follows a slow wave
        beats.append(beats[-1] + 60 / (bpm * (1 + 0.08 * np.sin(2 * np.pi * i / 24))))
    y = np.zeros(int(SR * (beats[-1] + 2.5)))
    rng = np.random.default_rng(7)
    bar = 0
    for chords, level in plan:
        for root, q in chords:
            a, e = beats[bar * 4], beats[bar * 4 + 4]
            t = np.arange(int(SR * (e - a + 0.6))) / SR
            env = np.minimum(1, t / 0.25) * np.minimum(1, (e - a + 0.6 - t) / 0.6)  # slow bow attack, overlap
            tones = [midi(root, 4) + iv for iv in CHORD[q]]
            if level > 0.5:
                tones += [n + 12 for n in tones]
            # each player's vibrato has its own speed and timing, as in a real section (in-sync vibrato would wobble
            # the whole orchestra together and fake a beat)
            sig = sum(0.08 * _partials(hz(n), t + rng.uniform(0, 1), (1, .6, .4, .25, .15), rng.uniform(4.8, 6.4), 18)
                      for n in tones)
            sig += 0.22 * _partials(hz(midi(root, 2)), t + rng.uniform(0, 1), (1, .7, .45, .3), rng.uniform(4.5, 5.5), 12)
            place(y, level * env * sig, a)
            bar += 1
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


EIGHT_BAR_E = [("E", "7"), ("B", "7"), ("A", "7"), ("A", "7"), ("E", "7"), ("B", "7"), ("E", "7"), ("B", "7")]
SIXTEEN_BAR_G = [("G", "7")] * 8 + [("C", "7"), ("C", "7"), ("G", "7"), ("G", "7"), ("D", "7"), ("C", "7"), ("G", "7"),
                                    ("G", "7")]
AABA_A = [("C", ""), ("A", "m"), ("F", ""), ("G", ""), ("C", ""), ("A", "m"), ("D", "m"), ("G", "")]
AABA_B = [("E", "7"), ("E", "7"), ("A", "m"), ("A", "m"), ("D", "7"), ("D", "7"), ("G", ""), ("G", "7")]


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
        "minor_blues": write(folder / "solo_guitar_minor_blues_Am_140bpm.wav", guitar_blues(minor=True)),
        "five_four": write(folder / "odd_meter_5-4_120bpm.wav",
                           song([("D", "m"), ("G", ""), ("C", ""), ("A", "m")], 120, 5, loops=4)),
        "seven_four": write(folder / "odd_meter_7-4_140bpm.wav",
                            song([("E", "m"), ("C", ""), ("G", ""), ("D", "")], 140, 7, loops=4)),
        "eight_bar": write(folder / "eight_bar_blues_E_110bpm.wav", song(EIGHT_BAR_E, 110, 4, loops=3)),
        "sixteen_bar": write(folder / "sixteen_bar_blues_G_130bpm.wav", song(SIXTEEN_BAR_G, 130, 4, loops=2)),
        "aaba": write(folder / "aaba_song_form_C_140bpm.wav", song(AABA_A + AABA_A + AABA_B + AABA_A, 140, 4, loops=2)),
        "electronic": write(folder / "electronic_house_Am_124bpm.wav", electronic_track()),
        "orchestral": write(folder / "orchestral_strings_Dm_rubato.wav", orchestral_track()),
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
    cello = {}
    for n in (33, 38):  # A1 is below a cello's C2 (an error), D2 is playable
        low = {"bass": {"low_part_only": True}, "low_end_missing": True, "low_part_instrument": "strings",
               "bass_transcription": [{"start": 0, "end": 1, "midi": n, "velocity": 90}]}
        ma.finish_low_part(low)
        cello[n] = low["bass_transcription"][0]["midi"]
    check("analysis", "each instrument's own range decides octave errors (cello: A1 is an error, D2 is real)",
          cello == {33: 45, 38: 38}, str(cello))
    check("analysis", "unknown instrument/genre values from the form are ignored",
          ma.clean_context(["drums", "kazoo<script>"], "polka") == {"instruments": ["drums"], "genre": None})
    r = res["minor_blues"]
    if check("analysis", "minor-key blues analyses", "error" not in r, r.get("error", "")):
        b = r.get("blues_form") or {}
        check("analysis", "a minor blues is named as one, with minor-blues variations (i m7, iv m7, bVI7, V7)",
              "minor" in b.get("name", "") and r.get("blues_variations")
              and r["blues_variations"][0]["chords"][:2] == ["Am7", "Dm7"], f"{b.get('name')} {r['key']}")
    for key, label in (("five_four", "5/4"), ("seven_four", "7/4")):
        r = res[key]
        if check("analysis", f"{label} song analyses", "error" not in r, r.get("error", "")):
            check("analysis", f"odd meter {label} recognised", r["meter"]["label"] == label,
                  f"{r['meter']['label']} {r['meter'].get('scores')}")
    # other song forms
    for key, want, label in (("eight_bar", "8-bar", "8-bar blues in E"), ("sixteen_bar", "16-bar", "16-bar blues in G")):
        r = res[key]
        if check("analysis", f"{label} analyses", "error" not in r, r.get("error", "")):
            check("analysis", f"{label} recognised", (r.get("blues_form") or {}).get("name", "").startswith(want),
                  str((r.get("blues_form") or {}).get("name")))
    r = res["aaba"]
    if check("analysis", "AABA (32-bar) song analyses", "error" not in r, r.get("error", "")):
        f = r.get("song_form") or {}
        check("analysis", "AABA song form recognised, from the start, both passes",
              f and f["starts"][0] < 2 and f["cycles"] == 2, str(f))
    no_form = [k for k in ("pop4", "power", "modulate", "bassline", "guitar_blues") if "error" not in res[k]]
    check("analysis", "no AABA claimed for songs that aren't one", not any(res[k].get("song_form") for k in no_form),
          ", ".join(k for k in no_form if res[k].get("song_form")))
    # styles the other tests don't cover
    r = res["electronic"]
    if check("analysis", "electronic (house) track analyses", "error" not in r, r.get("error", "")):
        check("analysis", "electronic: 124 BPM, 4/4, four-on-the-floor kick",
              near(r["tempo_bpm"], 124, 4) and r["meter"]["label"] == "4/4"
              and "four-on-the-floor" in (r["drums"].get("traits") or []), f"{r['tempo_bpm']} {r['drums'].get('traits')}")
        check("analysis", "electronic: the kick's low boom doesn't fool the key (A minor)", r["key"] == "A minor",
              f"{r['key']} {[c['chord'] for c in r['chords_used'][:4]]}")
        check("analysis", "electronic: synth bass found, house groove referenced",
              r["bass"].get("present") and "house" in (r["references"].get("groove") or ""), str(r["bass"].get("present")))
    r = res["orchestral"]
    if check("analysis", "orchestral (legato strings) track analyses", "error" not in r, r.get("error", "")):
        check("analysis", "orchestral: legato strings' tempo is flagged (approximate or no pulse), no odd meter invented",
              (r["tempo_bpm"] is None or r.get("tempo_approximate")) and r["meter"]["label"] in ("3/4", "4/4"),
              f"{r['tempo_bpm']} approx={r.get('tempo_approximate')} meter {r['meter']['label']}")
        check("analysis", "orchestral: key D minor, no drums", r["key"] == "D minor" and not r["drums"].get("present"),
              f"{r['key']} drums={r['drums'].get('present')}")
        check("analysis", "orchestral: the loud middle section is found (quiet - loud - quiet)",
              "peak in the middle" in r["feel"]["arc"] and r["feel"]["valence"] < 50,
              f"{r['feel']['arc']} / positivity {r['feel']['valence']}")
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
    err = ai.get("error", "")
    if err and any(k in err.lower() for k in AI_OUTAGE):
        warn("ai", f"AI provider didn't answer this time ({label}); the app shows the analysis without AI ideas", err)
        return
    if not check("ai", f"AI answers ({label})", not err, err or f"{ai.get('model')} in {secs:.0f}s"):
        return
    check("ai", "AI answer arrives within the deadline", secs <= ma.AI_DEADLINE_SECONDS, f"{secs:.0f}s")
    check("ai", "AI gives ideas and chord progressions", len(ai["ideas"]) >= 3 and ai["alt_progressions"],
          f"{len(ai['ideas'])} ideas, {len(ai['alt_progressions'])} progressions")
    flagged = [i["idea"] for i in ai["ideas"] if "⚠ check this" in i["idea"]]
    # An AI that gets the odd time wrong is what the fact-check is for (the page marks it). Most ideas being off
    # means the analysis summary it was given is wrong: that's ours to fix.
    check("ai", "AI ideas mostly match the analysis (most times/chords check out)",
          len(flagged) <= len(ai["ideas"]) // 2, f"{len(flagged)} of {len(ai['ideas'])} flagged")
    for f_ in flagged:
        warn("ai", "an AI idea was flagged by the fact-check (shown to the musician with a ⚠)", f_[:300])


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
    tl = [{"time": "0:04", "start": 4.2, "chord": "D5"}, {"time": "0:05", "start": 5.4, "chord": "F7"},
          {"time": "0:06", "start": 6.3, "chord": "G5"}, {"time": "0:09", "start": 9.1, "chord": "D5"}]
    fake3 = {"chord_timeline": tl, "sections": [], "drums": {}, "sound": {}, "duration_sec": 12}
    check("unit", "AI fact-check: a chord claimed at the wrong second is flagged, even one bar off",
          "⚠" in ma._check_times("Then resolve to D5 at 0:06 for a pivot", fake3)
          and "⚠" not in ma._check_times("Lift into G5 at 0:06", fake3))
    check("unit", "a 12-bar blues check accepts real blues and rejects a padded pop loop",
          ma.fits_twelve_bar("D7 G7 D7 D7 G7 G7 D7 D7 A7 G7 D7 A7".split(), 2)
          and not ma.fits_twelve_bar("D A Bm F#7 Gmaj7 D A Bm F#7 Gmaj7 D A".split(), 2))
    fake2 = {"chord_timeline": [{"time": "0:12", "chord": "D"}], "sections": [], "drums": {}, "sound": {}}
    check("unit", "AI fact-check ignores chords the AI suggests adding",
          "⚠" not in ma._check_times("Insert an A7 before the D chord at 0:12 (play A7 at 0:11)", fake2))
    # A suggestion stays a suggestion however long its description; a claim stays a claim (checked) however
    # it's dressed up, and a verb in another sentence or before a preposition doesn't excuse it.
    fake4 = {"chord_timeline": [{"time": "0:09", "start": 9.6, "chord": "D"}, {"time": "0:12", "start": 12.0, "chord": "Em"}],
             "sections": [], "drums": {}, "sound": {}, "duration_sec": 40}
    ok_ = lambda s: "⚠" not in ma._check_times(s, fake4)
    check("unit", "AI fact-check: long suggestions pass, claims are still checked",
          ok_("Introduce a secondary dominant A7 at 0:10 to pivot toward D, adding brightness.")
          and ok_("Add a bright, open, ringing Asus4 at 0:10 for lift.")
          and not ok_("The bright, ringing secondary dominant A7 at 0:10 lifts the chorus.")
          and not ok_("Play softly. The A7 at 0:10 already lifts it.")
          and not ok_("Add a fill before the A7 at 0:10."))
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
           "MUSIC_SEPARATION": "simple", "MUSIC_SELFTEST_CRASH_ONCE": str(Path(tmp) / "crash-flag"),
           "MUSIC_TAGGER": "fake", "MUSIC_OFFLINE": "1"}
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
        memory_web_checks(base, files)
        job5, s5, _, _ = run_job([files["guitar_blues"]], extra={"instruments": ["guitar", "voice"], "genre": "blues"})
        r5 = requests.get(f"{base}/jobs/{job5}/files/music_results.json", timeout=10).json()[0] if s5["state"] == "done" else {}
        sheet = requests.get(f"{base}/jobs/{job5}/files/sheets/solo_guitar_blues_A_140bpm_low_part.musicxml", timeout=10)
        check("web", "the upload form's instruments and genre reach the analysis",
              (r5.get("context") or {}).get("genre") == "Blues" and "voice" in r5["context"]["instruments"]
              and "<sign>G</sign>" in sheet.text, str(r5.get("context")))
        check("web", "a musician's analysis doesn't run the tag search",
              requests.get(f"{base}/jobs/{job5}/files/tags.json", timeout=10).status_code == 404)
        job6, s6, _, _ = run_job([files["guitar_blues"]], extra={"mode": "tags", "songs": "Portishead - Roads"})
        t6 = requests.get(f"{base}/jobs/{job6}/files/tags.json", timeout=10)
        check("web", "a tag job runs from the form and skips the musician's work (no report, MIDI or AI)",
              s6["state"] == "done" and t6.ok and "12-bar blues" in t6.text
              and requests.get(f"{base}/jobs/{job6}/files/music_report.md", timeout=10).status_code == 404
              and "Your tags" in requests.get(f"{base}/jobs/{job6}", timeout=10).text, s6.get("state"))
        up = [("tracks", (Path(files["pop4"]).name, open(files["pop4"], "rb"))),
              ("reference", ("my_reference.wav", open(files["waltz3"], "rb")))]
        job8 = requests.post(base + "/analyze", files=up, allow_redirects=False, timeout=60).headers["Location"].rsplit("/", 1)[-1]
        t0 = time.time()
        while time.time() - t0 < 300 and requests.get(f"{base}/jobs/{job8}/status", timeout=10).json()["state"] not in ("done", "error"):
            time.sleep(1)
        page8 = requests.get(f"{base}/jobs/{job8}", timeout=10).text
        for _ in range(20):  # the status says "done" a moment before the worker deletes the uploads
            if not (Path(tmp) / "jobs" / job8 / "reference").exists():
                break
            time.sleep(0.5)
        check("web", "a reference uploaded with the form is compared, and deleted with the audio afterwards",
              "Next to your reference" in page8 and "my_reference.wav" in page8
              and not (Path(tmp) / "jobs" / job8 / "reference").exists())
        demo1 = requests.get(base + "/demo/blues", allow_redirects=False, timeout=60).headers.get("Location", "")
        did = demo1.rsplit("/", 1)[-1]
        t0 = time.time()
        while time.time() - t0 < 300 and requests.get(f"{base}/jobs/{did}/status", timeout=10).json()["state"] not in ("done", "error"):
            time.sleep(1)
        dpage = requests.get(f"{base}/jobs/{did}", timeout=10).text
        again = requests.get(base + "/demo/blues", allow_redirects=False, timeout=30).headers.get("Location", "")
        check("web", "a demo builds once from generated audio, then opens straight away for everyone",
              "This is a demo report" in dpage and "12-bar" in dpage and "Save to history" not in dpage
              and again.endswith(did), f"{did[:8]} / {again.rsplit('/', 1)[-1][:8]}")
        check("web", "an unknown demo is a 404", requests.get(base + "/demo/nope", timeout=10).status_code == 404)
        r7 = requests.post(base + "/analyze", data={"mode": "tags", "songs": "Massive Attack - Teardrop"},
                           allow_redirects=False, timeout=30)
        check("web", "a tag job needs no upload when songs are named", r7.status_code == 302, f"HTTP {r7.status_code}")
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


def memory_web_checks(base, files):
    """The musician's memory end to end: off by default, switched on, learns from saved uploads, ratings and
    corrections, pre-fills the form, compares a new track with earlier ones, and forgets on request."""
    import requests
    dana, eve = requests.Session(), requests.Session()
    dana.post(base + "/signup", data={"username": "dana", "password": "selftest-pw4", "password2": "selftest-pw4"}, timeout=10)
    eve.post(base + "/signup", data={"username": "eve", "password": "selftest-pw5", "password2": "selftest-pw5"}, timeout=10)

    def upload(sess, path):
        up = [("tracks", (Path(path).name, open(path, "rb")))]
        loc = sess.post(base + "/analyze", files=up, data={"instruments": ["guitar"], "genre": "blues"},
                        allow_redirects=False, timeout=60).headers["Location"]
        job = loc.rsplit("/", 1)[-1]
        t0 = time.time()
        while time.time() - t0 < 300 and sess.get(f"{base}/jobs/{job}/status", timeout=10).json()["state"] not in ("done", "error"):
            time.sleep(1)
        return job

    off_page = dana.get(base + "/memory", timeout=10).text
    check("memory", "the memory is off until the musician switches it on", 'value="1" checked' not in off_page)
    dana.post(base + "/memory", data={"on": "1", "goals": "more odd meters please"}, timeout=10)
    j1, j2 = upload(dana, files["pop4"]), upload(dana, files["waltz3"])
    page1 = dana.get(f"{base}/jobs/{j1}", timeout=10).text
    res1 = dana.get(f"{base}/jobs/{j1}/files/music_results.json", timeout=10).json()[0]
    idea = res1["ideas"][0]
    fb = dana.post(f"{base}/jobs/{j1}/feedback", json={"track": res1["file"], "area": idea["area"],
                                                        "text": idea["idea"], "vote": -1}, timeout=10)
    dana.post(f"{base}/jobs/{j2}/correct", data={"track": "waltz", "field": "meter", "value": "3/4"}, timeout=10)
    stranger = eve.post(f"{base}/jobs/{j1}/feedback", json={"area": "x", "text": "y", "vote": 1}, timeout=10)
    check("memory", "the owner can rate ideas and correct results; someone else can't",
          fb.ok and stranger.status_code == 403 and "Something wrong?" in page1)
    mpage = dana.get(base + "/memory", timeout=10).text
    check("memory", "What we remember lists the habits, the rating and the correction",
          "G major" in mpage and "is 3/4" in mpage and idea["idea"][:40] in mpage and "more odd meters please" in mpage)
    home = dana.get(base + "/", timeout=10).text
    check("memory", "the upload form is pre-filled with the usual line-up and genre",
          'value="guitar" checked' in home and 'value="blues" selected' in home and "Ticked from your earlier uploads" in home)
    j3 = upload(dana, files["modulate"])
    page3 = dana.get(f"{base}/jobs/{j3}", timeout=10).text
    res3 = dana.get(f"{base}/jobs/{j3}/files/music_results.json", timeout=10).json()[0]
    check("memory", "a new track is compared with the earlier ones, and the disliked idea is gone",
          "Next to your earlier tracks" in page3 and res3.get("memory_tracks") == 2
          and not any(i["idea"] == idea["idea"] for i in res3["ideas"]), str(res3.get("past")))
    anon_home = requests.get(base + "/", timeout=10).text
    check("memory", "anonymous visitors get no pre-fill and no memory", "Ticked from your earlier uploads" not in anon_home)
    dana.post(base + "/memory/forget", timeout=10)
    gone = dana.get(base + "/memory", timeout=10).text
    check("memory", "Forget everything deletes ratings, corrections and goals and switches it off",
          idea["idea"][:40] not in gone and "is 3/4" not in gone and "more odd meters please" not in gone
          and 'value="1" checked' not in gone and "Ticked from your earlier uploads" not in
          dana.get(base + "/", timeout=10).text)


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
        if "form" in e:
            b = r.get("blues_form") or {}
            check("facts", f"{n}: {e['form']}" + (f", choruses of ~{e['chorus_sec']} s" if "chorus_sec" in e else ""),
                  b.get("name") == e["form"] and ("chorus_sec" not in e or near(b.get("chorus_sec"), e["chorus_sec"], 5)),
                  f"{b.get('name')}, {b.get('chorus_sec')} s" if b else "no blues form found")
        if e.get("no_song_form"):
            check("facts", f"{n}: no AABA song form claimed", not r.get("song_form"), str(r.get("song_form")))
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
        try:
            get(f"/jobs/{job}/files/tags.json")
            check("live", "a musician's analysis doesn't run the tag search", False, "tags.json was made")
        except urllib.error.HTTPError as e:
            check("live", "a musician's analysis doesn't run the tag search", e.code == 404, f"HTTP {e.code}")

    # "Find music like this" through the running app: one upload plus one song by name
    fields = {"mode": "tags", "songs": "Massive Attack - Teardrop"}
    body = b"".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
                    for k, v in fields.items())
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"tracks\"; filename=\"{track.name}\"\r\n"
             f"Content-Type: audio/wav\r\n\r\n").encode() + track.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(base + "/analyze", data=body, method="POST",
                                 headers={**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        urllib.request.build_opener(NoRedirect).open(req, timeout=60)
        check("live", "tag job accepted", False, "no redirect to a job page")
        return
    except urllib.error.HTTPError as e:
        if e.code != 302:
            check("live", "tag job accepted", False, f"HTTP {e.code}")
            return
        tjob = e.headers["Location"].rsplit("/", 1)[-1]
    t0 = time.time()
    while time.time() - t0 < 400:
        s = json.loads(get(f"/jobs/{tjob}/status"))
        if s["state"] in ("error", "done"):
            break
        time.sleep(2)
    if check("live", "a tag job finishes", s.get("state") == "done",
             f"{s.get('state')}: {s.get('message', '')} after {time.time() - t0:.0f}s"):
        tags = json.loads(get(f"/jobs/{tjob}/files/tags.json"))
        check("live", "the tag job's results page and tags", "Your tags" in get(f"/jobs/{tjob}")
              and any(x["audio"] for x in tags["songs"]), f"{len(tags['songs'])} songs, model: {tags.get('model')}")
        try:
            get(f"/jobs/{tjob}/files/music_report.md")
            check("live", "a tag job skips the musician's analysis, MIDI and AI", False, "music_report.md was made")
        except urllib.error.HTTPError as e:
            check("live", "a tag job skips the musician's analysis, MIDI and AI", e.code == 404, f"HTTP {e.code}")
        named = next((x for x in tags["songs"] if not x["audio"]), {})
        if not named.get("found"):
            warn("live", "the song typed by name was found on MusicBrainz", " | ".join(tags.get("notes", [])))

    # The demos: build them now, so no visitor waits for one, and check they finish
    started = {}
    for slug in ("blues", "band", "electronic"):
        try:
            urllib.request.build_opener(NoRedirect).open(urllib.request.Request(base + f"/demo/{slug}", headers=headers),
                                                         timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 302:
                started[slug] = e.headers["Location"].rsplit("/", 1)[-1]
    t0, states = time.time(), {}
    while time.time() - t0 < 600:
        states = {s: json.loads(get(f"/jobs/{j}/status"))["state"] for s, j in started.items()}
        if states and all(v in ("done", "error") for v in states.values()):
            break
        time.sleep(3)
    check("live", "the three demos are built and ready for visitors", len(started) == 3
          and all(v == "done" for v in states.values()), f"{states} after {time.time() - t0:.0f}s")


def precision_checks(tmp):
    """Measurements that must hold for any track, checked on generated audio where the answer is known:
    tempo to a fraction of a BPM, tuning claimed only when it's real, brightness that tells tracks apart,
    chord names marked when they're a guess, and half/double-speed readings that need real evidence."""
    import librosa

    import musicanalyze as ma
    print("\n== Precision on tracks with known answers ==")
    pops = {bpm: write(Path(tmp) / f"prec_{bpm}.wav", song([("A", "m"), ("F", ""), ("C", ""), ("G", "")], bpm, 4, loops=8))
            for bpm in (100, 128, 137)}
    got = {bpm: ma.analyze(f)["tempo_bpm"] for bpm, f in pops.items()}
    check("precision", "tempo within 0.5 BPM at speeds between the beat tracker's steps (100, 128, 137)",
          all(abs(got[b] - b) <= 0.5 for b in got), str(got))

    # Soft pads that only change chord once a bar: the beat tracker counts bars (25 BPM for a 100 BPM song);
    # the reading must come back to a tempo people tap along to, flagged as approximate.
    pads = {bpm: ma.analyze(write(Path(tmp) / f"prec_pads_{bpm}.wav",
                                  song([("A", "m"), ("F", ""), ("C", ""), ("G", "")], bpm, 4, loops=6, drums=False, soft=True)))
            for bpm in (100, 72)}
    check("precision", "soft pads changing once a bar read at their real tempo (not bars per minute), marked approximate",
          all(abs(r_["tempo_bpm"] - b) <= 1 and r_.get("tempo_approximate") for b, r_ in pads.items()),
          str({b: (r_["tempo_bpm"], r_.get("tempo_approximate")) for b, r_ in pads.items()}))

    # Tuning: the same song played 30 cents flat, in tune, and drifting from 25 flat to in tune halfway through
    base = song([("C", ""), ("A", "m"), ("F", ""), ("G", "")], 100, 4, loops=10)
    flat = lambda y, cents: librosa.resample(y, orig_sr=ma.SR, target_sr=int(ma.SR * 2 ** (cents / 1200)))
    half = len(base) // 2
    cases = {"30 cents flat": (write(Path(tmp) / "tune_flat.wav", flat(base, 30)), lambda c: -36 <= c <= -24),
             "in tune": (write(Path(tmp) / "tune_ok.wav", base), lambda c: abs(c) < 10),  # the report mentions >= 10
             "drifting": (write(Path(tmp) / "tune_drift.wav", np.concatenate([flat(base[:half], 25), base[half:]])),
                          lambda c: abs(c) < 10)}
    res = {n: ma.analyze(f)["tuning_cents"] for n, (f, _) in cases.items()}
    check("precision", "tuning: a real 30-cent detune is reported; in-tune and drifting recordings claim nothing",
          all(ok(res[n]) for n, (_, ok) in cases.items()), str(res))

    # Brightness: a dark string pad and a bright electronic track must not both read "dark/warm"
    dark = ma.analyze(write(Path(tmp) / "prec_orch.wav", orchestral_track()))["sound"]
    bright = ma.analyze(write(Path(tmp) / "prec_edm.wav", electronic_track()))["sound"]
    check("precision", "brightness tells a dark track from a brighter one",
          bright["brightness_hz"] > 1.5 * dark["brightness_hz"] and bright["tone"] != dark["tone"],
          f"{dark['brightness_hz']} Hz {dark['tone']} vs {bright['brightness_hz']} Hz {bright['tone']}")

    # Chord names: clean, known chords are never marked as guesses
    tl = ma.analyze(pops[100])["chord_timeline"]
    check("precision", "known chords are clear (no '?' on a clean recording)",
          all((c.get("clarity") or 0) >= ma.UNCLEAR_CHORD for c in tl), str([(c["chord"], c["clarity"]) for c in tl][:6]))

    # Loudness to the published standards (EBU test signals), and what a mono phone file reports
    sr48 = 48000
    tone = lambda db, secs=20: (10 ** (db / 20)) * np.sin(2 * np.pi * 1000 * np.arange(sr48 * secs) / sr48)
    st_ = lambda x: np.vstack([x, x])
    lufs = ma.integrated_lufs(st_(tone(-23)), sr48)
    lra = ma.loudness_range(st_(np.concatenate([tone(-20), tone(-30)])), sr48)
    tp = ma.true_peak_db(st_(tone(-23)), sr48)
    inter = np.sin(2 * np.pi * (sr48 / 4) * np.arange(sr48) / sr48 + np.pi / 4) * 0.99  # samples sit 3 dB below its peak
    check("precision", "loudness: EBU tone -23.0 LUFS, loudness range 10 LU, true peak of a tone and between samples",
          abs(lufs + 23) < 0.1 and abs(lra - 10) < 0.2 and abs(tp + 23) < 0.1
          and abs(ma.true_peak_db(inter[None, :], sr48) - 20 * np.log10(0.99)) < 0.2,
          f"{lufs:.2f} LUFS, LRA {lra:.2f}, TP {tp:.2f}, inter-sample {ma.true_peak_db(inter[None, :], sr48):.2f}")
    noise = np.random.default_rng(1).standard_normal(sr48 * 3)
    check("precision", "stereo correlation: mono +1, flipped channel -1, unrelated channels ~0",
          ma.stereo_correlation(st_(noise)) > 0.99 and ma.stereo_correlation(np.vstack([noise, -noise])) < -0.99
          and abs(ma.stereo_correlation(np.vstack([noise, np.random.default_rng(2).standard_normal(sr48 * 3)]))) < 0.05)
    s_ = ma.analyze(pops[100])["sound"].get("streaming") or {}
    check("precision", "the analysis reports streaming loudness (a mono file measured on both speakers)",
          s_.get("mono_file") is True and -40 < (s_.get("lufs") or 0) < 0 and s_.get("stereo_correlation") is None, str(s_))

    # Capo: open shapes where they exist, nothing where they don't
    cap = lambda chords, key: ma.guitar_capo({"chords_used": [{"chord": c, "seconds": 10} for c in chords],
                                              "key_index": ma.KEY_NAMES.index(key)})
    g1, g2, g3 = cap(["A", "E", "F#m", "D"], "A major"), cap(["G", "C", "D", "Em"], "G major"), cap(["Bb", "Eb", "F", "Gm"], "Bb major")
    check("precision", "capo: A-E-F#m-D -> capo 2 (G D Em C shapes); G-C-D-Em needs none; Bb-Eb-F-Gm -> capo 3",
          g1["capo"] == 2 and [s["shape"] for s in g1["shapes"]] == ["G", "D", "Em", "C"] and g2["capo"] == 0
          and g3["capo"] == 3, f"{g1 and g1['capo']} / {g2 and g2['capo']} / {g3 and g3['capo']}")
    check("precision", "capo: nothing suggested for power chords only, or a riff that's mostly power chords",
          cap(["E5", "A5", "D5"], "E major") is None and ma.guitar_capo(
              {"chords_used": [{"chord": "D5", "seconds": 60}, {"chord": "G5", "seconds": 30}, {"chord": "Am7", "seconds": 5}],
               "key_index": ma.KEY_NAMES.index("D major")}) is None)

    # Half/double bar length: needs to beat the detected one by more than chance
    on, off = (0.77, 60, "on"), (0.79, 24, "off")
    check("precision", "a half/double-speed form reading needs more than a lucky point or two",
          ma._pick_family({1.0: on, 2.0: off}) == "on"
          and ma._pick_family({1.0: (0.55, 60, "on"), 2.0: (0.95, 24, "off")}) == "off"
          and ma._pick_family({2.0: off}) == "off")


def comparison_checks(tmp):
    """Your track next to a reference: differences you'd hear are named, nothing else - and the reference never
    becomes part of your own results."""
    import accounts
    import musicanalyze as ma
    print("\n== Compared with a reference track ==")
    chords = [("A", "m"), ("F", ""), ("C", ""), ("G", "")]
    mine = write(Path(tmp) / "cmp_mine.wav", song(chords, 100, 4, loops=6))
    ref = write(Path(tmp) / "cmp_reference.wav", song(chords, 128, 4, loops=8))
    out = Path(tmp) / "cmp_out"
    results, report = ma.run_analysis([mine], out, cache_dir=None, reference=ref)
    r = results[0]
    areas = {i["area"]: i["idea"] for i in (r.get("comparison") or {}).get("ideas", [])}
    check("compare", "a reference is analysed and compared: a 100 vs 128 BPM tempo gap is named",
          "tempo" in areas and "128" in areas["tempo"], " | ".join(f"{a}: {i[:60]}" for a, i in areas.items()))
    bare = ma.analyze(write(Path(tmp) / "cmp_bare.wav", song(chords, 100, 4, loops=6, drums=False)))
    check("compare", "a reference with drums against a track without: the drums are named",
          any("drums" in i["idea"] for i in ma.compare_tracks(bare, results[-1])["ideas"]))
    check("compare", "a track compared with itself shows no differences", not ma.compare_tracks(r, r)["ideas"])
    vague = dict(r, tempo_approximate=True)
    check("compare", "no tempo claim when one tempo is only approximate (weak pulse)",
          not any(i["area"] == "tempo" for i in ma.compare_tracks(vague, results[-1])["ideas"]))
    check("compare", "the report shows the comparison and the reference, which stays out of the track table",
          "Next to your reference" in report and "Your reference: cmp_reference.wav" in report
          and "| cmp_reference.wav |" not in report and results[-1].get("role") == "reference")
    check("compare", "the reference isn't counted in the musician's own profile",
          [t["file"] for t in accounts.job_summary(results)["tracks"]] == ["cmp_mine.wav"])


def memory_checks():
    """A musician's memory: firsts and departures are named, matching tracks say nothing, close calls are only
    notes, feedback reorders ideas - on made-up histories where the right answer is known."""
    import memory
    print("\n== A musician's memory ==")
    past = [{"key": "G major", "mode": "Ionian (major)", "meter": "4/4", "tempo_bpm": b, "duration_sec": 180,
             "chords": {"G": 60, "C": 40, "D": 30}, "energy": 6} for b in (90, 100, 110)]
    ctx = [{"keys": ["guitar"], "genre_key": "rock"}, {"keys": ["guitar"], "genre_key": "rock"}, {"keys": ["guitar", "voice"]}]
    fb = [{"area": "harmony", "text": "Borrow from G minor: try a minor iv", "vote": 1},
          {"area": "drums", "text": "The kick stays on the main beats. Add one push", "vote": -1}]
    mem = memory.build(past, ctx, fb, [], goals="darker songs")
    check("memory", "habits: the usual line-up and genre (ticked in most saved analyses), tempo range",
          mem["usual_instruments"] == ["guitar"] and mem["usual_genre"] == "rock" and mem["tempo"]["min"] == 90
          and mem["tempo"]["max"] == 110, f"{mem['usual_instruments']} {mem['usual_genre']} {mem['tempo']}")
    new = {"key": "A minor", "mode": "Aeolian (natural minor)", "mode_confidence": "high", "tempo_bpm": 140,
           "meter": {"label": "3/4", "confidence": "detected"}, "duration_sec": 170,
           "chords_used": [{"chord": "Am", "seconds": 40}, {"chord": "G", "seconds": 20}], "sound": {}}
    notes = " ".join(memory.compare_with_past(new, mem))
    same = {"key": "G major", "mode": "Ionian (major)", "tempo_bpm": 100, "meter": {"label": "4/4", "confidence": "detected"},
            "duration_sec": 180, "chords_used": [{"chord": "G", "seconds": 40}], "sound": {}}
    check("memory", "a new track's firsts are named: minor key, faster than ever, 3/4, new chords",
          all(x in notes for x in ("first saved track in a minor key", "Faster than anything", "first track in 3/4", "Am")),
          notes)
    check("memory", "a track like your others gets no comparison, and with one past track there's no 'usual' yet",
          memory.compare_with_past(same, mem) == [] and
          memory.compare_with_past(new, memory.build(past[:1], ctx[:1], [], [])) == [])
    vague = dict(same, tempo_bpm=50, tempo_approximate=True)
    check("memory", "close call: an uncertain tempo gets a note pointing at your usual range; a clear one doesn't",
          "about 100 BPM" in " ".join(memory.close_call_notes(vague, mem))
          and memory.close_call_notes(dict(same, tempo_bpm=50), mem) == [])
    ideas = [{"area": "drums", "idea": "The kick stays on the main beats. Add one push - a kick on the 'a' of 2"},
             {"area": "sound", "idea": "Very little high-end air"},
             {"area": "harmony", "idea": "Borrow from A minor: try a minor iv - Dm"}]
    ranked = [i["area"] for i in memory.rank_ideas(ideas, mem)]
    check("memory", "ideas: a disliked one is dropped, a liked kind comes first", ranked == ["harmony", "sound"], str(ranked))
    thin = memory.build(past[:1] + [dict(past[1], tempo_approximate=True)], ctx[:2], [], [])
    check("memory", "one reliable tempo isn't a range: no tempo claim from it",
          thin["tempo"] is None and "Faster" not in " ".join(memory.compare_with_past(new, thin)))
    check("memory", "the AI gets the musician's habits and goals", "darker songs" in memory.ai_profile(mem)
          and "G major" in memory.ai_profile(mem))


def tag_checks(tmp, online=False):
    """"Find music like this": parsing song names, weighting, the light analysis, the style model, and that a tag
    job does none of the musician's work."""
    import discover
    import musicanalyze as ma
    import tagger
    print("\n== Tags for finding music ==")
    songs = discover.parse_song_list("1. Massive Attack - Teardrop\nRoads by Portishead\n"
                                     "Khruangbin – Maria También [Official Audio]\nBecky G - Mayores ft. Bad Bunny")
    check("tags", "song names in common formats are read",
          [(x["artist"], x["title"]) for x in songs] == [("Massive Attack", "Teardrop"), ("Portishead", "Roads"),
                                                         ("Khruangbin", "Maria También"), ("Becky G", "Mayores")],
          str(songs))
    check("tags", "a YouTube 'Title - Artist' is turned round using the channel name",
          discover.split_name("Teardrop - Massive Attack (Official Video)", "Massive Attack")
          == {"artist": "Massive Attack", "title": "Teardrop"})
    html = ('<html><script id="__NEXT_DATA__" type="application/json">' + json.dumps(
        {"props": {"pageProps": {"state": {"data": {"entity": {"name": "My list", "trackList": [
            {"title": "Roads", "subtitle": "Portishead"}, {"title": "Teardrop", "subtitle": "Massive Attack, Liz Fraser"}]}}}}}})
        + "</script></html>")
    check("tags", "a Spotify playlist page is read (names only)",
          discover.parse_spotify_embed(html) == ("My list", [{"artist": "Portishead", "title": "Roads"},
                                                            {"artist": "Massive Attack", "title": "Teardrop"}]))
    check("tags", "Spotify links of every shape are recognised",
          discover.spotify_id("https://open.spotify.com/intl-de/playlist/37i9dQZF1DX?si=1") == ("playlist", "37i9dQZF1DX")
          and discover.spotify_id("spotify:album:abc") == ("album", "abc"))
    agg = discover.aggregate([{"weight": 1.0, "tags": {"style": {"shoegaze": 1.0, "rock": 1.0}}},
                              {"weight": 0.4, "tags": {"style": {"dream pop": 1.0}}}])
    st_ = {t["tag"]: t["score"] for t in agg["style"]}
    check("tags", "a song we listened to counts more than one known only by name",
          abs(st_["shoegaze"] - 1 / 1.4) < 0.01 and abs(st_["dream pop"] - 0.4 / 1.4) < 0.01, str(st_))
    plan = discover.search_plan(agg, set())
    check("tags", "broad genres are shown but never searched",
          "rock" in [t["tag"] for t in agg.get("genre", [])] and "rock" not in [t["tag"] for t in plan["styles"]])
    check("tags", "tag search links use each site's tag address",
          discover.links("Trip Hop")["Bandcamp"] == "https://bandcamp.com/tag/trip-hop"
          and discover.links("Trip Hop")["Last.fm"] == "https://www.last.fm/tag/trip%20hop/artists")

    # Lesser-known artists, on canned MusicBrainz / ListenBrainz answers (known right result)
    fake_artists = [  # name, mbid, tags, listeners
        ("Massive Attack", "m1", {"trip hop": 20, "downtempo": 10}, 326000),   # already in the list: left out
        ("Hugo Kant", "a1", {"trip hop": 3, "downtempo": 2, "nu jazz": 1}, 4500),
        ("Bows", "a2", {"trip hop": 1}, 1800),
        ("Moby", "a3", {"trip hop": 5, "downtempo": 4}, 310000),              # too popular
        ("Bedroom Demo", "a4", {"trip hop": 1, "downtempo": 1}, 3),           # hardly any listeners yet
        ("Nobody Knows", "a5", {"trip hop": 1, "downtempo": 1}, None),        # no listening data
    ]
    saved_get = discover._get

    def fake_get(url, host_gap=0.0, ua=None, timeout=20, retries=2, data=None):
        if "musicbrainz.org/ws/2/artist?" in url:
            return json.dumps({"artists": [{"id": i, "name": n, "tags": [{"name": t, "count": c} for t, c in tg.items()]}
                                           for n, i, tg, _ in fake_artists]})
        if "listenbrainz" in url:
            ids = json.loads(data)["artist_mbids"]
            pop = {i: u for _, i, _, u in fake_artists}
            return json.dumps([{"artist_mbid": i, "total_user_count": pop.get(i)} for i in ids])
        raise OSError("unexpected " + url)
    discover._get = fake_get
    try:
        agg_ = {"style": [{"tag": "trip hop", "score": 0.6}, {"tag": "downtempo", "score": 0.5}], "mood": [], "sound": []}
        got, searched = discover.lesser_known(agg_, {"Massive Attack": "m1"}, discover.Cache(None))
        none_, _ = discover.lesser_known({"style": [], "mood": [], "sound": []}, {}, discover.Cache(None))
    finally:
        discover._get = saved_get
    check("tags", "lesser-known artists: your own artists, the famous and the barely-played are left out; "
          "more shared tags rank higher", [a["name"] for a in got] == ["Hugo Kant", "Bows"]
          and searched[:2] == ["trip hop", "downtempo"], str([(a["name"], a["listeners"]) for a in got]))
    check("tags", "nothing specific to search for -> no artist search", none_ == [])
    check("tags", "a description is read as tags of its own",
          discover.parse_description("Melancholic, trip hop; female vocals and lo-fi") ==
          ["melancholic", "trip hop", "female vocals", "lo-fi"])

    # The light analysis: what tags need, quicker, same verdicts on drums
    blues = write(Path(tmp) / "Test Artist - Tag Blues.wav", guitar_blues())
    edm = write(Path(tmp) / "tag_electronic.wav", electronic_track())
    t0 = time.time()
    lb, le = ma.analyze(blues, light=True), ma.analyze(edm, light=True)
    secs = time.time() - t0
    check("tags", "light analysis: solo guitar has no drums or bass, the electronic track has both",
          lb["drums"].get("present") is False and lb["bass"].get("present") is False
          and le["drums"].get("present") and le["bass"].get("present"),
          f"guitar {lb['drums'].get('present')}/{lb['bass'].get('present')}, electronic {le['drums'].get('present')}/{le['bass'].get('present')}")
    check("tags", "light analysis: tempos right, 12-bar blues found, musician-only parts skipped",
          # (the sparse guitar shuffle reads at half time, 70 - the full analysis reads it the same way)
          any(near(lb["tempo_bpm"], t, 4) for t in (140, 70)) and near(le["tempo_bpm"], 124, 4) and bool(lb.get("blues_form"))
          and lb.get("light") and not lb["top_line"].get("present") and not lb.get("bass_line"),
          f"{lb['tempo_bpm']} / {le['tempo_bpm']} BPM in {secs:.0f}s")

    # A whole tag job, offline, with the stand-in model: tags only, none of the musician's extras
    saved = {k: os.environ.get(k) for k in ("MUSIC_TAGGER", "MUSIC_OFFLINE")}
    os.environ.update(MUSIC_TAGGER="fake", MUSIC_OFFLINE="1")
    try:
        out = Path(tmp) / "tag_job"
        data, report = discover.run_tags([blues, edm], out, song_text="Portishead - Roads")
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    made = sorted(x.name for x in out.iterdir())
    check("tags", "a tag job writes tags only (no MIDI, sheet music, AI or musician report)",
          made == ["tags.json", "tags_report.md"], ", ".join(made))
    styles = [t["tag"] for t in data["tags"].get("style", [])]
    check("tags", "tags come from the audio (12-bar blues) and the model", "12-bar blues" in styles and
          any(t in styles for t in ("downtempo", "delta blues")), ", ".join(styles[:6]))
    check("tags", "the file's name 'Artist - Title' is used for the lookup",
          any(s_["name"] == "Test Artist - Tag Blues" for s_ in data["songs"]), str([s_["name"] for s_ in data["songs"]]))
    check("tags", "with lookups off, the named song is reported as not found (not silently dropped)",
          any("Portishead - Roads" in n for n in data["notes"]), " | ".join(data["notes"]))

    # The real style model (on the server)
    kind = tagger.backend()
    if kind == "essentia":
        try:
            t0 = time.time()
            pred = tagger.predict([str(blues), str(edm)], ma.load_audio)
            pb, pe = pred.get(str(blues), {}), pred.get(str(edm), {})
            check("tags", "the style model runs and names styles", bool(pb.get("styles")) and bool(pe.get("styles")),
                  f"{time.time() - t0:.0f}s")
            fam = lambda p_: {lab.split("---")[0] for lab, _ in p_.get("styles", [])[:3]}
            if "Electronic" not in fam(pe):
                warn("tags", "style model: the synthetic electronic track wasn't called Electronic",
                     str(pe.get("styles", [])[:3]))
            print(f"    model says: guitar blues → {[x for x, _ in pb.get('styles', [])[:3]]}; "
                  f"electronic → {[x for x, _ in pe.get('styles', [])[:3]]}")
        except Exception as e:
            check("tags", "the style model runs and names styles", False, str(e)[:300])
    elif os.environ.get("MUSIC_ESSENTIA_PYTHON"):
        warn("tags", "the style model isn't installed (the Docker build skipped it); tags still work without styles "
             "from the audio", "")
    else:
        print("    (style model not set up on this machine - skipped)")

    if online:  # outside services: a failure is a warning, the app copes without them
        try:
            got = discover.lookup({"artist": "Massive Attack", "title": "Teardrop"}, discover.Cache(None))
            (check if "trip hop" in got else warn)("tags", "MusicBrainz lookup by name works", ", ".join(got))
        except Exception as e:
            warn("tags", "MusicBrainz lookup by name works", str(e)[:200])
        try:
            agg_ = {"style": [{"tag": "trip hop", "score": 0.6}, {"tag": "downtempo", "score": 0.5}], "mood": [], "sound": []}
            got, _ = discover.lesser_known(agg_, {"Massive Attack": None}, discover.Cache(None))
            ok_ = got and all(discover.MIN_LISTENERS <= a["listeners"] <= discover.MAX_LISTENERS for a in got)
            (check if ok_ else warn)("tags", "lesser-known artists found live (MusicBrainz + ListenBrainz)",
                                     ", ".join(f"{a['name']} ({a['listeners']})" for a in got[:5]))
        except Exception as e:
            warn("tags", "lesser-known artists found live (MusicBrainz + ListenBrainz)", str(e)[:200])


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
        try:
            precision_checks(tmp)
        except Exception:
            check("precision", "precision checks ran", False, traceback.format_exc(limit=3))
        try:
            comparison_checks(tmp)
        except Exception:
            check("compare", "comparison checks ran", False, traceback.format_exc(limit=3))
        try:
            memory_checks()
        except Exception:
            check("memory", "memory checks ran", False, traceback.format_exc(limit=3))
        try:
            tag_checks(tmp, online=bool(args.live))
        except Exception:
            check("tags", "tag checks ran", False, traceback.format_exc(limit=3))
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
    print(f"\n{'=' * 60}\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed"
          + (f", {len(WARNINGS)} warning(s)" if WARNINGS else ""))
    for g, n, _, d in failed:
        print(f"  FAIL [{g}] {n}" + (f": {d}" if d else ""))
    for g, n, d in WARNINGS:
        print(f"  WARN [{g}] {n}" + (f": {d}" if d else ""))
    print("RESULT:", ("PASS ✅" + (" (with warnings: an outside service, see WARN lines)" if WARNINGS else ""))
          if not failed else "FAIL ❌")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()  # a crash inside a native library prints where it happened instead of dying silently
    main()
