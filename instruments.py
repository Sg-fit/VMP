"""Instrument-level analysis for musicanalyze.py.

1. Separate the drums from everything else ("music"): Demucs (an AI source-separation model)
   when installed, otherwise a lighter harmonic/percussive split.
2. Drums: kick / snare / hi-hat hits placed on a 16th-note grid -> the main groove pattern,
   how consistent it is, and likely fills.
3. Bass: pitch-track the lowest line in the drumless audio -> one bass note per beat.

Returns plain numbers (pitch classes as ints); musicanalyze.py names and formats them.
"""

import os

import numpy as np

# auto = Demucs if installed, else simple; "demucs" = require it; "simple" = never use it.
SEPARATION = os.environ.get("MUSIC_SEPARATION", "auto").lower()
DRUM_BANDS = {"Kick": (30, 120), "Snare": (180, 1200), "Hi-hat": (4000, 11000)}
_demucs_model = None


# ---------------------------------------------------------------- separation

def separate(y, sr):
    """Return (drums, music, method) where music is everything except the drums."""
    import librosa
    if SEPARATION != "simple":
        try:
            drums, music = _demucs_split(y, sr)
            return drums, music, "demucs"
        except ImportError:
            if SEPARATION == "demucs":
                raise
    music, drums = librosa.effects.hpss(y)
    return drums, music, "simple"


def _demucs_split(y, sr):
    global _demucs_model
    import librosa
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if _demucs_model is None:
        _demucs_model = get_model("htdemucs")
        _demucs_model.eval()
    model = _demucs_model
    y44 = librosa.resample(y, orig_sr=sr, target_sr=model.samplerate)
    wav = torch.from_numpy(np.stack([y44, y44])).float()[None]
    with torch.no_grad():
        stems = apply_model(model, wav, split=True, overlap=0.1, progress=False)[0].mean(dim=1).numpy()
    drums = stems[model.sources.index("drums")]
    music = stems.sum(axis=0) - drums  # bass + other + vocals

    def back(x):
        x = librosa.resample(x, orig_sr=model.samplerate, target_sr=sr)[:len(y)]
        return np.pad(x, (0, len(y) - len(x))).astype(np.float32)
    return back(drums), back(music)


# ---------------------------------------------------------------- drums

def drum_analysis(drums, sr, hop, beat_times, total_energy):
    """Where in the bar the drums hit, and whether each spot is kick- or snare-heavy.

    Rather than labelling every single hit (fragile on demos: one full-kit hit lights up every
    band), it averages each band's attack strength at each 16th-note position across all bars,
    then compares bands position by position. Assumes 4/4.
    """
    import librosa
    share = float(np.sum(drums ** 2) / (total_energy + 1e-12))
    out = {"present": share > 0.03, "energy_share_pct": round(100 * share)}
    if not out["present"] or len(beat_times) < 17:
        return out

    S = np.abs(librosa.stft(drums, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    frame_t = librosa.times_like(S[0], sr=sr, hop_length=hop)
    slot_t = np.concatenate([np.linspace(beat_times[i], beat_times[i + 1], 4, endpoint=False)
                             for i in range(len(beat_times) - 1)])
    idx = np.searchsorted(frame_t, slot_t)

    def at_slots(env):  # strongest attack within ±2 frames of each 16th-note position
        return np.array([env[max(0, i - 2):i + 3].max() for i in idx])

    db = librosa.amplitude_to_db(S, ref=np.max)
    overall = at_slots(librosa.onset.onset_strength(S=db, sr=sr, hop_length=hop))
    band = {}
    for name, (lo, hi) in DRUM_BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        if (S[sel] ** 2).sum() >= 0.01 * (S ** 2).sum():
            band[name] = at_slots(librosa.onset.onset_strength(S=db[sel], sr=sr, hop_length=hop))

    def bar_profile(v, offset):
        n = (len(v) - 4 * offset) // 16
        return v[4 * offset:4 * offset + 16 * n].reshape(n, 16)

    z = lambda p: (p - p.mean()) / (p.std() + 1e-9)
    kick, snare = band.get("Kick"), band.get("Snare")

    # Bar 1 starts where the strongest, most kick-heavy hit usually is.
    def downbeat_score(o):
        score = z(bar_profile(overall, o).mean(0))[0]
        if kick is not None and snare is not None:
            score += (z(bar_profile(kick, o).mean(0)) - z(bar_profile(snare, o).mean(0)))[0]
        return score
    offset = max(range(4), key=downbeat_score)
    bars = bar_profile(overall, offset)
    if len(bars) < 4:
        out["present"] = False
        return out
    # Ignore bars where the drums are silent (intros/breakdowns).
    active = bars.max(axis=1) > 0.25 * np.percentile(bars.max(axis=1), 90)
    bars = bars[active]
    hit = z(bars.mean(0))

    grid = []
    k_minus_s = (z(bar_profile(kick, offset)[active].mean(0)) - z(bar_profile(snare, offset)[active].mean(0))
                 if kick is not None and snare is not None else np.zeros(16))
    for i in range(16):
        if hit[i] <= 0.3:
            grid.append(".")
        elif k_minus_s[i] > 0.4:
            grid.append("K")
        elif k_minus_s[i] < -0.4:
            grid.append("S")
        else:
            grid.append("X" if hit[i] > 0.9 else "x")
    hats = None
    if "Hi-hat" in band:
        h = z(bar_profile(band["Hi-hat"], offset)[active].mean(0))
        hats = "".join("h" if v > 0.5 else "." for v in h)

    # How much bars repeat the main groove, and isolated bars that break it (likely fills).
    mean = bars.mean(0)
    corr = np.array([np.corrcoef(b, mean)[0, 1] if b.std() > 0 else 0 for b in bars])
    busy = bars.sum(axis=1)
    bar_times = np.array([beat_times[min(4 * offset + 4 * b, len(beat_times) - 1)]
                          for b in range(len(active))])[active]
    fills = [float(bar_times[i]) for i in range(1, len(bars) - 1)
             if corr[i] < 0.15 and busy[i] > 1.2 * np.median(busy) and corr[i - 1] > 0.3 and corr[i + 1] > 0.3]

    g = "".join(grid)
    traits = []
    if all(g[i] in "KX" for i in (0, 4, 8, 12)):
        traits.append("four-on-the-floor")
    if g[4] in "SX" and g[12] in "SX" and "S" in (g[4], g[12]):
        traits.append("backbeat (snare on 2 and 4)")
    elif g[8] in "SX" and g[4] == "." and g[12] == ".":
        traits.append("half-time feel (snare on 3)")
    if (g[0] != "." and g[3] != "." and g[6] != ".") or (g[8] != "." and g[11] != "." and g[14] != "."):
        traits.append("3+3+2 (tresillo) accents")
    if any(g[i] == "K" for i in range(16) if i % 4):
        traits.append("syncopated kick")
    if hats and hats.count("h") >= 7:
        traits.append("steady hi-hat/cymbal pulse")

    r = float(np.mean(corr))
    out.update({
        "grid": g,
        "hats": hats,
        "repetition": "tight (very repetitive)" if r > 0.6 else "medium" if r > 0.35 else "loose / varied",
        "repetition_score": round(r, 2),
        "fills": fills[:6],
        "traits": traits,
        "bars": int(active.sum()),
    })
    return out


# ---------------------------------------------------------------- bass

def bass_notes(music, sr, btimes, tuning):
    """One MIDI bass note per beat span (None where the bass rests), from the drumless audio."""
    import librosa
    import scipy.signal as ss
    low_sr = 8000
    yb = librosa.resample(music, orig_sr=sr, target_sr=low_sr)
    yb = ss.sosfiltfilt(ss.butter(4, 260, btype="low", fs=low_sr, output="sos"), yb)
    f0, voiced, prob = librosa.pyin(yb, fmin=30, fmax=280, sr=low_sr, frame_length=2048, hop_length=128)
    times = librosa.times_like(f0, sr=low_sr, hop_length=128)
    midi = librosa.hz_to_midi(np.where(voiced, f0, np.nan)) - tuning
    notes = []
    for a, b in zip(btimes[:-1], btimes[1:]):
        span = (times >= a) & (times < b)
        good = span & voiced & (prob > 0.1)
        notes.append(int(np.rint(np.nanmedian(midi[good]))) if span.sum() and good.sum() >= 0.3 * span.sum() else None)
    return notes


# ---------------------------------------------------------------- top line (best-effort melody)

def top_line(music, sr, btimes, tuning, lo_midi=45, hi_midi=81):
    """The most salient pitch (A2-A5) per 8th note, from a harmonic-sum of the CQT.

    This follows whatever pitched part is loudest. In a full-band phone recording that's often the
    chord instrument rather than the melody, so musicanalyze checks how much of it is just chord
    tones before calling it a melody. Returns a MIDI note (or None) per 8th note."""
    import librosa
    fmin_midi = lo_midi - 12
    n_bins = hi_midi - fmin_midi + 29
    C = np.abs(librosa.cqt(music, sr=sr, hop_length=512, fmin=librosa.midi_to_hz(fmin_midi),
                           n_bins=n_bins, bins_per_octave=12, tuning=tuning))
    sal = np.zeros_like(C)
    for off, w in ((0, 1.0), (12, 0.8), (19, 0.6), (24, 0.5), (28, 0.4)):  # fundamental + overtones
        sal[:n_bins - off] += w * C[off:]
    lo, hi = lo_midi - fmin_midi, hi_midi - fmin_midi
    best = sal[lo:hi].argmax(axis=0) + lo_midi
    strength = sal[lo:hi].max(axis=0)
    floor = np.percentile(strength, 30)
    times = librosa.times_like(C[0], sr=sr, hop_length=512)
    grid = np.concatenate([np.linspace(a, b, 2, endpoint=False) for a, b in zip(btimes[:-1], btimes[1:])]
                          + [[btimes[-1]]])
    notes = []
    for a, b in zip(grid[:-1], grid[1:]):
        sel = (times >= a) & (times < b)
        if not sel.any() or strength[sel].mean() < floor:
            notes.append(None)
            continue
        vals, counts = np.unique(best[sel], return_counts=True)
        notes.append(int(vals[np.argmax(counts)]))
    return notes, grid
