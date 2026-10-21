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


def separation_method():
    """The method separate() will actually use ("demucs" or "simple"), e.g. for cache keys."""
    if SEPARATION == "simple":
        return "simple"
    import importlib.util
    return "demucs" if importlib.util.find_spec("demucs") else "simple"


# ---------------------------------------------------------------- separation

def separate(y, sr, fast=False):
    """Return (drums, music, method) where music is everything except the drums.
    fast=True (tag mode): a coarser harmonic/percussive split, 3-4x quicker; on the test recordings it gave the same
    drums / no-drums verdicts as the full one."""
    import librosa
    if fast:
        D = librosa.stft(y, n_fft=1024, hop_length=512)
        H, P = librosa.decompose.hpss(D, kernel_size=15)
        return (librosa.istft(P, hop_length=512, length=len(y)), librosa.istft(H, hop_length=512, length=len(y)),
                "fast")
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

DRUMS_SURE = 0.0015  # hits at least this noise-like are certainly drums; 0.0008-0.0015 is reported as unsure


def hits_are_drums(drums, sr):
    """Separating drums from the rest also catches the attacks of plucked or struck strings (a guitar's
    pick, a piano's hammer), which can be a quarter of a guitar's energy. Real drums are noise (snare,
    hats) or a pitchless thump; string attacks still carry the strings' harmonics. Measured as the median
    spectral flatness (1-8 kHz) of the hits: solo guitar 0.0002, piano 0.0001; phone recordings of a
    band with drums 0.003-0.007; synthetic drums 0.07. Returns (is_drums, flatness)."""
    import librosa
    P = np.abs(librosa.stft(drums, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    env = librosa.onset.onset_strength(S=librosa.power_to_db(P), sr=sr, hop_length=512)
    hits = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=512)
    if len(hits) < 8:
        return False, 0.0
    flat = librosa.feature.spectral_flatness(S=P[(f >= 1000) & (f <= 8000)])[0]
    value = float(np.median(flat[hits]))
    return value >= 0.0008, value


OFFBEAT_KICK_RATIO = 2.0  # kicks at least twice as strong halfway between the beats as on them: the grid is off
EVEN_KICKS = 0.6          # four-on-the-floor: the weakest of the four beats' kicks is at least 60% of the strongest


def onto_kicks(drums, sr, beat_frames, hop):
    """In house and techno the off-beat hi-hat and bass are often the sharpest attacks, so the beat tracker can
    count them as the beat - then the kick lands halfway between "beats". The kick marks the beat in drum-based
    music, so: compare the kick band's attacks on the beats with those halfway between them, and if the kicks are
    clearly in between (OFFBEAT_KICK_RATIO), move every beat half a beat on and onto the nearest kick (within an
    8th of a beat). Returns the corrected beat frames, or None when the beats already sit on the kicks."""
    import librosa
    beat_frames = np.asarray(beat_frames)
    if len(beat_frames) < 9:
        return None
    S = np.abs(librosa.stft(drums, n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    lo, hi = DRUM_BANDS["Kick"]
    # the rise in power, not in dB: a bass note rising out of silence jumps as many dB as a kick, but the kick
    # carries far more power
    P = (S[(f >= lo) & (f < hi)] ** 2).sum(axis=0)
    env = np.maximum(0.0, np.diff(P, prepend=P[0]))
    peak = lambda i: env[max(0, i - 1):i + 2].max() if 0 <= i < len(env) else 0.0
    mids = (beat_frames[:-1] + beat_frames[1:]) // 2
    on = np.median([peak(int(i)) for i in beat_frames[:-1]])
    off = np.median([peak(int(i)) for i in mids])
    if not (off > 0 and off >= OFFBEAT_KICK_RATIO * max(on, 1e-6)):
        return None
    period = float(np.median(np.diff(beat_frames)))
    reach = max(1, int(round(period / 8)))
    moved = []
    for m in np.round(beat_frames + period / 2).astype(int):
        if m >= len(env):
            break
        a = max(0, m - reach)
        moved.append(a + int(np.argmax(env[a:m + reach + 1])))
    return np.array(sorted(set(moved)), dtype=int)


def drum_analysis(drums, sr, hop, beat_times, total_energy, beats_per_bar=4, are_drums=True):
    """Where in the bar the drums hit, and whether each spot is kick- or snare-heavy.

    Rather than labelling every single hit (fragile on demos: one full-kit hit lights up every
    band), it averages each band's attack strength at each 16th-note position across all bars,
    then compares bands position by position. Assumes 4/4.
    """
    import librosa
    share = float(np.sum(drums ** 2) / (total_energy + 1e-12))
    out = {"present": share > 0.03 and are_drums, "energy_share_pct": round(100 * share)}
    if share > 0.03 and not are_drums:
        out["string_attacks"] = True  # percussive, but it's picking/strumming, not a drum kit
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
    band = {}
    for name, (lo, hi) in DRUM_BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        if (S[sel] ** 2).sum() >= 0.01 * (S ** 2).sum():
            if name == "Kick":  # the kick is the big low-end hit: its rise in power, not in dB (see onto_kicks)
                P = (S[sel] ** 2).sum(axis=0)
                band[name] = at_slots(np.maximum(0.0, np.diff(P, prepend=P[0])))
            else:
                band[name] = at_slots(librosa.onset.onset_strength(S=db[sel], sr=sr, hop_length=hop))
    # Each drum voice counts equally: an onset curve over all frequencies is ruled by whatever covers the most
    # bins - the hi-hats - so a kick (a few low bins) could vanish from the grid of a house track.
    scaled = [v / (np.percentile(v, 90) + 1e-9) for v in band.values()]
    overall = (np.mean(scaled, axis=0) if len(scaled) >= 2 else
               at_slots(librosa.onset.onset_strength(S=db, sr=sr, hop_length=hop)))

    def bar_profile(v, offset):
        n = (len(v) - 4 * offset) // slots
        return v[4 * offset:4 * offset + slots * n].reshape(n, slots)

    z = lambda p: (p - p.mean()) / (p.std() + 1e-9)
    slots = 4 * beats_per_bar  # 16 in 4/4, 12 in 3/4
    kick, snare = band.get("Kick"), band.get("Snare")

    # Bar 1 starts where the strongest, most kick-heavy hit usually is.
    def downbeat_score(o):
        score = z(bar_profile(overall, o).mean(0))[0]
        if kick is not None and snare is not None:
            score += (z(bar_profile(kick, o).mean(0)) - z(bar_profile(snare, o).mean(0)))[0]
        return score
    offset = max(range(beats_per_bar), key=downbeat_score)
    bars = bar_profile(overall, offset)
    if len(bars) < 4:
        out["present"] = False
        return out
    # Ignore bars where the drums are silent (intros/breakdowns).
    active = bars.max(axis=1) > 0.25 * np.percentile(bars.max(axis=1), 90)
    bars = bars[active]
    hit = z(bars.mean(0))

    grid = []
    kz = z(bar_profile(kick, offset)[active].mean(0)) if kick is not None else np.zeros(slots)
    sz = z(bar_profile(snare, offset)[active].mean(0)) if snare is not None else np.zeros(slots)
    k_minus_s = kz - sz if kick is not None and snare is not None else np.zeros(slots)
    for i in range(slots):
        if hit[i] <= 0.3:
            grid.append(".")
        elif k_minus_s[i] > 0.4 and (kick is None or kz[i] > 0):  # and the kick band itself hits here
            grid.append("K")
        elif k_minus_s[i] < -0.4 and (snare is None or sz[i] > 0):
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
    bar_times = np.array([beat_times[min(offset + beats_per_bar * b, len(beat_times) - 1)]
                          for b in range(len(active))])[active]
    fills = [float(bar_times[i]) for i in range(1, len(bars) - 1)
             if corr[i] < 0.15 and busy[i] > 1.2 * np.median(busy) and corr[i - 1] > 0.3 and corr[i + 1] > 0.3]

    g = "".join(grid)
    traits = []
    if beats_per_bar == 3:
        traits.append("3/4 (waltz) pattern")
    if beats_per_bar != 4:  # the trait rules below are written for 16-step (4/4) bars
        out.update({"grid": g, "hats": hats, "beats_per_bar": beats_per_bar,
                    "repetition": "tight (very repetitive)" if float(np.mean(corr)) > 0.6 else
                    "medium" if float(np.mean(corr)) > 0.35 else "loose / varied",
                    "repetition_score": round(float(np.mean(corr)), 2), "fills": fills[:6], "traits": traits,
                    "bars": int(active.sum())})
        return out
    # every beat carries a kick: from the grid, or from the kick band on its own (in a house mix the hats and
    # the off-beat bass outvote the kick in the grid) - four kicks of about the same weight, louder than anything
    # in the kick band between the beats (a snare's low thump on 2 and 4 is much weaker than a kick)
    kp = bar_profile(kick, offset)[active].mean(0) if kick is not None else np.zeros(slots)
    on_beats, offs = kz[[0, 4, 8, 12]], np.delete(kz, [0, 4, 8, 12])
    if all(g[i] in "KX" for i in (0, 4, 8, 12)) or (
            kick is not None and on_beats.min() > 0.5 and on_beats.min() > offs.max()
            and kp[[0, 4, 8, 12]].min() >= EVEN_KICKS * kp[[0, 4, 8, 12]].max()):
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
        "beats_per_bar": beats_per_bar,
        "repetition": "tight (very repetitive)" if r > 0.6 else "medium" if r > 0.35 else "loose / varied",
        "repetition_score": round(r, 2),
        "fills": fills[:6],
        "traits": traits,
        "bars": int(active.sum()),
    })
    return out


# ---------------------------------------------------------------- bass

def bass_notes(music, sr, btimes, tuning):
    """Track the bass in the drumless audio. Returns (per_beat, notes):
    per_beat: one MIDI note (or None where the bass rests) per beat span - for summaries;
    notes:    a note-by-note transcription [(start s, end s, midi, peak dB)] - for the bass MIDI file."""
    import librosa
    import scipy.signal as ss
    low_sr = 4000  # bass lives below 280 Hz; 4 kHz is plenty and ~2.5x faster than 8 kHz
    yb = librosa.resample(music, orig_sr=sr, target_sr=low_sr)
    yb = ss.sosfiltfilt(ss.butter(4, 260, btype="low", fs=low_sr, output="sos"), yb)
    f0, voiced, prob = librosa.pyin(yb, fmin=30, fmax=280, sr=low_sr, frame_length=1024, hop_length=128)
    times = librosa.times_like(f0, sr=low_sr, hop_length=128)
    good_all = voiced & (prob > 0.1)
    midi = np.where(good_all, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)) - tuning, np.nan)
    per_beat = []
    for a, b in zip(btimes[:-1], btimes[1:]):
        span = (times >= a) & (times < b)
        good = span & good_all
        per_beat.append(int(np.rint(np.nanmedian(midi[good]))) if span.sum() and good.sum() >= 0.3 * span.sum() else None)
    return per_beat, _transcribe_bass(yb, low_sr, times, midi)


def _transcribe_bass(yb, low_sr, times, midi, hop=32):
    """Note-by-note bass: attacks in the bass band set the rhythm (so repeated notes stay separate),
    the pitch track names each note, and a span with no steady pitch (a kick drum) is skipped.
    Tested on synthetic bass lines with known notes: 88-100% of notes found, 96-100% of reported notes
    real, 94-100% right pitch, starts within 50 ms."""
    import librosa
    env = librosa.onset.onset_strength(y=yb, sr=low_sr, hop_length=hop, n_fft=256, lag=2, max_size=1)
    env = env / (env.max() + 1e-9)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=low_sr, hop_length=hop, backtrack=True,
                                        units="time", delta=0.12, wait=max(1, int(0.06 * low_sr / hop)))
    rms = librosa.feature.rms(y=yb, frame_length=256, hop_length=hop)[0]
    rt = librosa.times_like(rms, sr=low_sr, hop_length=hop)
    rdb = 20 * np.log10(rms + 1e-9)
    bounds = list(onsets) + [len(yb) / low_sr]
    notes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        vals = midi[(times >= a + 0.04) & (times < min(b, a + 0.5))]  # skip the attack, stay in this note
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2:
            continue  # no steady pitch: not a bass note
        seg = (rt >= a) & (rt < b)
        peak = float(rdb[seg].max()) if seg.any() else -60.0
        fade = rt[seg][rdb[seg] < peak - 18] if seg.any() else np.array([])
        fade = fade[fade > a + 0.05]
        n = int(np.rint(np.median(vals)))
        end = float(fade[0]) if len(fade) else float(b)
        if notes and notes[-1][2] == n and notes[-1][1] >= a - 0.01 and not _replucked(yb, low_sr, a, n):
            # Same note still ringing and no new attack at its pitch: a drum hit leaked in, not a new note.
            notes[-1] = (notes[-1][0], end, n, max(notes[-1][3], peak))
            continue
        notes.append((float(a), end, n, peak))
    return _fix_octaves(notes)


def _fix_octaves(notes, width=4):
    """Pitch trackers sometimes jump an octave (locking onto a harmonic, or below a weak fundamental).
    A note 10+ semitones away from the middle of its neighbours is moved by an octave toward them."""
    pitches = [n for _, _, n, _ in notes]
    out = []
    for i, (a, e, n, db) in enumerate(notes):
        near = pitches[max(0, i - width):i] + pitches[i + 1:i + 1 + width]
        if len(near) >= 3:
            med = float(np.median(near))
            while n - med >= 10:
                n -= 12
            while med - n >= 10:
                n += 12
        out.append((a, e, n, db))
    return out


def _replucked(yb, low_sr, t, note):
    """Did the bass re-attack this note at time t? Looks for a jump in energy at the note's own pitch
    (fundamental + 2nd harmonic, which phones keep when they lose the fundamental). A drum hit adds
    broadband noise, which barely moves these narrow bands."""
    import librosa
    import scipy.signal as ss
    lo, hi = int(max(0, t - 0.2) * low_sr), int(min(len(yb) / low_sr, t + 0.2) * low_sr)
    x, at = yb[lo:hi], int(t * low_sr) - lo
    if len(x) < 400 or at < 0.08 * low_sr:
        return True
    f0 = float(librosa.midi_to_hz(note))
    level = np.zeros(len(x))
    for f in (f0, 2 * f0):
        if f * 1.07 < low_sr / 2:
            level += ss.sosfiltfilt(ss.butter(2, [f / 1.07, f * 1.07], btype="band", fs=low_sr, output="sos"), x) ** 2
    w = int(0.03 * low_sr)
    before = level[max(0, at - int(0.07 * low_sr)):at - int(0.01 * low_sr)].mean()
    after = max(level[i:i + w].mean() for i in range(at, min(len(x) - w, at + int(0.08 * low_sr)), w // 3 or 1)) \
        if len(x) - w > at else before
    return 10 * np.log10((after + 1e-12) / (before + 1e-12)) >= 3


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
    # The melody is usually the highest clear voice, not the loudest (accompaniment is often louder):
    # take the highest local peak that reaches 45% of the strongest pitch in that frame.
    seg = sal[lo:hi]
    peak = (seg >= np.roll(seg, 1, axis=0)) & (seg >= np.roll(seg, -1, axis=0)) & (seg >= 0.45 * seg.max(axis=0))
    idx = np.arange(seg.shape[0])[:, None]
    best = np.where(peak, idx, -1).max(axis=0)
    best = np.where(best < 0, seg.argmax(axis=0), best) + lo_midi
    strength = seg.max(axis=0)
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
