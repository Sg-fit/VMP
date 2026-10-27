#!/usr/bin/env python3
"""Style, mood and instrument tags from a trained model (Essentia's Discogs-EffNet, MTG, UPF).

Our own analysis measures tempo, key, metre and feel. It can't name a style ("trip hop", "shoegaze", "Chicago
blues"); that needs a model trained on a large labelled catalogue. Essentia's models do exactly that:
one network turns the audio into an "embedding" (a 1280-number fingerprint of the sound), and small heads on top
of it each answer one question - which of 400 Discogs styles, which moods/themes, which instruments, vocals or
not, and so on.

Licence: the models are CC BY-NC-SA 4.0 (non-commercial use only, credit MTG/UPF, share alike).

How it runs
- MTG publishes the networks as ONNX files, which onnxruntime runs on any system (Windows included) without
  TensorFlow. The network's input - a log-compressed mel spectrogram - is rebuilt here exactly as Essentia's
  TensorflowPredictEffnetDiscogs makes it (16 kHz, 512-sample Hann frames every 256 samples, 96 Slaney mel bands
  up to 8 kHz with unit-area triangles, log10(1 + 10000 x), patches of 128 frames every 62): on a 4-minute song
  the averages match the TensorFlow version to the second decimal.
- The network sees the song in ~2-second patches, one a second. Besides each song's averages we keep the
  per-second scores for vocals and the instruments, so the report can show who plays when.
- The main process decodes each song (any format we can read) to 16 kHz and saves it as .npy; a short-lived
  subprocess loads the models, tags every song and exits, so the models' memory is freed when it's done and a
  crash there only loses the tags, never the job.
- MUSIC_TAGGER=fake gives deterministic made-up predictions so the rest of the pipeline can be tested without
  the model files.

Usage
    python tagger.py --download            # fetch the model files into MUSIC_MODELS_DIR
    python tagger.py --predict out.json a.npy b.npy   # (internal) run the models
"""

import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

SR = 16000
BASE = "https://essentia.upf.edu/models"
EMBEDDING = ("discogs-effnet-bsdynamic-1", f"{BASE}/feature-extractors/discogs-effnet/discogs-effnet-bsdynamic-1")
# The network's own 400 outputs are the Discogs styles; its 1280-number embedding feeds the heads.
# name -> what it answers. Each head has a .onnx (weights) and .json (class names).
HEADS = {
    "mtg_jamendo_moodtheme": "moods",
    "mtg_jamendo_instrument": "instruments",
    "voice_instrumental": "voice",
    "gender": "gender",
    "mood_acoustic": "acoustic",
    "mood_electronic": "electronic",
    "danceability": "danceable",
    "mood_aggressive": "aggressive",
    "mood_relaxed": "relaxed",
    "mood_happy": "happy",
    "mood_sad": "sad",
}
# Essentia's input for the EffNet networks (TensorflowInputMusiCNN + TensorflowPredictEffnetDiscogs defaults)
FRAME, HOP, MELS, PATCH, PATCH_HOP = 512, 256, 96, 128, 62
PATCH_SEC = PATCH_HOP * HOP / SR  # one patch a second (0.99 s); each covers 2.05 s
TIMELINE = ("instruments", "voice")  # the heads whose per-second scores are kept
TIMELINE_CLASSES = 12  # the instrument classes kept per second: the song's strongest
# For the two-way heads: the class whose probability we keep.
POSITIVE = {"voice": "voice", "gender": "female", "acoustic": "acoustic", "electronic": "electronic",
            "danceable": "danceable", "aggressive": "aggressive", "relaxed": "relaxed", "happy": "happy", "sad": "sad"}
# The mood/theme head also names uses ("advertising", "trailer"): not something a listener searches by.
USE_THEMES = {"advertising", "commercial", "corporate", "background", "documentary", "sport", "trailer", "game",
              "motivational", "children", "christmas", "holiday", "movie", "film", "travel"}
CREDIT = ("Style, mood and instrument tags use Essentia models by the Music Technology Group, Universitat Pompeu "
          "Fabra (CC BY-NC-SA 4.0, non-commercial use).")


def models_dir():
    return Path(os.environ.get("MUSIC_MODELS_DIR") or Path(__file__).with_name("models"))


def head_url(name):
    return f"{BASE}/classification-heads/{name}/{name}-discogs-effnet-1"


def _files():
    yield EMBEDDING[0], EMBEDDING[1]
    for name in HEADS:
        yield f"{name}-discogs-effnet-1", head_url(name)


def download():
    d = models_dir()
    d.mkdir(parents=True, exist_ok=True)
    for stem, url in _files():
        for ext in (".onnx", ".json"):
            target = d / (stem + ext)
            if target.exists() and target.stat().st_size:
                continue
            try:
                req = urllib.request.Request(url + ext, headers={"User-Agent": "MusicInspirationAnalyzer/1.0"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    target.write_bytes(r.read())
                print(f"downloaded {target.name}")
            except OSError as e:
                print(f"could not download {url + ext}: {e}", file=sys.stderr)


def backend():
    """'fake', 'onnx', or None (not set up here)."""
    if os.environ.get("MUSIC_TAGGER") == "fake":
        return "fake"
    if os.environ.get("MUSIC_TAGGER") == "off":
        return None
    d = models_dir()
    if not all((d / f"{EMBEDDING[0]}{ext}").exists() for ext in (".onnx", ".json")):
        return None
    try:
        import importlib.util
        return "onnx" if importlib.util.find_spec("onnxruntime") else None
    except (ImportError, ValueError):
        return None


def available():
    return backend() is not None


# ---------------------------------------------------------------- the model process
def mel_patches(audio):
    """16 kHz mono samples -> the network's input: (patches, 128 frames, 96 mel bands), as Essentia makes it."""
    import librosa
    import numpy as np
    import scipy.signal
    window = scipy.signal.get_window("hann", FRAME, fftbins=False).astype(np.float32)  # Essentia's symmetric Hann
    y = np.pad(np.asarray(audio, dtype=np.float32), (FRAME // 2, FRAME // 2))  # frames centred from t = 0
    n = 1 + (len(y) - FRAME) // HOP
    if n < PATCH:
        y = np.pad(y, (0, (PATCH - n) * HOP))  # a clip shorter than one patch: pad it with silence
        n = PATCH
    frames = np.lib.stride_tricks.as_strided(y, (n, FRAME), (y.strides[0] * HOP, y.strides[0]))
    weights = librosa.filters.mel(sr=SR, n_fft=FRAME, n_mels=MELS, fmin=0, fmax=SR / 2, htk=False, norm="slaney")
    mel = np.zeros((n, MELS), np.float32)
    for a in range(0, n, 4096):  # in blocks, so a long song doesn't hold every frame's spectrum at once
        power = np.abs(np.fft.rfft(frames[a:a + 4096] * window, axis=1)) ** 2
        mel[a:a + 4096] = np.log10(1 + 10000 * power @ weights.T)
    return np.stack([mel[s:s + PATCH] for s in range(0, n - PATCH + 1, PATCH_HOP)])


def _classes(name):
    return json.loads((models_dir() / f"{name}.json").read_text(encoding="utf-8"))["classes"]


def _predict_files(npy_files):
    import numpy as np
    import onnxruntime as ort
    d = models_dir()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    run = lambda path: ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    embed = run(d / f"{EMBEDDING[0]}.onnx")
    styles = _classes(EMBEDDING[0])
    heads = {key: (_classes(f"{name}-discogs-effnet-1"), run(d / f"{name}-discogs-effnet-1.onnx"))
             for name, key in HEADS.items() if (d / f"{name}-discogs-effnet-1.onnx").exists()}
    results = []
    for f in npy_files:
        patches = mel_patches(np.load(f))
        acts, embs = [], []
        for i in range(0, len(patches), 16):  # 16 patches at a time keeps the network's working memory small
            a, e = embed.run(["activations", "embeddings"], {"melspectrogram": patches[i:i + 16]})
            acts.append(a)
            embs.append(e)
        acts, embs = np.concatenate(acts), np.concatenate(embs)
        res = {"styles": {c: round(float(v), 4) for c, v in zip(styles, acts.mean(axis=0))}}
        timeline = {"patch_sec": round(PATCH_SEC, 4)}
        for key, (classes, model) in heads.items():
            p = model.run(None, {model.get_inputs()[0].name: embs})[0]
            res[key] = {c: round(float(v), 4) for c, v in zip(classes, p.mean(axis=0))}  # averaged over the patches
            if key in TIMELINE:
                keep = np.argsort(-p.mean(axis=0))[:TIMELINE_CLASSES]
                timeline[key] = {classes[i]: [round(float(v), 2) for v in p[:, i]] for i in keep}
        res["timeline"] = timeline
        results.append(res)
    return results


# ---------------------------------------------------------------- fake backend (tests)
def _fake_predict(npy_files):
    import numpy as np
    out = []
    for f in npy_files:
        a = np.load(f)
        bright = float(np.mean(np.abs(np.diff(a)))) / (float(np.mean(np.abs(a))) + 1e-9)  # crude: more highs -> more "electronic"
        n = max(1, int(len(a) / SR / PATCH_SEC) - 1)
        out.append({
            "styles": {"Electronic---Downtempo": round(min(0.9, bright), 3), "Rock---Indie Rock": 0.3,
                       "Blues---Delta Blues": round(max(0.05, 0.6 - bright), 3), "Pop---Ballad": 0.12},
            "moods": {"calm": 0.2, "melancholic": 0.15, "energetic": 0.05},
            "instruments": {"guitar": 0.6, "drums": 0.2, "piano": 0.1},
            "voice": {"voice": 0.2, "instrumental": 0.8}, "gender": {"female": 0.5, "male": 0.5},
            "acoustic": {"acoustic": 0.7, "non_acoustic": 0.3}, "electronic": {"electronic": 0.3, "non_electronic": 0.7},
            "danceable": {"danceable": 0.4, "not_danceable": 0.6}, "aggressive": {"aggressive": 0.1, "not_aggressive": 0.9},
            "relaxed": {"relaxed": 0.6, "not_relaxed": 0.4}, "happy": {"happy": 0.4, "non_happy": 0.6},
            "sad": {"sad": 0.3, "non_sad": 0.7},
            "timeline": {"patch_sec": round(PATCH_SEC, 4), "voice": {"voice": [0.2] * n},
                         "instruments": {"guitar": [0.6] * n}}})
    return out


# ---------------------------------------------------------------- main process
def summarise(raw):
    """Model probabilities -> what we report: top styles, moods, instruments, and the two-way answers."""
    def top(d, n, floor):
        return [(k, v) for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:n] if v >= floor]
    moods = {k: v for k, v in raw.get("moods", {}).items() if k not in USE_THEMES}
    out = {"styles": top(raw.get("styles", {}), 6, 0.08),
           "moods": top(moods, 5, 0.06),
           "instruments": top(raw.get("instruments", {}), 5, 0.12),
           "instrument_scores": raw.get("instruments", {})}  # all 40, for the musician's "instruments we hear"
    for key, pos in POSITIVE.items():
        if key in raw:
            out[key] = raw[key].get(pos, next((v for c, v in raw[key].items() if c.replace("_", "") == pos), None))
    if raw.get("timeline"):
        out["timeline"] = raw["timeline"]
    return out


# The model process (onnxruntime + the EffNet network and its heads) needs roughly this much memory on top of
# what's already running (measured: about 600 MB at its peak on a 4-minute song). In a container with a memory cap,
# starting it without that headroom gets something killed - possibly the analysis itself - so it isn't started
# then. It's a bonus: the analysis never depends on it.
MODEL_MEMORY_MB = int(os.environ.get("MUSIC_MODEL_MEMORY_MB", 800))


def free_memory_mb(cgroup="/sys/fs/cgroup", meminfo="/proc/meminfo"):
    """Memory free for a new process here, in MB: the container's cap minus what it uses (page cache that can be
    dropped doesn't count), or the machine's available memory, whichever is smaller. None where neither can be
    read (e.g. Windows): then no limit is assumed."""
    found = []
    root = Path(cgroup)
    try:  # cgroup v2
        cap = (root / "memory.max").read_text().strip()
        if cap != "max":
            used = int((root / "memory.current").read_text())
            stat = dict(line.split() for line in (root / "memory.stat").read_text().splitlines() if " " in line)
            used -= int(stat.get("inactive_file", 0))
            found.append((int(cap) - used) / 2 ** 20)
    except (OSError, ValueError):
        try:  # cgroup v1
            cap = int((root / "memory" / "memory.limit_in_bytes").read_text())
            if cap < 2 ** 60:
                used = int((root / "memory" / "memory.usage_in_bytes").read_text())
                found.append((cap - used) / 2 ** 20)
        except (OSError, ValueError):
            pass
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                found.append(int(line.split()[1]) / 1024)
    except (OSError, ValueError, IndexError):
        pass
    return round(min(found)) if found else None


def predict(paths, load_audio, timeout=900):
    """Tag each audio file. load_audio(path) -> (mono samples, sample rate). Returns {path: summary} or
    {} when the model isn't set up here. Raises RuntimeError if the model process fails, or isn't started because
    there isn't enough memory free for it (MODEL_MEMORY_MB)."""
    kind = backend()
    if not kind or not paths:
        return {}
    free = free_memory_mb() if kind != "fake" else None
    if free is not None and free < MODEL_MEMORY_MB:
        raise RuntimeError(f"not enough free memory to start it this time ({free} MB free, it needs about "
                           f"{MODEL_MEMORY_MB} MB)")
    import librosa
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        npys = []
        for i, p in enumerate(paths):
            y, sr = load_audio(p)
            y = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
            y = y[:SR * 600]  # at most 10 minutes per song
            f = Path(tmp) / f"{i}.npy"
            np.save(f, y.astype(np.float32))
            npys.append(str(f))
        if kind == "fake":
            raw = _fake_predict(npys)
        else:
            out = Path(tmp) / "out.json"
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--predict", str(out), *npys],
                                  capture_output=True, text=True, timeout=timeout,
                                  env={**os.environ, "MUSIC_MODELS_DIR": str(models_dir())})
            if proc.returncode != 0 or not out.exists():
                tail = (proc.stderr or "").strip().splitlines()[-3:]
                raise RuntimeError(f"style model failed (exit {proc.returncode}): {' | '.join(tail)[:300]}")
            raw = json.loads(out.read_text(encoding="utf-8"))
    return {p: summarise(r) for p, r in zip(paths, raw)}


if __name__ == "__main__":
    if sys.argv[1:2] == ["--download"]:
        download()
    elif sys.argv[1:2] == ["--predict"]:
        Path(sys.argv[2]).write_text(json.dumps(_predict_files(sys.argv[3:])), encoding="utf-8")
    elif sys.argv[1:2] == ["--check"]:
        print(backend() or "not available")
    else:
        print(__doc__)
