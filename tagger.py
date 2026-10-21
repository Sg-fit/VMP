#!/usr/bin/env python3
"""Style, mood and instrument tags from a trained model (Essentia's Discogs-EffNet, MTG, UPF).

Our own analysis measures tempo, key, metre and feel. It can't name a style ("trip hop", "shoegaze", "Chicago
blues"); that needs a model trained on a large labelled catalogue. Essentia's models do exactly that:
one network turns the audio into an "embedding" (a 1280-number fingerprint of the sound), and small heads on top
of it each answer one question - which of 400 Discogs styles, which moods/themes, which instruments, vocals or
not, and so on.

Licence: the models are CC BY-NC-SA 4.0 (non-commercial use only, credit MTG/UPF, share alike).

How it runs
- The models need TensorFlow (the `essentia-tensorflow` package, Linux only). It lives in its own Python
  environment (MUSIC_ESSENTIA_PYTHON, set up by the Dockerfile) so its pinned libraries can't clash with ours.
- The main process decodes each song (any format we can read) to 16 kHz and saves it as .npy; a short-lived
  subprocess running that other Python loads the models, tags every song and exits. TensorFlow's memory is
  freed when it exits, and a crash there only loses the style tags, never the job.
- MUSIC_TAGGER=fake gives deterministic made-up predictions so the rest of the pipeline can be tested on any
  machine (Windows included).

Usage
    python tagger.py --download            # fetch the model files into MUSIC_MODELS_DIR
    python tagger.py --predict out.json a.npy b.npy   # (internal) run in the Essentia environment
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
EMBEDDING = ("discogs-effnet-bs64-1", f"{BASE}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1")
EMBEDDING_OUTPUT = "PartitionedCall:1"  # the 1280-d embeddings (":0" is the model's own 400-style output)
# name -> what it answers. Each head has a .pb (weights) and .json (class names and input/output tensor names).
HEADS = {
    "genre_discogs400": "styles",
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
        for ext in (".pb", ".json"):
            target = d / (stem + ext)
            if target.exists() and target.stat().st_size:
                continue
            try:
                req = urllib.request.Request(url + ext, headers={"User-Agent": "MusicInspirationAnalyzer/1.0"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    target.write_bytes(r.read())
                print(f"downloaded {target.name}")
            except OSError as e:
                if ext == ".json" and stem == EMBEDDING[0]:  # the embedding's metadata isn't needed
                    continue
                print(f"could not download {url + ext}: {e}", file=sys.stderr)


def backend():
    """'fake', 'essentia', or None (not set up here)."""
    if os.environ.get("MUSIC_TAGGER") == "fake":
        return "fake"
    if os.environ.get("MUSIC_TAGGER") == "off":
        return None
    d = models_dir()
    if not (d / f"{EMBEDDING[0]}.pb").exists() or not (d / "genre_discogs400-discogs-effnet-1.pb").exists():
        return None
    return "essentia" if _essentia_python() else None


def available():
    return backend() is not None


def _essentia_python():
    exe = os.environ.get("MUSIC_ESSENTIA_PYTHON")
    if exe and Path(exe).exists():
        return exe
    try:  # essentia-tensorflow installed in our own environment
        import importlib.util
        if importlib.util.find_spec("essentia"):
            return sys.executable
    except (ImportError, ValueError):
        pass
    return None


# ---------------------------------------------------------------- inside the Essentia environment
def _head_meta(name):
    meta = json.loads((models_dir() / f"{name}-discogs-effnet-1.json").read_text(encoding="utf-8"))
    schema = meta.get("schema", {})
    inp = schema.get("inputs", [{}])[0].get("name", "model/Placeholder")
    outs = schema.get("outputs", [])
    out = next((o["name"] for o in outs if o.get("output_purpose") == "predictions"), outs[0]["name"] if outs else "model/Softmax")
    return meta["classes"], inp, out


def _predict_files(npy_files):
    import numpy as np
    from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs
    d = models_dir()
    embed = TensorflowPredictEffnetDiscogs(graphFilename=str(d / f"{EMBEDDING[0]}.pb"), output=EMBEDDING_OUTPUT)
    heads = {}
    for name, key in HEADS.items():
        if not (d / f"{name}-discogs-effnet-1.pb").exists():
            continue
        classes, inp, out = _head_meta(name)
        heads[key] = (classes, TensorflowPredict2D(graphFilename=str(d / f"{name}-discogs-effnet-1.pb"),
                                                   input=inp, output=out))
    results = []
    for f in npy_files:
        audio = np.load(f).astype(np.float32)
        emb = embed(audio)
        res = {}
        for key, (classes, model) in heads.items():
            p = np.mean(model(emb), axis=0)  # average over the ~3 s patches
            res[key] = {c: round(float(v), 4) for c, v in zip(classes, p)}
        results.append(res)
    return results


# ---------------------------------------------------------------- fake backend (tests, Windows)
def _fake_predict(npy_files):
    import numpy as np
    out = []
    for f in npy_files:
        a = np.load(f)
        bright = float(np.mean(np.abs(np.diff(a)))) / (float(np.mean(np.abs(a))) + 1e-9)  # crude: more highs -> more "electronic"
        out.append({
            "styles": {"Electronic---Downtempo": round(min(0.9, bright), 3), "Rock---Indie Rock": 0.3,
                       "Blues---Delta Blues": round(max(0.05, 0.6 - bright), 3), "Pop---Ballad": 0.12},
            "moods": {"calm": 0.2, "melancholic": 0.15, "energetic": 0.05},
            "instruments": {"guitar": 0.6, "drums": 0.2, "piano": 0.1},
            "voice": {"voice": 0.2, "instrumental": 0.8}, "gender": {"female": 0.5, "male": 0.5},
            "acoustic": {"acoustic": 0.7, "non_acoustic": 0.3}, "electronic": {"electronic": 0.3, "non_electronic": 0.7},
            "danceable": {"danceable": 0.4, "not_danceable": 0.6}, "aggressive": {"aggressive": 0.1, "not_aggressive": 0.9},
            "relaxed": {"relaxed": 0.6, "not_relaxed": 0.4}, "happy": {"happy": 0.4, "non_happy": 0.6},
            "sad": {"sad": 0.3, "non_sad": 0.7}})
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
    return out


def predict(paths, load_audio, timeout=900):
    """Tag each audio file. load_audio(path) -> (mono samples, sample rate). Returns {path: summary} or
    {} when the model isn't set up here. Raises RuntimeError if the model process fails."""
    kind = backend()
    if not kind or not paths:
        return {}
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
            proc = subprocess.run([_essentia_python(), str(Path(__file__).resolve()), "--predict", str(out), *npys],
                                  capture_output=True, text=True, timeout=timeout,
                                  env={**os.environ, "MUSIC_MODELS_DIR": str(models_dir()), "TF_CPP_MIN_LOG_LEVEL": "3"})
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
