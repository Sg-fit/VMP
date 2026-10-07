#!/usr/bin/env python3
"""Run web-app analysis jobs in a separate, long-lived worker process.

app.py starts this once (`--serve`) and feeds it job folders. Keeping one warm process means the
one-time start-up cost (loading libraries and compiling librosa's fast code paths, ~15 s) is paid
when the server starts, not on every upload. It still runs outside the web server: a crash, an
out-of-memory kill or a stuck job only fails that one job, and app.py starts a fresh worker.

Usage (normally only called by app.py):
    python job_runner.py --serve       # read job folders from stdin, one per line
    python job_runner.py <job folder>  # run a single job
"""

import json
import os
import sys
import time
from pathlib import Path

DONE_PREFIX = "@@done "  # protocol line on stdout: "@@done <job folder>"


def status_path(d):
    return Path(d) / "status.json"


def read_status(d):
    try:
        return json.loads(status_path(d).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "unknown", "message": ""}


def write_status(d, **changes):
    d = Path(d)
    s = read_status(d) if status_path(d).exists() else {}
    s.update(changes, updated=time.time())
    tmp = d / "status.tmp"
    tmp.write_text(json.dumps(s), encoding="utf-8")
    tmp.replace(status_path(d))


def run_job(job_dir):
    import musicanalyze as ma

    d = Path(job_dir)
    try:
        files = sorted((d / "input").iterdir())
        llm = ma.make_llm()[0] if read_status(d).get("use_ai") else None

        def analysis_ready(report):  # show the analysis now; the AI section is filled in later
            write_status(d, state="done", ai_pending=True, message="AI ideas on the way…")
        cache = os.environ.get("MUSIC_CACHE_DIR") or str(d.parent / "cache")  # shared by all jobs
        ma.run_analysis(files, d / "output", llm, progress=lambda msg: write_status(d, message=msg),
                        melody_text=read_status(d).get("melody"),
                        on_analysis_done=analysis_ready if llm else None, cache_dir=cache)
        write_status(d, state="done", ai_pending=False, message="Done")
    except Exception as e:
        if read_status(d).get("state") == "done":  # analysis already shown: keep it, just stop waiting for AI
            write_status(d, ai_pending=False, message="The AI ideas could not be added this time.")
        else:
            write_status(d, state="error", ai_pending=False, message=f"Analysis failed: {str(e)[:300]}")


def warm_up():
    """Run a tiny synthetic analysis so librosa's compiled code is ready before the first real job."""
    import tempfile

    import numpy as np
    import soundfile as sf

    import musicanalyze as ma
    sr, bpm = ma.SR, 120
    t = np.arange(int(sr * 12)) / sr
    y = sum(0.15 * np.sin(2 * np.pi * f * t) for f in (220, 277.2, 329.6, 110))
    for b in np.arange(0, 12, 60 / bpm):  # a kick on every beat
        i = int(b * sr)
        n = min(len(y) - i, 2000)
        y[i:i + n] += 0.6 * np.sin(2 * np.pi * 60 * t[:n]) * np.exp(-t[:n] * 30)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "warmup.wav"
        sf.write(path, (y / np.abs(y).max() * 0.8).astype(np.float32), sr)
        ma._analyze_audio(path)


def serve():
    # stdout carries the protocol; send everything else libraries might print to stderr.
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr
    started = time.time()
    try:
        warm_up()
        print(f"[worker] warmed up in {time.time() - started:.1f}s", file=sys.stderr)
    except Exception as e:  # warm-up is only an optimisation
        print(f"[worker] warm-up skipped: {e}", file=sys.stderr)
    proto.write("@@ready\n")
    for line in sys.stdin:
        job = line.strip()
        if job:
            run_job(job)
            proto.write(DONE_PREFIX + job + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        serve()
    else:
        run_job(sys.argv[1])
