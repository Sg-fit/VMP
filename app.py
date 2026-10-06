#!/usr/bin/env python3
"""Web front-end for the Music Inspiration Analyzer.

Upload tracks in the browser, watch progress, read the report, download MIDI sketches.

Run locally:  python app.py                      -> http://localhost:8000
Production:   gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 --timeout 120 app:app

Use ONE gunicorn worker: analysis jobs run in a background thread inside the process.

Environment variables (all optional):
  NVIDIA_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY   enable the AI brainstorm
  APP_PASSWORD     require this password (HTTP basic auth, any username)
  DATA_DIR         where jobs are stored (default ./jobs)
  MAX_UPLOAD_MB    max total upload size per request (default 95; Cloudflare free plan caps at 100)
  MAX_FILES        max tracks per job (default 10)
  JOB_TTL_HOURS    delete finished jobs after this long (default 24)
  JOB_TIMEOUT_MIN  stop a job that runs longer than this (default 30)
"""

import hmac
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

import markdown
from flask import (Flask, Response, abort, jsonify, redirect, render_template, request,
                   send_file, send_from_directory, url_for)
from werkzeug.utils import secure_filename

import musicanalyze as ma
from job_runner import read_status, write_status

DATA_DIR = Path(os.environ.get("DATA_DIR", "jobs"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 95))
MAX_FILES = int(os.environ.get("MAX_FILES", 10))
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", 24))
JOB_TIMEOUT_MIN = float(os.environ.get("JOB_TIMEOUT_MIN", 30))
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
JOB_ID = re.compile(r"^[0-9a-f]{32}$")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
DATA_DIR.mkdir(parents=True, exist_ok=True)

LLM, LLM_LABEL = ma.make_llm()
job_queue = queue.Queue()


# ---------------------------------------------------------------- job storage (one folder per job)

def get_job_dir(job_id):
    if not JOB_ID.match(job_id):
        abort(404)
    d = DATA_DIR / job_id
    if not d.is_dir():
        abort(404)
    return d


def cleanup_old_jobs():
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    for d in DATA_DIR.iterdir():
        if d.is_dir() and JOB_ID.match(d.name):
            s = read_status(d)
            if s.get("state") not in ("queued", "running") and s.get("updated", 0) < cutoff:
                shutil.rmtree(d, ignore_errors=True)


def fail_interrupted_jobs():
    """Jobs left queued/running by a previous process will never finish — mark them failed."""
    for d in DATA_DIR.iterdir():
        if d.is_dir() and JOB_ID.match(d.name) and read_status(d).get("state") in ("queued", "running"):
            write_status(d, state="error", message="The server restarted before this job finished. Please upload again.")
            shutil.rmtree(d / "input", ignore_errors=True)


def worker():
    while True:
        job_id = job_queue.get()
        d = DATA_DIR / job_id
        try:
            write_status(d, state="running", message="Starting…")
            # Each job runs in its own process: the web server stays responsive while it works,
            # and a crash or out-of-memory kill fails only this job, not the whole site.
            runner = Path(__file__).with_name("job_runner.py")
            proc = subprocess.run([sys.executable, str(runner), str(d)], timeout=JOB_TIMEOUT_MIN * 60)
            if read_status(d).get("state") != "done":
                if proc.returncode in (-9, 137):
                    msg = "The analysis was killed — the server probably ran out of memory. Try fewer or shorter tracks."
                elif read_status(d).get("state") == "error":
                    msg = read_status(d).get("message")
                else:
                    msg = f"The analysis process stopped unexpectedly (exit code {proc.returncode})."
                app.logger.error("job %s failed: %s", job_id, msg)
                write_status(d, state="error", message=msg)
        except subprocess.TimeoutExpired:
            write_status(d, state="error", message=f"The analysis took longer than {JOB_TIMEOUT_MIN:.0f} minutes and "
                                                   "was stopped. Try fewer or shorter tracks.")
        except Exception as e:  # never let one bad job kill the worker thread
            app.logger.exception("job %s failed", job_id)
            write_status(d, state="error", message=f"Analysis failed: {str(e)[:300]}")
        finally:
            shutil.rmtree(d / "input", ignore_errors=True)  # don't keep uploaded audio
            job_queue.task_done()


fail_interrupted_jobs()
threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- auth

@app.before_request
def require_password():
    if not APP_PASSWORD:
        return None
    auth = request.authorization
    if not auth or not hmac.compare_digest((auth.password or "").encode(), APP_PASSWORD.encode()):
        return Response("Password required.", 401, {"WWW-Authenticate": 'Basic realm="Music Analyzer"'})
    return None


# ---------------------------------------------------------------- pages

def form_page(error=None, code=200):
    return render_template("index.html", error=error, ai_label=LLM_LABEL, max_files=MAX_FILES,
                           max_mb=MAX_UPLOAD_MB, exts=" ".join(sorted(ma.AUDIO_EXTS)), ttl=JOB_TTL_HOURS), code


@app.get("/")
def index():
    return form_page()


@app.post("/analyze")
def analyze():
    uploads = [f for f in request.files.getlist("tracks") if f and f.filename]
    if not uploads:
        return form_page("Please choose at least one audio file.", 400)
    if len(uploads) > MAX_FILES:
        return form_page(f"Too many files — the limit is {MAX_FILES} per upload.", 400)

    cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    d = DATA_DIR / job_id
    (d / "input").mkdir(parents=True)
    saved = []
    for f in uploads:
        name = secure_filename(f.filename) or "track"
        if Path(name).suffix.lower() not in ma.AUDIO_EXTS:
            continue
        stem, ext, n = Path(name).stem, Path(name).suffix, 2
        while (d / "input" / name).exists():  # keep duplicate names distinct
            name, n = f"{stem}_{n}{ext}", n + 1
        f.save(d / "input" / name)
        saved.append(name)
    if not saved:
        shutil.rmtree(d, ignore_errors=True)
        return form_page("None of those files is a supported audio format.", 400)

    ahead = job_queue.qsize()
    write_status(d, state="queued", files=saved, use_ai=bool(LLM and request.form.get("ai")),
                 created=time.time(),
                 message=f"Waiting in line ({ahead} job{'s' if ahead != 1 else ''} ahead)…" if ahead else "Starting…")
    job_queue.put(job_id)
    return redirect(url_for("job", job_id=job_id))


@app.get("/jobs/<job_id>")
def job(job_id):
    d = get_job_dir(job_id)
    s = read_status(d)
    report_html, midi_files = None, []
    if s.get("state") == "done":
        report_md = (d / "output" / "music_report.md").read_text(encoding="utf-8")
        report_html = markdown.markdown(report_md, extensions=["tables", "fenced_code"])
        report_html = re.sub(  # turn `midi/x.mid` mentions into download links
            r"<code>midi/([\w.\-]+\.mid)</code>",
            lambda m: f'<a class="midi" href="{url_for("job_file", job_id=job_id, name="midi/" + m.group(1))}">'
                      f'⬇ {m.group(1)}</a>', report_html)
        midi_dir = d / "output" / "midi"
        midi_files = sorted(p.name for p in midi_dir.glob("*.mid")) if midi_dir.is_dir() else []
    return render_template("job.html", job_id=job_id, s=s, report_html=report_html, midi_files=midi_files)


@app.get("/jobs/<job_id>/status")
def job_status(job_id):
    s = read_status(get_job_dir(job_id))
    return jsonify(state=s.get("state"), message=s.get("message"))


@app.get("/jobs/<job_id>/files/<path:name>")
def job_file(job_id, name):
    return send_from_directory(get_job_dir(job_id) / "output", name, as_attachment=name.endswith(".mid"))


@app.get("/jobs/<job_id>/download.zip")
def job_zip(job_id):
    out = get_job_dir(job_id) / "output"
    if not out.is_dir():
        abort(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="music-analysis.zip")


@app.errorhandler(413)
def too_large(_e):
    return form_page(f"Upload too large — the limit is {MAX_UPLOAD_MB} MB per upload.", 413)


if __name__ == "__main__":
    # Development server only — use gunicorn (see top of file / README) on a real server.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
