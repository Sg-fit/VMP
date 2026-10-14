#!/usr/bin/env python3
"""Web front-end for the Music Inspiration Analyzer.

Upload tracks in the browser, watch progress, read the report, download MIDI sketches.

Run locally:  python app.py                      -> http://localhost:8000
Production:   gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 --timeout 120 app:app

Use ONE gunicorn worker: analysis jobs run in a background thread inside the process.

Environment variables (all optional):
  NVIDIA_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY   enable the AI brainstorm
  SECRET_KEY       signs the sign-in cookie (default: a random key kept in DATA_DIR/.secret_key)
  UPLOADS_PER_HOUR max uploads per visitor per hour (default 30)
  DATA_DIR         where jobs are stored (default ./jobs)
  MAX_UPLOAD_MB    max total upload size per request (default 95; Cloudflare free plan caps at 100)
  MAX_FILES        max tracks per job (default 10)
  JOB_TTL_HOURS    delete finished jobs after this long (default 24)
  JOB_TIMEOUT_MIN  stop a job that runs longer than this (default 30)
"""

import io
import json
import os
import secrets
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
                   send_file, send_from_directory, session, url_for)
from werkzeug.utils import secure_filename

import accounts
import musicanalyze as ma
from job_runner import read_status, write_status

DATA_DIR = Path(os.environ.get("DATA_DIR", "jobs"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 95))
MAX_FILES = int(os.environ.get("MAX_FILES", 10))
JOB_TTL_HOURS = float(os.environ.get("JOB_TTL_HOURS", 24))
JOB_TIMEOUT_MIN = float(os.environ.get("JOB_TIMEOUT_MIN", 30))
UPLOADS_PER_HOUR = int(os.environ.get("UPLOADS_PER_HOUR", 30))
JOB_ID = re.compile(r"^[0-9a-f]{32}$")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _secret_key():
    """The key that signs the sign-in cookie. Kept in the data volume, so people stay signed in across
    restarts and redeploys."""
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    f = DATA_DIR / ".secret_key"
    if not f.exists():
        f.write_text(secrets.token_hex(32))
    return f.read_text().strip()


app.secret_key = _secret_key()
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True,
                  PERMANENT_SESSION_LIFETIME=30 * 24 * 3600)
accounts.init(DATA_DIR / "users.db")

LLM, LLM_LABEL = ma.make_llm()
job_queue = queue.Queue()


# ---------------------------------------------------------------- job storage (one folder per job)

def get_job_dir(job_id):
    """The job's folder. Temporary results open for anyone with the link; an analysis saved to someone's
    history opens only for them (404 for everyone else, so its existence isn't revealed)."""
    if not JOB_ID.match(job_id):
        abort(404)
    d = DATA_DIR / job_id
    if not d.is_dir():
        abort(404)
    owner = accounts.owner_of(job_id)
    if owner is not None and owner != session.get("uid"):
        abort(404)
    return d


def cleanup_old_jobs():
    """Delete temporary results after JOB_TTL_HOURS. Analyses saved to someone's history are kept."""
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    keep = accounts.saved_job_ids()
    for d in DATA_DIR.iterdir():
        if d.is_dir() and JOB_ID.match(d.name) and d.name not in keep:
            s = read_status(d)
            if s.get("state") not in ("queued", "running") and s.get("updated", 0) < cutoff:
                shutil.rmtree(d, ignore_errors=True)


def fail_interrupted_jobs():
    """Jobs left queued/running by a previous process will never finish — mark them failed."""
    for d in DATA_DIR.iterdir():
        if d.is_dir() and JOB_ID.match(d.name) and read_status(d).get("state") in ("queued", "running"):
            write_status(d, state="error", message="The server restarted before this job finished. Please upload again.")
            shutil.rmtree(d / "input", ignore_errors=True)


class Runner:
    """One long-lived analysis worker process (job_runner.py --serve).

    It warms up once at start-up, then runs jobs one after another, so uploads don't pay the
    ~15 s start-up cost each time. If it crashes, runs out of memory or a job takes too long,
    it is killed and replaced; only that job fails."""

    def __init__(self):
        self.proc, self.lines, self.last_exit = None, queue.Queue(), None

    def ensure_started(self):
        if self.proc and self.proc.poll() is None:
            return
        runner = Path(__file__).with_name("job_runner.py")
        # Low priority and single-threaded maths libraries keep the site (and anything else on the
        # server) responsive while a job runs.
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "NUMBA_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
        # No preexec_fn: running Python code between fork and exec is unsafe in a multi-threaded server
        # (gunicorn threads) and crashed the worker with SIGSEGV. The worker lowers its own priority.
        self.proc = subprocess.Popen([sys.executable, str(runner), "--serve"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, env=env)
        self.lines = queue.Queue()
        proc, lines = self.proc, self.lines
        threading.Thread(target=lambda: [lines.put(l.strip()) for l in proc.stdout] + [lines.put(None)],
                         daemon=True).start()

    def run(self, job_dir, timeout):
        """Send a job and wait for it. Returns None on success, or an error message."""
        self.ensure_started()
        try:
            self.proc.stdin.write(str(job_dir) + "\n")
            self.proc.stdin.flush()
        except OSError:
            self.proc.kill()
            return "The analysis worker stopped unexpectedly. Please try again."
        deadline = time.time() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                self.proc.kill()  # stuck: replace the worker, fail only this job
                return (f"The analysis took longer than {JOB_TIMEOUT_MIN:.0f} minutes and was stopped. "
                        "Try fewer or shorter tracks.")
            if line is None:  # the worker died mid-job
                code = self.proc.wait()
                self.last_exit = code
                if code in (-9, 137):
                    return "The analysis was killed — the server probably ran out of memory. Try fewer or shorter tracks."
                if code in (-11, 139):
                    return "The analysis crashed inside an audio/maths library (segmentation fault, exit code -11)."
                return f"The analysis process stopped unexpectedly (exit code {code})."
            if line == f"@@done {job_dir}":
                return None


runner = Runner()


def worker():
    try:
        runner.ensure_started()  # warm up now, before the first upload arrives
    except Exception:
        app.logger.exception("could not start the analysis worker")
    while True:
        job_id = job_queue.get()
        d = DATA_DIR / job_id
        try:
            write_status(d, state="running", message="Starting…")
            problem = runner.run(d, JOB_TIMEOUT_MIN * 60)
            crashed = problem and runner.last_exit not in (None, -9, 137) and "longer than" not in problem
            if crashed and read_status(d).get("state") != "done":
                # A native crash can be a one-off: retry once in a fresh worker before giving up.
                where = read_status(d).get("message", "")
                app.logger.error("job %s crashed (%s) while: %s - retrying once", job_id, runner.last_exit, where)
                write_status(d, state="running", message="Retrying after a worker crash…")
                runner.last_exit = None
                problem2 = runner.run(d, JOB_TIMEOUT_MIN * 60)
                if problem2:
                    problem = (f"{problem2} It happened twice, while: {where}. "
                               "Please send the server log (docker compose logs --tail 150) so it can be fixed.")
                else:
                    problem = None
            if problem:
                app.logger.error("job %s failed: %s", job_id, problem)
                if read_status(d).get("state") == "done":  # analysis already shown; only the AI part was lost
                    write_status(d, ai_pending=False, message="The AI ideas could not be added this time.")
                else:
                    write_status(d, state="error", message=problem)
        except Exception as e:  # never let one bad job kill the worker thread
            app.logger.exception("job %s failed", job_id)
            write_status(d, state="error", message=f"Analysis failed: {str(e)[:300]}")
        finally:
            shutil.rmtree(d / "input", ignore_errors=True)  # don't keep uploaded audio
            job_queue.task_done()


fail_interrupted_jobs()
threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- accounts

@app.get("/health")
def health():
    """For deploy checks and monitoring: reveals nothing private."""
    alive = bool(runner.proc and runner.proc.poll() is None)
    return jsonify(status="ok", version=ma.CODE_VERSION, worker_running=alive, ai_configured=bool(LLM),
                   jobs_waiting=job_queue.qsize())


def current_user():
    uid = session.get("uid")
    return accounts.get_user(uid) if uid else None


@app.context_processor
def inject_user():
    return {"user": current_user(), "glossary": ma.GLOSSARY}


@app.template_filter("datetime")
def _datetime(ts, date_only=False):
    return time.strftime("%d %b %Y" if date_only else "%d %b %Y, %H:%M", time.localtime(ts or 0))


def _safe_next(default="/"):
    nxt = request.values.get("next") or default
    return nxt if nxt.startswith("/") and not nxt.startswith("//") else default  # only ever back into this site


def _sign_in(user_id):
    session.clear()  # a fresh session on sign-in
    session["uid"] = user_id
    session.permanent = True  # remembered for 30 days on this device


@app.route("/signup", methods=["GET", "POST"])
def signup():
    nxt, error = _safe_next(), None
    if request.method == "POST":
        if request.form.get("password") != request.form.get("password2"):
            error = "The two passwords don't match."
        else:
            uid, error = accounts.create_user(request.form.get("username"), request.form.get("password"))
            if uid:
                _sign_in(uid)
                return redirect(nxt)
    return render_template("account.html", mode="signup", error=error, next=nxt,
                           username=request.form.get("username", "")), (400 if error else 200)


@app.route("/login", methods=["GET", "POST"])
def login():
    nxt, error = _safe_next(), None
    if request.method == "POST":
        uid = accounts.check_login(request.form.get("username"), request.form.get("password"))
        if uid:
            _sign_in(uid)
            return redirect(nxt)
        time.sleep(1)  # slow down password guessing
        error = "That username and password don't match."
    return render_template("account.html", mode="login", error=error, next=nxt,
                           username=request.form.get("username", "")), (401 if error else 200)


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


def _job_title(d):
    files = read_status(d).get("files") or []
    return ", ".join(Path(f).stem for f in files[:3]) + (f" +{len(files) - 3} more" if len(files) > 3 else "") or "Analysis"


def _summaries(items):
    """Fill in the summary of saved analyses that finished since they were saved."""
    for it in items:
        if it["summary"] is None:
            res = DATA_DIR / it["job_id"] / "output" / "music_results.json"
            if res.exists() and read_status(DATA_DIR / it["job_id"]).get("state") == "done":
                it["summary"] = accounts.job_summary(json.loads(res.read_text(encoding="utf-8")))
                accounts.store_summary(it["job_id"], it["summary"])
    return items


def _require_user():
    u = current_user()
    if not u:
        abort(redirect(url_for("login", next=request.path)))
    return u


@app.get("/history")
def history():
    u = _require_user()
    items = _summaries(accounts.history(u["id"]))
    for it in items:
        it["state"] = read_status(DATA_DIR / it["job_id"]).get("state", "missing")
    return render_template("history.html", items=items, version=ma.CODE_VERSION)


@app.get("/profile")
def profile():
    u = _require_user()
    items = _summaries(accounts.history(u["id"]))
    return render_template("profile.html", p=accounts.profile_summary(items), recent=items[:5], version=ma.CODE_VERSION)


@app.post("/jobs/<job_id>/save")
def job_save(job_id):
    u = _require_user()
    d = get_job_dir(job_id)
    problem = accounts.save_job(u["id"], job_id, _job_title(d))
    if problem:
        abort(403, problem)
    return redirect(url_for("job", job_id=job_id))


@app.post("/jobs/<job_id>/rename")
def job_rename(job_id):
    u = _require_user()
    accounts.rename_job(u["id"], job_id, request.form.get("title") or "Analysis")
    return redirect(url_for("history"))


@app.post("/jobs/<job_id>/delete")
def job_delete(job_id):
    """Remove an analysis from your history and delete its files."""
    u = _require_user()
    if accounts.forget_job(u["id"], job_id) and read_status(DATA_DIR / job_id).get("state") not in ("queued", "running"):
        shutil.rmtree(DATA_DIR / job_id, ignore_errors=True)
    return redirect(url_for("history"))


_recent_uploads = {}


def _visitor():
    return (request.headers.get("CF-Connecting-IP") or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.remote_addr or "?")


def _too_many_uploads():
    """Analysis is open to everyone, so cap how often one visitor can start it."""
    now, who = time.time(), _visitor()
    recent = [t for t in _recent_uploads.get(who, []) if t > now - 3600]
    if len(recent) >= UPLOADS_PER_HOUR:
        _recent_uploads[who] = recent
        return True
    _recent_uploads[who] = recent + [now]
    return False


# ---------------------------------------------------------------- pages

def form_page(error=None, code=200):
    return render_template("index.html", error=error, ai_label=LLM_LABEL, max_files=MAX_FILES, version=ma.CODE_VERSION,
                           instruments=ma.INSTRUMENTS, genres={k: v[0] for k, v in ma.GENRES.items()},
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

    if _too_many_uploads():
        return form_page(f"You've started {UPLOADS_PER_HOUR} analyses in the last hour — please wait a little.", 429)
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
                 melody=(request.form.get("melody") or "")[:2000],
                 context=ma.clean_context(request.form.getlist("instruments"), request.form.get("genre")),
                 created=time.time(),
                 message=f"Waiting in line ({ahead} job{'s' if ahead != 1 else ''} ahead)…" if ahead else "Starting…")
    u = current_user()
    if u:  # signed in: goes straight into their history
        accounts.save_job(u["id"], job_id, _job_title(d))
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
        report_html = re.sub(  # turn `midi/x.mid` and `sheets/x.html` mentions into links
            r"<code>((?:midi|sheets)/[\w.\-]+\.(?:mid|musicxml|html))</code>",
            lambda m: f'<a class="midi" href="{url_for("job_file", job_id=job_id, name=m.group(1))}"'
                      + (' target="_blank">' if m.group(1).endswith(".html") else '>')
                      + m.group(1).split("/", 1)[1]
                      + (" ↗" if m.group(1).endswith(".html") else " ↓") + "</a>", report_html)
        # The page's typography carries the headings (docs/STYLE.md); drop the emoji the .md report uses.
        report_html = re.sub("(<(?:h[23]|summary|strong)>)\\s*[\U0001F300-\U0001FAFF☀-➿⬀-⯿️]+\\s*",
                             r"\1", report_html)
        midi_dir = d / "output" / "midi"
        midi_files = sorted(p.name for p in midi_dir.glob("*.mid")) if midi_dir.is_dir() else []
    return render_template("job.html", job_id=job_id, s=s, report_html=report_html, midi_files=midi_files,
                           version=ma.CODE_VERSION, saved=accounts.owner_of(job_id) is not None)


@app.get("/jobs/<job_id>/status")
def job_status(job_id):
    s = read_status(get_job_dir(job_id))
    return jsonify(state=s.get("state"), message=s.get("message"), ai_pending=bool(s.get("ai_pending")))


@app.get("/jobs/<job_id>/files/<path:name>")
def job_file(job_id, name):
    return send_from_directory(get_job_dir(job_id) / "output", name,
                               as_attachment=name.endswith((".mid", ".musicxml")))


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
