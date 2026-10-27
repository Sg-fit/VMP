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

import html
import io
import hashlib
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
from collections import Counter
from pathlib import Path

import markdown
from flask import (Flask, Response, abort, jsonify, redirect, render_template, request,
                   send_file, send_from_directory, session, url_for)
from werkzeug.utils import secure_filename

import accounts
import charts
import crosscompare
import demos
import trends
import memory
import musicanalyze as ma
import style
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

def _browser_id(create=False):
    """This browser's private id (a random value in the signed session cookie), hashed: it's what a temporary result
    is tied to, so only the browser that made it can open it. None until this browser has started an analysis."""
    v = session.get("visitor")
    if not v and create:
        v = session["visitor"] = secrets.token_hex(16)
        session.permanent = True
    return hashlib.sha256(v.encode()).hexdigest()[:32] if v else None


def is_creator(job_id, st=None):
    """The person this result belongs to: the signed-in owner of a saved analysis, or the browser that started a
    temporary one."""
    owner = accounts.owner_of(job_id)
    if owner is not None:
        return owner == session.get("uid")
    st = st if st is not None else read_status(DATA_DIR / job_id)
    return bool(st.get("visitor")) and st["visitor"] == _browser_id()


def get_job_dir(job_id):
    """The job's folder, for the person it belongs to (404 for everyone else, so its existence isn't revealed):
    a saved analysis opens for its owner, a temporary one for the browser that started it. A demo, or a result its
    owner chose to share by link, opens for anyone with the link."""
    if not can_open(job_id):
        abort(404)
    return DATA_DIR / job_id


def can_open(job_id):
    """May this visitor open this result? Its owner (saved) or the browser that made it (temporary), or anyone when
    it's a demo or its owner shared it by link."""
    if not JOB_ID.match(job_id or ""):
        return False
    d = DATA_DIR / job_id
    if not d.is_dir():
        return False
    st = read_status(d)
    if st.get("unreadable"):
        return False  # never open a result whose owner we can't check
    if st.get("demo") or st.get("shared"):
        return True
    owner = accounts.owner_of(job_id)
    if owner is not None:
        return owner == session.get("uid")
    # results from before the browser rule have no visitor and keep working by link until they expire
    return not st.get("visitor") or st["visitor"] == _browser_id()


def tracks_of(job_id):
    """[(index, result)] of a result's own tracks (songs it was compared with and failed files left out)."""
    try:
        results = json.loads((DATA_DIR / job_id / "output" / "music_results.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [(i, r) for i, r in enumerate(results) if r.get("role") != "reference" and "error" not in r]


def purge_job(d):
    """Delete a result's folder and the cached analyses made from its audio (derived data: keys, chords, ...)."""
    cache = Path(os.environ.get("MUSIC_CACHE_DIR") or DATA_DIR / "cache")
    for f in (d / "output").glob("*.json") if (d / "output").is_dir() else []:  # musician and tag results alike
        try:
            for keys in re.findall(r'"cache_keys":\s*\[([^\]]*)\]', f.read_text(encoding="utf-8")):
                for key in re.findall(r'"([\w\-]+)"', keys):
                    (cache / f"{key}.pkl").unlink(missing_ok=True)
        except OSError:
            pass
    shutil.rmtree(d, ignore_errors=True)


def cleanup_old_jobs():
    """Delete temporary results after JOB_TTL_HOURS. Analyses saved to someone's history are kept."""
    cutoff = time.time() - JOB_TTL_HOURS * 3600
    keep = accounts.saved_job_ids() | {demos.job_id(s, ma.CODE_VERSION) for s in demos.DEMOS}  # current demos stay
    for d in DATA_DIR.iterdir():
        if d.is_dir() and JOB_ID.match(d.name) and d.name not in keep:
            s = read_status(d)
            # counted from when it was made: sharing or other later changes don't extend a result's life
            if s.get("state") not in ("queued", "running") and (s.get("created") or s.get("updated", 0)) < cutoff:
                purge_job(d)


def fail_interrupted_jobs():
    """Jobs left queued/running by a previous process will never finish — mark them failed."""
    for d in DATA_DIR.iterdir():
        if d.is_dir() and JOB_ID.match(d.name) and read_status(d).get("state") in ("queued", "running"):
            write_status(d, state="error", message="The server restarted before we could finish. Sorry about that. Please upload it again.")
            for sub in ("input", "reference"):
                shutil.rmtree(d / sub, ignore_errors=True)


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
            return "Something stopped the analysis partway through. Please try again."
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
                    return ("The server ran out of memory partway through. Try fewer tracks at once, or "
                            "shorter ones.")
                if code in (-11, 139):
                    return ("One of the audio libraries we use crashed (a segmentation fault). Please try again. "
                            "If it keeps happening with this file, try exporting it as mp3 or wav.")
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
                write_status(d, state="running", message="That didn't work the first time, so we're trying again…")
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
                    write_status(d, ai_pending=False, message="We couldn't get the AI ideas this time.")
                else:
                    write_status(d, state="error", message=problem)
        except Exception as e:  # never let one bad job kill the worker thread
            app.logger.exception("job %s failed", job_id)
            write_status(d, state="error", message=f"Something went wrong during the analysis: {str(e)[:300]}")
        finally:
            for sub in ("input", "reference"):  # don't keep uploaded audio
                shutil.rmtree(d / sub, ignore_errors=True)
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


INTRO_COOKIE = "intro_seen"


def _intro_seen(user):
    """Has this visitor closed the "what the app can't do" notice? Remembered per account (any device) when
    signed in, otherwise by a cookie on this browser. Closing it before signing in carries over to the account."""
    cookie = request.cookies.get(INTRO_COOKIE) == "1"
    if user and cookie and not user.get("seen_intro"):
        accounts.mark_intro_seen(user["id"])
    return cookie or bool(user and user.get("seen_intro"))


@app.context_processor
def inject_user():
    user = current_user()
    return {"user": user, "glossary": ma.GLOSSARY, "show_intro": not _intro_seen(user)}


@app.post("/intro/seen")
def intro_seen():
    user = current_user()
    if user:
        accounts.mark_intro_seen(user["id"])
    resp = Response(status=204)
    resp.set_cookie(INTRO_COOKIE, "1", max_age=365 * 24 * 3600, samesite="Lax", httponly=True)
    return resp


app.jinja_env.globals.update(charts=charts, camelot_names=ma.CAMELOT_NAMES)


PRIVATE_PATHS = ("/jobs/", "/history", "/profile", "/memory", "/account", "/style")


@app.after_request
def privacy_headers(resp):
    """Unreleased music stays private: results and personal pages are never indexed or kept in shared caches, no
    page can be framed by another site, and a result's address never leaks to a site it links to."""
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), interest-cohort=()")
    if request.path.startswith(PRIVATE_PATHS):
        resp.headers["Cache-Control"] = "private, no-store"
        resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
    return resp


def demo_pictures():
    """A thumbnail for each demo that has been built: its main track's timeline (sections over loudness)."""
    out = {}
    for slug in demos.DEMOS:
        f = DATA_DIR / demos.job_id(slug, ma.CODE_VERSION) / "output" / "music_results.json"
        try:
            r = next(x for x in json.loads(f.read_text(encoding="utf-8")) if x.get("role") != "reference"
                     and "error" not in x)
            out[slug] = ma.timeline_svg(r)
        except (OSError, ValueError, StopIteration):
            pass
    return out


@app.template_filter("datetime")
def _datetime(ts, date_only=False):
    return time.strftime("%d %b %Y" if date_only else "%d %b %Y, %H:%M", time.localtime(ts or 0))


def _safe_next(default="/"):
    nxt = request.values.get("next") or default
    return nxt if nxt.startswith("/") and not nxt.startswith("//") else default  # only ever back into this site


def _sign_in(user_id):
    visitor = session.get("visitor")  # this browser's own temporary results stay its own
    session.clear()  # a fresh session on sign-in
    if visitor:
        session["visitor"] = visitor
    session["uid"] = user_id
    session.permanent = True  # remembered for 30 days on this device


@app.route("/signup", methods=["GET", "POST"])
def signup():
    nxt, error = _safe_next(), None
    if request.method == "POST":
        if request.form.get("password") != request.form.get("password2"):
            error = "Those two passwords don't match. Have another go."
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
        error = "That username and password don't match. Check for a typo and try again."
    return render_template("account.html", mode="login", error=error, next=nxt,
                           username=request.form.get("username", "")), (401 if error else 200)


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


def _job_title(d):
    st = read_status(d)
    files = st.get("files") or []
    title = ", ".join(Path(f).stem for f in files[:3]) + (f" +{len(files) - 3} more" if len(files) > 3 else "")
    if st.get("mode") == "tags":
        names = [ln.strip() for ln in (st.get("songs") or "").splitlines() if ln.strip()]
        what = title or ", ".join(names[:2]) + (f" +{len(names) - 2} more" if len(names) > 2 else "")
        return "Tags: " + (what or ("playlist" if st.get("playlist") else "songs"))
    return ("Style: " + title if st.get("mode") == "style" and title else title) or "Analysis"


def _summaries(items):
    """Fill in the summary of saved analyses that finished since they were saved."""
    for it in items:
        if it["summary"] is None and read_status(DATA_DIR / it["job_id"]).get("state") == "done":
            out = DATA_DIR / it["job_id"] / "output"
            if (out / "music_results.json").exists():
                it["summary"] = accounts.job_summary(json.loads((out / "music_results.json").read_text(encoding="utf-8")))
            elif (out / "tags.json").exists():
                it["summary"] = accounts.tags_summary(json.loads((out / "tags.json").read_text(encoding="utf-8")))
            if it["summary"] is not None:
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
    a = request.args
    f = {"q": a.get("q", "").strip(), "project": a.get("project", ""), "status": a.get("status", ""),
         "label": a.get("label", ""), "starred": a.get("starred") == "1"}

    def match(it):
        """Search and filters: the title, track names, project, labels, notes and key all count for search."""
        if f["project"] and it.get("project") != f["project"]:
            return False
        if f["status"] and it.get("status") != f["status"]:
            return False
        if f["label"] and f["label"] not in it.get("labels", []):
            return False
        if f["starred"] and not it.get("starred"):
            return False
        if f["q"]:
            tracks = (it.get("summary") or {}).get("tracks") or []
            hay = " ".join([it.get("title", ""), it.get("project", ""), it.get("notes", ""), " ".join(it.get("labels", []))]
                           + [f"{t.get('file', '')} {t.get('key', '')}" for t in tracks]).lower()
            return all(w in hay for w in f["q"].lower().split())
        return True
    shown = [it for it in items if match(it)]
    view = trends.overview(shown, request.args.get("range", trends.DEFAULT_FRAME))
    view["tiles"] = trends.tiles(view)
    def hurl(**changes):
        """This page's address with some filters changed (None or '' removes one)."""
        args = {**request.args.to_dict(), **changes}
        return url_for("history", **{k: v for k, v in args.items() if v})
    return render_template("history.html", items=items, v=view, version=ma.CODE_VERSION, f=f, hurl=hurl,
                           filtered=any(f.values()), projects=accounts.projects(u["id"]),
                           titles={it["job_id"]: it["title"] for it in items},
                           version_counts=Counter(it.get("version_of") for it in items if it.get("version_of")),
                           labels=accounts.all_labels(u["id"]), statuses=accounts.STATUSES)


MEMORY_SCAN = 15  # most recent saved analyses whose ideas count as "already shown"


def user_memory(u):
    """The signed-in musician's memory (memory.py), or None when not signed in or switched off."""
    if not u or not u.get("memory_on"):
        return None
    tracks, contexts, refs, seen = [], [], [], []
    for n, it in enumerate(_summaries(accounts.history(u["id"]))):
        sm = it.get("summary") or {}
        if sm.get("kind") == "tags":
            continue
        tracks += sm.get("tracks") or []
        st = read_status(DATA_DIR / it["job_id"])
        c = st.get("context")
        contexts.append({"keys": c.get("instruments") or [], "genre_key": c.get("genre")} if c else None)
        refs += st.get("references") or ([st["reference"]] if st.get("reference") else [])
        res = DATA_DIR / it["job_id"] / "output" / "music_results.json"
        if n < MEMORY_SCAN and res.exists():
            try:
                for r in json.loads(res.read_text(encoding="utf-8")):
                    seen += [memory.idea_key(i.get("area"), i.get("idea")) for i in r.get("ideas") or []]
            except (OSError, ValueError):
                pass
    return memory.build(tracks, contexts, accounts.list_feedback(u["id"]), accounts.list_corrections(u["id"]),
                        u.get("goals") or "", refs, seen)


@app.get("/memory")
def memory_page():
    u = _require_user()
    mem = memory.build([], [], [], [], u.get("goals") or "")
    if u.get("memory_on"):
        mem = user_memory(u)
    return render_template("memory.html", mem=mem, feedback=accounts.list_feedback(u["id"]),
                           interest=accounts.interest(u["id"]), section_names=SECTION_NAMES,
                           corrections=accounts.list_corrections(u["id"]), fields=accounts.CORRECTABLE,
                           genres={k: v[0] for k, v in ma.GENRES.items()}, instruments=ma.INSTRUMENTS,
                           version=ma.CODE_VERSION)


@app.post("/memory")
def memory_update():
    u = _require_user()
    accounts.set_memory(u["id"], request.form.get("on") == "1", request.form.get("goals"))
    return redirect(url_for("memory_page"))


@app.post("/memory/forget")
def memory_forget():
    u = _require_user()
    accounts.forget_everything(u["id"])
    return redirect(url_for("memory_page"))


@app.post("/memory/<table>/<int:item_id>/delete")
def memory_delete(table, item_id):
    u = _require_user()
    accounts.delete_memory_item(u["id"], table, item_id)
    return redirect(url_for("memory_page"))


@app.post("/interest")
def record_interest():
    """Which part of a report someone used: {section, kind}. Kept only for a signed-in musician with the memory
    on (section name and counts, nothing about the track); everyone else's stays in their own browser."""
    u = current_user()
    data = request.get_json(silent=True) or {}
    if u and u.get("memory_on"):
        accounts.add_interest(u["id"], str(data.get("section", ""))[:20], str(data.get("kind", ""))[:10])
    return "", 204


@app.post("/memory/interest/forget")
def interest_forget():
    u = _require_user()
    accounts.forget_interest(u["id"])
    return redirect(url_for("memory_page"))


def _teachable(job_id):
    """Feedback and corrections come only from the signed-in owner of a saved analysis, with the memory on."""
    u = current_user()
    if not u or not u.get("memory_on") or accounts.owner_of(job_id) != u["id"]:
        abort(403)
    return u


@app.post("/jobs/<job_id>/feedback")
def job_feedback(job_id):
    u = _teachable(job_id)
    data = request.get_json(silent=True) or {}
    accounts.add_feedback(u["id"], job_id, str(data.get("track", ""))[:200], str(data.get("area", ""))[:60],
                          str(data.get("text", ""))[:600], int(data.get("vote", 0) or 0))
    return jsonify(ok=True)


@app.post("/jobs/<job_id>/correct")
def job_correct(job_id):
    u = _teachable(job_id)
    problem = accounts.add_correction(u["id"], job_id, request.form.get("track", ""), request.form.get("field", ""),
                                      request.form.get("value", ""))
    return redirect(url_for("job", job_id=job_id, corrected="0" if problem else "1"))


@app.get("/profile")
def profile():
    u = _require_user()
    items = _summaries(accounts.history(u["id"]))
    return render_template("profile.html", p=accounts.profile_summary(items), recent=items[:5], version=ma.CODE_VERSION)


@app.post("/jobs/<job_id>/save")
def job_save(job_id):
    u = _require_user()
    d = get_job_dir(job_id)
    if read_status(d).get("demo"):  # shared with every visitor: saving would make it one person's private page
        abort(403, "Demos can't be saved to a history - upload your own track to keep one.")
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
        purge_job(DATA_DIR / job_id)
    return redirect(url_for("history"))


@app.post("/jobs/<job_id>/forget")
def job_forget(job_id):
    """Delete a result now, saved or not, instead of waiting for it to expire. Only the person it belongs to."""
    if not JOB_ID.match(job_id) or not (DATA_DIR / job_id).is_dir() or not is_creator(job_id):
        abort(404)
    if read_status(DATA_DIR / job_id).get("state") in ("queued", "running"):
        abort(409, "It's still being analysed. Delete it once it's done.")
    u = current_user()
    if u:
        accounts.forget_job(u["id"], job_id)
    purge_job(DATA_DIR / job_id)
    return redirect(url_for("history") + "?deleted=1" if u else url_for("index", deleted=1))


@app.post("/jobs/<job_id>/share")
def job_share(job_id):
    """Share by link, or stop sharing. Only the person the result belongs to."""
    if not JOB_ID.match(job_id) or not (DATA_DIR / job_id).is_dir() or not is_creator(job_id):
        abort(404)
    write_status(DATA_DIR / job_id, shared=request.form.get("on") == "1")
    return redirect(url_for("job", job_id=job_id))


@app.post("/jobs/<job_id>/organise")
def job_organise(job_id):
    """Project, status, labels, notes and the star of a saved analysis."""
    u = _require_user()
    f = request.form
    accounts.organise(u["id"], job_id, project=f.get("project"), status=f.get("status"), labels=f.get("labels"),
                      notes=f.get("notes"), starred=f.get("starred") == "1" if "starred_present" in f else None)
    if "version_of" in f:
        accounts.set_version(u["id"], job_id, f.get("version_of") or "")
    return redirect(_safe_next(url_for("job", job_id=job_id)))


@app.post("/jobs/<job_id>/star")
def job_star(job_id):
    u = _require_user()
    accounts.organise(u["id"], job_id, starred=request.form.get("on") == "1")
    return redirect(_safe_next(url_for("history")))


@app.post("/projects/rename")
def project_rename():
    u = _require_user()
    accounts.rename_project(u["id"], request.form.get("old", ""), request.form.get("new", ""))
    return redirect(url_for("history", project=request.form.get("new", "") or None))


@app.get("/compare")
def compare_page():
    """Your tracks side by side - any of your own (?t=job:index), a song's versions (?song=first job), and results
    other musicians shared with you (?link=their result's address). Nothing private of anyone else's can be added:
    an address you can't open is named as unavailable, and nothing about it is shown."""
    u = _require_user()
    picked, names, skipped, are_versions = [], [], [], False
    song = request.args.get("song", "")
    if song:
        chain = accounts.versions(u["id"], song)
        for k, v in enumerate(chain, 1):
            tr = tracks_of(v["job_id"]) if can_open(v["job_id"]) else []
            if tr:
                picked.append(tr[0][1])
                names.append(f"v{k} · {time.strftime('%d %b', time.localtime(v['created']))}")
        are_versions = len(picked) > 1
    for t in request.args.getlist("t"):
        job_id, _, idx = t.partition(":")
        tr = dict(tracks_of(job_id)) if can_open(job_id) else {}
        r = tr.get(int(idx) if idx.isdigit() else 0)
        if r is None:
            skipped.append(t)
            continue
        picked.append(r)
        names.append(Path(r["file"]).stem)
    for link in request.args.getlist("link"):
        m = re.search(r"/jobs/([0-9a-f]{32})", link or "")
        if not link.strip():
            continue
        if not m or not can_open(m.group(1)):
            skipped.append(link.strip()[:80])
            continue
        for _, r in tracks_of(m.group(1))[:1]:
            picked.append(r)
            names.append("Shared: " + Path(r["file"]).stem)
    names = [n[:40] for n in names]
    for i, n in enumerate(names):  # two tracks with one name stay apart
        if names.count(n) > 1:
            names[i] = f"{n} ({i + 1})"
    view = crosscompare.report(picked, names, are_versions) if len(picked) >= 2 else None
    pics = {}
    if view:
        tracks = picked[:crosscompare.MAX_TRACKS]
        others = [dict(r, file=n) for r, n in zip(tracks[1:], view["names"][1:])]
        rows = [(m, unit, [ma._measures(r)[m][0] for r in tracks]) for m, (_, unit) in ma._measures(tracks[0]).items()]
        pics = {"measures": charts.dumbbell(rows, view["names"]),
                "energy": ma.energy_overlay_svg(tracks[0], others, mine_label=view["names"][0]),
                "tonal": ma.tonal_overlay_svg(tracks[0], others, mine_label=view["names"][0]),
                "mood": charts.mood_map([("ABCD"[i], (r.get("feel") or {}).get("valence"), (r.get("feel") or {}).get("arousal"),
                                          f"{n}: {(r.get('feel') or {}).get('mood', '')}")
                                         for i, (r, n) in enumerate(zip(tracks, view["names"]))],
                                        size=220, path=are_versions, tags=True, label="Each track on the energy/mood map")}
    library = [{"job_id": it["job_id"], "title": it["title"], "created": it["created"],
                "tracks": [(i, Path(r["file"]).stem) for i, r in tracks_of(it["job_id"])]}
               for it in accounts.history(u["id"])[:200] if not ((it.get("summary") or {}).get("kind") == "tags")]
    return render_template("compare.html", view=view, pics=pics, skipped=skipped, library=library,
                           chosen=set(request.args.getlist("t")), song=song, max_tracks=crosscompare.MAX_TRACKS,
                           too_many=len(picked) > crosscompare.MAX_TRACKS, version=ma.CODE_VERSION)


@app.get("/account")
def account_page():
    u = _require_user()
    return render_template("settings.html", saved=len(accounts.history(u["id"])), version=ma.CODE_VERSION,
                           error=request.args.get("error"))


@app.get("/account/export")
def account_export():
    """Everything we hold about you, in one zip: your account data, and each saved analysis's report, data,
    MIDI and sheet music."""
    u = _require_user()
    data = accounts.export(u["id"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("account.json", json.dumps(data, indent=2, ensure_ascii=False))
        for it in data["saved"]:
            out = DATA_DIR / it["job_id"] / "output"
            folder = re.sub(r"[^\w\-. ]+", "_", it["title"])[:60] + f" ({it['job_id'][:8]})"
            for f in out.rglob("*") if out.is_dir() else []:
                if f.is_file():
                    z.write(f, f"analyses/{folder}/{f.relative_to(out).as_posix()}")
    return Response(buf.getvalue(), mimetype="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{u["username"]}-music-analyzer-data.zip"'})


@app.post("/account/delete")
def account_delete():
    """Delete the account and everything in it, after the password is typed again."""
    u = _require_user()
    if not accounts.check_login(u["username"], request.form.get("password")):
        time.sleep(1)
        return redirect(url_for("account_page", error="That password isn't right, so nothing was deleted."))
    for job_id in accounts.delete_account(u["id"]):
        if JOB_ID.match(job_id):
            purge_job(DATA_DIR / job_id)
    session.clear()
    return redirect(url_for("index", deleted="account"))


@app.get("/privacy")
def privacy():
    return render_template("privacy.html", ttl=JOB_TTL_HOURS, cache_hours=ma.CACHE_HOURS, version=ma.CODE_VERSION,
                           ai_on=bool(LLM))


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
                           max_refs=ma.MAX_REFERENCES,
                           instruments=ma.INSTRUMENTS, instrument_groups=ma.INSTRUMENT_GROUPS,
                           genres={k: v[0] for k, v in ma.GENRES.items()},
                           max_mb=MAX_UPLOAD_MB, exts=" ".join(sorted(ma.AUDIO_EXTS)), ttl=JOB_TTL_HOURS,
                           mode=request.form.get("mode") if request.form.get("mode") in ("tags", "style") else
                           "tags" if request.path == "/find" else "style" if request.path == "/style/new" else "musician",
                           library_tracks=len(style_tracks()) if current_user() else 0,
                           form=request.form, demos=demos.DEMOS, demo_pictures=demo_pictures(),
                           prefill=memory.prefill(user_memory(current_user()))), code


@app.get("/")
def index():
    return form_page()


@app.get("/find")
def find():
    """The same form, opened on "Find music like this"."""
    return form_page()


@app.get("/style/new")
def style_new():
    """The same form, opened on "Your style"."""
    return form_page()


def style_tracks(job_ids=None):
    """[{r, name, session, when, title, project}] for the style page: every track of yours across all your upload
    sessions - your saved analyses when you're signed in, else this browser's own results (kept a day). Songs you
    compared with and tag lists are left out: they aren't your music."""
    u, out = current_user(), []
    if u:
        items = [(it["job_id"], it["created"], it["title"], it.get("project") or "")
                 for it in accounts.history(u["id"]) if (it.get("summary") or {}).get("kind") != "tags"]
    else:
        me, items = _browser_id(), []
        if me and DATA_DIR.is_dir():
            for d in DATA_DIR.iterdir():
                if not JOB_ID.match(d.name):
                    continue
                st = read_status(d)
                if st.get("visitor") == me and st.get("state") == "done" and st.get("mode") in ("musician", "style") \
                        and not st.get("demo") and accounts.owner_of(d.name) is None:
                    items.append((d.name, st.get("created") or 0, _job_title(d), ""))
    for job_id, when, title, project in items:
        if job_ids is not None and job_id not in job_ids:
            continue
        for _, r in tracks_of(job_id):
            out.append({"r": r, "name": Path(r["file"]).stem[:60], "session": job_id, "when": when, "title": title,
                        "project": project})
    return out


@app.get("/style")
def style_page():
    """Your style across every track and upload session, or within one project or time span."""
    u = current_user()
    tracks = style_tracks()
    project, frame = request.args.get("project", ""), request.args.get("frame", "all")
    projects = sorted({t["project"] for t in tracks if t["project"]})
    if project:
        tracks = [t for t in tracks if t["project"] == project]
    if frame in trends.FRAMES and trends.FRAMES[frame][1]:
        since = time.time() - trends.FRAMES[frame][1] * 86400
        tracks = [t for t in tracks if t["when"] >= since]
    view = style.profile(tracks)
    return render_template("style.html", view=view, pics=style.pictures(view), projects=projects, project=project,
                           frame=frame, frames=trends.FRAMES, signed_in=bool(u), min_tracks=style.MIN_TRACKS,
                           hurl=lambda job_id: url_for("job", job_id=job_id), version=ma.CODE_VERSION)


_demo_lock = threading.Lock()


@app.get("/demo/<slug>")
def demo(slug):
    """A finished example report, made from generated audio the first time anyone opens it, then shared."""
    if slug not in demos.DEMOS:
        abort(404)
    job_id = demos.job_id(slug, ma.CODE_VERSION)
    d = DATA_DIR / job_id
    with _demo_lock:  # two first visitors at once must not both build it
        if read_status(d).get("state") not in ("queued", "running", "done"):
            shutil.rmtree(d, ignore_errors=True)
            fields = demos.prepare(slug, d)
            ahead = job_queue.qsize()
            write_status(d, state="queued", created=time.time(), **fields,
                         message=f"Making the demo: you're in line, with {ahead} ahead of you…" if ahead
                         else "Making the demo (only the first visitor waits)…")
            job_queue.put(job_id)
    return redirect(url_for("job", job_id=job_id))


@app.post("/analyze")
def analyze():
    uploads = [f for f in request.files.getlist("tracks") if f and f.filename]
    tags_mode = request.form.get("mode") == "tags"
    style_mode = request.form.get("mode") == "style"
    playlist = (request.form.get("playlist") or "").strip()[:500]
    songs = (request.form.get("songs") or "")[:8000]
    description = (request.form.get("description") or "").strip()[:300]
    if tags_mode and playlist and not re.match(r"^(https?://|spotify:)", playlist):
        return form_page("That playlist link doesn't look right. It should start with https://, like a link "
                         "copied from Spotify or YouTube.", 400)
    if not uploads and not (tags_mode and (playlist or songs.strip() or description)):
        return form_page("Choose at least one audio file to get started." if not tags_mode else
                         "Add a few songs first (upload them, paste a playlist link or type their names), or "
                         "describe what you're after.", 400)
    if style_mode and len(uploads) < 2:
        return form_page("Add at least two tracks: a style is what they have in common."
                         + (" Or open Your style to read it from the tracks you've saved." if current_user() else ""),
                         400)
    if len(uploads) > MAX_FILES:
        return form_page(f"That's more than {MAX_FILES} files. Upload up to {MAX_FILES} at a time.", 400)

    if _too_many_uploads():
        return form_page(f"You've started {UPLOADS_PER_HOUR} analyses in the last hour, which is our limit. "
                         "Give it a little while and try again.", 429)
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
    if not saved and uploads and not (tags_mode and (playlist or songs.strip() or description)):
        shutil.rmtree(d, ignore_errors=True)
        return form_page("We can't read any of those files. Try mp3, wav, m4a, flac, ogg or aac.", 400)
    references = []  # songs the musician admires, to measure their tracks against (up to ma.MAX_REFERENCES)
    ref_uploads = [f for f in request.files.getlist("reference") if f and f.filename] if not (tags_mode or style_mode) else []
    if len(ref_uploads) > ma.MAX_REFERENCES:
        shutil.rmtree(d, ignore_errors=True)
        return form_page(f"Up to {ma.MAX_REFERENCES} songs you admire at a time, please.", 400)
    for ref_upload in ref_uploads:
        name = secure_filename(ref_upload.filename) or "reference"
        if Path(name).suffix.lower() not in ma.AUDIO_EXTS:
            shutil.rmtree(d, ignore_errors=True)
            return form_page(f"We can't read {name}. Try mp3, wav, m4a, flac, ogg or aac.", 400)
        (d / "reference").mkdir(exist_ok=True)
        stem, ext, n = Path(name).stem, Path(name).suffix, 2
        while (d / "reference" / name).exists():
            name, n = f"{stem}_{n}{ext}", n + 1
        ref_upload.save(d / "reference" / name)
        references.append(name)

    ahead = job_queue.qsize()
    if tags_mode:  # finding music: none of the musician's options apply
        write_status(d, state="queued", mode="tags", files=saved, visitor=_browser_id(create=True), playlist=playlist, songs=songs,
                     description=description, created=time.time(),
                     message=f"You're in line, with {ahead} ahead of you…" if ahead else "Starting…")
    else:
        # "style": the same analysis of every track, read together on the result page (no AI: it's per track)
        write_status(d, state="queued", mode="style" if style_mode else "musician", files=saved, references=references,
                     visitor=_browser_id(create=True),
                     memory=user_memory(current_user()),  # None unless signed in with the memory switched on
                     use_ai=bool(LLM and request.form.get("ai")) and not style_mode,
                     melody=(request.form.get("melody") or "")[:2000],
                     context=ma.clean_context(request.form.getlist("instruments"), request.form.get("genre")),
                     created=time.time(),
                     message=f"You're in line, with {ahead} ahead of you…" if ahead else "Starting…")
    u = current_user()
    if u:  # signed in: goes straight into their history
        accounts.save_job(u["id"], job_id, _job_title(d))
    job_queue.put(job_id)
    return redirect(url_for("job", job_id=job_id))


# Each report section's key (for folding and for remembering which parts a musician uses), by a word of its title
SECTION_WORDS = [("admire", "you admire"), ("harmony", "Harmony"), ("structure", "Structure"),
                 ("energy", "Energy"), ("instruments", "Instruments"), ("melody", "Melody"), ("dna", "Shared DNA"),
                 ("groove", "Groove"), ("producer", "Producer"), ("past", "earlier tracks"),
                 ("gaps", "what's missing"), ("ideas", "Ideas to try"), ("ai", "AI brainstorm"),
                 ("sheet", "Sheet music"), ("midi", "MIDI")]
assert {k for k, _ in SECTION_WORDS} == set(accounts.INTEREST_SECTIONS)
SECTION_NAMES = {"admire": "Songs you admire", "harmony": "Harmony", "structure": "Structure", "energy": "Energy & mood",
                 "instruments": "Instruments", "melody": "Melody", "dna": "Shared DNA", "groove": "Groove & sound",
                 "producer": "Producer's view", "past": "Earlier tracks", "gaps": "What's missing", "ideas": "Ideas",
                 "ai": "AI ideas", "sheet": "Sheet music", "midi": "MIDI"}


def section_key(title):
    t = title.lower().replace("’", "'")
    return next((k for k, w in SECTION_WORDS if w.lower() in t), None)


def fold_sections(report_html):
    """Each report section (an <h3> and what follows it, up to the next heading) becomes a <section> with its key.
    Where the report marked the gist (<!--more-->, musicanalyze.MORE), what follows the mark goes into a folded
    "More" box, so each section shows its most important part first."""
    out = []
    for part in re.split(r"(?=<h[23][ >])", report_html):
        m = re.match(r"<h3[^>]*>(.*?)</h3>", part, re.S)
        if not m:
            out.append(part.replace(ma.MORE, ""))
            continue
        title = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
        key = section_key(title) or "other"
        body = part[m.end():]
        if ma.MORE in body:
            gist, rest = body.split(ma.MORE, 1)
            rest = rest.replace(ma.MORE, "")
            if re.sub(r"<[^>]+>|\s", "", rest) or re.search(r"<(svg|table)", rest):
                body = (f'{gist}<details class="more" data-sec="{key}"><summary>More: '
                        f'{html.escape(title)}</summary><div class="more-body">{rest}</div></details>')
            else:
                body = gist + rest
        out.append(f'<section class="rsec" data-sec="{key}" id="sec-{key}">{m.group(0)}{body}</section>')
    return "".join(out)


CHART_SVG = re.compile(r'<svg class="(?:chart|timeline)"[^>]*>.*?</svg>', re.S)


def _chart_name(svg):
    """A short name for a chart, from the start of its accessible label ("Chords over time, coloured by..." ->
    "Chords over time")."""
    m = re.search(r'aria-label="([^"]*)"', svg)
    label = html.unescape(m.group(1)) if m else ""
    label = re.split(r"[.:,(]| with | — ", label, maxsplit=1)[0].strip()
    return label[:42] or "Chart"


def tab_tracks(report_html):
    """Each track's report as a summary and tabs: its header, the few plain sentences and the at-a-glance pictures,
    then every other chart of the report as a small picture (a click opens its part), then one tab per section
    (fold_sections' <section>s), each with its own charts. The page script shows one tab at a time; without
    scripts, the tabs are links to the sections, which all stay on the page."""
    out, n = [], 0
    for part in re.split(r"(?=<h2[ >])", report_html):
        if not part.startswith("<h2") or '<section class="rsec"' not in part:
            out.append(part)
            continue
        n += 1
        h2_end, first = part.index("</h2>") + 5, part.index('<section class="rsec"')
        head, intro, body = part[:h2_end], part[h2_end:first], part[first:]
        tabs, thumbs, panels, used = [], [], [], set()
        charts_seen = 0
        for sec in re.split(r'(?=<section class="rsec")', body):
            m = re.match(r'<section class="rsec" data-sec="(\w+)" id="sec-\w+">(<h3[^>]*>(.*?)</h3>)', sec, re.S)
            if not m:
                panels.append(sec)
                continue
            key = m.group(1)
            title = html.unescape(re.sub(r"<[^>]+>", "", m.group(3))).strip()
            name = SECTION_NAMES.get(key) or title
            pid = f"t{n}-{key}" + (f"-{len(panels)}" if key in used else "")
            used.add(key)

            def mark(svg_m):
                nonlocal charts_seen
                charts_seen += 1
                cid = f"t{n}c{charts_seen}"
                svg = svg_m.group(0)
                # the summary's copy is a picture only: no timeline links, no playback, nothing to tab to
                thumb = re.sub(r' data-(?:dur|x0|x1|t0|t1|grid|hats|bpm|beats)="[^"]*"', "", svg)
                thumbs.append(f'<button type="button" class="thumb" data-go="{pid}" data-chart="{cid}">'
                              f'<span class="tl"><b>{html.escape(_chart_name(svg))}</b> · {html.escape(name)}</span>'
                              f'{thumb}</button>')
                return svg.replace("<svg ", f'<svg data-chart="{cid}" ', 1)
            sec = CHART_SVG.sub(mark, sec)
            sec = sec.replace(m.group(0)[:m.group(0).index(">") + 1],
                              f'<section class="rsec" data-sec="{key}" id="{pid}" role="tabpanel" '
                              f'aria-label="{html.escape(name)}">', 1)
            tabs.append(f'<a class="tab" role="tab" href="#{pid}" data-go="{pid}" data-sec="{key}">{html.escape(name)}</a>')
            panels.append(sec)
        gallery = (f'<div class="gallery" aria-label="Every picture in this report"><p class="eyebrow">All the '
                   f'pictures · click one to open its part</p><div class="thumbs">{"".join(thumbs)}</div></div>'
                   if thumbs else "")
        out.append(f'<article class="track" data-track="{n}">{head}<div class="tsum">{intro}{gallery}</div>'
                   f'<nav class="tabs" role="tablist" aria-label="Parts of the report">{"".join(tabs)}</nav>'
                   f'{"".join(panels)}</article>')
    return "".join(out)


@app.get("/jobs/<job_id>")
def job(job_id):
    d = get_job_dir(job_id)
    s = read_status(d)
    report_html, midi_files = None, []
    if s.get("state") == "done" and s.get("mode") == "tags":
        report_md = (d / "output" / "tags_report.md").read_text(encoding="utf-8")
        report_html = markdown.markdown(report_md, extensions=["tables", "fenced_code", "md_in_html"])
        # search links open in a new tab, so the tags stay open while browsing
        report_html = report_html.replace('<a href="http', '<a target="_blank" rel="noopener" href="http')
    elif s.get("state") == "done":
        report_md = (d / "output" / "music_report.md").read_text(encoding="utf-8")
        report_html = markdown.markdown(report_md, extensions=["tables", "fenced_code", "md_in_html"])
        report_html = re.sub(  # turn `midi/x.mid` and `sheets/x.html` mentions into links
            r"<code>((?:midi|sheets)/[\w.\-]+\.(?:mid|musicxml|html))</code>",
            lambda m: f'<a class="midi" href="{url_for("job_file", job_id=job_id, name=m.group(1))}"'
                      + (' target="_blank">' if m.group(1).endswith(".html") else '>')
                      + m.group(1).split("/", 1)[1]
                      + (" ↗" if m.group(1).endswith(".html") else " ↓") + "</a>", report_html)
        # The page's typography carries the headings (docs/STYLE.md); drop the emoji the .md report uses.
        report_html = re.sub("(<(?:h[23]|summary|strong)>)\\s*[\U0001F300-\U0001FAFF☀-➿⬀-⯿️]+\\s*",
                             r"\1", report_html)
        report_html = tab_tracks(fold_sections(report_html))
        midi_dir = d / "output" / "midi"
        midi_files = sorted(p.name for p in midi_dir.glob("*.mid")) if midi_dir.is_dir() else []
    u, owner = current_user(), accounts.owner_of(job_id)
    teach = bool(u and u.get("memory_on") and owner == u["id"] and s.get("mode") != "tags" and not s.get("demo"))
    meta = next((it for it in accounts.history(u["id"]) if it["job_id"] == job_id), None) \
        if u and owner == u["id"] else None
    used = accounts.interest(u["id"]) if u and u.get("memory_on") else {}
    style_view = None
    if s.get("state") == "done" and s.get("mode") == "style":  # these tracks read together, before each on its own
        style_view = style.profile([{"r": r, "name": Path(r["file"]).stem[:60], "session": job_id,
                                     "when": s.get("created") or 0, "title": ""} for _, r in tracks_of(job_id)])
    return render_template("job.html", job_id=job_id, s=s, report_html=report_html, midi_files=midi_files,
                           interest={k: v["score"] for k, v in used.items()}, remember=bool(u and u.get("memory_on")),
                           section_names=SECTION_NAMES, style_view=style_view, style_pics=style.pictures(style_view),
                           min_tracks=style.MIN_TRACKS,
                           version=ma.CODE_VERSION, saved=owner is not None, demos=demos.DEMOS, teach=teach,
                           creator=is_creator(job_id, s), meta=meta, statuses=accounts.STATUSES,
                           projects=accounts.projects(u["id"]) if meta else [],
                           expires=(s.get("created") or s.get("updated") or time.time()) + JOB_TTL_HOURS * 3600,
                           fields=accounts.CORRECTABLE, corrected=request.args.get("corrected"))


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
    return form_page(f"That upload is too big. Keep it under {MAX_UPLOAD_MB} MB in total.", 413)


if __name__ == "__main__":
    # Development server only — use gunicorn (see top of file / README) on a real server.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
