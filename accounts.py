"""User accounts and saved analyses (history), stored in one SQLite file next to the jobs.

Analysis itself needs no account. An account lets someone keep their results (saved jobs are never
deleted by the clean-up of temporary results) and see a summary of everything they've analysed.
"""

import json
import re
import sqlite3
import time
from collections import Counter

from werkzeug.security import check_password_hash, generate_password_hash

USERNAME = re.compile(r"^[A-Za-z0-9_.\-]{3,30}$")
MIN_PASSWORD = 8
MAX_SAVED = 300  # per account, so one account can't fill the disk

_db_path = None


def init(path):
    global _db_path
    _db_path = str(path)
    with _connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                pw_hash TEXT NOT NULL,
                created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS saved (
                job_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                title TEXT NOT NULL,
                created REAL NOT NULL,
                summary TEXT);
            CREATE INDEX IF NOT EXISTS saved_by_user ON saved(user_id, created);
        """)


def _connect():
    db = sqlite3.connect(_db_path, timeout=10)
    db.row_factory = sqlite3.Row
    return db


# ---------------------------------------------------------------- accounts

def create_user(username, password):
    """Returns (user_id, None) or (None, an error message for the form)."""
    username = (username or "").strip()
    if not USERNAME.match(username):
        return None, "Usernames are 3–30 letters, numbers, dots, dashes or underscores."
    if len(password or "") < MIN_PASSWORD:
        return None, f"Use a password of at least {MIN_PASSWORD} characters."
    try:
        with _connect() as db:
            cur = db.execute("INSERT INTO users (username, pw_hash, created) VALUES (?, ?, ?)",
                             (username, generate_password_hash(password), time.time()))
            return cur.lastrowid, None
    except sqlite3.IntegrityError:
        return None, "That username is taken — try another."


def check_login(username, password):
    """The user's id if the username and password match, else None."""
    with _connect() as db:
        row = db.execute("SELECT id, pw_hash FROM users WHERE username = ?", ((username or "").strip(),)).fetchone()
    if row and check_password_hash(row["pw_hash"], password or ""):
        return row["id"]
    return None


def get_user(user_id):
    with _connect() as db:
        row = db.execute("SELECT id, username, created FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------- history

def owner_of(job_id):
    with _connect() as db:
        row = db.execute("SELECT user_id FROM saved WHERE job_id = ?", (job_id,)).fetchone()
    return row["user_id"] if row else None


def save_job(user_id, job_id, title):
    """Add a job to a user's history. Returns None, or an error message."""
    with _connect() as db:
        row = db.execute("SELECT user_id FROM saved WHERE job_id = ?", (job_id,)).fetchone()
        if row:
            return None if row["user_id"] == user_id else "This analysis belongs to someone else."
        count = db.execute("SELECT COUNT(*) FROM saved WHERE user_id = ?", (user_id,)).fetchone()[0]
        if count >= MAX_SAVED:
            return f"Your history is full ({MAX_SAVED} analyses). Delete some old ones first."
        db.execute("INSERT INTO saved (job_id, user_id, title, created) VALUES (?, ?, ?, ?)",
                   (job_id, user_id, title[:200], time.time()))
    return None


def forget_job(user_id, job_id):
    with _connect() as db:
        return db.execute("DELETE FROM saved WHERE job_id = ? AND user_id = ?", (job_id, user_id)).rowcount > 0


def rename_job(user_id, job_id, title):
    with _connect() as db:
        db.execute("UPDATE saved SET title = ? WHERE job_id = ? AND user_id = ?", (title.strip()[:200], job_id, user_id))


def saved_job_ids():
    with _connect() as db:
        return {r["job_id"] for r in db.execute("SELECT job_id FROM saved")}


def history(user_id):
    with _connect() as db:
        rows = db.execute("SELECT job_id, title, created, summary FROM saved WHERE user_id = ? ORDER BY created DESC",
                          (user_id,)).fetchall()
    return [{**dict(r), "summary": json.loads(r["summary"]) if r["summary"] else None} for r in rows]


def store_summary(job_id, summary):
    with _connect() as db:
        db.execute("UPDATE saved SET summary = ? WHERE job_id = ?", (json.dumps(summary), job_id))


# ---------------------------------------------------------------- summaries

def job_summary(results):
    """A compact per-track summary of one analysis (from music_results.json), kept in the database so the
    history and profile pages don't have to re-read every result file."""
    tracks = []
    for r in results:
        if "error" in r:
            tracks.append({"file": r.get("file", "?"), "error": True})
            continue
        feel = r.get("feel") or {}
        tracks.append({
            "file": r["file"], "duration_sec": r.get("duration_sec"), "duration": r.get("duration"),
            "tempo_bpm": r.get("tempo_bpm"), "meter": (r.get("meter") or {}).get("label"),
            "key": r.get("key"), "camelot": r.get("camelot"), "mode": r.get("mode"),
            "energy": feel.get("energy"), "mood": feel.get("mood"), "valence": feel.get("valence"),
            "form": " ".join(s["label"] for s in r.get("sections", [])),
            "chords": {c["chord"]: c["seconds"] for c in r.get("chords_used", [])},
            "drums": bool((r.get("drums") or {}).get("present")), "bass": bool((r.get("bass") or {}).get("present")),
        })
    return {"tracks": tracks}


def profile_summary(items):
    """Everything a user has saved, summed up: what keys, tempos, modes, chords and moods they gravitate to."""
    tracks = [t for it in items if it.get("summary") for t in it["summary"]["tracks"] if not t.get("error")]
    out = {"analyses": len(items), "tracks": len(tracks)}
    if not tracks:
        return out
    out["minutes"] = round(sum(t.get("duration_sec") or 0 for t in tracks) / 60, 1)
    out["keys"] = Counter(t["key"] for t in tracks if t.get("key")).most_common(5)
    out["modes"] = Counter(t["mode"] for t in tracks if t.get("mode")).most_common(4)
    out["meters"] = Counter(t["meter"] for t in tracks if t.get("meter")).most_common(3)
    out["moods"] = Counter(t["mood"] for t in tracks if t.get("mood")).most_common(4)
    tempos = sorted(t["tempo_bpm"] for t in tracks if t.get("tempo_bpm"))
    if tempos:
        out["tempo"] = {"min": round(tempos[0]), "max": round(tempos[-1]),
                        "median": round(tempos[len(tempos) // 2])}
    energies = [t["energy"] for t in tracks if t.get("energy")]
    if energies:
        out["energy_avg"] = round(sum(energies) / len(energies), 1)
    chords = Counter()
    for t in tracks:
        chords.update(t.get("chords") or {})
    out["chords"] = [c for c, _ in chords.most_common(8)]
    out["drums_pct"] = round(100 * sum(t.get("drums", False) for t in tracks) / len(tracks))
    out["bass_pct"] = round(100 * sum(t.get("bass", False) for t in tracks) / len(tracks))
    # One plain sentence a musician can read at a glance.
    parts = []
    if out["keys"]:
        parts.append("mostly in " + " and ".join(k for k, _ in out["keys"][:2]))
    if tempos:
        parts.append(f"around {out['tempo']['min']}–{out['tempo']['max']} BPM" if out["tempo"]["min"] != out["tempo"]["max"]
                     else f"at {out['tempo']['min']} BPM")
    if out["meters"]:
        parts.append("in " + out["meters"][0][0])
    if out["moods"]:
        mood = out["moods"][0][0]
        parts.append(("with an " if mood[0] in "aeiou" else "with a ") + mood + " feel")
    out["sentence"] = "Your music is " + ", ".join(parts) + "." if parts else ""
    return out
