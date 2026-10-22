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
        # Added later: has this user closed the "what the app can't do" notice? (kept per account, so it doesn't
        # come back on a new device)
        cols = [c[1] for c in db.execute("PRAGMA table_info(users)")]
        if "seen_intro" not in cols:
            db.execute("ALTER TABLE users ADD COLUMN seen_intro INTEGER NOT NULL DEFAULT 0")
        # The musician's memory (memory.py): off until they switch it on; their goals in their own words
        if "memory_on" not in cols:
            db.execute("ALTER TABLE users ADD COLUMN memory_on INTEGER NOT NULL DEFAULT 0")
        if "goals" not in cols:
            db.execute("ALTER TABLE users ADD COLUMN goals TEXT NOT NULL DEFAULT ''")
        # Keeping the work organised: a project, where it's at, labels, private notes, a star
        saved_cols = [c[1] for c in db.execute("PRAGMA table_info(saved)")]
        for col, kind in (("project", "TEXT NOT NULL DEFAULT ''"), ("status", "TEXT NOT NULL DEFAULT ''"),
                          ("labels", "TEXT NOT NULL DEFAULT ''"), ("notes", "TEXT NOT NULL DEFAULT ''"),
                          ("starred", "INTEGER NOT NULL DEFAULT 0")):
            if col not in saved_cols:
                db.execute(f"ALTER TABLE saved ADD COLUMN {col} {kind}")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                job_id TEXT NOT NULL, track TEXT NOT NULL, idea_key TEXT NOT NULL,
                area TEXT NOT NULL, text TEXT NOT NULL, vote INTEGER NOT NULL, created REAL NOT NULL,
                UNIQUE (user_id, job_id, track, idea_key));
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                job_id TEXT NOT NULL, track TEXT NOT NULL, field TEXT NOT NULL, value TEXT NOT NULL,
                created REAL NOT NULL);
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
        return None, "Pick a username of 3–30 letters, numbers, dots, dashes or underscores, with no spaces."
    if len(password or "") < MIN_PASSWORD:
        return None, f"Your password needs at least {MIN_PASSWORD} characters."
    try:
        with _connect() as db:
            cur = db.execute("INSERT INTO users (username, pw_hash, created) VALUES (?, ?, ?)",
                             (username, generate_password_hash(password), time.time()))
            return cur.lastrowid, None
    except sqlite3.IntegrityError:
        return None, "Someone's already using that username. Try another one."


def check_login(username, password):
    """The user's id if the username and password match, else None."""
    with _connect() as db:
        row = db.execute("SELECT id, pw_hash FROM users WHERE username = ?", ((username or "").strip(),)).fetchone()
    if row and check_password_hash(row["pw_hash"], password or ""):
        return row["id"]
    return None


def get_user(user_id):
    with _connect() as db:
        row = db.execute("SELECT id, username, created, seen_intro, memory_on, goals FROM users WHERE id = ?",
                         (user_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------- the musician's memory (see memory.py)

CORRECTABLE = {"tempo": "Tempo (BPM)", "meter": "Time signature", "key": "Key", "drums": "Drums", "bass": "Bass"}


def set_memory(user_id, on, goals=None):
    with _connect() as db:
        db.execute("UPDATE users SET memory_on = ? WHERE id = ?", (1 if on else 0, user_id))
        if goals is not None:
            db.execute("UPDATE users SET goals = ? WHERE id = ?", (goals.strip()[:500], user_id))


def add_feedback(user_id, job_id, track, area, text, vote):
    """One vote per idea and track: voting again replaces it; vote 0 takes it back."""
    import memory
    k = memory.idea_key(area, text)
    with _connect() as db:
        db.execute("DELETE FROM feedback WHERE user_id = ? AND job_id = ? AND track = ? AND idea_key = ?",
                   (user_id, job_id, track[:200], k))
        if vote:
            db.execute("INSERT INTO feedback (user_id, job_id, track, idea_key, area, text, vote, created) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (user_id, job_id, track[:200], k, area[:60], text[:600],
                                                           1 if vote > 0 else -1, time.time()))


def add_correction(user_id, job_id, track, field, value):
    if field not in CORRECTABLE or not str(value).strip():
        return "Pick what to correct and give the right value."
    with _connect() as db:
        db.execute("INSERT INTO corrections (user_id, job_id, track, field, value, created) VALUES (?, ?, ?, ?, ?, ?)",
                   (user_id, job_id, track[:200], field, str(value).strip()[:60], time.time()))
    return None


def list_feedback(user_id):
    with _connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM feedback WHERE user_id = ? ORDER BY created DESC", (user_id,))]


def list_corrections(user_id):
    with _connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM corrections WHERE user_id = ? ORDER BY created DESC",
                                            (user_id,))]


def delete_memory_item(user_id, table, item_id):
    if table not in ("feedback", "corrections"):
        return
    with _connect() as db:
        db.execute(f"DELETE FROM {table} WHERE id = ? AND user_id = ?", (item_id, user_id))


def forget_everything(user_id):
    """Every vote, correction and goal gone, and the memory switched off. Saved analyses stay (that's History)."""
    with _connect() as db:
        db.execute("DELETE FROM feedback WHERE user_id = ?", (user_id,))
        db.execute("DELETE FROM corrections WHERE user_id = ?", (user_id,))
        db.execute("UPDATE users SET memory_on = 0, goals = '' WHERE id = ?", (user_id,))


def mark_intro_seen(user_id):
    with _connect() as db:
        db.execute("UPDATE users SET seen_intro = 1 WHERE id = ?", (user_id,))


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
        rows = db.execute("SELECT job_id, title, created, summary, project, status, labels, notes, starred FROM saved "
                          "WHERE user_id = ? ORDER BY created DESC", (user_id,)).fetchall()
    return [{**dict(r), "summary": json.loads(r["summary"]) if r["summary"] else None,
             "labels": [x for x in (r["labels"] or "").split(",") if x], "starred": bool(r["starred"])} for r in rows]


# ---------------------------------------------------------------- organising

STATUSES = ["idea", "writing", "recording", "mixing", "finished", "released"]
MAX_LABELS, LABEL_LEN, PROJECT_LEN, NOTES_LEN = 8, 24, 60, 2000


def clean_labels(text):
    """'Chill, demo,  EP 2 , chill' -> ['chill', 'demo', 'ep 2']: lower case, no duplicates, at most 8."""
    out = []
    for x in (text or "").replace(";", ",").split(","):
        x = " ".join(x.strip().lower().split())[:LABEL_LEN]
        if x and x not in out:
            out.append(x)
    return out[:MAX_LABELS]


def organise(user_id, job_id, project=None, status=None, labels=None, notes=None, starred=None):
    """Update how one saved analysis is filed. Only its owner can; None leaves a field as it is."""
    sets, args = [], []
    if project is not None:
        sets.append("project = ?"), args.append(" ".join(project.split())[:PROJECT_LEN])
    if status is not None:
        sets.append("status = ?"), args.append(status if status in STATUSES else "")
    if labels is not None:
        sets.append("labels = ?"), args.append(",".join(clean_labels(labels)))
    if notes is not None:
        sets.append("notes = ?"), args.append(notes.strip()[:NOTES_LEN])
    if starred is not None:
        sets.append("starred = ?"), args.append(1 if starred else 0)
    if not sets:
        return False
    with _connect() as db:
        return db.execute(f"UPDATE saved SET {', '.join(sets)} WHERE job_id = ? AND user_id = ?",
                          (*args, job_id, user_id)).rowcount > 0


def projects(user_id):
    """[(project, analyses, last saved)] for this user, most recently used first."""
    with _connect() as db:
        rows = db.execute("SELECT project, COUNT(*) AS n, MAX(created) AS last FROM saved WHERE user_id = ? AND "
                          "project != '' GROUP BY project ORDER BY last DESC", (user_id,)).fetchall()
    return [(r["project"], r["n"], r["last"]) for r in rows]


def rename_project(user_id, old, new):
    new = " ".join((new or "").split())[:PROJECT_LEN]
    with _connect() as db:
        db.execute("UPDATE saved SET project = ? WHERE user_id = ? AND project = ?", (new, user_id, old))


def all_labels(user_id):
    labels = Counter()
    for it in history(user_id):
        labels.update(it["labels"])
    return labels.most_common()


# ---------------------------------------------------------------- your data: take it, or delete it all

def export(user_id):
    """Everything we hold about a user, as plain data (the analyses' files are added by the app)."""
    u = get_user(user_id)
    return {"username": u["username"], "created": u["created"], "memory_on": bool(u.get("memory_on")),
            "goals": u.get("goals", ""), "saved": history(user_id), "feedback": list_feedback(user_id),
            "corrections": list_corrections(user_id)}


def delete_account(user_id):
    """Remove the user and every row about them. Returns the job ids whose files the app must delete."""
    with _connect() as db:
        jobs = [r["job_id"] for r in db.execute("SELECT job_id FROM saved WHERE user_id = ?", (user_id,))]
        for table in ("feedback", "corrections", "saved"):
            db.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
        db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return jobs


def store_summary(job_id, summary):
    with _connect() as db:
        db.execute("UPDATE saved SET summary = ? WHERE job_id = ?", (json.dumps(summary), job_id))


# ---------------------------------------------------------------- summaries

def job_summary(results):
    """A compact per-track summary of one analysis (from music_results.json), kept in the database so the
    history and profile pages don't have to re-read every result file."""
    tracks = []
    for r in results:
        if r.get("role") == "reference":  # someone else's song: not part of the musician's own profile
            continue
        if "error" in r:
            tracks.append({"file": r.get("file", "?"), "error": True})
            continue
        feel = r.get("feel") or {}
        tracks.append({
            "file": r["file"], "duration_sec": r.get("duration_sec"), "duration": r.get("duration"),
            "tempo_bpm": r.get("tempo_bpm"), "meter": (r.get("meter") or {}).get("label"),
            "key": r.get("key"), "camelot": r.get("camelot"), "mode": r.get("mode"),
            "energy": feel.get("energy"), "mood": feel.get("mood"), "valence": feel.get("valence"),
            "arousal": feel.get("arousal"),
            "energy_curve": [round(x, 2) for x in (r.get("sound") or {}).get("energy_curve") or []],
            "form": " ".join(s["label"] for s in r.get("sections", [])),
            "chords": {c["chord"]: c["seconds"] for c in r.get("chords_used", [])},
            "drums": bool((r.get("drums") or {}).get("present")), "bass": bool((r.get("bass") or {}).get("present")),
            "genre": (r.get("context") or {}).get("genre"),
            # for the musician's memory
            "tempo_approximate": bool(r.get("tempo_approximate")),
            "lufs": ((r.get("sound") or {}).get("streaming") or {}).get("lufs"),
            "form_name": (r.get("blues_form") or r.get("song_form") or {}).get("name"),
        })
    return {"tracks": tracks}


def tags_summary(data):
    """A "find music like this" job (from tags.json): the list's top tags, for the history and profile pages."""
    t = data.get("tags") or {}
    top = lambda cat, n: [(x["tag"], x["score"]) for x in t.get(cat, [])[:n]]
    return {"kind": "tags", "tracks": [], "songs": len(data.get("songs") or []), "list_name": data.get("list_name"),
            "styles": top("style", 6), "moods": top("mood", 4) + top("sound", 3)}


def profile_summary(items):
    """Everything a user has saved, summed up: what keys, tempos, modes, chords and moods they gravitate to,
    and (from their tag lists) the styles they listen to."""
    tracks = [t for it in items if it.get("summary") for t in it["summary"]["tracks"] if not t.get("error")]
    out = {"analyses": len(items), "tracks": len(tracks)}
    listening = Counter()
    for it in items:
        sm = it.get("summary") or {}
        if sm.get("kind") == "tags":
            for tag, score in sm.get("styles", []) + sm.get("moods", []):
                listening[tag] += score
    out["listening"] = [(tag, round(v, 2)) for tag, v in listening.most_common(12)]
    if not tracks:
        return out
    out["minutes"] = round(sum(t.get("duration_sec") or 0 for t in tracks) / 60, 1)
    out["keys"] = Counter(t["key"] for t in tracks if t.get("key")).most_common(5)
    out["modes"] = Counter(t["mode"] for t in tracks if t.get("mode")).most_common(4)
    out["meters"] = Counter(t["meter"] for t in tracks if t.get("meter")).most_common(3)
    out["moods"] = Counter(t["mood"] for t in tracks if t.get("mood")).most_common(4)
    out["genres"] = Counter(t["genre"] for t in tracks if t.get("genre")).most_common(5)
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
    # for the pictures: keys on the Camelot wheel, every tempo, every track on the energy/mood map
    out["camelot_counts"] = dict(Counter(t["camelot"] for t in tracks if t.get("camelot")))
    out["tempo_list"] = [t["tempo_bpm"] for t in tracks if t.get("tempo_bpm") and not t.get("tempo_approximate")]
    out["mood_points"] = [(None, t["valence"], t["arousal"], f"{t['file']}: {t.get('mood')}") for t in tracks
                          if t.get("valence") is not None and t.get("arousal") is not None]
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
