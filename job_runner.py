#!/usr/bin/env python3
"""Run one web-app analysis job in its own process.

The web app starts this as a subprocess for every job, so a heavy analysis can't block the web
server, and a crash or out-of-memory kill only fails that one job instead of the whole site.

Usage (normally only called by app.py):  python job_runner.py <job folder>
"""

import json
import sys
import time
from pathlib import Path


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


def main(job_dir):
    import musicanalyze as ma  # imported here so the status helpers above stay lightweight for app.py

    d = Path(job_dir)
    try:
        files = sorted((d / "input").iterdir())
        llm = ma.make_llm()[0] if read_status(d).get("use_ai") else None
        ma.run_analysis(files, d / "output", llm, progress=lambda msg: write_status(d, message=msg),
                        melody_text=read_status(d).get("melody"))
        write_status(d, state="done", message="Done")
    except Exception as e:
        write_status(d, state="error", message=f"Analysis failed: {str(e)[:300]}")
        raise


if __name__ == "__main__":
    main(sys.argv[1])
