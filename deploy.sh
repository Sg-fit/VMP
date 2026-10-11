#!/usr/bin/env bash
# Update the server in one go: pull the latest code, rebuild, wait until the app is healthy, then run
# the self-test inside the container: music with known answers, a real request to your AI model, and a
# real upload through the running website (so a broken analysis worker shows up here, not on your upload).
#
#   cd ~/VMP && bash deploy.sh
#
# Ends with "RESULT: PASS" or "RESULT: FAIL" and the reasons. Your other sites aren't touched.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Pulling the latest code"
git pull --ff-only

echo "==> Rebuilding and restarting (the first build after a change takes a few minutes)"
docker compose up -d --build

PORT=$(grep -E '^HOST_PORT=' .env 2>/dev/null | cut -d= -f2 | tr -d '"'"'" || true)
PORT=${PORT:-8090}
echo "==> Waiting for the app on 127.0.0.1:$PORT"
for _ in $(seq 1 90); do
  curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 2
done
if ! curl -fs "http://127.0.0.1:$PORT/health"; then
  echo
  echo "The app did not come up. Last log lines:"
  docker compose logs --tail 60
  exit 1
fi
echo

echo "==> Self-test (about 2-4 minutes)"
rc=0
docker compose exec -T music-analyzer python selftest.py --live http://127.0.0.1:8000 || rc=$?
if [ "$rc" -ne 0 ]; then
  echo
  case "$rc" in
    137) echo "The self-test was KILLED (exit 137): the container ran out of memory (MEM_LIMIT in .env)."
         echo "Raise MEM_LIMIT (e.g. MEM_LIMIT=2500m), then run: bash deploy.sh" ;;
    139) echo "The self-test CRASHED inside a native library (exit 139, segmentation fault)." ;;
    1)   echo "Some self-test checks failed (see FAIL lines above)." ;;
    *)   echo "The self-test stopped with exit code $rc." ;;
  esac
  echo "Memory right now:"; docker stats --no-stream --format '  {{.Name}}: {{.MemUsage}}' 2>/dev/null || true
  echo "Last app log lines:"; docker compose logs --tail 15 music-analyzer
  exit "$rc"
fi
