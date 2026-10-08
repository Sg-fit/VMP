#!/usr/bin/env bash
# Update the server in one go: pull the latest code, rebuild, wait until the app is healthy, then run
# the self-test inside the container (music with known answers + a real request to your AI model).
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
docker compose exec -T music-analyzer python selftest.py
