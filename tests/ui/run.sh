#!/bin/sh
# Starts the API and a tiny feed server, drives the real page with jsdom, then stops both.
# Needs: node, and `npm install jsdom` in this folder. Usage: sh tests/ui/run.sh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; WORK="$(mktemp -d)"; export REPO
cd "$REPO" && python3 - "$WORK" <<'PY'
import sys
from datetime import date, timedelta
from app import forecast
S = forecast.simulate(n=40, days=240, seed=2); D0 = date(2025, 1, 1)
rows = ["date," + ",".join(f"trend {i}" for i in range(40))] + [(D0 + timedelta(days=d)).isoformat() + "," + ",".join(f"{S['main'][i, d]:.2f}" for i in range(40)) for d in range(240)]
open(sys.argv[1] + "/feed.csv", "w").write("\n".join(rows))
PY
(cd "$WORK" && python3 -m http.server 8801 --bind 127.0.0.1 >/dev/null 2>&1 &)
(DE_DB="$WORK/ui.db" DE_ALLOW_HTTP_FEEDS=1 python3 -m uvicorn app.main:app --port 8765 --host 127.0.0.1 >"$WORK/server.log" 2>&1 &)
for i in 1 2 3 4 5 6 7 8 9 10; do curl -s -m 2 http://127.0.0.1:8765/api/v1/health >/dev/null && break; sleep 1; done
(cd "$REPO/tests/ui" && timeout 240 node smoke.js); CODE=$?
pkill -f "uvicorn app.main"; pkill -f "http.server 8801"; exit $CODE
