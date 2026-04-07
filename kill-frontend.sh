#!/usr/bin/env bash
# kill-frontend.sh — Gracefully stop the background frontend server.
# Reads the PID written by start-frontend.sh.

set -euo pipefail

PID_FILE="${PID_FILE:-.frontend.pid}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$PID_FILE" ]]; then
    echo "[kill-frontend] No PID file found ($PID_FILE). Is the frontend running?"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo "[kill-frontend] Process $PID is not running (stale PID file). Removing."
    rm -f "$PID_FILE"
    exit 0
fi

echo "[kill-frontend] Sending SIGTERM to PID $PID …"
kill -TERM "$PID"

# Wait up to 10 seconds for graceful shutdown
TIMEOUT=10
for i in $(seq 1 "$TIMEOUT"); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[kill-frontend] Frontend stopped (took ${i}s)."
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force-kill if still alive after timeout
echo "[kill-frontend] Frontend did not stop after ${TIMEOUT}s — sending SIGKILL."
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "[kill-frontend] Frontend killed."
