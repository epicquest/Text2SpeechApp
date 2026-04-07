#!/usr/bin/env bash
# kill-server.sh — Gracefully stop the background uvicorn server.
# Reads the PID written by start-server.sh.

set -euo pipefail

PID_FILE="${PID_FILE:-.server.pid}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$PID_FILE" ]]; then
    echo "[kill-server] No PID file found ($PID_FILE). Is the server running?"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo "[kill-server] Process $PID is not running (stale PID file). Removing."
    rm -f "$PID_FILE"
    exit 0
fi

echo "[kill-server] Sending SIGTERM to PID $PID …"
kill -TERM "$PID"

# Wait up to 10 seconds for graceful shutdown
TIMEOUT=10
for i in $(seq 1 "$TIMEOUT"); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[kill-server] Server stopped (took ${i}s)."
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force-kill if still alive after timeout
echo "[kill-server] Server did not stop after ${TIMEOUT}s — sending SIGKILL."
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "[kill-server] Server killed."
