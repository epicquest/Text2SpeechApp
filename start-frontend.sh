#!/usr/bin/env bash
# start-frontend.sh — Serve the static frontend in the background.
# Usage: ./start-frontend.sh [--port 3000] [--host 127.0.0.1]

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via env vars or flags)
# ---------------------------------------------------------------------------
HOST="${FRONTEND_HOST:-127.0.0.1}"
PORT="${FRONTEND_PORT:-3000}"
FRONTEND_DIR="${FRONTEND_DIR:-frontend}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/frontend.log"
PID_FILE="${PID_FILE:-.frontend.pid}"

# Parse optional CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[start-frontend] Frontend is already running (PID $OLD_PID)."
        echo "                 Run ./kill-frontend.sh first, or send: kill $OLD_PID"
        exit 1
    else
        echo "[start-frontend] Stale PID file found (process $OLD_PID is gone) — removing."
        rm -f "$PID_FILE"
    fi
fi

if [[ ! -d "$FRONTEND_DIR" ]]; then
    echo "[start-frontend] Frontend directory not found at '$FRONTEND_DIR'."
    echo "                 Create it and place your HTML/CSS/JS files there."
    exit 1
fi

# ---------------------------------------------------------------------------
# Prepare log directory
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "[start-frontend] Serving '$FRONTEND_DIR' on http://${HOST}:${PORT}"
echo "[start-frontend] Log file: $LOG_FILE"

nohup python3 -m http.server "$PORT" \
    --bind "$HOST" \
    --directory "$FRONTEND_DIR" \
    >> "$LOG_FILE" 2>&1 &

FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$PID_FILE"

# Give the process a moment to confirm it's alive
sleep 1
if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "[start-frontend] ERROR: Frontend server exited immediately. Check $LOG_FILE for details."
    rm -f "$PID_FILE"
    tail -n 20 "$LOG_FILE"
    exit 1
fi

echo "[start-frontend] Frontend running — PID $FRONTEND_PID"
echo "[start-frontend] Tail logs:  tail -f $LOG_FILE"
echo "[start-frontend] Stop:       ./kill-frontend.sh"
