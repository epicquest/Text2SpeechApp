#!/usr/bin/env bash
# start-server.sh — Activate venv, start uvicorn in the background, log to file.
# Usage: ./start-server.sh [--port 8000] [--host 0.0.0.0]

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via env vars or flags)
# ---------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-.venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/server.log"
PID_FILE="${PID_FILE:-.server.pid}"

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
        echo "[start-server] Server is already running (PID $OLD_PID)."
        echo "               Run ./kill-server.sh first, or send: kill $OLD_PID"
        exit 1
    else
        echo "[start-server] Stale PID file found (process $OLD_PID is gone) — removing."
        rm -f "$PID_FILE"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[start-server] Virtual environment not found at '$VENV_DIR'."
    echo "               Create it with:  python3.12 -m venv $VENV_DIR"
    echo "               Then install:    $VENV_DIR/bin/pip install -r requirements.txt"
    exit 1
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
    if [[ -f ".env" ]]; then
        echo "[start-server] Sourcing .env for DATABASE_URL"
        set -o allexport
        # shellcheck source=/dev/null
        source .env
        set +o allexport
    else
        echo "[start-server] WARNING: DATABASE_URL is not set and no .env file found."
        echo "               The server will fail unless it is configured."
    fi
fi

# ---------------------------------------------------------------------------
# Prepare log directory
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
UVICORN="$SCRIPT_DIR/$VENV_DIR/bin/uvicorn"

if [[ ! -x "$UVICORN" ]]; then
    echo "[start-server] uvicorn not found at $UVICORN"
    echo "               Install it with: $VENV_DIR/bin/pip install -r requirements.txt"
    exit 1
fi

echo "[start-server] Starting TTS server on http://${HOST}:${PORT}"
echo "[start-server] Log file: $LOG_FILE"

nohup "$UVICORN" app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1 \
    --log-level info \
    >> "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Give the process a moment to confirm it's alive
sleep 1
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[start-server] ERROR: Server exited immediately. Check $LOG_FILE for details."
    rm -f "$PID_FILE"
    tail -n 30 "$LOG_FILE"
    exit 1
fi

echo "[start-server] Server running — PID $SERVER_PID"
echo "[start-server] Tail logs:  tail -f $LOG_FILE"
echo "[start-server] Stop:       ./kill-server.sh"
