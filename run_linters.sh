#!/usr/bin/env bash
set -euo pipefail

# Run all project linters and type checks using the local .venv environment.
# This script assumes the venv is located at ./ .venv and already has the
# required dependencies installed.

VENV_BIN="$(pwd)/.venv/bin"
if [[ ! -d "$VENV_BIN" ]]; then
  echo "ERROR: .venv not found. Please create a virtualenv at .venv." >&2
  exit 1
fi

echo "Running lint checks (black/isort/flake8/ruff/pylint/mypy) in .venv..."

"$VENV_BIN/black" --check .
"$VENV_BIN/isort" --check-only .
"$VENV_BIN/flake8" .
"$VENV_BIN/ruff" check .
"$VENV_BIN/pylint" app
"$VENV_BIN/python" -m mypy app

echo "All linters passed ✅"
