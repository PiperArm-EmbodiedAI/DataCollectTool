#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$TOOL_DIR/.venv/bin/activate" ]]; then
  source "$TOOL_DIR/.venv/bin/activate"
  exec tool-piper --help
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$TOOL_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m tool_piper.cli --help
