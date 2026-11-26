#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/backend/.env" ]; then
    set -a
    source "$SCRIPT_DIR/backend/.env"
    set +a
fi
cleanup() {
    kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
(cd "$SCRIPT_DIR/backend" && exec .venv/bin/python -m uvicorn agent_k.api:app --host 0.0.0.0 --port 9000) &
BACKEND_PID=$!
(cd "$SCRIPT_DIR/frontend" && exec pnpm dev) &
FRONTEND_PID=$!
wait "$BACKEND_PID" "$FRONTEND_PID"
