#!/bin/bash
# Restart both backend and frontend servers
# Backend runs on port 9000, Frontend runs on port 3000

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load environment variables from backend/.env if it exists
if [ -f "$SCRIPT_DIR/backend/.env" ]; then
    echo "📦 Loading environment from backend/.env..."
    set -a
    source "$SCRIPT_DIR/backend/.env"
    set +a
fi

echo "🔄 Stopping any existing servers..."

# Kill existing processes on ports 9000 and 3000
lsof -ti:9000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Give processes time to fully terminate
sleep 1

# Check for OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable is not set"
    echo "   The backend will fail to start without it."
    echo "   Set it with: export OPENAI_API_KEY='your-api-key'"
    echo ""
fi

# Cleanup function to kill child processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    lsof -ti:9000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "🚀 Starting backend server on http://localhost:9000..."
cd "$SCRIPT_DIR/backend"
source .venv/bin/activate && python -m agent_k.ui.ag_ui &
BACKEND_PID=$!

echo "🚀 Starting frontend server on http://localhost:3000..."
cd "$SCRIPT_DIR/frontend"
pnpm dev &
FRONTEND_PID=$!

echo ""
echo "✅ Both servers are starting..."
echo "   Backend:  http://localhost:9000"
echo "   Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID

