#!/usr/bin/env bash
# ==============================================================================
# Fake Job Posting Detection System - Unified Startup Script
# ==============================================================================
# This script starts all three services in parallel:
#   1. Python FastAPI ML Service (port 8000)
#   2. Express Backend API (port 5000)
#   3. React Vite Frontend (port 5173)
#
# Press Ctrl+C to terminate all services simultaneously.
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "  🚀 Starting Fake Job Posting Detection System"
echo "=========================================================="

# Check Python environment
if [ ! -d "ml/venv" ]; then
    echo "⚠️  Python virtual environment not found in ml/venv."
    echo "   Setting up ml/venv now..."
    python3 -m venv ml/venv
    source ml/venv/bin/activate
    pip install -r ml/requirements.txt
else
    source ml/venv/bin/activate
fi

# Cleanup handler for graceful shutdown
cleanup() {
    echo ""
    echo "🛑 Shutting down all services..."
    kill $(jobs -p) 2>/dev/null || true
    echo "Done. Goodbye!"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start Python ML FastAPI service (Port 8000)
echo "🐍 Starting Python FastAPI ML Service on http://localhost:8000 ..."
python -m uvicorn ml.service:app --port 8000 --host 0.0.0.0 &
ML_PID=$!

# Wait for ML service health check
echo "   Waiting for ML service to initialize..."
for i in {1..20}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "   ✅ ML Service is up and healthy!"
        break
    fi
    sleep 1
done

# 2. Start Express Backend
echo "🟢 Starting Express Backend on http://localhost:5001 ..."
(cd server && npm start) &
SERVER_PID=$!

# 3. Start React Frontend (Port 5173)
echo "⚛️  Starting React Vite Frontend on http://localhost:5173 ..."
(cd client && npm run dev) &
CLIENT_PID=$!

echo ""
echo "=========================================================="
echo "  🎉 All services running!"
echo "  - Client UI:    http://localhost:5173"
echo "  - Express API:  http://localhost:5001"
echo "  - ML Service:   http://localhost:8000"
echo "  - ML API Docs:  http://localhost:8000/docs"
echo "=========================================================="
echo "Press Ctrl+C to stop all services."

# Wait indefinitely
wait
