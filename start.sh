#!/bin/bash
# Start both Node.js server and Python backend together
# This allows deploying everything in a single Render service

set -e

echo "🚀 Starting StockVue application (Full Stack)..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Render Node.js environment needs Python installed."
    echo "💡 Tip: Use separate services or ensure Python is available in build environment"
    exit 1
fi

# Start Python backend in background
echo "📦 Starting Python Flask backend..."
cd python_backend

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv || python3 -m venv venv || python -m venv .venv
fi

# Activate virtual environment (try different paths)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found, using system Python"
fi

# Install dependencies (always check, as Render may clear cache)
echo "Installing Python dependencies (this may take 5-10 minutes on first run)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Start Flask backend in background
export FLASK_APP=app.py
export FLASK_ENV=production
PORT_PYTHON=${PORT_PYTHON:-5001}

# Check if gunicorn is installed
if ! command -v gunicorn &> /dev/null; then
    echo "Installing gunicorn..."
    pip install --quiet gunicorn
fi

echo "Starting Python backend on port $PORT_PYTHON..."
gunicorn app:app --bind 0.0.0.0:$PORT_PYTHON --workers 1 --timeout 120 --log-level warning --access-logfile - --error-logfile - &
PYTHON_PID=$!

echo "✅ Python backend started (PID: $PYTHON_PID)"

# Go back to root directory
cd ..

# Wait for Python backend to be ready
echo "Waiting for Python backend to start..."
sleep 3

# Check if Python backend is running
if ! kill -0 $PYTHON_PID 2>/dev/null; then
    echo "❌ Python backend failed to start"
    exit 1
fi

# Set Python API base for Node.js (use localhost since they're in same container)
export PYTHON_API_BASE=http://localhost:$PORT_PYTHON
echo "🔗 Python API Base: $PYTHON_API_BASE"

# Start Node.js server (foreground - this keeps the container alive)
echo "🌐 Starting Node.js server..."
npm start

# Cleanup: If Node.js exits, kill Python backend
trap "kill $PYTHON_PID 2>/dev/null || true" EXIT

