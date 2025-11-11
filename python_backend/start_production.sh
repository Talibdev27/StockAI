#!/bin/bash
# Production start script for Python backend

cd python_backend

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# Use gunicorn for production
if command -v gunicorn &> /dev/null; then
    gunicorn app:app --bind 0.0.0.0:${PORT:-5001} --workers 2 --timeout 120
else
    echo "Gunicorn not found, installing..."
    pip install gunicorn
    gunicorn app:app --bind 0.0.0.0:${PORT:-5001} --workers 2 --timeout 120
fi

