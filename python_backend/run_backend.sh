#!/bin/bash

# Navigate to backend directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Install/update dependencies if needed
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Set Flask environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Run the Flask server
echo "Starting Flask backend on http://localhost:5001"
echo "Press Ctrl+C to stop"
python app.py

