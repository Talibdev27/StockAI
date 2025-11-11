#!/bin/bash

# python_backend/start_improved.sh
# Optimized configuration for balanced accuracy and training time

echo "🚀 Starting backend with optimized model settings..."
echo ""

# LSTM Configuration - Balanced for accuracy and speed
export LSTM_EPOCHS=80               # Balanced (not 100) - faster training
export LSTM_PATIENCE=15             # Prevent premature stopping
export LSTM_UNITS1=128              # Increased capacity
export LSTM_UNITS2=64               # Deeper architecture
export LSTM_BIDIRECTIONAL=true      # Better pattern recognition
export LSTM_DROPOUT=0.3             # Prevent overfitting
export LSTM_BATCH_SIZE=64           # Stable gradients

echo "📊 LSTM Configuration:"
echo "   Epochs: $LSTM_EPOCHS (balanced for speed)"
echo "   Architecture: $LSTM_UNITS1 → $LSTM_UNITS2 (bidirectional)"
echo "   Dropout: $LSTM_DROPOUT, Batch: $LSTM_BATCH_SIZE"
echo ""

# Model Performance Tracking (for Phase 2)
export TRACK_MODEL_PERFORMANCE=true
export PERFORMANCE_LOG_FILE="model_performance.json"

echo "📈 Performance Tracking: Enabled"
echo "   Log file: $PERFORMANCE_LOG_FILE"
echo ""

# Navigate to backend directory
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "✅ Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "📦 Installing dependencies..."
    pip install -q -r requirements.txt
fi

echo ""
echo "🎯 Starting Flask API with optimized settings..."
echo "   Training will take 8-12 minutes (worth the wait!)"
echo ""

# Start the application
export FLASK_APP=app.py
export FLASK_ENV=development
python app.py

