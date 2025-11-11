# LSTM Model Tuning with Environment Variables

This document explains how to tune the LSTM model hyperparameters using environment variables.

## Available Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LSTM_EPOCHS` | 50 | Number of training epochs |
| `LSTM_PATIENCE` | 5 | Early stopping patience (epochs) |
| `LSTM_DROPOUT` | 0.2 | Dropout rate between layers (0.0-1.0) |
| `LSTM_BATCH_SIZE` | 32 | Batch size for training |
| `LSTM_UNITS1` | 50 | Units in first LSTM layer |
| `LSTM_UNITS2` | 50 | Units in second LSTM layer |
| `LSTM_BIDIRECTIONAL` | false | Use Bidirectional LSTM layers (true/false) |

## Usage Examples

### One-time configuration (current shell session)
```bash
export LSTM_EPOCHS=80
export LSTM_PATIENCE=8
export LSTM_DROPOUT=0.3
export LSTM_BATCH_SIZE=64
export LSTM_UNITS1=64
export LSTM_UNITS2=32
export LSTM_BIDIRECTIONAL=true
python python_backend/app.py
```

### Per-run configuration
```bash
LSTM_EPOCHS=80 LSTM_BIDIRECTIONAL=true python python_backend/app.py
```

### Balanced configuration (recommended for production)
```bash
export LSTM_EPOCHS=30
export LSTM_PATIENCE=5
export LSTM_DROPOUT=0.2
export LSTM_BATCH_SIZE=32
export LSTM_UNITS1=50
export LSTM_UNITS2=50
export LSTM_BIDIRECTIONAL=false
```

## Performance vs Accuracy Trade-offs

- **Higher EPOCHS**: More training time, potentially better accuracy
- **Higher UNITS**: More parameters, slower training, better pattern capture
- **BIDIRECTIONAL=true**: Better pattern recognition, 2x slower training
- **Higher BATCH_SIZE**: Faster training, more memory usage
- **Higher DROPOUT**: Better generalization, potentially lower accuracy

## Model Persistence

Trained models are automatically saved to `python_backend/models_store/` and reused when available, so you only need to tune once per symbol/interval combination.

## Notes

- Environment variables override constructor defaults
- Invalid values fall back to defaults
- Changes require restarting the Flask server
- The ensemble uses balanced defaults (30 epochs) for speed
