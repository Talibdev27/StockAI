# Railway Backend Environment Variables

## Required Environment Variables

**None!** The backend works without any environment variables. Railway automatically sets `PORT`.

## Optional Environment Variables (LSTM Model Tuning)

These are optional and have sensible defaults. Only set them if you want to customize model performance:

### LSTM Hyperparameters (Optional)

| Variable | Default | Description | Recommended |
|----------|---------|-------------|-------------|
| `LSTM_EPOCHS` | `30` | Number of training epochs | `30-80` (higher = more accurate but slower) |
| `LSTM_PATIENCE` | `5` | Early stopping patience | `5-15` |
| `LSTM_UNITS1` | `50` | Units in first LSTM layer | `50-128` |
| `LSTM_UNITS2` | `50` | Units in second LSTM layer | `50-64` |
| `LSTM_DROPOUT` | `0.2` | Dropout rate (0.0-1.0) | `0.2-0.3` |
| `LSTM_BATCH_SIZE` | `32` | Batch size for training | `32-64` |
| `LSTM_BIDIRECTIONAL` | `false` | Use bidirectional LSTM | `true` (better accuracy, slower) |

### Example Configuration (Balanced)

For Railway, you can set these in the Railway dashboard:

```
LSTM_EPOCHS=30
LSTM_PATIENCE=5
LSTM_UNITS1=50
LSTM_UNITS2=50
LSTM_DROPOUT=0.2
LSTM_BATCH_SIZE=32
LSTM_BIDIRECTIONAL=false
```

### Example Configuration (Optimized for Accuracy)

```
LSTM_EPOCHS=80
LSTM_PATIENCE=15
LSTM_UNITS1=128
LSTM_UNITS2=64
LSTM_DROPOUT=0.3
LSTM_BATCH_SIZE=64
LSTM_BIDIRECTIONAL=true
```

## Required Environment Variables

**`DATABASE_URL`** - PostgreSQL connection string (set automatically by Railway when you add PostgreSQL service)

## Auto-Set by Railway

- `PORT` - Automatically set by Railway (don't set manually)
- `DATABASE_URL` - Automatically set when you add PostgreSQL service

## How to Set in Railway

1. Go to Railway Dashboard → Your Project → Variables
2. Click "New Variable"
3. Add each variable name and value
4. Railway will automatically restart the service

## Database Setup

### PostgreSQL (Production - Recommended)

1. **Add PostgreSQL Service**:
   - Railway Dashboard → Your Project → "+ New" → "Database" → "Add PostgreSQL"
   - Railway automatically sets `DATABASE_URL`

2. **Backend will automatically**:
   - Detect `DATABASE_URL` environment variable
   - Use PostgreSQL instead of SQLite
   - Create tables on first run

### SQLite (Local Development)

- If `DATABASE_URL` is not set, backend uses SQLite (`predictions.db`)
- Works automatically for local development

## Notes

- **Database**: PostgreSQL in production (Railway), SQLite locally
- **All LSTM variables are optional**: Backend works with defaults
- **Changes require restart**: Railway auto-restarts when you add/modify variables
- **Performance vs Speed**: Higher values = better accuracy but slower predictions

## Recommended for Production

Start with **no environment variables** (use defaults). Only add LSTM tuning variables if you need:
- Faster predictions: Lower epochs (20-30)
- Better accuracy: Higher epochs (80-100), bidirectional=true
- Memory constraints: Lower batch size (16-32)

