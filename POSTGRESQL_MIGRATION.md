# PostgreSQL Migration Guide

## ✅ Migration Complete!

The Python backend now supports **PostgreSQL** for production and **SQLite** for local development.

## How It Works

- **Production (Railway)**: Uses PostgreSQL when `DATABASE_URL` is set
- **Local Development**: Falls back to SQLite (`predictions.db`) if `DATABASE_URL` is not set

## What Data Is Stored

The backend stores:
1. **Predictions**: Predicted prices, confidence scores, model breakdowns
2. **Evaluations**: Comparison of predictions vs actual prices, accuracy metrics
3. **Statistics**: Performance metrics (RMSE, MAE, direction accuracy)

## Railway Setup

### Step 1: Add PostgreSQL Service

1. Go to Railway Dashboard → Your Project
2. Click **"+ New"** → **"Database"** → **"Add PostgreSQL"**
3. Railway will automatically create a PostgreSQL database
4. Railway will automatically set `DATABASE_URL` environment variable

### Step 2: Verify DATABASE_URL

1. Go to Railway Dashboard → Your Project → Variables
2. Verify `DATABASE_URL` is set (Railway sets this automatically)
3. It should look like: `postgresql://user:password@host:port/database`

### Step 3: Deploy

The backend will automatically:
- Detect `DATABASE_URL` environment variable
- Use PostgreSQL instead of SQLite
- Create tables automatically on first run
- Store all predictions and evaluations in PostgreSQL

## Local Development

For local development (without PostgreSQL):

1. **Don't set `DATABASE_URL`** in your local environment
2. The backend will automatically use SQLite (`predictions.db`)
3. All data will be stored locally in `python_backend/predictions.db`

## Database Schema

### Predictions Table
- `id`: Primary key
- `symbol`: Stock symbol (e.g., "AAPL")
- `timestamp`: When prediction was made
- `interval`: Time interval (e.g., "1d", "1h")
- `horizon`: Prediction horizon (number of periods ahead)
- `current_price`: Price at time of prediction
- `predicted_price`: Predicted future price
- `confidence`: Model confidence score (0-1)
- `model_breakdown`: JSON with individual model predictions
- `actual_price`: Actual price (filled after evaluation)
- `evaluated`: Whether prediction has been evaluated
- `created_at`: Record creation timestamp

### Evaluations Table
- `id`: Primary key
- `prediction_id`: Foreign key to predictions table
- `actual_price`: Actual market price
- `error`: Absolute error
- `error_percent`: Percentage error
- `direction_actual`: Actual direction ("up", "down", "neutral")
- `direction_predicted`: Predicted direction
- `correct`: Whether direction prediction was correct
- `evaluated_at`: When evaluation was performed

## Benefits of PostgreSQL

✅ **Persistent**: Data survives deployments and restarts  
✅ **Scalable**: Handles concurrent requests better  
✅ **Reliable**: Production-grade database  
✅ **Backed up**: Railway provides automatic backups  
✅ **Queryable**: Can run complex queries and analytics  

## Migration Notes

- **No data loss**: Existing SQLite data stays in `predictions.db` (local)
- **Fresh start**: PostgreSQL starts with empty tables (new predictions)
- **Automatic**: Tables are created automatically on first run
- **Backward compatible**: Local development still uses SQLite

## Troubleshooting

### Backend can't connect to PostgreSQL

**Error**: `psycopg2.OperationalError` or connection refused

**Fix**:
1. Verify `DATABASE_URL` is set in Railway Variables
2. Check Railway PostgreSQL service is running (green status)
3. Verify connection string format is correct

### Tables not created

**Error**: Table doesn't exist

**Fix**:
- Tables are created automatically on first API call
- Check Railway logs for initialization messages
- Manually trigger: Make a prediction from the frontend

### psycopg2 not installed

**Error**: `ModuleNotFoundError: No module named 'psycopg2'`

**Fix**:
- Already added to `requirements.txt`
- Railway will install it automatically on deploy
- If local: `pip install psycopg2-binary`

## Summary

✅ **Production (Railway)**: Uses PostgreSQL (set `DATABASE_URL` automatically)  
✅ **Local Dev**: Uses SQLite (no `DATABASE_URL` needed)  
✅ **Automatic**: Tables created on first run  
✅ **No changes needed**: Just add PostgreSQL service in Railway  

The backend will automatically detect which database to use based on the `DATABASE_URL` environment variable!

