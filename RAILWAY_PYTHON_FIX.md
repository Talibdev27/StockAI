# Railway Python Backend Fix

## Problem
Railway is detecting Node.js project and running `npm start` instead of Python Flask backend.

## Solution

### Option 1: Configure Railway Service Settings (Recommended)

1. Go to Railway Dashboard → Your Project → Service Settings
2. Under "Build & Deploy" → "Root Directory": Set to `python_backend`
3. Under "Start Command": Set to `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
4. Under "Build Command": Set to `pip install -r requirements.txt`

### Option 2: Use nixpacks.toml (Alternative)

The `nixpacks.toml` file has been created to force Python detection.

### Option 3: Separate Services

Create two separate Railway services:
- **Frontend Service**: Deploy Node.js/Vite frontend (or use Vercel)
- **Backend Service**: Deploy Python Flask backend

## Current Configuration Files

- `Procfile`: Contains `web: cd python_backend && gunicorn app:app ...`
- `railway.json`: Contains build and deploy commands
- `nixpacks.toml`: Forces Python detection

## Verification

After deploying, check Railway logs:
- Should see: `Starting gunicorn` or `[INFO] Starting gunicorn`
- Should NOT see: `npm start` or Node.js messages
- Should see Flask app starting on the PORT

## If Still Not Working

1. **Check Railway Service Settings**:
   - Make sure "Root Directory" is set correctly
   - Make sure "Start Command" matches Procfile
   - Make sure Python runtime is selected

2. **Check Build Logs**:
   - Should see Python packages installing
   - Should see `pip install -r requirements.txt` running

3. **Check Deploy Logs**:
   - Should see gunicorn starting
   - Should see Flask app initializing

