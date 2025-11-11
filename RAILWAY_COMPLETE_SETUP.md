# Railway Deployment - Complete Setup Guide

## The Problem
Railway was detecting Node.js instead of Python, causing build failures.

## Solution: Configure Railway Dashboard Settings

### CRITICAL: Manual Configuration Required

Railway needs to be configured manually in the dashboard. Configuration files alone won't work if Railway auto-detects Node.js.

### Step 1: Set Root Directory
1. Go to Railway Dashboard → Your Project → Service Settings
2. Scroll to "Root Directory" 
3. Set to: `python_backend`
4. Click "Save"

### Step 2: Set Start Command
1. In Service Settings → "Deploy" section
2. Find "Start Command"
3. Set to: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
4. Click "Save"

### Step 3: Set Build Command
1. In Service Settings → "Build" section  
2. Find "Build Command"
3. Set to: `pip install -r requirements.txt`
4. Click "Save"

### Step 4: Verify Python Runtime
1. Railway should auto-detect Python when Root Directory is set
2. If not, check "Runtime" settings

### Step 5: Redeploy
1. Go to "Deployments" tab
2. Click "..." on latest deployment
3. Select "Redeploy"

## What You Should See in Logs

**✅ Success:**
```
Using Nixpacks
Installing Python dependencies...
Starting gunicorn...
[INFO] Starting gunicorn
```

**❌ Failure (if still wrong):**
```
npm start
node dist/index.js
```

## Environment Variables Needed

### Required:
- `PORT` - Auto-set by Railway (don't add manually)
- `DATABASE_URL` - Auto-set if you added PostgreSQL service

### Optional (LSTM tuning):
- `LSTM_EPOCHS`, `LSTM_PATIENCE`, etc. - Only if you want to customize

### NOT Needed:
- `FRONTEND_URL` - Backend doesn't need this (CORS allows all origins)
- `NODE_ENV` - Not needed for Python backend

## Troubleshooting

### Build still fails?
1. Delete `nixpacks.toml` - Let Railway auto-detect Python
2. Set Root Directory to `python_backend` in dashboard
3. Set Start Command manually in dashboard
4. Redeploy

### Still seeing Node.js?
- Railway is ignoring config files
- MUST configure in dashboard manually
- Root Directory is the key setting

### Backend not accessible?
- Check Railway service is running (green status)
- Check Public Domain URL is correct
- Verify VITE_API_BASE in Vercel matches Railway URL

