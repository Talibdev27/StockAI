# Deployment Guide: Frontend (Vercel) + Backend (Railway)

## Quick Start

### 1. Deploy Backend to Railway

1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `StockAI` repository
4. Railway will auto-detect Python
5. Configure:
   - **Root Directory**: `python_backend`
   - **Start Command**: Uses `Procfile` automatically
6. Copy the Railway URL (e.g., `https://your-app.railway.app`)

### 2. Deploy Frontend to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "Add New Project" → Import GitHub repo
3. Select your `StockAI` repository
4. Configure:
   - **Framework Preset**: Vite (auto-detected)
   - **Root Directory**: `./` (root)
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `dist/public` (auto-detected)
5. Add Environment Variable:
   - **Key**: `VITE_API_BASE`
   - **Value**: Your Railway backend URL (e.g., `https://your-app.railway.app`)
6. Click "Deploy"

### 3. Update CORS (if needed)

CORS is already configured to allow all origins. For production, you can restrict it to your Vercel domain in `python_backend/app.py`.

## Environment Variables

### Railway (Backend)
- `PORT` - Auto-set by Railway
- `FLASK_APP=app.py` - Optional
- Add any other backend env vars

### Vercel (Frontend)
- `VITE_API_BASE` - Your Railway backend URL (required)

## Testing Deployment

1. **Backend**: Visit `https://your-backend.railway.app/api/quote/AAPL`
   - Should return JSON data
   
2. **Frontend**: Visit `https://your-frontend.vercel.app`
   - Should load the dashboard
   - Check browser console for API calls

## Troubleshooting

### Backend Issues
- Check Railway logs for errors
- Verify `gunicorn` is in `requirements.txt`
- Ensure `Procfile` exists in root
- Check that port is set correctly (`$PORT`)

### Frontend Issues
- Verify `VITE_API_BASE` is set correctly
- Check Vercel build logs
- Ensure backend URL is accessible
- Check browser console for CORS errors

### CORS Errors
- CORS is configured to allow all origins (`*`)
- For production, update `app.py` to restrict to your Vercel domain

## File Structure

```
StockAI/
├── vercel.json          # Vercel config (frontend)
├── railway.json         # Railway config (backend)
├── Procfile             # Railway start command
├── client/              # Frontend React app
├── python_backend/      # Backend Flask app
│   ├── app.py
│   └── requirements.txt
└── server/              # Node.js API gateway (not needed for separate deployment)
```

## Notes

- **Separate Deployments**: Frontend and backend are deployed separately
- **API Communication**: Frontend calls backend via `VITE_API_BASE`
- **CORS**: Backend allows all origins (configure for production)
- **Environment Variables**: Set in respective platform dashboards

