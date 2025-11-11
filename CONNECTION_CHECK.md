# Frontend-Backend Connection Checklist

## ✅ Configuration Check

### Frontend (Vercel)
- ✅ API Base URL: Uses `VITE_API_BASE` environment variable
- ✅ Fallback: `http://localhost:5001` (for local dev)
- ✅ API calls: All use `/api/*` endpoints

### Backend (Railway)
- ✅ CORS: Configured to allow all origins (`*`)
- ✅ Port: Auto-set by Railway via `$PORT`
- ✅ Routes: All `/api/*` endpoints available

## 🔗 Connection Requirements

### 1. Vercel Environment Variable (CRITICAL)

**Must be set in Vercel Dashboard:**
- Variable: `VITE_API_BASE`
- Value: Your Railway backend URL
  - Example: `https://your-project.railway.app`
  - **Important**: Include `https://` and no trailing slash

**How to check:**
1. Go to Vercel Dashboard → Your Project → Settings → Environment Variables
2. Verify `VITE_API_BASE` is set to your Railway URL
3. Make sure it's set for **Production**, **Preview**, and **Development**

### 2. Railway Backend URL

**Get your Railway URL:**
1. Go to Railway Dashboard → Your Project
2. Click on your service
3. Go to "Settings" → "Networking"
4. Copy the "Public Domain" URL
5. It should look like: `https://your-project.up.railway.app`

### 3. Backend Environment Variables (Railway)

**Current variables look good:**
- ✅ LSTM configs (all optional, have defaults)
- ⚠️ `DATABASE_URL` - Not needed for SQLite (can remove if causing issues)

**Required:**
- ✅ `PORT` - Auto-set by Railway (don't add manually)

## 🧪 Testing Connection

### Test Backend Directly

1. **Test Railway backend:**
   ```bash
   curl https://your-railway-url.railway.app/api/quote/AAPL
   ```
   Should return JSON with stock quote data.

2. **Test stocks endpoint:**
   ```bash
   curl https://your-railway-url.railway.app/api/stocks?popular=true
   ```
   Should return list of popular stocks.

### Test Frontend Connection

1. **Open browser console** on your Vercel frontend
2. **Check Network tab** when loading the dashboard
3. **Look for API calls:**
   - Should call: `https://your-railway-url.railway.app/api/quote/AAPL`
   - Should NOT call: `http://localhost:5001/api/...`

### Common Issues

#### ❌ CORS Errors
**Symptom**: Browser console shows CORS error
**Fix**: Backend CORS is already configured (`origins: "*"`), should work

#### ❌ 404 on API Calls
**Symptom**: API calls return 404
**Fix**: Check that `VITE_API_BASE` includes full URL with `https://`

#### ❌ Connection Refused
**Symptom**: Network error, can't connect
**Fix**: 
- Verify Railway backend is running (check Railway logs)
- Verify Railway URL is correct
- Check Railway service is deployed and healthy

#### ❌ Wrong API URL
**Symptom**: Calls going to `localhost:5001` instead of Railway
**Fix**: `VITE_API_BASE` not set in Vercel, or wrong value

## ✅ Quick Verification Steps

1. **Railway Backend:**
   - ✅ Service is running (green status)
   - ✅ Public domain is accessible
   - ✅ Test: `curl https://your-railway-url/api/quote/AAPL`

2. **Vercel Frontend:**
   - ✅ `VITE_API_BASE` environment variable is set
   - ✅ Value is your Railway URL (with `https://`)
   - ✅ Deployed successfully

3. **Connection:**
   - ✅ Open frontend in browser
   - ✅ Open browser DevTools → Network tab
   - ✅ Check API calls go to Railway URL (not localhost)
   - ✅ Verify responses are successful (200 status)

## 📝 Summary

**Backend (Railway):**
- ✅ CORS configured correctly
- ✅ No required env vars (all optional)
- ✅ `DATABASE_URL` not needed (uses SQLite)

**Frontend (Vercel):**
- ✅ Must set `VITE_API_BASE` = Railway backend URL
- ✅ API calls will use this URL

**Connection:**
- ✅ Should work if `VITE_API_BASE` is set correctly
- ✅ CORS allows all origins, so no CORS issues expected

