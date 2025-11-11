# Vercel Frontend 404 Fix

## Issue
Frontend deployed but showing 404 error.

## Common Causes & Solutions

### 1. **Routing Configuration** ✅ Fixed
- Updated `vercel.json` to use `rewrites` for SPA routing
- All routes now redirect to `/index.html` for client-side routing

### 2. **Environment Variable Missing**
Make sure `VITE_API_BASE` is set in Vercel:
- Go to Vercel Dashboard → Your Project → Settings → Environment Variables
- Add: `VITE_API_BASE` = Your Railway backend URL
- Example: `https://your-backend.railway.app`

### 3. **Build Output Directory**
Verify in Vercel Settings:
- Output Directory: `dist/public`
- Build Command: `npm run build:frontend`

### 4. **Check Build Logs**
- Go to Vercel Dashboard → Deployments
- Click on the latest deployment
- Check "Build Logs" for errors
- Verify `index.html` is being created

### 5. **Clear Cache & Redeploy**
- In Vercel Dashboard → Deployments
- Click "..." on latest deployment
- Select "Redeploy" (with cache cleared)

## Testing

After fixing, test:
1. Visit your Vercel URL (root path)
2. Should load the Dashboard
3. Check browser console for API errors
4. If API errors, verify `VITE_API_BASE` is set correctly

## Current Configuration

```json
{
  "buildCommand": "npm run build:frontend",
  "outputDirectory": "dist/public",
  "installCommand": "npm install",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

This ensures all routes serve `index.html` for client-side routing.

