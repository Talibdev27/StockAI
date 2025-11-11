# Vercel Environment Variable Setup

## Critical: Set VITE_API_BASE

The frontend needs to know where your Railway backend is located.

### Step 1: Get Your Railway Backend URL

1. Go to Railway Dashboard → Your Project
2. Click on your backend service
3. Go to "Settings" → "Networking"
4. Copy the "Public Domain" URL
   - Example: `https://your-project.up.railway.app`
   - **Important**: Include `https://` and no trailing slash

### Step 2: Set in Vercel

1. Go to Vercel Dashboard → Your Project
2. Go to "Settings" → "Environment Variables"
3. Click "Add New"
4. Add:
   - **Key**: `VITE_API_BASE`
   - **Value**: Your Railway backend URL (e.g., `https://your-project.up.railway.app`)
   - **Environment**: Select all (Production, Preview, Development)
5. Click "Save"

### Step 3: Redeploy

After adding the environment variable:
1. Go to "Deployments" tab
2. Click "..." on latest deployment
3. Select "Redeploy"

### Step 4: Verify

After redeploy, check:
1. Open browser DevTools (F12) → Console
2. Look for API calls - they should go to your Railway URL
3. Should NOT see calls to `localhost:5001` (unless testing locally)

## Troubleshooting

### Still seeing localhost:5001?
- Environment variable not set correctly
- Need to redeploy after setting variable
- Check Vercel deployment logs

### Getting CORS errors?
- Backend CORS is configured to allow all origins
- Check Railway backend is running
- Verify Railway URL is correct

### Getting HTML instead of JSON?
- Backend is not running or returning error page
- Check Railway logs
- Verify backend service is deployed and healthy

