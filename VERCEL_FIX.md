# Vercel Deployment Fix - Manual Steps

## Issue
Vercel is still deploying from old commit `645f497` instead of the latest fixes.

## Solution: Manual Configuration in Vercel Dashboard

Since Vercel might be caching or not detecting the new commits, configure it manually:

### Steps:

1. **Go to Vercel Dashboard**
   - Visit [vercel.com](https://vercel.com)
   - Open your StockAI project

2. **Go to Settings → General**
   - Scroll to "Build & Development Settings"

3. **Override Build Settings** (Important!)
   - **Framework Preset**: Select "Other" (not "Vite")
   - **Build Command**: `npm run build:frontend`
   - **Output Directory**: `dist/public`
   - **Install Command**: `npm install`
   - **Root Directory**: `./` (leave empty or use `./`)

4. **Save Settings**

5. **Redeploy**
   - Go to "Deployments" tab
   - Click "..." on the latest deployment
   - Select "Redeploy"
   - OR push a new commit to trigger auto-deploy

### Alternative: Force New Deployment

If manual settings don't work, trigger a new deployment:

```bash
# Make a small change to trigger deployment
echo "" >> README.md
git add README.md
git commit -m "Trigger Vercel redeploy"
git push origin main
```

### Verify Configuration

After updating settings, check that:
- ✅ Build Command shows: `npm run build:frontend`
- ✅ Output Directory shows: `dist/public`
- ✅ Framework is set to "Other" (not auto-detected)

### Why This Happens

Vercel sometimes:
- Caches build configurations
- Auto-detects frameworks and overrides vercel.json
- Needs manual settings to override auto-detection

The `vercel.json` file is correct, but Vercel dashboard settings take precedence.

