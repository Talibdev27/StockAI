# Step-by-Step Guide: Deploy Python Backend on Render

## Prerequisites
- ✅ Node.js service already deployed (StockAI)
- ✅ Node.js service URL: `https://stockai-qfss.onrender.com`

---

## Step 1: Create Python Backend Service

1. **Go to Render Dashboard**
   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Make sure you're logged in

2. **Create New Service**
   - Click the **"New +"** button (top right)
   - Select **"Web Service"**

3. **Connect Repository**
   - Click **"Connect account"** if not already connected
   - Select **GitHub**
   - Authorize Render to access your repositories
   - Find and select **`Talibdev27/StockAI`** repository
   - Click **"Connect"**

---

## Step 2: Configure Python Service

### Basic Settings:

1. **Name**: `stockvue-python` (or any name you prefer)

2. **Region**: Choose closest to you (or same as Node.js service)

3. **Branch**: `main`

4. **Root Directory**: `python_backend` ⚠️ **IMPORTANT**

5. **Runtime**: **Python 3** (Render will auto-detect)

6. **Python Version**: 
   - In the settings, look for "Python Version" or "Runtime"
   - Select **Python 3.11** (NOT 3.13 - pandas compatibility issue)

### Build & Start Commands:

7. **Build Command**: 
   ```
   pip install --upgrade pip && pip install -r requirements.txt
   ```

8. **Start Command**: 
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```

9. **Instance Type**: 
   - **Free** (for testing)
   - Or **Starter** ($7/month) for better performance

---

## Step 3: Set Environment Variables

Click **"Advanced"** → **"Add Environment Variable"**:

### Required Variables:

1. **FLASK_APP**
   - Key: `FLASK_APP`
   - Value: `app.py`

2. **FLASK_ENV**
   - Key: `FLASK_ENV`
   - Value: `production`

3. **CORS_ORIGINS**
   - Key: `CORS_ORIGINS`
   - Value: `https://stockai-qfss.onrender.com`
   - (Use your Node.js service URL)

### Optional Variables:

4. **PORT** (auto-set by Render, don't need to add)

---

## Step 4: Deploy

1. Click **"Create Web Service"** at the bottom
2. Render will start building (this takes 5-10 minutes)
3. Watch the build logs:
   - Installing dependencies (pandas, numpy, tensorflow, etc.)
   - This is slow but normal!
   - Wait for "Build successful 🎉"

---

## Step 5: Get Python Service URL

After deployment succeeds:

1. Look for **"Your service is live at"** message
2. Copy the URL (e.g., `https://stockvue-python.onrender.com`)
3. **Save this URL** - you'll need it next!

---

## Step 6: Connect Node.js to Python Backend

1. **Go back to your Node.js service** (`StockAI`)
2. Click **"Environment"** tab
3. Click **"Add Environment Variable"**
4. Add:
   - **Key**: `PYTHON_API_BASE`
   - **Value**: `https://your-python-service-url.onrender.com`
   - (Use the URL from Step 5)
5. Click **"Save Changes"**
6. Render will **automatically redeploy** your Node.js service

---

## Step 7: Verify Connection

1. **Wait for both services to be "Live"**
2. Visit your Node.js service URL: `https://stockai-qfss.onrender.com`
3. Open browser console (F12)
4. Check Network tab - API calls should now work!
5. Test a stock prediction - should work without errors

---

## Troubleshooting

### Python Build Fails:
- **Error**: "pandas compilation error"
  - **Fix**: Make sure Python version is 3.11 (not 3.13)
  - Go to Settings → Change Python version

### Python Service Won't Start:
- Check logs for errors
- Verify `gunicorn` is in requirements.txt ✅ (already added)
- Check if port is correct

### Node.js Can't Connect to Python:
- Verify `PYTHON_API_BASE` is set correctly
- Check Python service is "Live" (not "Building")
- Verify `CORS_ORIGINS` includes Node.js URL

### CORS Errors:
- Make sure `CORS_ORIGINS` in Python service = Node.js URL
- Should be: `https://stockai-qfss.onrender.com`
- Redeploy Python service after changing CORS_ORIGINS

---

## Quick Checklist

- [ ] Python service created
- [ ] Root directory set to `python_backend`
- [ ] Python version set to 3.11
- [ ] Build command: `pip install --upgrade pip && pip install -r requirements.txt`
- [ ] Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
- [ ] Environment variables set (FLASK_APP, FLASK_ENV, CORS_ORIGINS)
- [ ] Python service deployed and live
- [ ] Python service URL copied
- [ ] PYTHON_API_BASE set in Node.js service
- [ ] Node.js service redeployed
- [ ] Both services are "Live"
- [ ] Tested - API calls work!

---

## Expected Timeline

- **Python Build**: 5-10 minutes (installing TensorFlow, Prophet, etc.)
- **Python Start**: 1-2 minutes
- **Node.js Redeploy**: 2-3 minutes
- **Total**: ~15 minutes

---

## After Deployment

Your setup will be:
- **Frontend + Node.js**: `https://stockai-qfss.onrender.com`
- **Python Backend**: `https://stockvue-python.onrender.com` (example)
- **Communication**: Node.js proxies API calls to Python backend

Everything should work now! 🎉

