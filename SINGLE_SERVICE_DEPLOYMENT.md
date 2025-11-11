# Single Service Deployment Guide

## Deploy Everything Together on Render

You can now deploy both frontend and backend together in **ONE** Render web service!

### Why This Works

Unlike your other projects that might be pure Node.js or have the backend in the same language, this project has:
- **Frontend**: React (Node.js)
- **Backend**: Python Flask (ML models)

Render's single web service can only run **one runtime** (Node.js OR Python), BUT we can run both processes together using a shell script!

### How It Works

The `start.sh` script:
1. Starts Python Flask backend in the background (port 5001)
2. Starts Node.js Express server in the foreground (port from Render)
3. Node.js proxies API calls to Python backend on localhost
4. Both run in the same container!

### Deployment Steps

**Option 1: Using render-single.yaml (Easiest)**

1. Go to Render Dashboard
2. Click "New +" → "Blueprint"
3. Connect your GitHub repo
4. Render will detect `render-single.yaml` automatically
5. Deploy!

**Option 2: Manual Setup**

1. Create new **Web Service** on Render
2. Connect GitHub repository
3. Configure:
   - **Environment**: Node.js
   - **Build Command**: `npm ci && npm run build && cd python_backend && pip install -r requirements.txt`
   - **Start Command**: `./start.sh`
   - **Root Directory**: `/` (root)
4. Add Environment Variables:
   - `NODE_ENV=production`
   - `PORT_PYTHON=5001`
   - `DATABASE_URL` (your Neon PostgreSQL URL)
   - `CORS_ORIGINS=*` (or your domain)
5. Deploy!

### Important Notes

⚠️ **Python Installation**: Render's Node.js environment should have Python 3 available. If not, you may need to:
- Use a custom Dockerfile
- Or deploy as separate services (original method)

✅ **Advantages**:
- Single service = simpler deployment
- No need to configure service URLs
- Everything runs together
- Lower cost (one service instead of two)

⚠️ **Limitations**:
- Both processes share the same resources (CPU/RAM)
- If one crashes, both go down
- Python dependencies add to build time (5-10 minutes)

### Troubleshooting

**Python not found error:**
- Render's Node.js environment should have Python
- If not, use separate services (original `render.yaml`)

**Python backend won't start:**
- Check logs for Python errors
- Verify `requirements.txt` is correct
- Check if gunicorn is installed

**Port conflicts:**
- Python uses port 5001 (internal)
- Node.js uses Render's PORT (external)
- They don't conflict!

### Alternative: Separate Services

If single service doesn't work, use the original `render.yaml` with two separate services (more reliable but requires configuring URLs).

