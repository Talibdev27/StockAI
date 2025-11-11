# Railway Deployment Configuration

## Python Backend Deployment

This project uses Railway for Python backend deployment.

### Setup Steps:

1. **Install Railway CLI** (optional, can use web UI):
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Create Railway Project**:
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure Service**:
   - Railway will auto-detect Python
   - Set root directory: `python_backend`
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

4. **Environment Variables**:
   - `FLASK_APP=app.py`
   - `PORT` (auto-set by Railway)
   - Add any other env vars your app needs

5. **Add Gunicorn to requirements.txt** (if not already):
   ```bash
   echo "gunicorn>=21.2.0" >> python_backend/requirements.txt
   ```

6. **Deploy**:
   - Railway will auto-deploy on git push
   - Or manually trigger from Railway dashboard

### Notes:
- Railway provides a public URL for your backend
- Update `VITE_API_BASE` in Vercel with Railway backend URL
- Backend will be accessible at: `https://your-project.railway.app`

