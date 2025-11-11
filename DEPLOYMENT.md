# Deployment Guide for StockVue

## Architecture Overview

StockVue consists of three main components:
1. **Frontend**: React + Vite (static files)
2. **Node.js Server**: Express API gateway (port 3000)
3. **Python Backend**: Flask ML service (port 5001)

## Deployment Options

### Option 1: Vercel (Recommended - Easiest)

**Best for**: Frontend + Node.js server together

#### Steps:

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Create `vercel.json`** (already created below)

3. **Deploy**:
   ```bash
   vercel
   ```

4. **Set Environment Variables** in Vercel dashboard:
   - `DATABASE_URL`: Your Neon PostgreSQL connection string
   - `NODE_ENV`: `production`
   - `VITE_API_BASE`: Your Vercel deployment URL (auto-set)

5. **Deploy Python Backend separately** (see Option 3)

---

### Option 2: Railway (All-in-One)

**Best for**: Deploying everything together

#### Steps:

1. **Install Railway CLI**:
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Create Railway project**:
   ```bash
   railway init
   ```

3. **Add services**:
   - Service 1: Node.js (from root directory)
   - Service 2: Python (from `python_backend/` directory)

4. **Set Environment Variables**:
   - `DATABASE_URL`: Your Neon PostgreSQL connection string
   - `PORT`: `3000` (for Node.js), `5001` (for Python)
   - `NODE_ENV`: `production`
   - `FLASK_APP`: `app.py`

5. **Deploy**:
   ```bash
   railway up
   ```

---

### Option 3: Render (Free Tier Available)

**Best for**: Separate services with free hosting

#### Frontend + Node.js Server:

1. Create new **Web Service** on Render
2. Connect GitHub repository
3. Build command: `npm install --include=dev && npm run build`
   - Note: `--include=dev` ensures devDependencies (like vite) are installed
4. Start command: `npm start`
5. Set environment variables:
   - `DATABASE_URL`
   - `NODE_ENV=production`
   - `PYTHON_API_BASE` (URL of your Python backend service)

#### Python Backend:

1. Create new **Web Service** on Render
2. Root directory: `python_backend`
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Set environment variables:
   - `FLASK_APP=app.py`
   - `PORT` (auto-set by Render)

---

### Option 4: DigitalOcean App Platform

**Best for**: Production-ready deployment

1. Connect GitHub repository
2. Add two components:
   - **Component 1**: Node.js (root directory)
   - **Component 2**: Python (python_backend directory)
3. Set environment variables
4. Deploy

---

## Environment Variables Required

### Node.js Server:
```bash
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
NODE_ENV=production
PORT=3000
```

### Python Backend:
```bash
FLASK_APP=app.py
FLASK_ENV=production
PORT=5001
```

### Frontend (Build-time):
```bash
VITE_API_BASE=https://your-node-server-url.com
```

---

## Pre-Deployment Checklist

- [ ] Update `client/src/lib/api.ts` with production API URL
- [ ] Set up Neon PostgreSQL database
- [ ] Run `npm run db:push` to create database schema
- [ ] Test production build locally: `npm run build && npm start`
- [ ] Ensure Python backend can run: `cd python_backend && python app.py`
- [ ] Configure CORS in Python backend for production domain
- [ ] Set up environment variables on hosting platform

---

## Post-Deployment Steps

1. **Verify Database Connection**:
   - Check Node.js server logs
   - Verify tables exist in Neon dashboard

2. **Test API Endpoints**:
   - Frontend: `https://your-domain.com`
   - Node.js API: `https://your-domain.com/api/stocks`
   - Python API: `https://your-python-backend.com/api/predict/AAPL`

3. **Monitor Logs**:
   - Check for errors in hosting platform logs
   - Monitor Python backend for model loading issues

4. **Performance Optimization**:
   - Enable CDN for static assets
   - Set up caching headers
   - Monitor API response times

---

## Troubleshooting

### Python Backend Won't Start:
- Check Python version (3.9+ required)
- Verify all dependencies installed
- Check for missing model files (they'll be generated on first use)

### CORS Errors:
- Update `python_backend/app.py` CORS settings
- Add your frontend domain to allowed origins

### Database Connection Issues:
- Verify `DATABASE_URL` format
- Check Neon database is running
- Ensure SSL mode is correct (`?sslmode=require`)

### Build Failures:
- Check Node.js version (18+)
- Verify all dependencies in `package.json`
- Check for TypeScript errors: `npm run check`

---

## Recommended Production Setup

**Frontend**: Vercel (automatic CDN, fast)
**Node.js Server**: Railway or Render
**Python Backend**: Railway or Render (same platform as Node.js for easier communication)
**Database**: Neon PostgreSQL (already set up)

This setup provides:
- Fast frontend delivery (Vercel CDN)
- Reliable backend services
- Managed database
- Easy scaling

