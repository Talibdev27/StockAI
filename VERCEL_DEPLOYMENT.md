# Vercel Frontend Deployment

## Frontend Deployment to Vercel

This guide explains how to deploy the React frontend to Vercel.

### Setup Steps:

1. **Install Vercel CLI** (optional, can use web UI):
   ```bash
   npm i -g vercel
   vercel login
   ```

2. **Deploy via Vercel Dashboard**:
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New Project"
   - Import your GitHub repository
   - Configure:
     - **Framework Preset**: Vite
     - **Root Directory**: `./` (root)
     - **Build Command**: `npm run build`
     - **Output Directory**: `dist/public`

3. **Environment Variables**:
   Add in Vercel dashboard → Settings → Environment Variables:
   - `VITE_API_BASE`: Your Railway backend URL (e.g., `https://your-backend.railway.app`)

4. **Deploy**:
   - Click "Deploy"
   - Vercel will build and deploy automatically
   - Frontend will be live at: `https://your-project.vercel.app`

### Alternative: Deploy via CLI

```bash
# From project root
vercel

# For production
vercel --prod
```

### Important Notes:

- **API Base URL**: Make sure `VITE_API_BASE` points to your Railway backend
- **CORS**: Ensure Railway backend allows requests from your Vercel domain
- **Build Output**: Vite builds to `dist/public` (configured in `vite.config.ts`)

### Troubleshooting:

- If build fails, check Vercel build logs
- Ensure all dependencies are in `package.json`
- Check that `VITE_API_BASE` is set correctly
- Verify backend is accessible from Vercel domain

