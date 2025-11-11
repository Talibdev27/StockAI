# Technical Context: StockVue

## Technology Stack

### Frontend Technologies

**Core Framework**
- **React 18.3.1**: UI library
- **TypeScript 5.6.3**: Type safety
- **Vite 5.4.20**: Build tool and dev server

**UI Libraries**
- **Radix UI**: Accessible component primitives
  - Dialog, Dropdown, Select, Tabs, Toast, etc.
- **Tailwind CSS 3.4.17**: Utility-first CSS framework
- **Lucide React**: Icon library
- **Lightweight Charts 4.2.3**: TradingView charting library

**State Management**
- **TanStack Query (React Query) 5.60.5**: Server state management
- **React Hook Form 7.55.0**: Form handling

**Routing**
- **Wouter 3.3.5**: Lightweight router

**Other**
- **Framer Motion 11.13.1**: Animations
- **Recharts 2.15.2**: Additional charting (for metrics)
- **date-fns 3.6.0**: Date manipulation
- **Zod 3.24.2**: Schema validation

### Backend Technologies

**Node.js Server**
- **Express 4.21.2**: Web framework
- **TypeScript**: Type safety
- **tsx 4.20.5**: TypeScript execution
- **Drizzle ORM 0.39.1**: Database ORM
- **PostgreSQL (Neon)**: Primary database
- **SQLite**: Fallback storage

**Python ML Backend**
- **Flask 3.0.3**: Web framework
- **Flask-CORS 4.0.1**: Cross-origin support
- **Python 3.9+**: Runtime

**Machine Learning Libraries**
- **TensorFlow**: Deep learning framework
- **Keras**: High-level neural network API
- **scikit-learn 1.5.2**: Traditional ML algorithms
- **XGBoost 2.0.0**: Gradient boosting
- **statsmodels 0.14.0**: Statistical models
- **pmdarima 2.0.3**: ARIMA model selection
- **Prophet 1.1.5**: Facebook's time series forecasting

**Data Processing**
- **pandas 2.2.2**: Data manipulation
- **numpy 1.26.4**: Numerical computing
- **yahooquery 2.3.0**: Yahoo Finance API wrapper
- **ta 0.11.0**: Technical analysis library

## Development Setup

### Prerequisites

**Node.js**
- Version: 18+ recommended
- Package manager: npm (comes with Node.js)

**Python**
- Version: 3.9+ required
- Virtual environment: `.venv` in `python_backend/`

**Database**
- PostgreSQL (Neon) for production
- SQLite fallback for development

### Installation Steps

**1. Frontend Setup**
```bash
npm install
```

**2. Python Backend Setup**
```bash
cd python_backend
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Environment Variables**
Create `.env` in project root:
```
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
```

**4. Database Setup**
```bash
npm run db:push  # Creates tables in PostgreSQL
```

### Running the Application

**Development Mode**
```bash
# Terminal 1: Node.js server + Frontend
npm run dev

# Terminal 2: Python backend
npm run dev:py
# OR
cd python_backend
source .venv/bin/activate
FLASK_APP=app.py flask run -p 5001
```

**Production Build**
```bash
npm run build
npm start
```

## Technical Constraints

### Python Backend

**Model Dependencies**
- Prophet requires C++ compiler (can be slow to install)
- XGBoost requires compiled dependencies
- TensorFlow requires compatible Python version

**Performance**
- Model training: 10-30 seconds per stock
- Prediction generation: 1-5 seconds (with cached models)
- First Prophet run: 5-10 seconds (compilation)

**Storage**
- Models stored in `python_backend/models_store/`
- Predictions in SQLite (`predictions.db`)
- Each model ~1-10 MB

### Frontend

**Browser Support**
- Modern browsers (Chrome, Firefox, Safari, Edge)
- ES6+ features required
- Canvas API for charts

**Performance**
- Initial bundle: ~500KB (gzipped)
- Chart rendering: Smooth 60fps
- API calls: Cached by React Query

### Data Source

**Yahoo Finance (via yahooquery)**
- Free tier: No API key required
- Rate limits: Not officially documented, but reasonable
- Data availability: Real-time quotes, historical data
- Coverage: S&P 500 stocks, major indices

**Limitations**
- No guarantee of uptime
- Rate limiting possible with high traffic
- Historical data limited to available range

## Dependencies Management

### Frontend
- **package.json**: All dependencies listed
- **package-lock.json**: Locked versions
- Updates: `npm update` or manual version bumping

### Python Backend
- **requirements.txt**: Pinned versions for stability
- Updates: `pip install -r requirements.txt --upgrade`
- Virtual environment: Isolated from system Python

## Development Tools

**Code Quality**
- TypeScript: Type checking
- ESLint: Linting (if configured)
- Prettier: Code formatting (if configured)

**Build Tools**
- Vite: Fast HMR, optimized builds
- esbuild: Fast TypeScript bundling for server

**Version Control**
- Git: Source control
- `.gitignore`: Excludes `node_modules`, `.venv`, `.env`

## Deployment Considerations

### Frontend
- **Build Output**: `dist/` directory
- **Static Hosting**: Can be deployed to Vercel, Netlify, etc.
- **Environment**: Production build optimized

### Backend
- **Node.js Server**: Requires Node.js runtime
- **Python Backend**: Requires Python 3.9+ and virtual environment
- **Database**: Requires PostgreSQL connection

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `NODE_ENV`: `production` or `development`
- `PORT`: Server port (default: 3000)
- `FLASK_APP`: Flask app file (default: `app.py`)

## Known Technical Issues

### Warnings (Non-Critical)
1. **urllib3 OpenSSL Warning**: 
   - Message: "urllib3 v2 only supports OpenSSL 1.1.1+"
   - Impact: None, functionality works
   - Fix: Can downgrade urllib3 or ignore

2. **Plotly Import Failure**:
   - Message: "Importing plotly failed"
   - Impact: Interactive plots unavailable (not used in production)
   - Fix: Optional dependency, can be ignored

### Model Installation
- **Prophet**: Can take 5-10 minutes first install (compiles C++)
- **XGBoost**: Requires compatible compiler
- **TensorFlow**: Large download (~500MB)

## Performance Benchmarks

**Model Training (per stock)**
- LSTM: 15-25 seconds
- XGBoost: 5-10 seconds
- ARIMA: 2-5 seconds
- Prophet: 10-20 seconds
- Total Ensemble: 30-60 seconds (first time)

**Prediction Generation (with cached models)**
- Ensemble: 1-3 seconds
- Individual models: 0.1-0.5 seconds each

**API Response Times**
- `/api/predict`: 1-30 seconds (depends on model caching)
- `/api/backtest`: 5-15 seconds
- `/api/indicators`: <1 second
- `/api/history`: <1 second

## Future Technical Considerations

**Scalability**
- Model caching reduces load
- Could add Redis for distributed caching
- Could add Celery for async model training

**Monitoring**
- Add logging/monitoring (e.g., Sentry)
- Track API response times
- Monitor model accuracy over time

**Optimization**
- Pre-train models for popular stocks
- Batch prediction requests
- Optimize model architectures further

