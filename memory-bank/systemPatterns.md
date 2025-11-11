# System Patterns: StockVue Architecture

## System Architecture

StockVue follows a **three-tier architecture**:

```
┌─────────────────┐
│   React Client  │  (Frontend - TypeScript/React)
└────────┬────────┘
         │ HTTP/REST
┌────────▼────────┐
│  Node.js Server │  (Express API Gateway)
└────────┬────────┘
         │ HTTP/REST
┌────────▼────────┐
│  Python Backend │  (Flask ML Service)
└─────────────────┘
```

### Frontend (Client)
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Routing**: Wouter (lightweight router)
- **State Management**: React Query (TanStack Query) for server state
- **UI Components**: Radix UI primitives + custom components
- **Styling**: Tailwind CSS with custom design system
- **Charts**: Lightweight Charts (TradingView library)

### Middleware (Node.js Server)
- **Framework**: Express.js
- **Purpose**: API gateway, static file serving, session management
- **Storage**: PostgreSQL (Neon) for user data, SQLite fallback
- **Session**: Express-session with PostgreSQL store

### Backend (Python ML Service)
- **Framework**: Flask
- **Purpose**: ML model training, prediction generation, backtesting
- **Storage**: SQLite (`predictions.db`) for prediction history
- **Data Source**: Yahoo Finance via `yahooquery`

## Key Technical Decisions

### 1. Ensemble Model Architecture

**Pattern**: Weighted Voting Ensemble

```python
# Models are initialized conditionally
self.models = {
    "linear": LinearRegressionModel(),
    "lstm": LSTMModel(),
    "arima": ARIMAModel(),
    # Optional models loaded if dependencies available
}
if _HAS_XGB:
    self.models["xgboost"] = XGBoostModel()
# ... etc
```

**Rationale**:
- Combines strengths of different model types
- Graceful degradation if optional models unavailable
- Confidence-based weighting improves accuracy

### 2. Adaptive Lag Selection

**Pattern**: Strategy Pattern for Timeframe Adaptation

```python
def _get_lag(self):
    if self.interval in ["1h", "30m", "15m", "5m", "1m"]:
        return 20  # Intraday: more recent data
    elif self.interval in ["1d"]:
        return 20  # Daily: standard lag
    # ... etc
```

**Rationale**:
- Different timeframes require different historical windows
- Intraday needs more recent data (20 periods)
- Weekly/Monthly need less frequent data (12/6 periods)

### 3. Model Persistence

**Pattern**: File-based Model Storage

- **Location**: `python_backend/models_store/`
- **Format**: 
  - LSTM: `.h5` (Keras), `.pkl` (scaler), `.json` (metadata)
  - XGBoost/DT/SVM: `.pkl` (model), `.pkl` (scaler), `.pkl` (metadata)
- **Naming**: `{SYMBOL}_{INTERVAL}_{MODEL_TYPE}.{ext}`

**Rationale**:
- Avoids retraining on every request
- Models are symbol+interval specific
- Metadata stores training parameters and timestamps

### 4. Graceful Degradation

**Pattern**: Try-Except with Feature Flags

```python
try:
    from .prophet_model import ProphetModel
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False
```

**Rationale**:
- Optional dependencies don't break the system
- System works with minimum required models
- User experience remains consistent

### 5. Data Fetching Strategy

**Pattern**: Caching with TTL

```python
_cache: Dict[str, Dict[str, Any]] = {}

def _cache_get(key: str, ttl_seconds: int) -> Any:
    # Check cache, validate TTL
```

**Rationale**:
- Reduces API calls to Yahoo Finance
- Improves response time
- Respects rate limits

### 6. Frontend State Management

**Pattern**: React Query for Server State

```typescript
const { data: prediction } = usePrediction(symbol, interval);
const { data: historical } = useHistorical(symbol, interval);
```

**Rationale**:
- Automatic caching and refetching
- Loading and error states handled
- Optimistic updates possible

## Component Relationships

### Frontend Components

```
App
├── Header (Navigation, Theme Toggle)
├── Router
    ├── Dashboard (Main prediction interface)
    │   ├── StockSelector
    │   ├── TimeframeSelector
    │   ├── PriceChart
    │   ├── PredictionSummary
    │   ├── TechnicalIndicators
    │   └── PerformanceMetrics
    ├── Predictions (Prediction history)
    ├── Backtest (Backtesting interface)
    └── About (Project information)
```

### Backend Services

```
Flask App (app.py)
├── /api/predict → EnsembleModel.predict()
├── /api/backtest → BacktestEngine.run()
├── /api/indicators → Technical indicators calculation
├── /api/history → Prediction history from SQLite
└── /api/stats → Prediction accuracy statistics
```

## Design Patterns in Use

### 1. Factory Pattern
- Model initialization based on availability
- Component creation based on props

### 2. Strategy Pattern
- Different lag strategies per timeframe
- Different model types for different patterns

### 3. Observer Pattern
- React Query subscriptions
- Chart updates on data changes

### 4. Repository Pattern
- `evaluation.py` handles prediction storage/retrieval
- `storage.ts` handles user data persistence

### 5. Facade Pattern
- `EnsembleModel` provides simple interface to complex model system
- API routes abstract backend complexity

## Data Flow

### Prediction Request Flow

1. User selects stock + timeframe
2. Frontend calls `/api/predict` via React Query
3. Node.js server proxies to Flask backend
4. Flask backend:
   - Fetches historical data from Yahoo Finance
   - Checks for cached models
   - Trains models if needed
   - Generates predictions
   - Saves prediction to SQLite
   - Returns prediction + confidence
5. Frontend displays prediction on chart

### Backtesting Flow

1. User configures backtest parameters
2. Frontend calls `/api/backtest`
3. Flask backend:
   - Loads historical data for date range
   - Runs walk-forward analysis
   - Calculates performance metrics
   - Returns equity curve + metrics
4. Frontend visualizes results

## Error Handling Patterns

### Backend
- Try-except blocks around model initialization
- Graceful degradation if models fail
- Error messages returned as JSON

### Frontend
- React Query error boundaries
- Loading states for async operations
- User-friendly error messages

## Performance Optimizations

1. **Model Caching**: Trained models saved to disk
2. **Data Caching**: API responses cached with TTL
3. **Lazy Loading**: Models loaded only when needed
4. **Code Splitting**: Vite automatically splits bundles
5. **Chart Optimization**: Lightweight Charts is performant

## Security Considerations

- CORS enabled for Flask backend
- Input validation on API endpoints
- SQL injection prevention (parameterized queries)
- XSS prevention (React escapes by default)

