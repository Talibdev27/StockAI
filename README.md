# StockVue - AI-Powered Stock Market Prediction Platform

StockVue is an AI-powered stock market prediction platform that uses an ensemble of 7 machine learning models to predict short-term price movements with confidence scores.

## 🚀 Quick Start

### Prerequisites
- **Node.js** v18+ and npm
- **Python** 3.9+
- **Git**

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Talibdev27/StockAI.git
   cd StockAI
   ```

2. **Install Frontend Dependencies**
   ```bash
   npm install
   ```

3. **Setup Python Backend**
   ```bash
   cd python_backend
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the Application

**Terminal 1 - Backend (Python Flask):**
```bash
cd python_backend
source .venv/bin/activate
FLASK_APP=app.py flask run -p 5001
```
Backend runs on `http://localhost:5001`

**Terminal 2 - Frontend (React + Node.js):**
```bash
npm run dev
```
Frontend runs on `http://localhost:3000`

Open `http://localhost:3000` in your browser.

## 📊 Features

- **7-Model Ensemble**: Linear, LSTM, ARIMA, XGBoost, Decision Tree, SVM, Prophet
- **Real-time Predictions**: Yahoo Finance data integration
- **Advanced Charting**: Interactive charts with technical indicators
- **Backtesting Engine**: Walk-forward validation with realistic transaction costs
- **Prediction Tracking**: Historical accuracy evaluation
- **S&P 500 Coverage**: Support for 500+ stocks

## 🏗️ Architecture

- **Frontend**: React + TypeScript + Tailwind CSS + Vite
- **Backend**: Python Flask + TensorFlow/Keras
- **Database**: SQLite (local) / PostgreSQL (production)
- **Data Source**: Yahoo Finance (yahooquery)

## 📈 Performance Metrics

- **Direction Accuracy**: 55.3% (5.3% above random)
- **Annual Return**: 12.4% (backtesting)
- **Sharpe Ratio**: 1.08
- **Win Rate**: 56.2% (high confidence predictions)

## 🔧 Key Components

### Models
- `python_backend/models/ensemble.py` - Main ensemble orchestrator
- `python_backend/models/lstm_model.py` - Dual-head LSTM (price + direction)
- `python_backend/models/*_model.py` - Individual model implementations

### API Endpoints
- `GET /api/predict/<symbol>` - Get stock prediction
- `GET /api/backtest/<symbol>` - Run backtest simulation
- `GET /api/predictions/bias` - Directional bias analysis
- `GET /api/predictions/history` - Prediction history

### Frontend Pages
- `client/src/pages/Dashboard.tsx` - Main prediction interface
- `client/src/pages/Backtest.tsx` - Backtesting interface
- `client/src/pages/PredictionAccuracy.tsx` - Accuracy analysis

## 📚 Documentation

- [Quick Reference](QUICK_REFERENCE.md) - Project overview and metrics
- [Deployment Guide](DEPLOYMENT.md) - Production deployment instructions
- [Database Setup](DATABASE_SETUP.md) - Database configuration
- [Confidence Explanation](CONFIDENCE_EXPLANATION.md) - How confidence scores work

## 🧪 Testing

### Test a Prediction
```bash
# Backend must be running
curl http://localhost:5001/api/predict/AAPL?interval=1d
```

### Retrain Models
```bash
cd python_backend
source .venv/bin/activate
python retrain_model.py AAPL 1d
```

### Batch Retrain S&P 500
```bash
python batch_retrain.py --range 100d --limit 20
```

## 🛠️ Development

### Project Structure
```
StockVue/
├── client/              # React frontend
├── python_backend/      # Flask backend
│   ├── models/         # ML models
│   ├── models_store/   # Trained models
│   └── app.py         # Flask API
├── server/             # Node.js gateway
└── shared/             # Shared schemas
```

### Environment Variables
- `DATABASE_URL` - PostgreSQL connection (optional, defaults to SQLite)
- `LSTM_EPOCHS` - LSTM training epochs (default: 30)
- `ENSEMBLE_DIRECTION_WEIGHT` - Direction weight in ensemble (default: 0.7)

## 📝 Recent Improvements

- ✅ Dual-head LSTM architecture (price + direction classification)
- ✅ Post-processing filters for mean-reversion bias
- ✅ Performance-based ensemble weighting
- ✅ Directional bias analysis and metrics
- ✅ Batch retraining tools for S&P 500 stocks

## ⚠️ Important Notes

- First prediction per stock may take 30-60 seconds (model training)
- Subsequent predictions are cached (1-3 seconds)
- XGBoost and Prophet are optional (system works without them)
- Models are stored in `python_backend/models_store/`

## 🤝 Contributing

This is a university project. For questions or issues, please contact the repository owner.

## 📄 License

MIT License

## 🙏 Acknowledgments

- Yahoo Finance for market data
- TensorFlow/Keras for deep learning
- React community for UI components

