# StockVue - Quick Reference Card

## 🎯 Project Name
**StockVue** (also referred to as StockPredict AI)

## 📊 Key Numbers
- **7 Models**: Linear, LSTM, ARIMA, XGBoost, Decision Tree, SVM, Prophet
- **~30 Features**: 20 lags + 2 returns + 6 rolling stats + 6 technical indicators
- **55.3% Accuracy**: Direction prediction (5.3% above random)
- **12.4% Return**: Annual return from backtesting
- **1.08 Sharpe**: Risk-adjusted return metric
- **7 Timeframes**: 5m, 15m, 1h, 4h, 1d, 1wk, 1mo
- **500+ Stocks**: S&P 500 coverage
- **1-3 seconds**: Prediction time (with caching)

## 🏗️ Architecture
- **Frontend**: React + TypeScript + Tailwind CSS
- **Gateway**: Node.js Express
- **Backend**: Python Flask
- **Database**: PostgreSQL (prod) / SQLite (local)
- **Data Source**: Yahoo Finance (yahooquery)

## 🤖 The 7 Models
1. **Linear Regression** - Fast baseline
2. **LSTM** - Deep learning, temporal patterns
3. **ARIMA** - Statistical time series
4. **XGBoost** - Gradient boosting (best individual)
5. **Decision Tree** - Interpretable
6. **SVM** - Support vector machine
7. **Prophet** - Facebook time series (optional)

## 💡 Key Innovation
**Performance-Based Ensemble Weighting**
- Tracks each model's historical accuracy
- Automatically adjusts weights (70% RMSE + 30% Direction)
- Self-improving system as evaluations accumulate

## 📈 Performance Metrics
- Direction Accuracy: **55.3%**
- Annual Return: **12.4%**
- Sharpe Ratio: **1.08**
- Max Drawdown: **8.9%**
- Win Rate: **56.2%** (high confidence)

## 🔧 Key Features
✅ Multi-model ensemble (7 models)
✅ Performance-based weighting
✅ Real-time market data (Yahoo Finance)
✅ Advanced charting (Lightweight Charts)
✅ Backtesting engine (walk-forward)
✅ Prediction evaluation system
✅ ~30 engineered features

## 🎓 Common Questions

**Q: Why 7 models?**
A: Different models capture different patterns. Ensemble reduces overfitting and improves robustness.

**Q: How accurate?**
A: 55.3% directional accuracy (5.3% above random). Small edge, but meaningful in finance.

**Q: What's unique?**
A: Performance-based weighting that adapts over time, complete end-to-end system, realistic backtesting.

**Q: Can it trade live?**
A: Currently educational/research prototype. Would need brokerage integration for live trading.

**Q: How long to predict?**
A: 1-3 seconds with model caching, 10-30 seconds if training needed.

**Q: What data?**
A: Yahoo Finance OHLCV data, technical indicators only (no fundamental data).

## 📝 Technical Details
- **Training**: Temporal split (80/20), walk-forward validation
- **Regularization**: Dropout (LSTM), L1/L2 (XGBoost), early stopping
- **Parallel Execution**: ThreadPoolExecutor for faster predictions
- **Model Persistence**: Saved to `models_store/` directory
- **Evaluation**: Automated after prediction horizon passes

## 🚀 Deployment
- **Frontend**: Vercel
- **Backend**: Railway
- **Database**: Neon (PostgreSQL)

## 📧 Contact
- **Student**: Mukhammadamin Esaev
- **SID**: 221873
- **Email**: mukhammadaminkhonesaev@gmail.com

---

## 🎤 Presentation Tips

### Opening (30 sec)
"StockVue is an AI-powered stock prediction platform that combines 7 machine learning models to predict short-term price movements. The key innovation is performance-based ensemble weighting—the system automatically learns which models work best and adjusts their influence accordingly."

### Demo Flow
1. Show dashboard with chart
2. Select a stock (e.g., AAPL)
3. Generate prediction (show 7 models + ensemble)
4. Show backtesting results
5. Explain performance-based weighting

### Closing (30 sec)
"In backtesting, the system achieved 55% directional accuracy and 12% annual returns, outperforming buy-and-hold strategies. The complete system includes real-time data, professional charts, and comprehensive backtesting—all open-source and production-ready."

---

## ⚠️ Important Notes
- **7 models** (not 6 - includes Prophet)
- **Backtesting results** (not live trading)
- **Performance-based weighting** (key differentiator)
- **Complete system** (not just models)
- **Open-source** (reproducible research)

---

*Print this card or keep on phone for quick reference during presentations.*

