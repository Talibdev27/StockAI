# Project Brief: StockVue (StockPredict AI)

## Project Overview

StockVue is an AI-powered stock market prediction platform that combines multiple machine learning models to provide accurate stock price forecasts, technical analysis, and backtesting capabilities. The platform serves as a comprehensive tool for traders and investors to make informed trading decisions.

## Core Objectives

1. **Accurate Predictions**: Provide reliable stock price predictions using an ensemble of 7 machine learning models
2. **Technical Analysis**: Offer comprehensive technical indicators and candlestick pattern recognition
3. **Backtesting**: Enable users to test trading strategies using historical data
4. **User Experience**: Deliver a modern, professional interface with real-time data visualization
5. **Performance Tracking**: Track prediction accuracy over time and evaluate model performance

## Key Features

### Machine Learning Ensemble
- **7 Models Total**: Linear Regression, LSTM, ARIMA, XGBoost, Decision Tree, SVM, and Prophet
- **Weighted Voting**: Combines predictions based on individual model confidence scores
- **Adaptive Lag**: Adjusts historical data window based on timeframe (5m to 1M)

### Stock Coverage
- **S&P 500 Stocks**: Access to 500+ stocks
- **Multiple Timeframes**: 5m, 15m, 1H, 4H, 1D, 1W, 1M
- **Real-time Data**: Live data from Yahoo Finance via yahooquery

### Analysis Tools
- **Technical Indicators**: RSI, MACD, Moving Averages, Bollinger Bands
- **Candlestick Patterns**: Automatic detection of Doji, Hammer, Engulfing, etc.
- **Price Charts**: Professional TradingView-style candlestick charts
- **Prediction Visualization**: Historical data with future price predictions

### Backtesting Engine
- **Walk-forward Analysis**: Test strategies on historical data
- **Performance Metrics**: Sharpe Ratio, Win Rate, Max Drawdown
- **Equity Curve**: Visualize strategy performance over time

## Target Users

- Individual traders and investors
- Algorithmic trading enthusiasts
- Financial analysts
- Students learning about ML in finance

## Success Criteria

- Prediction accuracy: Target 40%+ (ideally 50%+)
- Model ensemble: All 7 models operational
- Real-time data: Reliable Yahoo Finance integration
- User experience: Intuitive, professional interface
- Performance: Fast predictions (<30 seconds per stock)

## Project Scope

### In Scope
- Stock price prediction using ML ensemble
- Technical analysis and indicators
- Backtesting capabilities
- Prediction history and accuracy tracking
- Modern web interface

### Out of Scope (Current Phase)
- User authentication (future enhancement)
- Payment processing
- Real-time trading execution
- Portfolio management
- Social features or community

## Constraints

- Data source: Yahoo Finance (free tier)
- Model training: Requires historical data (minimum 60 days)
- Prediction latency: 10-30 seconds per stock
- Storage: SQLite for Python backend, PostgreSQL for Node.js backend

