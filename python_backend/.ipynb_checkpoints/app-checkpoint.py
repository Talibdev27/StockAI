import os
import time
from typing import Dict, Any, List

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from yahooquery import Ticker
from sklearn.linear_model import LinearRegression


app = Flask(__name__)
CORS(app)

# yahooquery handles headers and caching automatically

_cache: Dict[str, Dict[str, Any]] = {}


def _cache_get(key: str, ttl_seconds: int) -> Any:
    entry = _cache.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > ttl_seconds:
        return None
    return entry["value"]


def _cache_set(key: str, value: Any):
    _cache[key] = {"value": value, "ts": time.time()}


@app.get("/api/stocks")
def stocks():
    return jsonify([
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "GOOGL", "name": "Alphabet Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corp."},
        {"symbol": "TSLA", "name": "Tesla, Inc."},
        {"symbol": "AMZN", "name": "Amazon.com, Inc."},
    ])


@app.get("/api/quote/<symbol>")
def quote(symbol: str):
    cache_key = f"quote:{symbol}"
    cached = _cache_get(cache_key, ttl_seconds=30)  # 30 second cache for live quotes
    if cached is not None:
        return jsonify(cached)
    
    try:
        ticker = Ticker(symbol)
        quote_data = ticker.price[symbol]
        
        result = {
            "symbol": symbol,
            "price": quote_data.get("regularMarketPrice"),
            "previousClose": quote_data.get("regularMarketPreviousClose"),
            "currency": quote_data.get("currency", "USD"),
            "change": quote_data.get("regularMarketChange"),
            "changePercent": quote_data.get("regularMarketChangePercent")
        }
        
        _cache_set(cache_key, result)
        return jsonify(result)
    except Exception as e:
        print(f"Quote fetch failed for {symbol}: {e}")
        return jsonify({"error": f"Unable to fetch quote for {symbol}"}), 503


@app.get("/api/historical/<symbol>")
def historical(symbol: str):
    range_param = request.args.get("range", "1y")
    interval = request.args.get("interval", "1d")

    cache_key = f"hist:{symbol}:{range_param}:{interval}"
    cached = _cache_get(cache_key, ttl_seconds=60 * 10)
    if cached is not None:
        return jsonify(cached)

    # Use yahooquery which is more reliable than yfinance
    df = pd.DataFrame()
    try:
        ticker = Ticker(symbol)
        df = ticker.history(period=range_param, interval=interval)
    except Exception as e:
        print(f"yahooquery failed: {e}")
    
    if df is None or df.empty:
        try:
            # Try with shorter period as fallback
            time.sleep(0.5)
            ticker = Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")
        except Exception as e:
            print(f"3mo fallback failed: {e}")
    
    if df is None or df.empty:
        try:
            # Try with minimal period
            time.sleep(0.5)
            ticker = Ticker(symbol)
            df = ticker.history(period="1mo", interval="1d")
        except Exception as e:
            print(f"1mo fallback failed: {e}")
    
    # No mock data - only real Yahoo Finance or error
    if df is None or df.empty:
        return jsonify({"error": "Unable to fetch real market data. Please try again later."}), 503
    
    # yahooquery returns MultiIndex DF - flatten it
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    df = df.reset_index()
    
    # yahooquery uses lowercase column names - capitalize them
    df.columns = [col.capitalize() for col in df.columns]
    
    # Standardize keys for client
    # For intraday (1h, 4h), include time; for daily+, just date
    date_key = "Datetime" if "Datetime" in df.columns else "Date"
    date_format = "%Y-%m-%d %H:%M:%S" if interval in ["1h", "2h", "4h", "30m", "15m", "5m", "1m"] else "%Y-%m-%d"
    
    records: List[Dict[str, Any]] = [
        {
            "date": pd.to_datetime(row[date_key]).strftime(date_format),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
        }
        for _, row in df.iterrows()
    ]

    _cache_set(cache_key, records)
    return jsonify(records)


def _train_linear_regression(series: pd.Series, lag: int = 10) -> Dict[str, Any]:
    values = series.astype(float).values
    if len(values) <= lag + 1:
        raise ValueError("Not enough data to train model")

    X, y = [], []
    for i in range(lag, len(values)):
        X.append(values[i - lag : i])
        y.append(values[i])
    X_arr = np.array(X)
    y_arr = np.array(y)

    model = LinearRegression()
    model.fit(X_arr, y_arr)

    # Next-step prediction using last window
    last_window = values[-lag:]
    next_pred = float(model.predict(last_window.reshape(1, -1))[0])

    # Naive confidence proxy based on R^2 clipped to [0, 1]
    r2 = float(max(0.0, min(1.0, model.score(X_arr, y_arr))))

    return {"model": model, "next_pred": next_pred, "confidence": r2}


@app.get("/api/predict/<symbol>")
def predict(symbol: str):
    horizon = int(request.args.get("horizon", 5))
    range_param = request.args.get("range", "1y")
    interval = request.args.get("interval", "1d")

    cache_key = f"pred:{symbol}:{range_param}:{interval}:{horizon}"
    cached = _cache_get(cache_key, ttl_seconds=60 * 5)
    if cached is not None:
        return jsonify(cached)

    try:
        ticker = Ticker(symbol)
        df = ticker.history(period=range_param, interval=interval)
        if df is None or df.empty:
            df = ticker.history(period="6mo", interval=interval)
        if df is None or df.empty:
            df = ticker.history(period="30d", interval="1d")
    except Exception as e:
        print(f"yahooquery prediction failed: {e}")
        df = pd.DataFrame()
    
    # No mock data - only real Yahoo Finance or error
    if df is None or df.empty:
        return jsonify({"error": "Unable to fetch real market data for prediction. Please try again later."}), 503
    
    # yahooquery returns MultiIndex DF - flatten it
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    df = df.reset_index()
    
    # yahooquery uses lowercase column names - capitalize them
    df.columns = [col.capitalize() for col in df.columns]
    
    closes = df["Close"].dropna()

    result = {"symbol": symbol, "predictedPrice": None, "confidence": None, "forecast": []}

    try:
        trained = _train_linear_regression(closes, lag=min(20, len(closes) - 1))
        result["predictedPrice"] = round(trained["next_pred"], 2)
        result["confidence"] = round(trained["confidence"] * 100, 1)

        # Produce simple recursive forecast by rolling window
        window = closes.values[-min(20, len(closes) - 1):].astype(float)
        model: LinearRegression = trained["model"]
        forecast_vals: List[float] = []
        for _ in range(horizon):
            pred = float(model.predict(window.reshape(1, -1))[0])
            forecast_vals.append(pred)
            window = np.append(window[1:], pred)

        result["forecast"] = [round(x, 2) for x in forecast_vals]
    except Exception as e:
        return jsonify({"message": str(e)}), 500

    _cache_set(cache_key, result)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)


