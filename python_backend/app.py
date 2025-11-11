import os
import time
import json
from typing import Dict, Any, List

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from yahooquery import Ticker
from models.ensemble import EnsembleModel
from backtest import BacktestEngine
from evaluation import save_prediction, evaluate_pending_predictions, get_prediction_history, get_prediction_stats

# Technical indicators
try:
    import ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False
    print("Warning: 'ta' library not available. Technical indicators will not work.")


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


def _calculate_indicators(df: pd.DataFrame, interval: str = "1d") -> Dict[str, Any]:
    """
    Calculate technical indicators from historical data.
    
    Args:
        df: DataFrame with columns: Date, Open, High, Low, Close, Volume
        interval: Data interval for adaptive periods
        
    Returns:
        Dictionary with calculated indicators
    """
    if not _HAS_TA:
        return {}
    
    if df.empty or len(df) < 20:
        return {}
    
    # Create a clean copy with timezone-naive data
    df_clean = df.copy()
    
    # Ensure Close column is numeric
    df_clean["Close"] = pd.to_numeric(df_clean["Close"], errors="coerce")
    df_clean = df_clean.dropna(subset=["Close"])
    
    if len(df_clean) < 20:
        return {}
    
    # Ensure Close Series is timezone-naive by creating a fresh Series
    close_series = pd.Series(df_clean["Close"].values, name="Close")
    close_series.index = pd.RangeIndex(len(close_series))
    
    # Adaptive periods based on interval
    if interval in ["5m", "15m", "1h"]:
        ma_periods = [20, 50, 100]
    elif interval in ["1wk", "1mo"]:
        ma_periods = [10, 20, 50]
    else:  # daily
        ma_periods = [20, 50, 200]
    
    volumes = df_clean["Volume"].values if "Volume" in df_clean.columns else None
    
    indicators = {}
    
    # RSI
    try:
        rsi_indicator = ta.momentum.RSIIndicator(close_series, window=14)
        rsi_values = rsi_indicator.rsi()
        rsi = float(rsi_values.iloc[-1]) if not rsi_values.empty else None
        
        if rsi is not None:
            if rsi > 70:
                signal = "overbought"
            elif rsi < 30:
                signal = "oversold"
            else:
                signal = "neutral"
            
            indicators["rsi"] = {
                "value": round(rsi, 2),
                "signal": signal,
                "period": 14
            }
    except Exception as e:
        print(f"RSI calculation error: {e}")
    
    # MACD
    try:
        macd_indicator = ta.trend.MACD(close_series, window_fast=12, window_slow=26, window_sign=9)
        macd_line = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()
        histogram = macd_indicator.macd_diff()
        
        if not macd_line.empty and not signal_line.empty:
            macd_val = float(macd_line.iloc[-1])
            signal_val = float(signal_line.iloc[-1])
            hist_val = float(histogram.iloc[-1]) if not histogram.empty else (macd_val - signal_val)
            
            if macd_val > signal_val:
                trend = "bullish"
            elif macd_val < signal_val:
                trend = "bearish"
            else:
                trend = "neutral"
            
            indicators["macd"] = {
                "macd": round(macd_val, 4),
                "signal": round(signal_val, 4),
                "histogram": round(hist_val, 4),
                "trend": trend
            }
    except Exception as e:
        print(f"MACD calculation error: {e}")
    
    # Moving Averages (SMA)
    try:
        sma_dict = {}
        for period in ma_periods:
            if len(close_series) >= period:
                sma_indicator = ta.trend.SMAIndicator(close_series, window=period)
                sma_values = sma_indicator.sma_indicator()
                if not sma_values.empty:
                    sma_dict[f"sma{period}"] = round(float(sma_values.iloc[-1]), 2)
        
        if sma_dict:
            indicators["movingAverages"] = sma_dict
    except Exception as e:
        print(f"SMA calculation error: {e}")
    
    # Bollinger Bands
    try:
        if len(close_series) >= 20:
            bb_indicator = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
            bb_high = bb_indicator.bollinger_hband()
            bb_low = bb_indicator.bollinger_lband()
            bb_mid = bb_indicator.bollinger_mavg()
            
            if not bb_high.empty and not bb_low.empty and not bb_mid.empty:
                upper = float(bb_high.iloc[-1])
                lower = float(bb_low.iloc[-1])
                middle = float(bb_mid.iloc[-1])
                current_price = float(close_series.iloc[-1])
                
                width = ((upper - lower) / middle) * 100 if middle > 0 else 0
                
                # Determine position
                if current_price >= upper * 0.98:
                    position = "upper"
                elif current_price <= lower * 1.02:
                    position = "lower"
                else:
                    position = "middle"
                
                indicators["bollingerBands"] = {
                    "upper": round(upper, 2),
                    "middle": round(middle, 2),
                    "lower": round(lower, 2),
                    "width": round(width, 2),
                    "position": position
                }
    except Exception as e:
        print(f"Bollinger Bands calculation error: {e}")
    
    # Volume
    try:
        if volumes is not None and len(volumes) > 0:
            current_volume = float(volumes[-1])
            avg_volume = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            indicators["volume"] = {
                "current": int(current_volume),
                "average": int(avg_volume),
                "ratio": round(volume_ratio, 2)
            }
    except Exception as e:
        print(f"Volume calculation error: {e}")
    
    return indicators


# Load S&P 500 stocks list
_SP500_STOCKS = None

def _load_sp500_stocks():
    """Load S&P 500 stocks from JSON file, cache in memory"""
    global _SP500_STOCKS
    if _SP500_STOCKS is None:
        file_path = os.path.join(os.path.dirname(__file__), "data", "sp500.json")
        try:
            with open(file_path, "r") as f:
                _SP500_STOCKS = json.load(f)
        except FileNotFoundError:
            # Fallback to hardcoded list if file not found
            _SP500_STOCKS = [
                {"symbol": "AAPL", "name": "Apple Inc."},
                {"symbol": "GOOGL", "name": "Alphabet Inc."},
                {"symbol": "MSFT", "name": "Microsoft Corp."},
                {"symbol": "TSLA", "name": "Tesla, Inc."},
                {"symbol": "AMZN", "name": "Amazon.com, Inc."},
            ]
    return _SP500_STOCKS


@app.get("/api/stocks")
def stocks():
    """Return list of S&P 500 stocks"""
    popular = request.args.get("popular", "").lower() == "true"
    all_stocks = _load_sp500_stocks()
    
    if popular:
        # Return top 20 most popular/well-known stocks
        popular_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", 
                          "BRK.B", "V", "JNJ", "WMT", "MA", "PG", "UNH", "HD", 
                          "DIS", "AVGO", "BAC", "ADBE", "COST"]
        popular_stocks = [s for s in all_stocks if s["symbol"] in popular_symbols]
        # Ensure we have exactly popular_symbols in that order
        result = []
        for sym in popular_symbols:
            found = next((s for s in popular_stocks if s["symbol"] == sym), None)
            if found:
                result.append(found)
        return jsonify(result)
    
    return jsonify(all_stocks)


@app.get("/api/indicators/<symbol>")
def indicators(symbol: str):
    """Calculate and return technical indicators for a stock."""
    range_param = request.args.get("range", "1y")
    interval = request.args.get("interval", "1d")
    
    cache_key = f"indicators:{symbol}:{range_param}:{interval}"
    cached = _cache_get(cache_key, ttl_seconds=60 * 5)  # 5 minute cache
    if cached is not None:
        return jsonify(cached)
    
    try:
        # Fetch historical data
        ticker = Ticker(symbol)
        df = ticker.history(period=range_param, interval=interval)
        
        if df is None or df.empty:
            # Try fallback
            df = ticker.history(period="6mo", interval=interval)
        if df is None or df.empty:
            df = ticker.history(period="3mo", interval="1d")
        
        if df is None or df.empty:
            return jsonify({"error": "Unable to fetch historical data for indicators."}), 503
        
        # Process DataFrame
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        df = df.reset_index()
        df.columns = [col.capitalize() for col in df.columns]
        
        # Convert datetime columns to timezone-naive if needed
        date_key = "Datetime" if "Datetime" in df.columns else "Date"
        if date_key in df.columns:
            df[date_key] = pd.to_datetime(df[date_key], utc=False)
            # Remove timezone info if present
            if df[date_key].dt.tz is not None:
                df[date_key] = df[date_key].dt.tz_localize(None)
        
        # Ensure Close column is numeric and timezone-naive
        if "Close" in df.columns:
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Close"])
        
        if "Volume" in df.columns:
            df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        
        if len(df) < 20:
            return jsonify({"error": "Insufficient data. Need at least 20 data points."}), 400
        
        # Calculate indicators
        calculated_indicators = _calculate_indicators(df, interval)
        
        if not calculated_indicators:
            return jsonify({"error": "Unable to calculate indicators. 'ta' library may not be available."}), 503
        
        # Get last date
        last_date = df[date_key].iloc[-1].strftime("%Y-%m-%d") if date_key in df.columns else ""
        
        result = {
            "symbol": symbol,
            "lastUpdated": last_date,
            "indicators": calculated_indicators
        }
        
        _cache_set(cache_key, result)
        return jsonify(result)
        
    except Exception as e:
        print(f"Indicators calculation failed for {symbol}: {e}")
        return jsonify({"error": f"Failed to calculate indicators: {str(e)}"}), 500


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
        
        # Special handling for 4h interval - Yahoo doesn't reliably support it
        if interval == "4h":
            # Try 4h first, then fallback to 1h with longer range
            df = ticker.history(period=range_param, interval="4h")
            if df is None or df.empty:
                print("4h interval failed, trying 1h with longer range")
                df = ticker.history(period="1mo", interval="1h")
        else:
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
    
    closes = df["Close"].dropna().values

    try:
        # Use ensemble model
        ensemble = EnsembleModel(interval=interval)
        
        # Train all models
        training_results = ensemble.train_all(closes, symbol=symbol, interval=interval, force_retrain=False)
        
        # Get ensemble prediction
        result = ensemble.predict_ensemble(closes, horizon)
        result["symbol"] = symbol
        result["training"] = training_results  # Optional: for debugging
        if getattr(ensemble, "warnings", None):
            result["warnings"] = list(dict.fromkeys(ensemble.warnings))
        
        # Get current price for saving prediction
        current_price = float(closes[-1]) if len(closes) > 0 else 0.0
        predicted_price = result.get("predictedPrice", 0.0)
        confidence = result.get("confidence", 0.0)
        model_breakdown = result.get("models", {})
        
        # Save prediction to database (only save next-period prediction, horizon=1)
        if horizon >= 1:
            try:
                prediction_id = save_prediction(
                    symbol=symbol,
                    interval=interval,
                    horizon=1,  # Save as next-period prediction
                    current_price=current_price,
                    predicted_price=predicted_price,
                    confidence=confidence,
                    model_breakdown=model_breakdown,
                )
                result["prediction_id"] = prediction_id
            except Exception as e:
                print(f"Warning: Failed to save prediction: {e}")
                # Don't fail the request if saving fails
        
    except ValueError as e:
        return jsonify({
            "error": "Insufficient data for ensemble predictions.",
            "details": str(e)
        }), 400
    except Exception as e:
        return jsonify({"message": str(e)}), 500

    _cache_set(cache_key, result)
    return jsonify(result)


@app.get("/api/backtest/<symbol>")
def backtest(symbol: str):
    """Run backtest simulation on historical data."""
    # Get parameters
    range_param = request.args.get("range", "1y")
    interval = request.args.get("interval", "1d")
    strategy = request.args.get("strategy", "simple_signals")
    initial_capital = float(request.args.get("initial_capital", 10000))
    commission = float(request.args.get("commission", 0.001))
    position_size = float(request.args.get("position_size", 1.0))
    threshold = float(request.args.get("threshold", 0.0))
    
    # Validate strategy
    valid_strategies = ["simple_signals", "threshold", "momentum"]
    if strategy not in valid_strategies:
        return jsonify({"error": f"Invalid strategy. Must be one of: {valid_strategies}"}), 400
    
    # Limit range for performance (max 5 years)
    max_ranges = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}
    if range_param not in max_ranges:
        range_param = "1y"  # Default
    
    try:
        # Fetch historical data
        ticker = Ticker(symbol)
        df = ticker.history(period=range_param, interval=interval)
        
        if df is None or df.empty:
            # Try fallbacks
            if range_param == "1y":
                df = ticker.history(period="6mo", interval=interval)
            if df is None or df.empty:
                df = ticker.history(period="3mo", interval=interval)
        
        if df is None or df.empty:
            return jsonify({"error": "Unable to fetch historical data for backtest."}), 503
        
        # Process DataFrame
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        df = df.reset_index()
        df.columns = [col.capitalize() for col in df.columns]
        
        # Get closes and dates
        closes = df["Close"].dropna().values
        
        if len(closes) < 50:
            return jsonify({"error": "Insufficient historical data. Need at least 50 data points."}), 400
        
        # Format dates
        date_key = "Datetime" if "Datetime" in df.columns else "Date"
        dates = pd.to_datetime(df[date_key]).dt.strftime("%Y-%m-%d").tolist()
        
        # Generate predictions for each point using walk-forward analysis
        predictions = []
        horizon = 1  # We only need next period prediction
        
        print(f"Running backtest for {symbol} with {len(closes)} data points...")
        
        # Walk-forward: train on data up to point i, predict for i+1
        for i in range(20, len(closes) - 1):  # Start from index 20 to have enough training data
            try:
                # Use data up to current point for training
                training_data = closes[:i+1]
                
                # Train ensemble
                ensemble = EnsembleModel(interval=interval)
                ensemble.train_all(training_data, symbol=symbol, interval=interval, force_retrain=False)
                
                # Predict next value
                result = ensemble.predict_ensemble(training_data, horizon)
                next_pred = result["predictedPrice"]
                predictions.append(next_pred)
                
            except Exception:
                # If prediction fails, use current price as fallback
                predictions.append(float(closes[i]))
        
        # Pad predictions (use last value for remaining points)
        while len(predictions) < len(closes):
            predictions.append(float(closes[len(predictions)]))
        
        # Align predictions with closes (predictions[i] predicts closes[i+1])
        # For backtest, we'll use prediction[i] to decide action at closes[i]
        aligned_predictions = [float(closes[0])]  # First prediction = first price
        for i in range(min(len(predictions), len(closes) - 1)):
            aligned_predictions.append(predictions[i])
        
        # Ensure same length
        min_len = min(len(closes), len(dates), len(aligned_predictions))
        closes = closes[:min_len]
        dates = dates[:min_len]
        aligned_predictions = aligned_predictions[:min_len]
        
        # Run backtest
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=initial_capital,
            commission=commission,
            position_size=position_size,
            threshold=threshold,
        )
        
        results = engine.run_backtest(
            np.array(closes),
            dates,
            aligned_predictions,
            interval=interval,
        )
        results["symbol"] = symbol
        
        return jsonify(results)
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Backtest error: {e}")
        return jsonify({"error": f"Backtest failed: {str(e)}"}), 500


@app.get("/api/predictions/evaluate")
def evaluate_predictions():
    """Evaluate pending predictions by comparing with actual prices."""
    symbol = request.args.get("symbol", None)
    max_predictions = int(request.args.get("max", 100))
    
    try:
        result = evaluate_pending_predictions(symbol=symbol, max_predictions=max_predictions)
        return jsonify(result)
    except Exception as e:
        print(f"Evaluation error: {e}")
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500


@app.get("/api/predictions/history")
@app.get("/api/predictions/history/<symbol>")
def prediction_history(symbol: str = None):
    """Get prediction history with evaluations."""
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    interval = request.args.get("interval", None)
    
    try:
        history = get_prediction_history(symbol=symbol, limit=limit, offset=offset, interval=interval)
        return jsonify(history)
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500


@app.get("/api/predictions/stats")
@app.get("/api/predictions/stats/<symbol>")
def prediction_stats(symbol: str = None):
    """Get prediction performance statistics."""
    interval = request.args.get("interval", None)
    
    try:
        stats = get_prediction_stats(symbol=symbol, interval=interval)
        return jsonify(stats)
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({"error": f"Failed to fetch stats: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)


