#!/usr/bin/env python3
"""
Script to retrain models for specific symbols and intervals.

Usage:
    python retrain_model.py AAPL 1d
    python retrain_model.py NVDA 1d
    python retrain_model.py TSLA 1d
    python retrain_model.py AAPL 1d --all-models  # Retrain all models, not just LSTM
"""

import sys
import os
import argparse
from yahooquery import Ticker
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from models.ensemble import EnsembleModel
from models.lstm_model import LSTMModel


def retrain_model(symbol: str, interval: str = "1d", all_models: bool = False, range_param: str = None):
    """
    Retrain model(s) for a specific symbol and interval.
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
        interval: Time interval (e.g., "1d", "1h", "15m")
        all_models: If True, retrain all ensemble models. If False, only retrain LSTM.
        range_param: Historical data range (default: auto-selected based on interval)
    """
    print(f"\n{'='*60}")
    print(f"Retraining model for {symbol} ({interval})")
    print(f"{'='*60}\n")
    
    # Auto-select range based on interval if not provided
    if range_param is None:
        if interval in ["5m", "15m"]:
            range_param = "5d"  # 5 days for 5m/15m (Yahoo limit)
        elif interval in ["30m", "1h"]:
            range_param = "60d"  # 60 days for hourly
        elif interval in ["4h", "2h"]:
            range_param = "60d"  # 60 days for 4h
        else:
            range_param = "1y"  # 1 year for daily/weekly/monthly
    
    # Fetch historical data
    print(f"Fetching historical data for {symbol} (range: {range_param})...")
    try:
        ticker = Ticker(symbol)
        df = ticker.history(period=range_param, interval=interval)
        
        # For intraday intervals, try longer ranges if initial fetch fails
        if df is None or df.empty:
            print(f"⚠️  Failed to fetch data with {range_param}, trying fallback...")
            if interval in ["5m", "15m", "30m", "1h"]:
                # Intraday intervals need more recent data, try 60 days
                df = ticker.history(period="60d", interval=interval)
            else:
                df = ticker.history(period="6mo", interval=interval)
        
        if df is None or df.empty:
            # Last resort: try daily data (will be converted/used if needed)
            if interval != "1d":
                print(f"⚠️  Trying daily interval as fallback...")
                df = ticker.history(period="3mo", interval="1d")
            else:
                df = ticker.history(period="3mo", interval="1d")
        
        if df is None or df.empty:
            print(f"❌ Error: Unable to fetch historical data for {symbol}")
            return False
        
        # Process DataFrame
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        df = df.reset_index()
        df.columns = [col.capitalize() for col in df.columns]
        
        closes = df["Close"].dropna().values
        
        # Calculate minimum required data points based on interval
        if interval in ["1h", "30m", "15m", "5m", "1m"]:
            min_lag = 20
        elif interval in ["4h", "2h"]:
            min_lag = 15
        elif interval in ["1wk"]:
            min_lag = 12
        elif interval in ["1mo"]:
            min_lag = 6
        else:  # 1d
            min_lag = 20
        
        # LSTM needs lag * 3 (up to 60) + 10 extra points
        min_required = min(70, min_lag * 3 + 10)
        
        if len(closes) < min_required:
            print(f"❌ Error: Only {len(closes)} data points available.")
            print(f"   Need at least {min_required} points for {interval} interval training.")
            print(f"   Try using a longer range (e.g., --range 60d for intraday intervals)")
            return False
        
        if len(closes) < 100:
            print(f"⚠️  Warning: Only {len(closes)} data points available. More data (100+) recommended for better training.")
        
        print(f"✓ Fetched {len(closes)} data points")
        print(f"  Price range: ${closes.min():.2f} - ${closes.max():.2f}")
        print(f"  Current price: ${closes[-1]:.2f}\n")
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return False
    
    # Retrain model(s)
    if all_models:
        print("Training all ensemble models...")
        ensemble = EnsembleModel(interval=interval)
        training_results = ensemble.train_all(
            closes, 
            symbol=symbol, 
            interval=interval, 
            force_retrain=True
        )
        
        print("\nTraining Results:")
        for model_name, result in training_results.items():
            if result.get("trained"):
                conf = result.get("confidence", 0)
                print(f"  ✓ {model_name}: confidence={conf:.3f}")
                if "metrics" in result:
                    metrics = result["metrics"]
                    if metrics:
                        print(f"    RMSE: {metrics.get('rmse', 'N/A')}, MAE: {metrics.get('mae', 'N/A')}")
            else:
                error = result.get("error", "Unknown error")
                print(f"  ✗ {model_name}: {error}")
        
        print(f"\n✓ All models retrained for {symbol} ({interval})")
        
    else:
        print("Training LSTM model only...")
        # For LSTM, we need to determine lag based on interval
        if interval in ["1h", "30m", "15m", "5m", "1m"]:
            lag = 20
        elif interval in ["4h", "2h"]:
            lag = 15
        elif interval in ["1wk"]:
            lag = 12
        elif interval in ["1mo"]:
            lag = 6
        else:  # 1d
            lag = 20
        
        lstm = LSTMModel(lag=min(60, lag * 3))
        confidence = lstm.train(closes, symbol=symbol, interval=interval, force_retrain=True)
        
        print(f"\n✓ LSTM model retrained for {symbol} ({interval})")
        print(f"  Confidence: {confidence:.3f}")
        if lstm.metrics:
            print(f"  RMSE: {lstm.metrics.get('rmse', 'N/A')}")
            print(f"  MAE: {lstm.metrics.get('mae', 'N/A')}")
        print(f"  Model saved to: {lstm._model_path(symbol, interval)}")
    
    print(f"\n{'='*60}")
    print("Retraining complete!")
    print(f"{'='*60}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Retrain ML models for stock predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Retrain LSTM only for AAPL daily
  python retrain_model.py AAPL 1d
  
  # Retrain all ensemble models for NVDA daily
  python retrain_model.py NVDA 1d --all-models
  
  # Retrain with more historical data
  python retrain_model.py TSLA 1d --range 2y
  
  # Retrain intraday intervals (use longer range for more data points)
  python retrain_model.py AAPL 15m --range 60d
  python retrain_model.py AAPL 1h --range 60d
        """
    )
    
    parser.add_argument("symbol", help="Stock symbol (e.g., AAPL, NVDA, TSLA)")
    parser.add_argument("interval", nargs="?", default="1d", help="Time interval (default: 1d)")
    parser.add_argument("--all-models", action="store_true", help="Retrain all ensemble models, not just LSTM")
    parser.add_argument("--range", default=None, help="Historical data range (default: auto-selected based on interval)")
    
    args = parser.parse_args()
    
    success = retrain_model(
        symbol=args.symbol.upper(),
        interval=args.interval,
        all_models=args.all_models,
        range_param=args.range
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

