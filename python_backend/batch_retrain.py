#!/usr/bin/env python3
"""
Batch retrain models for S&P 500 stocks.

Usage:
    python batch_retrain.py                    # Retrain all S&P 500 stocks
    python batch_retrain.py --limit 10        # Retrain first 10 stocks
    python batch_retrain.py --symbols AAPL,MSFT,GOOGL  # Retrain specific stocks
    python batch_retrain.py --interval 1h      # Retrain for 1h interval
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from retrain_model import retrain_model


def load_sp500_stocks():
    """Load S&P 500 stocks list."""
    file_path = os.path.join(os.path.dirname(__file__), "data", "sp500.json")
    try:
        with open(file_path, "r") as f:
            stocks = json.load(f)
        return stocks
    except FileNotFoundError:
        # Fallback to popular stocks if file not found
        return [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corp."},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com, Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corp."},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "TSLA", "name": "Tesla, Inc."},
        ]


def batch_retrain(
    symbols: list = None,
    interval: str = "1d",
    range_param: str = "60d",  # 2 months
    limit: int = None,
    all_models: bool = False,
    delay: float = 2.0,  # Delay between stocks to avoid rate limiting
):
    """
    Batch retrain models for multiple stocks.
    
    Args:
        symbols: List of stock symbols (if None, uses all S&P 500)
        interval: Time interval (default: "1d")
        range_param: Historical data range (default: "60d" = 2 months)
        limit: Maximum number of stocks to process (None = all)
        all_models: If True, retrain all ensemble models
        delay: Seconds to wait between stocks
    """
    # Load stock list
    if symbols is None:
        stocks = load_sp500_stocks()
        symbols = [s["symbol"] for s in stocks]
    
    # Apply limit if specified
    if limit:
        symbols = symbols[:limit]
    
    total = len(symbols)
    print(f"\n{'='*70}")
    print(f"Batch Retraining: {total} stocks")
    print(f"Interval: {interval}, Range: {range_param}")
    print(f"Mode: {'All models' if all_models else 'LSTM only'}")
    print(f"{'='*70}\n")
    
    results = {
        "success": [],
        "failed": [],
        "skipped": [],
    }
    
    start_time = time.time()
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{total}] Processing {symbol}...")
        print("-" * 70)
        
        try:
            success = retrain_model(
                symbol=symbol,
                interval=interval,
                all_models=all_models,
                range_param=range_param,
            )
            
            if success:
                results["success"].append(symbol)
                print(f"✓ {symbol} completed successfully")
            else:
                results["skipped"].append(symbol)
                print(f"⚠ {symbol} skipped (insufficient data or other issue)")
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Batch retraining interrupted by user")
            print(f"Processed {i-1}/{total} stocks before interruption")
            break
        
        except Exception as e:
            results["failed"].append((symbol, str(e)))
            print(f"✗ {symbol} failed: {e}")
        
        # Delay between stocks to avoid rate limiting
        if i < total:
            print(f"Waiting {delay}s before next stock...")
            time.sleep(delay)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("BATCH RETRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Total stocks: {total}")
    print(f"✓ Successful: {len(results['success'])}")
    print(f"⚠ Skipped: {len(results['skipped'])}")
    print(f"✗ Failed: {len(results['failed'])}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"{'='*70}\n")
    
    if results["success"]:
        print("Successfully retrained:")
        print(f"  {', '.join(results['success'][:20])}")
        if len(results["success"]) > 20:
            print(f"  ... and {len(results['success']) - 20} more")
        print()
    
    if results["failed"]:
        print("Failed stocks:")
        for symbol, error in results["failed"][:10]:
            print(f"  {symbol}: {error[:80]}")
        if len(results["failed"]) > 10:
            print(f"  ... and {len(results['failed']) - 10} more failures")
        print()
    
    if results["skipped"]:
        print("Skipped stocks (insufficient data):")
        print(f"  {', '.join(results['skipped'][:20])}")
        if len(results["skipped"]) > 20:
            print(f"  ... and {len(results['skipped']) - 20} more")
        print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch retrain models for S&P 500 stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Retrain all S&P 500 stocks (daily, 2 months data, LSTM only)
  python batch_retrain.py
  
  # Retrain first 20 stocks
  python batch_retrain.py --limit 20
  
  # Retrain specific stocks
  python batch_retrain.py --symbols AAPL,MSFT,GOOGL,NVDA,TSLA
  
  # Retrain all models (not just LSTM)
  python batch_retrain.py --limit 10 --all-models
  
  # Retrain for hourly interval
  python batch_retrain.py --interval 1h --limit 10
  
  # Use 3 months of data
  python batch_retrain.py --range 90d --limit 20
        """
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL)"
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Time interval (default: 1d)"
    )
    parser.add_argument(
        "--range",
        default="60d",
        help="Historical data range (default: 60d = 2 months)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of stocks to process"
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Retrain all ensemble models (not just LSTM)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between stocks (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Run batch retraining
    results = batch_retrain(
        symbols=symbols,
        interval=args.interval,
        range_param=args.range,
        limit=args.limit,
        all_models=args.all_models,
        delay=args.delay,
    )
    
    # Exit with error code if all failed
    if len(results["success"]) == 0 and len(results["failed"]) > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

