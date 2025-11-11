"""
Prediction tracking and evaluation system.
Stores predictions, compares with actual prices, and calculates accuracy metrics.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from yahooquery import Ticker


DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")


def get_db_connection():
    """Get SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn


def init_db():
    """Initialize database tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            interval TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            current_price REAL NOT NULL,
            predicted_price REAL NOT NULL,
            confidence REAL NOT NULL,
            model_breakdown TEXT,
            actual_price REAL,
            evaluated BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Evaluations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            actual_price REAL NOT NULL,
            error REAL NOT NULL,
            error_percent REAL NOT NULL,
            direction_actual TEXT NOT NULL,
            direction_predicted TEXT NOT NULL,
            correct BOOLEAN NOT NULL,
            evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_evaluated ON predictions(evaluated)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_evaluations_prediction_id ON evaluations(prediction_id)')
    
    conn.commit()
    conn.close()


def save_prediction(
    symbol: str,
    interval: str,
    horizon: int,
    current_price: float,
    predicted_price: float,
    confidence: float,
    model_breakdown: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Save a prediction to the database.
    
    Returns:
        prediction_id: ID of the saved prediction
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    model_breakdown_json = json.dumps(model_breakdown) if model_breakdown else None
    
    cursor.execute('''
        INSERT INTO predictions 
        (symbol, timestamp, interval, horizon, current_price, predicted_price, confidence, model_breakdown, evaluated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
    ''', (
        symbol,
        datetime.utcnow().isoformat(),
        interval,
        horizon,
        current_price,
        predicted_price,
        confidence,
        model_breakdown_json,
    ))
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return prediction_id


def classify_direction(predicted_price: float, current_price: float, threshold: float = 0.001) -> str:
    """
    Classify prediction direction.
    
    Args:
        predicted_price: Predicted price
        current_price: Current price
        threshold: Threshold for classification (0.001 = 0.1%)
    
    Returns:
        "up", "down", or "neutral"
    """
    if predicted_price > current_price * (1 + threshold):
        return "up"
    elif predicted_price < current_price * (1 - threshold):
        return "down"
    else:
        return "neutral"


def evaluate_prediction(prediction_id: int, actual_price: float) -> Dict[str, Any]:
    """
    Evaluate a prediction by comparing with actual price.
    
    Args:
        prediction_id: ID of the prediction to evaluate
        actual_price: Actual market price
    
    Returns:
        Dictionary with evaluation results
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get prediction
    cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,))
    pred = cursor.fetchone()
    
    if not pred:
        conn.close()
        raise ValueError(f"Prediction {prediction_id} not found")
    
    current_price = pred["current_price"]
    predicted_price = pred["predicted_price"]
    
    # Calculate errors
    error = abs(predicted_price - actual_price)
    error_percent = (error / current_price) * 100 if current_price > 0 else 0
    
    # Classify directions
    threshold = 0.005  # 0.5% threshold - more realistic for stock volatility
    predicted_dir = classify_direction(predicted_price, current_price, threshold)
    actual_dir = classify_direction(actual_price, current_price, threshold)
    correct = predicted_dir == actual_dir
    
    # Save evaluation
    cursor.execute('''
        INSERT INTO evaluations 
        (prediction_id, actual_price, error, error_percent, direction_actual, direction_predicted, correct)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        prediction_id,
        actual_price,
        error,
        error_percent,
        actual_dir,
        predicted_dir,
        correct,
    ))
    
    # Update prediction
    cursor.execute('''
        UPDATE predictions 
        SET actual_price = ?, evaluated = 1
        WHERE id = ?
    ''', (actual_price, prediction_id))
    
    conn.commit()
    conn.close()
    
    return {
        "prediction_id": prediction_id,
        "actual_price": actual_price,
        "error": error,
        "error_percent": error_percent,
        "direction_actual": actual_dir,
        "direction_predicted": predicted_dir,
        "correct": correct,
    }


def evaluate_pending_predictions(symbol: Optional[str] = None, max_predictions: int = 100) -> Dict[str, Any]:
    """
    Evaluate pending predictions by fetching actual prices.
    
    Args:
        symbol: If provided, only evaluate predictions for this symbol
        max_predictions: Maximum number of predictions to evaluate
    
    Returns:
        Dictionary with evaluation statistics
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get pending predictions
    if symbol:
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE evaluated = 0 AND symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (symbol, max_predictions))
    else:
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE evaluated = 0
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (max_predictions,))
    
    predictions = cursor.fetchall()
    
    # Check if there are any predictions at all for this symbol
    if symbol:
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE symbol = ?', (symbol,))
        total_count = cursor.fetchone()[0]
    else:
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_count = cursor.fetchone()[0]
    
    conn.close()
    
    if not predictions:
        if symbol and total_count == 0:
            return {
                "evaluated": 0,
                "errors": 0,
                "total": 0,
                "message": f"No predictions found for {symbol}. Make predictions first from the Dashboard."
            }
        elif symbol and total_count > 0:
            return {
                "evaluated": 0,
                "errors": 0,
                "total": total_count,
                "message": f"All {total_count} prediction(s) for {symbol} are already evaluated."
            }
        else:
            return {
                "evaluated": 0,
                "errors": 0,
                "total": total_count,
                "message": "No pending predictions to evaluate"
            }
    
    evaluated_count = 0
    error_count = 0
    
    # Group by symbol for batch fetching
    symbols_to_fetch = list(set([p["symbol"] for p in predictions]))
    
    # Fetch current prices for all symbols
    ticker = Ticker(symbols_to_fetch)
    quotes = ticker.price
    
    for pred in predictions:
        try:
            symbol = pred["symbol"]
            quote_data = quotes.get(symbol, {})
            actual_price = quote_data.get("regularMarketPrice")
            
            if actual_price is None:
                continue
            
            # For horizon > 1, we'd need historical data, but for now evaluate as if horizon=1
            # TODO: For multi-period predictions, fetch historical price at the predicted time
            evaluate_prediction(pred["id"], float(actual_price))
            evaluated_count += 1
            
        except Exception as e:
            print(f"Error evaluating prediction {pred['id']}: {e}")
            error_count += 1
    
    return {
        "evaluated": evaluated_count,
        "errors": error_count,
        "total": len(predictions)
    }


def get_prediction_stats(symbol: Optional[str] = None, interval: Optional[str] = None) -> Dict[str, Any]:
    """
    Get prediction performance statistics.
    
    Args:
        symbol: Filter by symbol
        interval: Filter by interval
    
    Returns:
        Dictionary with statistics
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Build query
    query = '''
        SELECT 
            p.*,
            e.actual_price,
            e.error,
            e.error_percent,
            e.direction_actual,
            e.direction_predicted,
            e.correct
        FROM predictions p
        JOIN evaluations e ON p.id = e.prediction_id
    '''
    params = []
    
    conditions = []
    if symbol:
        conditions.append("p.symbol = ?")
        params.append(symbol)
    if interval:
        conditions.append("p.interval = ?")
        params.append(interval)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return {
            "total": 0,
            "direction_accuracy": 0,
            "avg_error": 0,
            "avg_error_percent": 0,
            "rmse": 0,
            "mae": 0,
        }
    
    # Calculate statistics
    total = len(rows)
    correct_directions = sum(1 for row in rows if row["correct"])
    direction_accuracy = (correct_directions / total) * 100 if total > 0 else 0
    
    errors = [row["error"] for row in rows]
    error_percents = [row["error_percent"] for row in rows]
    
    avg_error = np.mean(errors) if errors else 0
    avg_error_percent = np.mean(error_percents) if error_percents else 0
    rmse = np.sqrt(np.mean([e**2 for e in errors])) if errors else 0
    mae = np.mean(np.abs(errors)) if errors else 0
    
    return {
        "total": total,
        "direction_accuracy": round(direction_accuracy, 2),
        "avg_error": round(avg_error, 2),
        "avg_error_percent": round(avg_error_percent, 2),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
    }


def get_prediction_history(
    symbol: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    interval: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get prediction history with evaluations.
    
    Args:
        symbol: Filter by symbol
        limit: Maximum number of results
        offset: Offset for pagination
        interval: Filter by interval
    
    Returns:
        List of prediction dictionaries with evaluations
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = '''
        SELECT 
            p.*,
            e.actual_price,
            e.error,
            e.error_percent,
            e.direction_actual,
            e.direction_predicted,
            e.correct,
            e.evaluated_at
        FROM predictions p
        LEFT JOIN evaluations e ON p.id = e.prediction_id
    '''
    params = []
    
    conditions = []
    if symbol:
        conditions.append("p.symbol = ?")
        params.append(symbol)
    if interval:
        conditions.append("p.interval = ?")
        params.append(interval)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY p.timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to dictionaries
    results = []
    for row in rows:
        result = {
            "id": row["id"],
            "symbol": row["symbol"],
            "timestamp": row["timestamp"],
            "interval": row["interval"],
            "horizon": row["horizon"],
            "current_price": row["current_price"],
            "predicted_price": row["predicted_price"],
            "confidence": row["confidence"],
            "evaluated": bool(row["evaluated"]),
        }
        
        if row["model_breakdown"]:
            try:
                result["model_breakdown"] = json.loads(row["model_breakdown"])
            except:
                result["model_breakdown"] = None
        
        if row["actual_price"] is not None:
            result["actual_price"] = row["actual_price"]
            result["error"] = row["error"]
            result["error_percent"] = row["error_percent"]
            result["direction_actual"] = row["direction_actual"]
            result["direction_predicted"] = row["direction_predicted"]
            result["correct"] = bool(row["correct"])
            result["evaluated_at"] = row["evaluated_at"]
        
        results.append(result)
    
    return results

