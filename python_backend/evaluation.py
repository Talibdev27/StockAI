"""
Prediction tracking and evaluation system.
Stores predictions, compares with actual prices, and calculates accuracy metrics.

Supports both PostgreSQL (production) and SQLite (local development).
Uses DATABASE_URL environment variable to determine which database to use.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from yahooquery import Ticker

# Database connection setup
DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = DATABASE_URL is not None and DATABASE_URL.startswith("postgres")

if USE_POSTGRES:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor, execute_values
        print("Using PostgreSQL database")
    except ImportError:
        print("Warning: psycopg2 not installed, falling back to SQLite")
        USE_POSTGRES = False
        import sqlite3
        DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")
else:
    import sqlite3
    DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")
    print("Using SQLite database (local development)")


def get_db_connection():
    """Get database connection (PostgreSQL or SQLite)."""
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def _get_placeholder():
    """Get placeholder for parameterized queries."""
    return "%s" if USE_POSTGRES else "?"


# Interval duration mapping for evaluation timing
INTERVAL_DURATIONS = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "1wk": timedelta(weeks=1),
    "1mo": timedelta(days=30),  # Approximate month
}


def get_interval_duration(interval: str) -> timedelta:
    """
    Get the time duration for a given interval.
    
    Args:
        interval: Interval string (e.g., "1d", "1h", "15m")
    
    Returns:
        timedelta representing the duration
    """
    return INTERVAL_DURATIONS.get(interval, timedelta(days=1))  # Default to 1 day


def is_prediction_ready_for_evaluation(prediction: Dict[str, Any]) -> bool:
    """
    Check if a prediction is ready for evaluation based on its timestamp and interval.
    
    A prediction is ready if enough time has passed since it was made.
    For example, a "1d" prediction should only be evaluated after 1 day has passed.
    
    Args:
        prediction: Dictionary containing prediction data with 'timestamp' and 'interval' keys
    
    Returns:
        True if prediction is ready for evaluation, False otherwise
    """
    timestamp = prediction.get("timestamp")
    interval = prediction.get("interval")
    
    if not timestamp or not interval:
        return False
    
    # Parse timestamp if it's a string
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            try:
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return False
    
    # Get the duration for this interval
    duration = get_interval_duration(interval)
    
    # Calculate when this prediction should be evaluated
    evaluation_time = timestamp + duration
    
    # Check if enough time has passed
    now = datetime.now()
    
    # Handle timezone-aware timestamps
    if timestamp.tzinfo is not None and now.tzinfo is None:
        now = now.replace(tzinfo=timestamp.tzinfo)
    elif timestamp.tzinfo is None and now.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=now.tzinfo)
    
    return evaluation_time <= now


def _row_to_dict(row):
    """Convert database row to dictionary."""
    if USE_POSTGRES:
        return dict(row)
    else:
        return dict(row)


def init_db():
    """Initialize database tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if USE_POSTGRES:
        # PostgreSQL schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                interval VARCHAR(10) NOT NULL,
                horizon INTEGER NOT NULL,
                current_price DOUBLE PRECISION NOT NULL,
                predicted_price DOUBLE PRECISION NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                model_breakdown TEXT,
                actual_price DOUBLE PRECISION,
                evaluated BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id SERIAL PRIMARY KEY,
                prediction_id INTEGER NOT NULL,
                actual_price DOUBLE PRECISION NOT NULL,
                error DOUBLE PRECISION NOT NULL,
                error_percent DOUBLE PRECISION NOT NULL,
                direction_actual VARCHAR(10) NOT NULL,
                direction_predicted VARCHAR(10) NOT NULL,
                correct BOOLEAN NOT NULL,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
            )
        ''')
    else:
        # SQLite schema
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
    placeholder = _get_placeholder()
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_predictions_evaluated ON predictions(evaluated)",
        "CREATE INDEX IF NOT EXISTS idx_evaluations_prediction_id ON evaluations(prediction_id)",
    ]
    
    for index_sql in indexes:
        try:
            cursor.execute(index_sql)
        except Exception as e:
            print(f"Warning: Could not create index: {e}")
    
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
    placeholder = _get_placeholder()
    
    if USE_POSTGRES:
        cursor.execute(f'''
            INSERT INTO predictions 
            (symbol, timestamp, interval, horizon, current_price, predicted_price, confidence, model_breakdown, evaluated)
            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, FALSE)
            RETURNING id
        ''', (
            symbol,
            datetime.utcnow(),
            interval,
            horizon,
            current_price,
            predicted_price,
            confidence,
            model_breakdown_json,
        ))
        prediction_id = cursor.fetchone()[0]
    else:
        cursor.execute(f'''
            INSERT INTO predictions 
            (symbol, timestamp, interval, horizon, current_price, predicted_price, confidence, model_breakdown, evaluated)
            VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, 0)
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
    
    placeholder = _get_placeholder()
    
    # Get prediction
    cursor.execute(f'SELECT * FROM predictions WHERE id = {placeholder}', (prediction_id,))
    pred_row = cursor.fetchone()
    
    if not pred_row:
        conn.close()
        raise ValueError(f"Prediction {prediction_id} not found")
    
    pred = _row_to_dict(pred_row)
    
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
    cursor.execute(f'''
        INSERT INTO evaluations 
        (prediction_id, actual_price, error, error_percent, direction_actual, direction_predicted, correct)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
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
    evaluated_value = True if USE_POSTGRES else 1
    cursor.execute(f'''
        UPDATE predictions 
        SET actual_price = {placeholder}, evaluated = {placeholder}
        WHERE id = {placeholder}
    ''', (actual_price, evaluated_value, prediction_id))
    
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
    
    Only evaluates predictions where enough time has passed based on their interval.
    For example, a "1d" prediction will only be evaluated after 1 day has passed.
    
    Args:
        symbol: If provided, only evaluate predictions for this symbol
        max_predictions: Maximum number of predictions to evaluate
    
    Returns:
        Dictionary with evaluation statistics
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    placeholder = _get_placeholder()
    evaluated_value = False if USE_POSTGRES else 0
    
    # Get pending predictions
    if symbol:
        cursor.execute(f'''
            SELECT * FROM predictions 
            WHERE evaluated = {placeholder} AND symbol = {placeholder}
            ORDER BY timestamp DESC
            LIMIT {placeholder}
        ''', (evaluated_value, symbol, max_predictions))
    else:
        cursor.execute(f'''
            SELECT * FROM predictions 
            WHERE evaluated = {placeholder}
            ORDER BY timestamp DESC
            LIMIT {placeholder}
        ''', (evaluated_value, max_predictions))
    
    pred_rows = cursor.fetchall()
    predictions = [_row_to_dict(row) for row in pred_rows]
    
    # Check if there are any predictions at all for this symbol
    if symbol:
        cursor.execute(f'SELECT COUNT(*) FROM predictions WHERE symbol = {placeholder}', (symbol,))
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
    
    # Filter predictions by time eligibility
    ready_predictions = []
    not_ready_predictions = []
    
    for pred in predictions:
        if is_prediction_ready_for_evaluation(pred):
            ready_predictions.append(pred)
        else:
            not_ready_predictions.append(pred)
    
    if not ready_predictions:
        interval_counts = {}
        for pred in not_ready_predictions:
            interval = pred.get("interval", "unknown")
            interval_counts[interval] = interval_counts.get(interval, 0) + 1
        
        interval_msg = ", ".join([f"{count} {interval}" for interval, count in interval_counts.items()])
        
        return {
            "evaluated": 0,
            "errors": 0,
            "total": len(predictions),
            "ready": 0,
            "not_ready": len(not_ready_predictions),
            "message": f"No predictions are ready for evaluation yet. {len(not_ready_predictions)} prediction(s) pending ({interval_msg}). Predictions will be ready after their prediction horizon has passed."
        }
    
    evaluated_count = 0
    error_count = 0
    skipped_count = 0
    
    # Group ready predictions by symbol for batch processing
    symbols_to_fetch = list(set([p["symbol"] for p in ready_predictions]))
    
    # Process each ready prediction
    for pred in ready_predictions:
        try:
            symbol = pred["symbol"]
            interval = pred.get("interval", "1d")
            timestamp = pred.get("timestamp")
            
            # Parse timestamp
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    try:
                        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        error_count += 1
                        continue
            
            # Calculate the evaluation time (when the prediction should be evaluated)
            duration = get_interval_duration(interval)
            evaluation_time = timestamp + duration
            
            # Fetch historical price at the evaluation time
            actual_price = fetch_historical_price_at_time(symbol, evaluation_time, interval)
            
            if actual_price is None:
                # If we can't get historical price, try current price as fallback
                # but only if evaluation_time is very recent (within last hour)
                now = datetime.now()
                if timestamp.tzinfo is not None and now.tzinfo is None:
                    now = now.replace(tzinfo=timestamp.tzinfo)
                elif timestamp.tzinfo is None and now.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=now.tzinfo)
                
                time_since_eval = now - evaluation_time
                if time_since_eval <= timedelta(hours=1):
                    # Fallback to current price if evaluation was very recent
                    ticker = Ticker(symbol)
                    quotes = ticker.price
                    quote_data = quotes.get(symbol, {})
                    actual_price = quote_data.get("regularMarketPrice")
                
                if actual_price is None:
                    skipped_count += 1
                    print(f"Could not fetch price for {symbol} at evaluation time {evaluation_time}")
                    continue
            
            # Evaluate the prediction with the correct historical price
            evaluate_prediction(pred["id"], float(actual_price))
            evaluated_count += 1
            
        except Exception as e:
            print(f"Error evaluating prediction {pred.get('id', 'unknown')}: {e}")
            error_count += 1
    
    return {
        "evaluated": evaluated_count,
        "errors": error_count,
        "total": len(predictions),
        "ready": len(ready_predictions),
        "not_ready": len(not_ready_predictions),
        "skipped": skipped_count,
        "message": f"Evaluated {evaluated_count} prediction(s). {len(not_ready_predictions)} prediction(s) not yet ready (waiting for prediction horizon to pass)."
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
    
    placeholder = _get_placeholder()
    
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
        conditions.append(f"p.symbol = {placeholder}")
        params.append(symbol)
    if interval:
        conditions.append(f"p.interval = {placeholder}")
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
    
    # Convert rows to dicts
    rows_dict = [_row_to_dict(row) for row in rows]
    
    # Calculate statistics
    total = len(rows_dict)
    correct_directions = sum(1 for row in rows_dict if row["correct"])
    direction_accuracy = (correct_directions / total) * 100 if total > 0 else 0
    
    errors = [row["error"] for row in rows_dict]
    error_percents = [row["error_percent"] for row in rows_dict]
    
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
    
    placeholder = _get_placeholder()
    
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
        conditions.append(f"p.symbol = {placeholder}")
        params.append(symbol)
    if interval:
        conditions.append(f"p.interval = {placeholder}")
        params.append(interval)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += f" ORDER BY p.timestamp DESC LIMIT {placeholder} OFFSET {placeholder}"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to dictionaries
    results = []
    for row in rows:
        row_dict = _row_to_dict(row)
        result = {
            "id": row_dict["id"],
            "symbol": row_dict["symbol"],
            "timestamp": row_dict["timestamp"].isoformat() if hasattr(row_dict["timestamp"], "isoformat") else str(row_dict["timestamp"]),
            "interval": row_dict["interval"],
            "horizon": row_dict["horizon"],
            "current_price": row_dict["current_price"],
            "predicted_price": row_dict["predicted_price"],
            "confidence": row_dict["confidence"],
            "evaluated": bool(row_dict["evaluated"]),
        }
        
        if row_dict["model_breakdown"]:
            try:
                result["model_breakdown"] = json.loads(row_dict["model_breakdown"])
            except:
                result["model_breakdown"] = None
        
        if row_dict.get("actual_price") is not None:
            result["actual_price"] = row_dict["actual_price"]
            result["error"] = row_dict["error"]
            result["error_percent"] = row_dict["error_percent"]
            result["direction_actual"] = row_dict["direction_actual"]
            result["direction_predicted"] = row_dict["direction_predicted"]
            result["correct"] = bool(row_dict["correct"])
            evaluated_at = row_dict.get("evaluated_at")
            if evaluated_at:
                result["evaluated_at"] = evaluated_at.isoformat() if hasattr(evaluated_at, "isoformat") else str(evaluated_at)
        
        results.append(result)
    
    return results


def get_model_performance_metrics(
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    min_evaluations: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate performance metrics for each individual model from evaluated predictions.
    
    Extracts individual model predictions from model_breakdown JSON and compares
    with actual prices to calculate per-model accuracy metrics.
    
    Args:
        symbol: Filter by symbol (optional)
        interval: Filter by interval (optional)
        min_evaluations: Minimum number of evaluations needed to calculate metrics
    
    Returns:
        Dictionary mapping model names to their performance metrics:
        {
            "linear": {"rmse": 1.2, "mae": 0.8, "direction_accuracy": 65.0, "count": 10, "score": 0.75},
            "lstm": {...},
            ...
        }
        Returns empty dict if insufficient data
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    placeholder = _get_placeholder()
    
    # Query evaluated predictions with model breakdown
    query = '''
        SELECT 
            p.model_breakdown,
            p.current_price,
            e.actual_price,
            e.error_percent,
            e.direction_actual,
            e.correct
        FROM predictions p
        JOIN evaluations e ON p.id = e.prediction_id
    '''
    params = []
    
    conditions = []
    if symbol:
        conditions.append(f"p.symbol = {placeholder}")
        params.append(symbol)
    if interval:
        conditions.append(f"p.interval = {placeholder}")
        params.append(interval)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY p.timestamp DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return {}
    
    # Collect per-model predictions and actuals
    model_data: Dict[str, List[Dict[str, Any]]] = {}
    
    for row in rows:
        row_dict = _row_to_dict(row)
        model_breakdown = row_dict.get("model_breakdown")
        actual_price = row_dict.get("actual_price")
        current_price = row_dict.get("current_price")
        direction_actual = row_dict.get("direction_actual")
        
        if not model_breakdown or actual_price is None or current_price is None:
            continue
        
        # Parse model breakdown JSON
        try:
            if isinstance(model_breakdown, str):
                breakdown = json.loads(model_breakdown)
            else:
                breakdown = model_breakdown
        except:
            continue
        
        # Extract individual model predictions
        for model_name, model_info in breakdown.items():
            if not isinstance(model_info, dict):
                continue
            
            model_prediction = model_info.get("prediction")
            if model_prediction is None:
                continue
            
            # Calculate error for this model's prediction
            error = abs(model_prediction - actual_price)
            error_percent = (error / current_price) * 100 if current_price > 0 else 0
            
            # Determine if direction was correct
            predicted_dir = classify_direction(model_prediction, current_price, threshold=0.005)
            direction_correct = predicted_dir == direction_actual
            
            if model_name not in model_data:
                model_data[model_name] = []
            
            model_data[model_name].append({
                "predicted": model_prediction,
                "actual": actual_price,
                "current": current_price,
                "error": error,
                "error_percent": error_percent,
                "direction_correct": direction_correct,
            })
    
    # Calculate metrics per model
    model_metrics = {}
    
    for model_name, data_list in model_data.items():
        if len(data_list) < min_evaluations:
            continue
        
        errors = [d["error"] for d in data_list]
        error_percents = [d["error_percent"] for d in data_list]
        direction_correct_count = sum(1 for d in data_list if d["direction_correct"])
        
        # Calculate metrics
        rmse = np.sqrt(np.mean([e**2 for e in errors])) if errors else float('inf')
        mae = np.mean(errors) if errors else float('inf')
        avg_error_percent = np.mean(error_percents) if error_percents else float('inf')
        direction_accuracy = (direction_correct_count / len(data_list)) * 100 if data_list else 0
        
        # Calculate performance score (higher is better)
        # Combine inverse RMSE (normalized) and direction accuracy
        # Lower RMSE = better, higher direction accuracy = better
        if rmse > 0 and not np.isinf(rmse):
            # Normalize RMSE: convert to score (0-1), lower RMSE = higher score
            # Use inverse RMSE normalized by average price
            avg_price = np.mean([d["current"] for d in data_list])
            normalized_rmse = rmse / avg_price if avg_price > 0 else 1.0
            rmse_score = max(0.0, min(1.0, 1.0 / (1.0 + normalized_rmse * 10)))
            
            # Combine RMSE score (70%) and direction accuracy (30%)
            performance_score = (rmse_score * 0.7) + (direction_accuracy / 100 * 0.3)
        else:
            performance_score = direction_accuracy / 100
        
        model_metrics[model_name] = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "avg_error_percent": round(avg_error_percent, 2),
            "direction_accuracy": round(direction_accuracy, 2),
            "count": len(data_list),
            "score": round(performance_score, 4),  # Overall performance score (0-1)
        }
    
    return model_metrics


def get_trading_performance_metrics(
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    lookback_days: int = 30
) -> Dict[str, Any]:
    """
    Calculate trading performance metrics from evaluated predictions.
    
    Calculates metrics assuming a simple trading strategy:
    - Buy when prediction is "up", Sell when "down", Hold when "neutral"
    - Returns are calculated based on actual price movements
    
    Args:
        symbol: Filter by symbol (optional)
        interval: Filter by interval (optional)
        lookback_days: Number of days to look back (default 30)
    
    Returns:
        Dictionary with performance metrics:
        {
            "prediction_accuracy": 75.5,  # Direction accuracy %
            "sharpe_ratio": 1.85,         # Risk-adjusted return
            "win_rate": 68.0,             # % of profitable predictions
            "total_return": 12.5,         # Cumulative return %
            "total_predictions": 50,       # Number of evaluated predictions
            "profitable_predictions": 34   # Number of correct direction predictions
        }
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    placeholder = _get_placeholder()
    
    # Query evaluated predictions
    query = '''
        SELECT 
            p.current_price,
            p.predicted_price,
            p.timestamp,
            e.actual_price,
            e.direction_actual,
            e.direction_predicted,
            e.correct,
            e.error_percent
        FROM predictions p
        JOIN evaluations e ON p.id = e.prediction_id
    '''
    params = []
    
    conditions = []
    if symbol:
        conditions.append(f"p.symbol = {placeholder}")
        params.append(symbol)
    if interval:
        conditions.append(f"p.interval = {placeholder}")
        params.append(interval)
    
    # Filter by lookback period if timestamp is available
    # Note: This assumes timestamp is stored correctly
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY p.timestamp DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return {
            "prediction_accuracy": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "total_predictions": 0,
            "profitable_predictions": 0,
        }
    
    # Convert rows to dicts
    predictions = [_row_to_dict(row) for row in rows]
    
    # Limit to lookback period if needed
    if lookback_days > 0:
        # Keep only recent predictions (approximate)
        predictions = predictions[:min(len(predictions), lookback_days * 2)]  # Rough estimate
    
    if not predictions:
        return {
            "prediction_accuracy": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "total_predictions": 0,
            "profitable_predictions": 0,
        }
    
    # Calculate returns for each prediction
    # Return = (actual_price - current_price) / current_price * 100
    returns = []
    correct_predictions = 0
    profitable_predictions = 0
    
    for pred in predictions:
        current_price = pred.get("current_price")
        actual_price = pred.get("actual_price")
        direction_predicted = pred.get("direction_predicted")
        direction_actual = pred.get("direction_actual")
        is_correct = pred.get("correct", False)
        
        if current_price is None or actual_price is None or current_price == 0:
            continue
        
        # Calculate return (%)
        return_pct = ((actual_price - current_price) / current_price) * 100
        returns.append(return_pct)
        
        # Count correct predictions
        if is_correct:
            correct_predictions += 1
        
        # Count profitable predictions (correct direction AND positive return)
        if direction_predicted == "up" and return_pct > 0:
            profitable_predictions += 1
        elif direction_predicted == "down" and return_pct < 0:
            profitable_predictions += 1
        elif direction_predicted == "neutral" and abs(return_pct) < 0.5:  # Small movement
            profitable_predictions += 1
    
    total_predictions = len(returns)
    
    if total_predictions == 0:
        return {
            "prediction_accuracy": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "total_predictions": 0,
            "profitable_predictions": 0,
        }
    
    # Calculate metrics
    # 1. Prediction Accuracy (Direction Accuracy)
    prediction_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    
    # 2. Win Rate (% of profitable predictions)
    win_rate = (profitable_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
    
    # 3. Total Return (cumulative return)
    total_return = sum(returns) if returns else 0.0
    
    # 4. Sharpe Ratio (risk-adjusted return)
    # Sharpe = (mean_return - risk_free_rate) / std_dev
    # Using 0 as risk-free rate for simplicity
    sharpe_ratio = 0.0
    if len(returns) > 1:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return > 0:
            sharpe_ratio = mean_return / std_return
        elif mean_return > 0:
            sharpe_ratio = 10.0  # Very high Sharpe if no volatility but positive returns
        else:
            sharpe_ratio = 0.0
    
    return {
        "prediction_accuracy": round(prediction_accuracy, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "win_rate": round(win_rate, 2),
        "total_return": round(total_return, 2),
        "total_predictions": total_predictions,
        "profitable_predictions": profitable_predictions,
    }
