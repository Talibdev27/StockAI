"""
Ensemble model that combines multiple prediction models for better accuracy.

This module provides a weighted ensemble that combines Linear Regression,
LSTM, and ARIMA models to produce more robust and accurate predictions.
"""

import os
import time
import numpy as np
from typing import Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from .linear_model import LinearRegressionModel
from .lstm_model import LSTMModel
from .arima_model import ARIMAModel
try:
    from .xgboost_model import XGBoostModel  # may raise if deps missing
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from .decision_tree_model import DecisionTreeModel
    _HAS_DT = True
except Exception:
    _HAS_DT = False

try:
    from .svm_model import SVMModel
    _HAS_SVM = True
except Exception:
    _HAS_SVM = False

try:
    from .prophet_model import ProphetModel
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    from evaluation import get_model_performance_metrics
    _HAS_EVALUATION = True
except Exception:
    _HAS_EVALUATION = False


class EnsembleModel:
    """
    Ensemble model that combines multiple prediction models.
    
    This class trains multiple models and combines their predictions
    using performance-based weighting when historical evaluation data is available,
    falling back to confidence-based weighting otherwise.
    
    Performance-based weighting:
    - Uses historical prediction accuracy (RMSE, MAE, direction accuracy) from evaluated predictions
    - Models with better historical performance get higher weights
    - Automatically adapts as more evaluations accumulate
    
    Confidence-based weighting (fallback):
    - Uses validation metrics from training (RMSE, R², AIC)
    - Used when insufficient performance data is available
    """
    
    def __init__(self, interval="1d"):
        """
        Initialize the ensemble model.
        
        Args:
            interval (str): Time interval for adaptive lag calculation
        """
        self.interval = interval
        
        # Read LSTM hyperparameters from environment variables
        u1 = int(os.getenv("LSTM_UNITS1", "50"))
        u2 = int(os.getenv("LSTM_UNITS2", "50"))
        dp = float(os.getenv("LSTM_DROPOUT", "0.2"))
        bs = int(os.getenv("LSTM_BATCH_SIZE", "32"))
        ep = int(os.getenv("LSTM_EPOCHS", "30"))  # Balanced default for ensemble
        pt = int(os.getenv("LSTM_PATIENCE", "5"))
        bi = os.getenv("LSTM_BIDIRECTIONAL", "false").lower() in ("1","true","yes","on")
        
        self.models = {
            "linear": LinearRegressionModel(lag=self._get_lag()),
            "lstm": LSTMModel(
                lag=min(60, self._get_lag() * 3),
                epochs=ep,
                batch_size=bs,
                units1=u1,
                units2=u2,
                dropout=dp,
                patience=pt,
                bidirectional=bi,
            ),
            "arima": ARIMAModel(),
        }
        if _HAS_XGB:
            self.models["xgboost"] = XGBoostModel(lags=self._get_lag())
        if _HAS_DT:
            self.models["decision_tree"] = DecisionTreeModel(lags=self._get_lag())
        if _HAS_SVM:
            self.models["svm"] = SVMModel(lags=self._get_lag())
        
        # Prophet (optional, time series specialist)
        self.prophet = None
        self.prophet_failed = False
        if _HAS_PROPHET:
            try:
                self.prophet = ProphetModel(interval=self.interval)
            except Exception as e:
                print(f"Prophet initialization failed: {e}")
                self.prophet_failed = True
        else:
            self.prophet_failed = True
        
        self.confidences = {}
        self.warnings = []
        self.performance_metrics = {}  # Cache for performance metrics
        
        # Direction weighting configuration (configurable via environment variable)
        # Higher default weight for directional accuracy when performance data is available.
        self.direction_weight = float(os.getenv("ENSEMBLE_DIRECTION_WEIGHT", "0.7"))
        
    def _predict_single_model(self, name, model, closes, horizon):
        """
        Wrapper for single model prediction (for parallel execution).
        
        Args:
            name: Model name
            model: Model instance
            closes: Historical closing prices
            horizon: Number of future periods to predict
            
        Returns:
            Tuple of (name, forecast, next_pred, error)
        """
        try:
            if self.confidences.get(name, 0) > 0:
                forecast, next_pred = model.predict(closes, horizon)
                return name, forecast, next_pred, None
            return name, None, None, None
        except Exception as e:
            return name, None, None, str(e)
        
    def _get_lag(self):
        """
        Get adaptive lag based on interval.
        
        Different intervals require different amounts of historical data
        for optimal performance.
        
        Returns:
            int: Optimal lag for the given interval
        """
        if self.interval in ["1h", "30m", "15m", "5m", "1m"]:
            return 20  # Intraday: need more recent data
        elif self.interval in ["4h", "2h"]:
            return 15  # 4-hour: moderate lag
        elif self.interval in ["1wk"]:
            return 12  # Weekly: less frequent data
        elif self.interval in ["1mo"]:
            return 6   # Monthly: very infrequent data
        else:  # 1d
            return 20  # Daily: standard lag
            
    def train_all(self, closes, symbol: str = None, interval: str = None, force_retrain: bool = False):
        """
        Train all models and store their confidence scores.
        
        Args:
            closes (np.ndarray): Array of closing prices
            
        Returns:
            dict: Training results for each model
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                if name in ("lstm", "xgboost", "decision_tree", "svm"):
                    # Attempt load-before-train within model.train
                    confidence = model.train(
                        closes,
                        symbol=symbol,
                        interval=interval or self.interval,
                        force_retrain=force_retrain,
                    )
                else:
                    confidence = model.train(closes)
                self.confidences[name] = confidence
                # Add model-specific diagnostics
                model_result = {"trained": True, "confidence": confidence}
                if name in ("lstm", "xgboost", "decision_tree", "svm"):
                    model_result["metrics"] = getattr(model, "metrics", None)
                    model_result["status"] = getattr(model, "status", None)
                    model_result["version"] = getattr(model, "version", None)
                    # propagate warnings
                    lstm_warnings = getattr(model, "warnings", []) or []
                    self.warnings.extend(lstm_warnings)
                results[name] = model_result
            except Exception as e:
                self.confidences[name] = 0.0
                results[name] = {"trained": False, "error": str(e)}
        
        # Prophet (optional, time series specialist)
        if self.prophet and not self.prophet_failed:
            try:
                print("\n[Training Prophet...]")
                prophet_conf = self.prophet.train(closes)
                if prophet_conf > 0:
                    self.confidences['prophet'] = prophet_conf
                    results['prophet'] = {"trained": True, "confidence": prophet_conf}
                    print(f"✓ Prophet: confidence={prophet_conf:.3f}")
                else:
                    print("✗ Prophet: training failed")
                    self.prophet_failed = True
                    results['prophet'] = {"trained": False, "error": "Training returned zero confidence"}
            except Exception as e:
                print(f"✗ Prophet error: {e}")
                self.prophet_failed = True
                results['prophet'] = {"trained": False, "error": str(e)}
                
        return results
    
    def _calculate_performance_weights(
        self, 
        valid_models: dict, 
        symbol: str = None, 
        interval: str = None,
        prioritize_direction: bool = False,
        closes: np.ndarray = None
    ) -> Tuple[Dict[str, float], bool]:
        """
        Calculate weights based on historical performance metrics.
        
        Uses direction-based weighting that considers both price prediction accuracy (RMSE)
        and direction prediction accuracy (up/down/neutral). The balance between these
        two metrics is configurable via ENSEMBLE_DIRECTION_WEIGHT environment variable.
        
        Args:
            valid_models: Dictionary of model names with valid predictions
            symbol: Stock symbol for filtering performance data
            interval: Time interval for filtering performance data
            prioritize_direction: If True, use 50/50 split or higher direction weight
        
        Returns:
            Tuple of (weights_dict, use_performance_based)
            - weights_dict: Normalized weights for each model
            - use_performance_based: True if performance data was used, False if fell back to confidence
        """
        if not _HAS_EVALUATION:
            # Fallback to confidence-based if evaluation module not available
            total_confidence = sum(self.confidences[k] for k in valid_models.keys())
            weights = {k: self.confidences[k] / total_confidence for k in valid_models.keys()}
            return weights, False
        
        try:
            # Get performance metrics from database
            performance_metrics = get_model_performance_metrics(
                symbol=symbol,
                interval=interval or self.interval,
                min_evaluations=3  # Minimum evaluations needed
            )
            
            if not performance_metrics:
                # No performance data available, use confidence-based
                total_confidence = sum(self.confidences[k] for k in valid_models.keys())
                weights = {k: self.confidences[k] / total_confidence for k in valid_models.keys()}
                return weights, False
            
            # Calculate weights based on performance scores
            # Use performance score if available, otherwise use confidence
            # Recalculate performance score with configurable direction weight
            performance_scores = {}
            # When prioritize_direction is True (e.g. daily interval),
            # tilt weights even more strongly toward directional accuracy.
            direction_weight = self.direction_weight if prioritize_direction else min(self.direction_weight, 0.5)
            
            # Estimate average price from closes for RMSE normalization
            # Use median of recent closes as representative price
            if closes is not None and len(closes) > 0:
                avg_price_estimate = float(np.median(closes[-30:]))
            else:
                avg_price_estimate = 100.0  # Default fallback
            
            for model_name in valid_models.keys():
                if model_name in performance_metrics:
                    perf = performance_metrics[model_name]
                    # Recalculate score with configurable direction weight
                    # Get RMSE and direction_accuracy from metrics
                    rmse = perf.get("rmse", float('inf'))
                    direction_accuracy = perf.get("direction_accuracy", 0.0)
                    
                    if rmse > 0 and not np.isinf(rmse) and avg_price_estimate > 0:
                        # Normalize RMSE: convert to score (0-1), lower RMSE = higher score
                        # Use same normalization approach as evaluation.py
                        normalized_rmse = rmse / avg_price_estimate
                        rmse_score = max(0.0, min(1.0, 1.0 / (1.0 + normalized_rmse * 10)))
                        
                        # Use configurable direction weight
                        rmse_weight = 1.0 - direction_weight
                        performance_score = (rmse_score * rmse_weight) + (direction_accuracy / 100 * direction_weight)
                    else:
                        # Fallback to direction accuracy only
                        performance_score = direction_accuracy / 100
                    
                    performance_scores[model_name] = performance_score
                else:
                    # Model not in performance data, use confidence as fallback
                    performance_scores[model_name] = self.confidences.get(model_name, 0.1)
            
            # Normalize performance scores to weights
            total_score = sum(performance_scores.values())
            if total_score > 0:
                weights = {k: performance_scores[k] / total_score for k in valid_models.keys()}
                # Store performance metrics for reference
                self.performance_metrics = performance_metrics
                return weights, True
            else:
                # All scores are zero, fallback to confidence
                total_confidence = sum(self.confidences[k] for k in valid_models.keys())
                weights = {k: self.confidences[k] / total_confidence for k in valid_models.keys()}
                return weights, False
                
        except Exception as e:
            print(f"Warning: Failed to calculate performance weights: {e}")
            # Fallback to confidence-based on error
            total_confidence = sum(self.confidences[k] for k in valid_models.keys())
            weights = {k: self.confidences[k] / total_confidence for k in valid_models.keys()}
            return weights, False
        
    def predict_ensemble(self, closes, horizon, symbol: str = None, interval: str = None):
        """
        Generate weighted ensemble prediction.
        
        Args:
            closes (np.ndarray): Historical closing prices
            horizon (int): Number of future periods to predict
            symbol (str): Stock symbol for performance-based weighting (optional)
            interval (str): Time interval for performance-based weighting (optional)
            
        Returns:
            dict: Ensemble prediction results with model breakdown
            
        Raises:
            ValueError: If no models produced valid predictions
        """
        predictions = {}
        forecasts = {}
        
        # Parallel prediction for all models using ThreadPoolExecutor
        # This speeds up ensemble inference by running models concurrently
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=min(len(self.models), 7)) as executor:
            # Submit all model predictions
            futures = {
                executor.submit(self._predict_single_model, name, model, closes, horizon): name
                for name, model in self.models.items()
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                name, forecast, next_pred, error = future.result()
                if error:
                    print(f"Model {name} prediction failed: {error}")
                    predictions[name] = None
                    forecasts[name] = None
                else:
                    predictions[name] = next_pred
                    forecasts[name] = forecast
        
        parallel_execution_time = time.time() - start_time
        
        # Prophet prediction
        if self.prophet and not self.prophet_failed and self.confidences.get('prophet', 0) > 0:
            try:
                forecast, next_pred = self.prophet.predict(closes, horizon)
                predictions['prophet'] = next_pred
                forecasts['prophet'] = forecast
            except Exception as e:
                print(f"Prophet prediction failed: {e}")
                self.prophet_failed = True
                predictions['prophet'] = None
                forecasts['prophet'] = None
                    
        # Filter valid predictions
        valid_models = {k: v for k, v in predictions.items() if v is not None}
        
        if not valid_models:
            raise ValueError("No models produced valid predictions")
        
        # Calculate weights using performance-based or confidence-based method.
        # For daily predictions, explicitly prioritize directional accuracy
        # when performance data is available.
        eff_interval = interval or self.interval
        prioritize_direction = eff_interval == "1d"
        weights, use_performance_based = self._calculate_performance_weights(
            valid_models,
            symbol=symbol,
            interval=eff_interval,
            closes=closes,
            prioritize_direction=prioritize_direction,
        )
        
        # Weighted next prediction
        ensemble_next = sum(predictions[k] * weights[k] for k in valid_models.keys())
        
        # Weighted forecast for all horizons
        ensemble_forecast = []
        for i in range(horizon):
            weighted_val = sum(
                forecasts[k][i] * weights[k] 
                for k in valid_models.keys() 
                if forecasts[k] is not None
            )
            ensemble_forecast.append(weighted_val)
        
        # Ensemble confidence is weighted average of individual confidences
        ensemble_confidence = sum(
            self.confidences[k] * weights[k] 
            for k in valid_models.keys()
        )
        
        # Build result with model breakdown
        model_breakdown = {}
        for k in valid_models.keys():
            model_info = {
                "prediction": round(predictions[k], 2),
                "confidence": round(self.confidences[k] * 100, 1),
                "weight": round(weights[k] * 100, 1)
            }
            
            # Add performance metrics if available and using performance-based weighting
            if use_performance_based and k in self.performance_metrics:
                perf = self.performance_metrics[k]
                model_info["performance"] = {
                    "rmse": perf["rmse"],
                    "mae": perf["mae"],
                    "direction_accuracy": perf["direction_accuracy"],
                    "evaluations": perf["count"]
                }
            
            model_breakdown[k] = model_info
        
        result = {
            "predictedPrice": round(ensemble_next, 2),
            "confidence": round(ensemble_confidence * 100, 1),
            "forecast": [round(x, 2) for x in ensemble_forecast],
            "models": model_breakdown,
            "weighting_method": "performance" if use_performance_based else "confidence"
        }
        
        # Add performance metrics (optional, for monitoring)
        result["parallel_execution_time"] = round(parallel_execution_time, 3)
        
        if self.warnings:
            result["warnings"] = list(dict.fromkeys(self.warnings))
        return result
