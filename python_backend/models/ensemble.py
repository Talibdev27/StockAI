"""
Ensemble model that combines multiple prediction models for better accuracy.

This module provides a weighted ensemble that combines Linear Regression,
LSTM, and ARIMA models to produce more robust and accurate predictions.
"""

import os
import numpy as np
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


class EnsembleModel:
    """
    Ensemble model that combines multiple prediction models.
    
    This class trains multiple models and combines their predictions
    using weighted voting based on individual model confidence scores.
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
        
    def predict_ensemble(self, closes, horizon):
        """
        Generate weighted ensemble prediction.
        
        Args:
            closes (np.ndarray): Historical closing prices
            horizon (int): Number of future periods to predict
            
        Returns:
            dict: Ensemble prediction results with model breakdown
            
        Raises:
            ValueError: If no models produced valid predictions
        """
        predictions = {}
        forecasts = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            if self.confidences.get(name, 0) > 0:
                try:
                    forecast, next_pred = model.predict(closes, horizon)
                    predictions[name] = next_pred
                    forecasts[name] = forecast
                except Exception as e:
                    predictions[name] = None
                    forecasts[name] = None
        
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
            
        # Calculate weighted average based on confidence scores
        total_confidence = sum(self.confidences[k] for k in valid_models.keys())
        weights = {k: self.confidences[k] / total_confidence for k in valid_models.keys()}
        
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
        
        result = {
            "predictedPrice": round(ensemble_next, 2),
            "confidence": round(ensemble_confidence * 100, 1),
            "forecast": [round(x, 2) for x in ensemble_forecast],
            "models": {
                k: {
                    "prediction": round(predictions[k], 2),
                    "confidence": round(self.confidences[k] * 100, 1),
                    "weight": round(weights[k] * 100, 1)
                }
                for k in valid_models.keys()
            }
        }
        if self.warnings:
            result["warnings"] = list(dict.fromkeys(self.warnings))
        return result
