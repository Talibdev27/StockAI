#!/usr/bin/env python3
# python_backend/utils/performance_tracker.py
"""
Track and log model performance over time.
Essential for Phase 2 (performance-based weighting).
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class PerformanceTracker:
    """Track model performance metrics over time"""
    
    def __init__(self, log_file: str = "model_performance.json"):
        """
        Initialize performance tracker
        
        Args:
            log_file: Path to JSON file for storing performance data
        """
        self.log_file = log_file
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load existing performance data"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {self.log_file}: {e}")
                return self._init_data_structure()
        return self._init_data_structure()
    
    def _init_data_structure(self) -> Dict:
        """Initialize data structure"""
        return {
            "created_at": datetime.now().isoformat(),
            "predictions": [],
            "model_stats": {},
            "stock_stats": {}
        }
    
    def _save_data(self):
        """Save performance data to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving performance data: {e}")
    
    def log_prediction(self, 
                      symbol: str,
                      model_name: str,
                      prediction: float,
                      confidence: float,
                      actual_price: Optional[float] = None):
        """
        Log a single prediction
        
        Args:
            symbol: Stock symbol
            model_name: Name of the model making the prediction
            prediction: Predicted value
            confidence: Confidence score (0-1)
            actual_price: Actual price (if known, for accuracy calculation)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "model": model_name,
            "prediction": prediction,
            "confidence": confidence,
            "actual_price": actual_price
        }
        
        self.data["predictions"].append(entry)
        self._save_data()
    
    def log_ensemble_prediction(self,
                               symbol: str,
                               ensemble_prediction: float,
                               ensemble_confidence: float,
                               individual_predictions: Dict,
                               actual_price: Optional[float] = None):
        """
        Log an ensemble prediction with all individual model predictions
        
        Args:
            symbol: Stock symbol
            ensemble_prediction: Final ensemble prediction
            ensemble_confidence: Ensemble confidence score
            individual_predictions: Dict of {model_name: {prediction, confidence}}
            actual_price: Actual price for validation
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "ensemble_prediction": ensemble_prediction,
            "ensemble_confidence": ensemble_confidence,
            "models": individual_predictions,
            "actual_price": actual_price
        }
        
        self.data["predictions"].append(entry)
        self._save_data()
    
    def calculate_model_accuracy(self, model_name: str, lookback: int = 100) -> Dict:
        """
        Calculate accuracy metrics for a specific model
        
        Args:
            model_name: Name of the model
            lookback: Number of recent predictions to analyze
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Filter predictions for this model that have actual prices
        model_predictions = [
            p for p in self.data["predictions"][-lookback:]
            if "models" in p and model_name in p["models"]
            and p.get("actual_price") is not None
        ]
        
        if not model_predictions:
            return {"error": "No data available for this model"}
        
        # Calculate metrics
        total = len(model_predictions)
        correct_direction = 0
        total_error = 0
        
        for pred in model_predictions:
            model_data = pred["models"][model_name]
            predicted = model_data["prediction"]
            actual = pred["actual_price"]
            
            # Direction accuracy (simplified - assumes price movement direction)
            if (predicted > pred.get("ensemble_prediction", predicted) and 
                actual > pred.get("ensemble_prediction", actual)) or \
               (predicted < pred.get("ensemble_prediction", predicted) and 
                actual < pred.get("ensemble_prediction", actual)):
                correct_direction += 1
            
            # Absolute error
            total_error += abs(predicted - actual)
        
        direction_accuracy = correct_direction / total if total > 0 else 0
        avg_error = total_error / total if total > 0 else 0
        
        return {
            "model": model_name,
            "total_predictions": total,
            "direction_accuracy": direction_accuracy,
            "average_error": avg_error,
            "sample_size": lookback
        }
    
    def get_best_model(self, symbol: Optional[str] = None, lookback: int = 100) -> Dict:
        """
        Identify the best performing model
        
        Args:
            symbol: Specific stock symbol (None for all stocks)
            lookback: Number of recent predictions to analyze
            
        Returns:
            Dictionary with best model info
        """
        # Get all unique model names
        model_names = set()
        for pred in self.data["predictions"][-lookback:]:
            if "models" in pred:
                if symbol is None or pred.get("symbol") == symbol:
                    model_names.update(pred["models"].keys())
        
        if not model_names:
            return {"error": "No model data available"}
        
        # Calculate accuracy for each model
        model_accuracies = {}
        for model_name in model_names:
            accuracy = self.calculate_model_accuracy(model_name, lookback)
            if "error" not in accuracy:
                model_accuracies[model_name] = accuracy["direction_accuracy"]
        
        if not model_accuracies:
            return {"error": "Could not calculate accuracies"}
        
        # Find best model
        best_model = max(model_accuracies.items(), key=lambda x: x[1])
        
        return {
            "best_model": best_model[0],
            "accuracy": best_model[1],
            "all_models": model_accuracies
        }
    
    def get_stock_stats(self, symbol: str, lookback: int = 100) -> Dict:
        """
        Get statistics for a specific stock
        
        Args:
            symbol: Stock symbol
            lookback: Number of recent predictions
            
        Returns:
            Dictionary with stock statistics
        """
        stock_predictions = [
            p for p in self.data["predictions"][-lookback:]
            if p.get("symbol") == symbol
        ]
        
        if not stock_predictions:
            return {"error": f"No data for {symbol}"}
        
        avg_confidence = sum(
            p.get("ensemble_confidence", 0) for p in stock_predictions
        ) / len(stock_predictions)
        
        return {
            "symbol": symbol,
            "total_predictions": len(stock_predictions),
            "average_confidence": avg_confidence,
            "sample_size": lookback
        }
    
    def get_summary(self) -> Dict:
        """Get overall summary of tracked performance"""
        total_predictions = len(self.data["predictions"])
        
        if total_predictions == 0:
            return {"message": "No predictions tracked yet"}
        
        # Get unique stocks and models
        stocks = set(p.get("symbol") for p in self.data["predictions"] if p.get("symbol"))
        models = set()
        for pred in self.data["predictions"]:
            if "models" in pred:
                models.update(pred["models"].keys())
        
        return {
            "total_predictions": total_predictions,
            "unique_stocks": len(stocks),
            "unique_models": len(models),
            "stocks": list(stocks),
            "models": list(models),
            "oldest_prediction": self.data["predictions"][0].get("timestamp") if self.data["predictions"] else None,
            "newest_prediction": self.data["predictions"][-1].get("timestamp") if self.data["predictions"] else None
        }
    
    def export_for_analysis(self, output_file: str = "performance_export.json"):
        """Export data in format suitable for analysis"""
        export_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_predictions": len(self.data["predictions"])
            },
            "predictions": self.data["predictions"],
            "summary": self.get_summary()
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            return {"success": True, "file": output_file}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Example usage in your API endpoint
def integrate_with_api():
    """
    Add this to your api.py predict endpoint:
    
    from utils.performance_tracker import PerformanceTracker
    
    tracker = PerformanceTracker()
    
    # After getting predictions:
    tracker.log_ensemble_prediction(
        symbol=symbol,
        ensemble_prediction=ensemble_pred,
        ensemble_confidence=ensemble_conf,
        individual_predictions=predictions
    )
    """
    pass


if __name__ == "__main__":
    # Test the tracker
    tracker = PerformanceTracker("test_performance.json")
    
    # Simulate some predictions
    print("Testing performance tracker...\n")
    
    tracker.log_prediction(
        symbol="AAPL",
        model_name="LSTM",
        prediction=150.5,
        confidence=0.72,
        actual_price=151.2
    )
    
    tracker.log_ensemble_prediction(
        symbol="AAPL",
        ensemble_prediction=150.8,
        ensemble_confidence=0.68,
        individual_predictions={
            "LSTM": {"prediction": 150.5, "confidence": 0.72},
            "XGBoost": {"prediction": 151.1, "confidence": 0.65}
        },
        actual_price=151.2
    )
    
    print("Summary:", json.dumps(tracker.get_summary(), indent=2))
    print("\nTest complete. Check test_performance.json")

