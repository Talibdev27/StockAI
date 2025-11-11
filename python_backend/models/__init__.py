"""
Ensemble prediction models for stock price forecasting.

This module contains individual prediction models and ensemble logic:
- LinearRegressionModel: Fast baseline linear regression
- LSTMModel: Deep learning LSTM for pattern recognition  
- ARIMAModel: Statistical ARIMA for trend analysis
- XGBoostModel: Gradient boosted trees (optional)
- DecisionTreeModel: Decision tree regressor (optional)
- SVMModel: Support vector regressor (optional)
- ProphetModel: Facebook Prophet for time series with seasonality (optional)
- EnsembleModel: Weighted ensemble combining all models
"""

from .linear_model import LinearRegressionModel
from .lstm_model import LSTMModel
from .arima_model import ARIMAModel
from .ensemble import EnsembleModel

try:
    from .decision_tree_model import DecisionTreeModel
except ImportError:
    DecisionTreeModel = None

try:
    from .svm_model import SVMModel
except ImportError:
    SVMModel = None

try:
    from .prophet_model import ProphetModel
except ImportError:
    ProphetModel = None

__all__ = [
    "LinearRegressionModel",
    "LSTMModel", 
    "ARIMAModel",
    "EnsembleModel",
    "DecisionTreeModel",
    "SVMModel",
    "ProphetModel",
]
