"""
ARIMA (AutoRegressive Integrated Moving Average) model for stock price prediction.

This module provides a statistical ARIMA model that automatically finds
the best parameters for time series forecasting.
"""

from pmdarima import auto_arima
import numpy as np


class ARIMAModel:
    """
    ARIMA model for price prediction using statistical time series analysis.
    
    This model uses auto_arima to automatically find the best ARIMA parameters
    and provides statistical confidence based on AIC (Akaike Information Criterion).
    """
    
    def __init__(self):
        """Initialize the ARIMA model."""
        self.model = None
        
    def train(self, closes):
        """
        Train ARIMA on price data using auto_arima.
        
        Args:
            closes (np.ndarray): Array of closing prices
            
        Returns:
            float: Model confidence score based on AIC
            
        Raises:
            ValueError: If insufficient data for training
        """
        if len(closes) < 30:
            raise ValueError(f"Not enough data for ARIMA training. Need at least 30 points, got {len(closes)}")
            
        # Auto ARIMA finds best parameters automatically
        self.model = auto_arima(
            closes,
            start_p=1, start_q=1,  # Start with simple ARIMA(1,1,1)
            max_p=5, max_q=5,      # Maximum order of 5
            seasonal=False,         # No seasonal component for stock prices
            stepwise=True,         # Use stepwise search for efficiency
            suppress_warnings=True, # Suppress convergence warnings
            error_action="ignore",  # Ignore errors and continue
            max_order=None,        # No maximum order limit
            trace=False            # Don't print search progress
        )
        
        # Calculate confidence from AIC (lower AIC is better)
        aic = self.model.aic()
        # Convert AIC to confidence: lower AIC = higher confidence
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + abs(aic) / 1000)))
        
        return confidence
        
    def predict(self, closes, horizon):
        """
        Generate forecast using trained ARIMA model.
        
        Args:
            closes (np.ndarray): Historical closing prices
            horizon (int): Number of future periods to predict
            
        Returns:
            tuple: (forecast_list, next_prediction)
                - forecast_list: List of predicted prices for all horizons
                - next_prediction: Single prediction for next period
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Generate forecast
        forecast = self.model.predict(n_periods=horizon)
        
        # Convert to list and return
        forecast_list = forecast.tolist()
        return forecast_list, forecast_list[0]  # Return forecast and next prediction
