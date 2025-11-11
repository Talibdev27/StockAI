"""
Linear Regression model for stock price prediction.

This module provides a simple but effective linear regression model
that uses a sliding window approach to predict future prices.
"""

from sklearn.linear_model import LinearRegression
import numpy as np


class LinearRegressionModel:
    """
    Linear Regression model for price prediction using sliding window.
    
    This model uses the last 'lag' prices to predict the next price,
    then recursively uses its own predictions for multi-step forecasting.
    """
    
    def __init__(self, lag=20):
        """
        Initialize the Linear Regression model.
        
        Args:
            lag (int): Number of previous prices to use for prediction
        """
        self.lag = lag
        self.model = None
        
    def train(self, closes):
        """
        Train linear regression on price data.
        
        Args:
            closes (np.ndarray): Array of closing prices
            
        Returns:
            float: Model confidence score (R²)
            
        Raises:
            ValueError: If insufficient data for training
        """
        if len(closes) <= self.lag + 1:
            raise ValueError(f"Not enough data for training. Need at least {self.lag + 2} points, got {len(closes)}")
            
        # Prepare training data with sliding window
        X, y = [], []
        for i in range(self.lag, len(closes)):
            X.append(closes[i - self.lag : i])
            y.append(closes[i])
            
        X = np.array(X)
        y = np.array(y)
        
        # Train the model
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # Calculate confidence from R² score
        confidence = max(0.0, min(1.0, self.model.score(X, y)))
        return confidence
        
    def predict(self, closes, horizon):
        """
        Generate forecast using trained model.
        
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
            
        # Start with the last 'lag' prices
        window = closes[-self.lag:].astype(float)
        forecast = []
        
        # Generate recursive predictions
        for _ in range(horizon):
            pred = float(self.model.predict(window.reshape(1, -1))[0])
            forecast.append(pred)
            # Update window: remove oldest, add new prediction
            window = np.append(window[1:], pred)
            
        return forecast, forecast[0]  # Return forecast and next prediction
