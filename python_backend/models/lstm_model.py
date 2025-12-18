"""
LSTM (Long Short-Term Memory) model for stock price prediction.

This module provides a deep learning LSTM model that can capture
complex patterns and long-term dependencies in price data.
Enhanced with:
- EarlyStopping and ReduceLROnPlateau
- Configurable hyperparameters
- Validation metrics (RMSE, MAE)
- Save/Load persistence per symbol+interval
- Long-horizon warning helper
"""

import os
import json
import math
import pickle
from datetime import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# Helper functions for environment variable parsing
def _env_bool(val: str) -> bool:
    return str(val).lower() in ("1", "true", "yes", "on")

def _env_or(current_value, key: str, cast):
    raw = os.environ.get(key)
    if raw is None:
        return current_value
    try:
        if cast is bool:
            return _env_bool(raw)
        return cast(raw)
    except Exception:
        return current_value


class LSTMModel:
    """
    LSTM model for price prediction using deep learning.
    
    This model uses LSTM layers to capture temporal patterns in price data,
    with data normalization and dropout for better generalization.
    """
    
    def __init__(
        self,
        lag: int = 60,
        epochs: int = 50,
        batch_size: int = 32,
        units1: int = 50,
        units2: int = 50,
        dropout: float = 0.2,
        patience: int = 5,
        bidirectional: bool = False,
    ):
        """
        Initialize the LSTM model.

        Args:
            lag: Number of previous prices to use for prediction
            epochs: Number of training epochs
            batch_size: Batch size for training
            units1: Units in first LSTM layer
            units2: Units in second LSTM layer
            dropout: Dropout rate between layers
            patience: Early stopping patience (epochs)
            bidirectional: Whether to use Bidirectional LSTM layers
        """
        # Base values from args
        self.lag = lag
        self.epochs = epochs
        self.batch_size = batch_size
        self.units1 = units1
        self.units2 = units2
        self.dropout = dropout
        self.patience = patience
        self.bidirectional = bidirectional

        # Environment variable overrides (if provided)
        self.epochs = _env_or(self.epochs, "LSTM_EPOCHS", int)
        self.batch_size = _env_or(self.batch_size, "LSTM_BATCH_SIZE", int)
        self.units1 = _env_or(self.units1, "LSTM_UNITS1", int)
        self.units2 = _env_or(self.units2, "LSTM_UNITS2", int)
        self.dropout = _env_or(self.dropout, "LSTM_DROPOUT", float)
        self.patience = _env_or(self.patience, "LSTM_PATIENCE", int)
        self.bidirectional = _env_or(self.bidirectional, "LSTM_BIDIRECTIONAL", bool)

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.metrics = {"rmse": None, "mae": None}
        self.version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.warnings = []

    # ---------- Persistence helpers ----------
    def _base_name(self, symbol: str, interval: str) -> str:
        return f"{symbol}_{interval}"

    def _store_dir(self) -> str:
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_store")

    def _model_path(self, symbol: str, interval: str) -> str:
        return os.path.join(self._store_dir(), f"{self._base_name(symbol, interval)}_lstm_model.h5")

    def _scaler_path(self, symbol: str, interval: str) -> str:
        return os.path.join(self._store_dir(), f"{self._base_name(symbol, interval)}_scaler.pkl")

    def _meta_path(self, symbol: str, interval: str) -> str:
        return os.path.join(self._store_dir(), f"{self._base_name(symbol, interval)}_lstm_meta.json")

    def save(self, symbol: str, interval: str, data_points: int) -> None:
        os.makedirs(self._store_dir(), exist_ok=True)
        if self.model is not None:
            self.model.save(self._model_path(symbol, interval), include_optimizer=True)
        with open(self._scaler_path(symbol, interval), "wb") as f:
            pickle.dump(self.scaler, f)
        meta = {
            "version": self.version,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "data_points": int(data_points),
            "metrics": self.metrics,
        }
        with open(self._meta_path(symbol, interval), "w") as f:
            json.dump(meta, f)

    def load(self, symbol: str, interval: str) -> bool:
        try:
            model_p = self._model_path(symbol, interval)
            scaler_p = self._scaler_path(symbol, interval)
            if os.path.exists(model_p) and os.path.exists(scaler_p):
                self.model = load_model(model_p)
                with open(scaler_p, "rb") as f:
                    self.scaler = pickle.load(f)
                # meta optional
                meta_p = self._meta_path(symbol, interval)
                if os.path.exists(meta_p):
                    with open(meta_p, "r") as f:
                        meta = json.load(f)
                        self.metrics = meta.get("metrics", self.metrics)
                        self.version = meta.get("version", self.version)
                return True
        except Exception:
            return False
        return False
        
    def train(self, closes, symbol: str = None, interval: str = None, force_retrain: bool = False):
        """
        Train LSTM on price data.
        
        Args:
            closes (np.ndarray): Array of closing prices
            
        Returns:
            float: Model confidence score based on validation loss
            
        Raises:
            ValueError: If insufficient data for training
        """
        if len(closes) < self.lag + 10:
            raise ValueError(f"Not enough data for LSTM training. Need at least {self.lag + 10} points, got {len(closes)}")
        
        # Try load existing model if available and not forcing retrain
        if not force_retrain and symbol and interval and self.load(symbol, interval):
            return 1.0  # loaded model assumed valid; confidence handled by ensemble weighting

        # Scale data to [0, 1] range for better LSTM performance
        scaled_data = self.scaler.fit_transform(closes.reshape(-1, 1))
        
        # Prepare sequences for LSTM training
        X, y = [], []
        for i in range(self.lag, len(scaled_data)):
            X.append(scaled_data[i-self.lag:i, 0])
            y.append(scaled_data[i, 0])
            
        X = np.array(X)
        y = np.array(y)
        # Reshape for LSTM: (samples, timesteps, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build shared LSTM feature extractor (optionally bidirectional)
        inputs = layers.Input(shape=(self.lag, 1), name="price_sequence")
        lstm1 = layers.LSTM(self.units1, return_sequences=True)
        x = layers.Bidirectional(lstm1)(inputs) if self.bidirectional else lstm1(inputs)
        x = layers.Dropout(self.dropout)(x)

        lstm2 = layers.LSTM(self.units2, return_sequences=False)
        x = layers.Bidirectional(lstm2)(x) if self.bidirectional else lstm2(x)
        x = layers.Dropout(self.dropout)(x)

        shared = layers.Dense(25, name="shared_dense")(x)

        # Price regression head
        price_output = layers.Dense(1, name="price")(shared)

        # Direction classification head (Up vs non-Up)
        direction_output = layers.Dense(1, activation="sigmoid", name="direction")(shared)

        self.model = keras.Model(inputs=inputs, outputs=[price_output, direction_output])
        
        # Prepare direction labels: 1 if next price meaningfully higher than current, else 0
        directions = []
        closes_arr = np.asarray(closes, dtype=float)
        for i in range(self.lag, len(closes_arr)):
            prev_price = closes_arr[i - 1]
            next_price = closes_arr[i]
            if prev_price <= 0:
                directions.append(0.0)
                continue
            change = (next_price - prev_price) / prev_price
            directions.append(1.0 if change > 0.001 else 0.0)
        y_dir = np.array(directions).reshape(-1, 1)

        # Compile model with dual-objective loss:
        # - price: MSE (regression)
        # - direction: binary cross-entropy (classification)
        # Direction gets higher weight to emphasize correct sign.
        self.model.compile(
            optimizer="adam",
            loss={"price": "mse", "direction": "binary_crossentropy"},
            loss_weights={"price": 0.4, "direction": 0.6},
            metrics={"price": ["mae"], "direction": ["accuracy"]},
        )
        
        # Callbacks: EarlyStopping and ReduceLROnPlateau
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(1, self.patience // 2),
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        # Train with validation split
        history = self.model.fit(
            X,
            {"price": y, "direction": y_dir},
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0,
        )
        
        # Metrics from validation (price head)
        # Prefer dedicated price loss key if available.
        if "val_price_loss" in history.history:
            val_price_losses = history.history["val_price_loss"]
            best_idx = int(np.argmin(val_price_losses))
            val_loss = float(val_price_losses[best_idx])
        else:
            val_losses = history.history.get("val_loss", [0.0])
            best_idx = int(np.argmin(val_losses))
            val_loss = float(val_losses[best_idx])

        val_mae_series = history.history.get("val_price_mae") or history.history.get("val_mae")
        val_mae = float(val_mae_series[best_idx]) if val_mae_series else 0.0
        val_rmse = math.sqrt(val_loss) if val_loss > 0 else 0.0

        self.metrics = {"rmse": round(val_rmse, 6), "mae": round(val_mae, 6)}

        # Persist model if symbol+interval provided
        if symbol and interval:
            self.save(symbol, interval, data_points=len(closes))

        # Confidence proxy from RMSE (bounded 0..1)
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + val_rmse)))
        return confidence
        
    def predict(self, closes, horizon):
        """
        Generate forecast using trained LSTM model.
        
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
            
        # Scale the input data using the same scaler
        scaled_data = self.scaler.transform(closes.reshape(-1, 1))
        last_window = scaled_data[-self.lag:]
        
        forecast = []
        current_window = last_window.copy()
        
        # Generate recursive predictions
        for _ in range(horizon):
            # Reshape for LSTM prediction
            X_pred = np.reshape(current_window, (1, self.lag, 1))
            raw_pred = self.model.predict(X_pred, verbose=0)

            # Support both legacy single-output and new dual-output models
            if isinstance(raw_pred, (list, tuple)):
                price_head = raw_pred[0]
            else:
                price_head = raw_pred

            pred_scaled = price_head[0, 0]
            
            # Inverse transform to get actual price
            pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
            forecast.append(pred)
            
            # Update window: remove oldest, add new prediction
            current_window = np.append(current_window[1:], [[pred_scaled]], axis=0)
            
        return forecast, forecast[0]  # Return forecast and next prediction

    # ---------- Warnings ----------
    def long_horizon_warning(self, horizon: int, series_len: int) -> None:
        if horizon > max(1, int(0.25 * series_len)):
            self.warnings.append(
                "Forecast horizon is large relative to available history; uncertainty increases."
            )
