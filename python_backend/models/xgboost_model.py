import os
import math
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    # XGBoost may be unavailable on some macs without libomp; handle gracefully
    _HAS_XGB = False
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import ta
    _HAS_TA = True
except Exception:
    _HAS_TA = False


class XGBoostModel:
    """
    Gradient boosted trees model for next-step price prediction.

    Uses feature-engineering over close prices: lags/returns/rolling stats
    and common technical indicators (RSI, MACD, Bollinger) when available.
    Persisted per symbol+interval under python_backend/models_store/.
    """

    def __init__(self, lags: int = 20, seed: int = 42):
        self.lags = lags
        self.model: Optional[XGBRegressor] = None
        self.metrics = {"rmse": None, "mae": None}
        self.status = "uninitialized"
        self.seed = seed

    def _paths(self, symbol: Optional[str], interval: Optional[str]) -> Tuple[str, str]:
        base = "python_backend/models_store"
        os.makedirs(base, exist_ok=True)
        key = f"xgb_{symbol or 'sym'}_{interval or 'int'}"
        return os.path.join(base, key + ".pkl"), os.path.join(base, key + "_meta.pkl")

    def _build_features(self, closes: np.ndarray) -> pd.DataFrame:
        s = pd.Series(closes.astype(float), name="close").reset_index(drop=True)
        df = pd.DataFrame({"close": s})

        # Lags and returns
        for k in range(1, self.lags + 1):
            df[f"lag_{k}"] = df["close"].shift(k)
        df["ret_1"] = df["close"].pct_change(1)
        df["ret_5"] = df["close"].pct_change(5)
        # Rolling stats
        for w in [5, 10, 20]:
            df[f"roll_mean_{w}"] = df["close"].rolling(w).mean()
            df[f"roll_std_{w}"] = df["close"].rolling(w).std()

        # Indicators
        if _HAS_TA:
            try:
                df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
                macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
                df["macd"] = macd.macd()
                df["macd_signal"] = macd.macd_signal()
                bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
                df["bb_high"] = bb.bollinger_hband()
                df["bb_low"] = bb.bollinger_lband()
                df["bb_pct"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"]).replace(0, np.nan)
            except Exception:
                # If TA fails, continue with existing features
                pass

        df = df.dropna().reset_index(drop=True)

        # Next-step target
        df["target"] = df["close"].shift(-1)
        df = df.dropna().reset_index(drop=True)
        return df

    def load(self, symbol: Optional[str], interval: Optional[str]) -> bool:
        path, meta = self._paths(symbol, interval)
        if os.path.exists(path) and os.path.exists(meta):
            try:
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                with open(meta, "rb") as f:
                    stored = pickle.load(f)
                    self.metrics = stored.get("metrics", self.metrics)
                self.status = "cached"
                return True
            except Exception:
                return False
        return False

    def save(self, symbol: Optional[str], interval: Optional[str]):
        if self.model is None:
            return
        path, meta = self._paths(symbol, interval)
        try:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            with open(meta, "wb") as f:
                pickle.dump({"metrics": self.metrics}, f)
            self.status = "cached"
        except Exception:
            pass

    def train(self, closes: np.ndarray, symbol: Optional[str] = None, interval: Optional[str] = None, force_retrain: bool = False) -> float:
        if not _HAS_XGB:
            raise RuntimeError("XGBoost not available on this system")
        if len(closes) < max(60, self.lags + 10):
            raise ValueError("Not enough data for XGBoost training")

        if not force_retrain and self.load(symbol, interval):
            # Confidence based on prior metrics if available
            rmse = self.metrics.get("rmse")
            if rmse:
                return max(0.1, 1.0 / (1.0 + rmse))
            return 0.6

        df = self._build_features(closes)
        X = df.drop(columns=["target"]).values
        y = df["target"].values

        # Train/validation split (80/20)
        n = len(df)
        split = int(n * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        self.model = XGBRegressor(
            n_estimators=500,        # Increased from 300 - more trees
            max_depth=8,             # Increased from 6 - capture complex patterns
            learning_rate=0.04,      # Decreased from 0.05 - better convergence
            subsample=0.8,           # Feature sampling per tree
            colsample_bytree=0.8,    # Column sampling per tree
            min_child_weight=3,      # Prevent overfitting on small samples
            gamma=0.1,               # Minimum loss reduction required
            reg_alpha=0.1,           # L1 regularization
            reg_lambda=1.0,          # L2 regularization
            objective="reg:squarederror",
            random_state=self.seed,
            n_jobs=-1,               # Use all CPU cores
            tree_method="hist",
            verbosity=0              # Reduce console spam
        )
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=20,  # Stop if no improvement after 20 rounds
            verbose=False,
        )

        # Metrics and confidence
        preds = self.model.predict(X_val)
        rmse = math.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        self.metrics = {"rmse": round(rmse, 3), "mae": round(mae, 3)}
        self.status = "trained"

        # Confidence scaled by RMSE
        confidence = max(0.1, min(1.0, 1.0 / (1.0 + rmse)))

        self.save(symbol, interval)
        return confidence

    def predict(self, closes: np.ndarray, horizon: int) -> Tuple[List[float], float]:
        if self.model is None or not _HAS_XGB:
            raise ValueError("XGBoost model must be trained before predicting")

        df = self._build_features(closes)
        X_all = df.drop(columns=["target"]).values
        # Use last available row as seed features and roll forward
        from copy import deepcopy
        last_row = deepcopy(X_all[-1])

        # Forecast next step recursively
        forecast = []
        current_close = df["close"].iloc[-1]
        for _ in range(horizon):
            next_val = float(self.model.predict(last_row.reshape(1, -1))[0])
            forecast.append(next_val)
            # Shift features: move lags and update rolling-like proxies crudely
            # For simplicity, we just update lag_1 with predicted and shift right
            # Find lag_1 column index
            cols = list(df.drop(columns=["target"]).columns)
            try:
                lag1_idx = cols.index("lag_1")
            except ValueError:
                lag1_idx = None
            if lag1_idx is not None:
                # shift lag features to the right (lag_k becomes next step)
                for k in range(self.lags, 1, -1):
                    name = f"lag_{k}"
                    prev_name = f"lag_{k-1}"
                    if name in cols and prev_name in cols:
                        last_row[cols.index(name)] = last_row[cols.index(prev_name)]
                last_row[lag1_idx] = next_val
            current_close = next_val

        return forecast, forecast[0]


