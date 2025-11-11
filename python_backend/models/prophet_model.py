"""
Prophet model adapted to match ensemble interface.

Handles seasonality and trends in stock price data.

Prophet (Facebook) time series forecasting model - excellent for capturing
seasonality and trends in financial data. Adapted to work seamlessly with
the ensemble system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")


class ProphetModel:
    """
    Prophet time series model adapted for ensemble interface.
    
    Interface matches other ensemble models:
    - train(closes) -> confidence
    - predict(closes, horizon) -> (forecast_list, next_pred)
    """
    
    def __init__(self, interval='1d'):
        """
        Initialize Prophet model
        
        Args:
            interval: Data frequency ('1d', '1wk', '1mo', '3mo')
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet library not installed")
        
        self.model = None
        self.confidence = 0.0
        self.interval = interval
        self.last_date = None  # Track last known date for predictions
        
        # Map interval to pandas frequency
        self.freq_map = {
            '1d': 'D',
            '1wk': 'W',
            '1mo': 'MS',  # Month start
            '3mo': '3MS'  # 3 months start
        }
    
    def _create_dataframe(self, closes, start_date=None):
        """
        Convert closes array to Prophet-compatible DataFrame
        
        Args:
            closes: numpy array of close prices
            start_date: Starting date (if None, uses historical reference)
            
        Returns:
            DataFrame with 'ds' (date) and 'y' (price) columns
        """
        if start_date is None:
            # Default: assume data ends today, work backwards
            end_date = datetime.now()
            start_date = end_date - timedelta(days=len(closes) * self._interval_to_days())
        
        # Generate date range with appropriate frequency
        freq = self.freq_map.get(self.interval, 'D')
        dates = pd.date_range(
            start=start_date,
            periods=len(closes),
            freq=freq
        )
        
        self.last_date = dates[-1]  # Store for future predictions
        
        return pd.DataFrame({
            'ds': dates,
            'y': closes
        })
    
    def _interval_to_days(self):
        """Convert interval to approximate number of days"""
        interval_days = {
            '1d': 1,
            '1wk': 7,
            '1mo': 30,
            '3mo': 90
        }
        return interval_days.get(self.interval, 1)
    
    def train(self, closes):
        """
        Train Prophet model (matches ensemble interface)
        
        Args:
            closes: numpy array of close prices
            
        Returns:
            confidence: float (0-1) indicating model reliability
        """
        try:
            if len(closes) < 30:
                print("Prophet: Insufficient data (need 30+ points)")
                self.confidence = 0.0
                return 0.0
            
            # Split data internally (80/20)
            split_idx = int(len(closes) * 0.8)
            train_closes = closes[:split_idx]
            test_closes = closes[split_idx:]
            
            if len(test_closes) < 5:
                print("Prophet: Insufficient test data")
                self.confidence = 0.0
                return 0.0
            
            # Create DataFrames
            train_df = self._create_dataframe(train_closes)
            
            # Configure Prophet based on interval
            seasonality_config = self._get_seasonality_config()
            
            # Initialize Prophet
            self.model = Prophet(
                daily_seasonality=seasonality_config['daily'],
                weekly_seasonality=seasonality_config['weekly'],
                yearly_seasonality=seasonality_config['yearly'],
                seasonality_mode='multiplicative',  # Better for stocks
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.95
            )
            
            # Suppress Prophet's verbose output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(train_df)
            
            # Validate on test set
            future = self.model.make_future_dataframe(
                periods=len(test_closes),
                freq=self.freq_map.get(self.interval, 'D')
            )
            forecast = self.model.predict(future)
            
            # Get test predictions
            test_preds = forecast['yhat'].tail(len(test_closes)).values
            
            # Calculate confidence based on test error
            mae = np.mean(np.abs(test_closes - test_preds))
            price_range = np.max(closes) - np.min(closes)
            
            if price_range > 0:
                normalized_error = mae / price_range
                self.confidence = max(0.0, min(1.0, 1 - normalized_error))
            else:
                self.confidence = 0.5
            
            print(f"Prophet trained: confidence={self.confidence:.3f}, MAE={mae:.4f}")
            return self.confidence
            
        except Exception as e:
            print(f"Prophet training failed: {e}")
            self.confidence = 0.0
            return 0.0
    
    def _get_seasonality_config(self):
        """
        Get appropriate seasonality settings based on interval
        
        Returns:
            dict with seasonality flags
        """
        if self.interval == '1d':
            return {
                'daily': True,
                'weekly': True,
                'yearly': True
            }
        elif self.interval == '1wk':
            return {
                'daily': False,
                'weekly': True,
                'yearly': True
            }
        elif self.interval in ['1mo', '3mo']:
            return {
                'daily': False,
                'weekly': False,
                'yearly': True
            }
        else:
            return {
                'daily': False,
                'weekly': True,
                'yearly': True
            }
    
    def predict(self, closes, horizon=10):
        """
        Make predictions (matches ensemble interface)
        
        Args:
            closes: numpy array of recent close prices
            horizon: number of periods to forecast ahead
            
        Returns:
            tuple: (forecast_list, next_prediction)
                - forecast_list: list of horizon predictions
                - next_prediction: next single period prediction
        """
        if self.model is None:
            print("Prophet: Model not trained")
            # Return neutral predictions
            last_price = closes[-1] if len(closes) > 0 else 0.0
            return [last_price] * horizon, last_price
        
        try:
            # Create DataFrame from recent data
            df = self._create_dataframe(closes)
            
            # Make future dataframe
            future = self.model.make_future_dataframe(
                periods=horizon,
                freq=self.freq_map.get(self.interval, 'D')
            )
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Extract future predictions (last 'horizon' rows)
            future_preds = forecast['yhat'].tail(horizon).values.tolist()
            
            # Next prediction is the first future value
            next_pred = future_preds[0] if len(future_preds) > 0 else closes[-1]
            
            return future_preds, next_pred
            
        except Exception as e:
            print(f"Prophet prediction failed: {e}")
            # Fallback: return last price
            last_price = closes[-1] if len(closes) > 0 else 0.0
            return [last_price] * horizon, last_price
    
    def get_trend_components(self):
        """
        Get trend and seasonality components (optional, for analysis)
        
        Returns:
            dict with trend, weekly, yearly components or None
        """
        if self.model is None:
            return None
        
        try:
            future = self.model.make_future_dataframe(periods=30)
            forecast = self.model.predict(future)
            
            components = {
                'trend': forecast['trend'].tail(30).tolist()
            }
            
            if 'weekly' in forecast.columns:
                components['weekly'] = forecast['weekly'].tail(30).tolist()
            if 'yearly' in forecast.columns:
                components['yearly'] = forecast['yearly'].tail(30).tolist()
            
            return components
            
        except Exception as e:
            print(f"Failed to get components: {e}")
            return None


# Test if Prophet is available
def test_prophet_availability():
    """Test if Prophet can be imported and used"""
    if not PROPHET_AVAILABLE:
        print("❌ Prophet not available")
        print("   Install with: pip install prophet")
        print("   Note: May require additional dependencies (pystan)")
        return False
    
    print("✅ Prophet available")
    return True


if __name__ == "__main__":
    # Test the adapter
    print("Testing Prophet Model Adapter...")
    
    if not test_prophet_availability():
        exit(1)
    
    # Create sample data
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(100) * 2)
    
    # Test training
    print("\n1. Testing train() method...")
    model = ProphetModel(interval='1d')
    confidence = model.train(closes)
    print(f"   Confidence: {confidence:.3f}")
    
    # Test prediction
    print("\n2. Testing predict() method...")
    forecast, next_pred = model.predict(closes, horizon=10)
    print(f"   Next prediction: {next_pred:.2f}")
    print(f"   10-day forecast: {[f'{x:.2f}' for x in forecast[:3]]}...")
    
    # Test with weekly data
    print("\n3. Testing with weekly interval...")
    model_weekly = ProphetModel(interval='1wk')
    confidence_weekly = model_weekly.train(closes)
    print(f"   Confidence: {confidence_weekly:.3f}")
    
    print("\n✅ All tests passed!")

