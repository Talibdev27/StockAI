"""
Backtesting engine for evaluating trading strategies based on ML predictions.
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from models.ensemble import EnsembleModel


class BacktestEngine:
    """Backtest engine for simulating trading strategies."""
    
    def __init__(
        self,
        strategy: str = "simple_signals",
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        position_size: float = 1.0,
        threshold: float = 0.0,
    ):
        """
        Initialize backtest engine.
        
        Args:
            strategy: Strategy type ("simple_signals", "threshold", "momentum")
            initial_capital: Starting capital
            commission: Commission per trade (0.001 = 0.1%)
            position_size: Position size as fraction of capital (1.0 = 100%)
            threshold: Threshold for threshold/momentum strategies
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.position_size = position_size
        self.threshold = threshold
        
        # State tracking
        self.cash = initial_capital
        self.shares = 0.0
        self.equity_curve = []
        self.trades = []
        self.prices = []
        self.predictions = []
        self.dates = []
        
    def run_backtest(
        self,
        closes: np.ndarray,
        dates: List[str],
        predictions: List[float],
        interval: str = "1d",
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data with predictions.
        
        Args:
            closes: Array of closing prices
            dates: List of date strings
            predictions: List of predicted prices (one step ahead)
            interval: Data interval for annualization
            
        Returns:
            Dictionary with backtest results and metrics
        """
        if len(closes) != len(predictions) or len(closes) != len(dates):
            raise ValueError("closes, dates, and predictions must have same length")
        
        # Reset state
        self.cash = self.initial_capital
        self.shares = 0.0
        self.equity_curve = []
        self.trades = []
        self.prices = closes.tolist()
        self.predictions = predictions
        self.dates = dates
        
        # Walk-forward backtest
        for i in range(len(closes)):
            current_price = closes[i]
            predicted_price = predictions[i] if i < len(predictions) else current_price
            
            # Generate signal based on strategy
            signal = self._generate_signal(i, current_price, predicted_price)
            
            # Execute trade
            if signal == "buy" and self.shares == 0:
                self._buy(current_price, dates[i])
            elif signal == "sell" and self.shares > 0:
                self._sell(current_price, dates[i])
            
            # Update equity curve
            equity = self.cash + self.shares * current_price
            self.equity_curve.append({
                "date": dates[i],
                "value": equity,
                "price": float(current_price),
            })
        
        # Close any open position at end
        if self.shares > 0:
            final_price = closes[-1]
            self._sell(final_price, dates[-1])
            # Update final equity
            self.equity_curve[-1]["value"] = self.cash
        
        # Calculate metrics
        metrics = self._calculate_metrics(interval)
        
        return {
            "symbol": "",  # Will be set by endpoint
            "period": {
                "start": dates[0] if dates else "",
                "end": dates[-1] if dates else "",
            },
            "initialCapital": self.initial_capital,
            "finalCapital": self.cash,
            "totalReturn": (self.cash - self.initial_capital) / self.initial_capital,
            "metrics": metrics,
            "equityCurve": self.equity_curve,
            "trades": self.trades,
        }
    
    def _generate_signal(self, index: int, current_price: float, predicted_price: float) -> str:
        """Generate buy/sell signal based on strategy."""
        if self.strategy == "simple_signals":
            if predicted_price > current_price * (1 + self.threshold):
                return "buy"
            elif predicted_price < current_price * (1 - self.threshold):
                return "sell"
        
        elif self.strategy == "threshold":
            if index > 0:
                prev_pred = self.predictions[index - 1] if index > 0 else current_price
                change_pct = (predicted_price - prev_pred) / prev_pred if prev_pred > 0 else 0
                
                if change_pct > self.threshold:
                    return "buy"
                elif change_pct < -self.threshold:
                    return "sell"
        
        elif self.strategy == "momentum":
            if index > 2:
                # Check trend: last 3 predictions
                recent_preds = self.predictions[max(0, index-2):index+1]
                if len(recent_preds) >= 3:
                    trend = (recent_preds[-1] - recent_preds[0]) / recent_preds[0] if recent_preds[0] > 0 else 0
                    
                    if predicted_price > current_price and trend > 0:
                        return "buy"
                    elif predicted_price < current_price or trend < -0.01:
                        return "sell"
            elif index > 0:
                # Simple check for early periods
                if predicted_price > current_price:
                    return "buy"
                elif predicted_price < current_price * 0.98:
                    return "sell"
        
        return "hold"
    
    def _buy(self, price: float, date: str):
        """Execute buy order."""
        capital_to_use = self.cash * self.position_size
        shares_to_buy = capital_to_use / price * (1 - self.commission)
        cost = shares_to_buy * price * (1 + self.commission)
        
        if cost <= self.cash:
            self.shares = shares_to_buy
            self.cash -= cost
            
            self.trades.append({
                "date": date,
                "type": "buy",
                "price": float(price),
                "shares": float(shares_to_buy),
                "cost": float(cost),
            })
    
    def _sell(self, price: float, date: str):
        """Execute sell order."""
        if self.shares > 0:
            proceeds = self.shares * price * (1 - self.commission)
            
            # Calculate PnL from buy price
            buy_trades = [t for t in self.trades if t["type"] == "buy"]
            if buy_trades:
                # Use last buy price and cost
                last_buy = buy_trades[-1]
                buy_price = last_buy["price"]
                buy_cost = last_buy.get("cost", buy_price * self.shares)
                pnl = proceeds - buy_cost
            else:
                # Fallback if no buy found (shouldn't happen)
                pnl = 0.0
            
            self.cash += proceeds
            
            self.trades.append({
                "date": date,
                "type": "sell",
                "price": float(price),
                "shares": float(self.shares),
                "pnl": float(pnl),
            })
            
            self.shares = 0.0
    
    def _calculate_metrics(self, interval: str) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.equity_curve:
            return {
                "pnl": 0.0,
                "returnPercent": 0.0,
                "sharpeRatio": 0.0,
                "winRate": 0.0,
                "maxDrawdown": 0.0,
                "totalTrades": 0,
                "avgWin": 0.0,
                "avgLoss": 0.0,
            }
        
        # Basic metrics
        final_value = self.equity_curve[-1]["value"]
        pnl = final_value - self.initial_capital
        return_percent = (pnl / self.initial_capital) * 100
        
        # Calculate returns for Sharpe ratio
        values = [point["value"] for point in self.equity_curve]
        returns = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                returns.append((values[i] - values[i-1]) / values[i-1])
        
        # Sharpe ratio
        sharpe_ratio = 0.0
        if returns and np.std(returns) > 0:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            # Annualize based on interval
            periods_per_year = {
                "1d": 252,
                "1wk": 52,
                "1mo": 12,
                "1h": 252 * 6.5,  # Trading hours
                "4h": 252 * 1.625,
                "15m": 252 * 26,
                "5m": 252 * 78,
            }.get(interval, 252)
            
            sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year) if std_return > 0 else 0.0
        
        # Max drawdown
        max_drawdown = 0.0
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Trade statistics
        sell_trades = [t for t in self.trades if t["type"] == "sell" and "pnl" in t]
        total_trades = len(sell_trades)
        
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        
        if sell_trades:
            winning_trades = [t for t in sell_trades if t["pnl"] > 0]
            losing_trades = [t for t in sell_trades if t["pnl"] <= 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
            avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0.0
        
        return {
            "pnl": round(float(pnl), 2),
            "returnPercent": round(float(return_percent), 2),
            "sharpeRatio": round(float(sharpe_ratio), 2),
            "winRate": round(float(win_rate), 3),
            "maxDrawdown": round(float(-max_drawdown), 4),  # Negative for convention
            "totalTrades": total_trades,
            "avgWin": round(float(avg_win), 2),
            "avgLoss": round(float(avg_loss), 2),
        }

