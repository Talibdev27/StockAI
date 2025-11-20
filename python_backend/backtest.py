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
        slippage: float = 0.0005,
        slippage_type: str = "hybrid",
    ):
        """
        Initialize backtest engine.
        
        Args:
            strategy: Strategy type ("simple_signals", "threshold", "momentum")
            initial_capital: Starting capital
            commission: Commission per trade (0.001 = 0.1%)
            position_size: Position size as fraction of capital (1.0 = 100%)
            threshold: Threshold for threshold/momentum strategies
            slippage: Base slippage percentage (0.0005 = 0.05%)
            slippage_type: Slippage calculation method ("fixed", "volatility", "hybrid")
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.position_size = position_size
        self.threshold = threshold
        self.slippage = slippage
        self.slippage_type = slippage_type
        
        # State tracking
        self.cash = initial_capital
        self.shares = 0.0
        self.equity_curve = []
        self.trades = []
        self.prices = []
        self.predictions = []
        self.dates = []
        self.total_slippage_cost = 0.0
        
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
        self.total_slippage_cost = 0.0
        
        # Initialize random seed for random strategy (for reproducibility)
        if self.strategy == "random":
            np.random.seed(42)
        
        # Walk-forward backtest
        for i in range(len(closes)):
            current_price = closes[i]
            predicted_price = predictions[i] if i < len(predictions) else current_price
            
            # Generate signal based on strategy
            signal = self._generate_signal(i, current_price, predicted_price)
            
            # Execute trade
            if signal == "buy" and self.shares == 0:
                self._buy(current_price, dates[i], i)
            elif signal == "sell" and self.shares > 0:
                self._sell(current_price, dates[i], i)
            
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
            final_index = len(closes) - 1
            self._sell(final_price, dates[-1], final_index)
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
        
        elif self.strategy == "buy_and_hold":
            # Buy-and-Hold: Buy on first day, sell on last day
            if index == 0:  # First day
                return "buy"
            elif index == len(self.prices) - 1:  # Last day
                return "sell"
            return "hold"
        
        elif self.strategy == "random":
            # Random Trading: Random buy/sell decisions (50% probability each)
            # Seed is set once at start of backtest for reproducibility
            random_value = np.random.random()
            
            if self.shares == 0:  # Not holding
                return "buy" if random_value > 0.5 else "hold"
            else:  # Holding
                return "sell" if random_value > 0.5 else "hold"
        
        return "hold"
    
    def _calculate_slippage(self, index: int, price: float) -> float:
        """
        Calculate slippage based on selected method.
        
        Args:
            index: Current price index
            price: Current price
            
        Returns:
            Slippage percentage (e.g., 0.0005 for 0.05%)
        """
        if self.slippage_type == "fixed":
            return self.slippage
        
        elif self.slippage_type == "volatility":
            # Need at least 2 prices to calculate volatility
            if len(self.prices) < 2 or index < 1:
                return self.slippage
            
            # Calculate rolling volatility from recent returns
            window = min(20, index)  # Use up to 20 periods, or available data
            if window < 2:
                return self.slippage
            
            recent_prices = self.prices[max(0, index - window):index + 1]
            if len(recent_prices) < 2:
                return self.slippage
            
            # Calculate returns
            returns = []
            for i in range(1, len(recent_prices)):
                if recent_prices[i-1] > 0:
                    ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                    returns.append(ret)
            
            if not returns:
                return self.slippage
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns)
            
            # Scale volatility to slippage (volatility_factor = 0.5 means 50% of volatility becomes slippage)
            volatility_factor = 0.5
            volatility_component = volatility * volatility_factor
            
            # Cap maximum slippage at 0.5% (0.005)
            max_slippage = 0.005
            slippage = min(self.slippage + volatility_component, max_slippage)
            
            return max(slippage, 0.0)  # Ensure non-negative
        
        elif self.slippage_type == "hybrid":
            # Combine fixed base with volatility component
            base_slippage = self.slippage
            
            # Need at least 2 prices to calculate volatility
            if len(self.prices) < 2 or index < 1:
                return base_slippage
            
            # Calculate rolling volatility from recent returns
            window = min(20, index)
            if window < 2:
                return base_slippage
            
            recent_prices = self.prices[max(0, index - window):index + 1]
            if len(recent_prices) < 2:
                return base_slippage
            
            # Calculate returns
            returns = []
            for i in range(1, len(recent_prices)):
                if recent_prices[i-1] > 0:
                    ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                    returns.append(ret)
            
            if not returns:
                return base_slippage
            
            # Calculate volatility
            volatility = np.std(returns)
            
            # Volatility component (capped at 0.2%)
            volatility_factor = 0.5
            volatility_component = min(volatility * volatility_factor, 0.002)
            
            slippage = base_slippage + volatility_component
            
            # Cap maximum slippage at 0.5%
            max_slippage = 0.005
            return min(slippage, max_slippage)
        
        else:
            # Default to fixed if unknown type
            return self.slippage
    
    def _buy(self, price: float, date: str, index: int):
        """Execute buy order with slippage."""
        # Calculate slippage for this trade using current price index
        slippage_pct = self._calculate_slippage(index, price)
        
        # Apply slippage: buy orders pay more (price * (1 + slippage))
        execution_price = price * (1 + slippage_pct)
        
        capital_to_use = self.cash * self.position_size
        shares_to_buy = capital_to_use / execution_price * (1 - self.commission)
        cost = shares_to_buy * execution_price * (1 + self.commission)
        
        # Calculate slippage cost
        slippage_cost = shares_to_buy * price * slippage_pct
        
        if cost <= self.cash:
            self.shares = shares_to_buy
            self.cash -= cost
            self.total_slippage_cost += slippage_cost
            
            self.trades.append({
                "date": date,
                "type": "buy",
                "price": float(price),  # Original price
                "execution_price": float(execution_price),  # Price with slippage
                "shares": float(shares_to_buy),
                "cost": float(cost),
                "slippage_cost": float(slippage_cost),
                "slippage_pct": float(slippage_pct * 100),  # As percentage
            })
    
    def _sell(self, price: float, date: str, index: int):
        """Execute sell order with slippage."""
        if self.shares > 0:
            # Calculate slippage for this trade using current price index
            slippage_pct = self._calculate_slippage(index, price)
            
            # Apply slippage: sell orders receive less (price * (1 - slippage))
            execution_price = price * (1 - slippage_pct)
            
            proceeds = self.shares * execution_price * (1 - self.commission)
            
            # Calculate slippage cost
            slippage_cost = self.shares * price * slippage_pct
            
            # Calculate PnL from buy price (using original buy price, not execution price)
            buy_trades = [t for t in self.trades if t["type"] == "buy"]
            if buy_trades:
                # Use last buy cost (which already includes buy slippage)
                last_buy = buy_trades[-1]
                buy_cost = last_buy.get("cost", last_buy.get("execution_price", last_buy["price"]) * self.shares)
                pnl = proceeds - buy_cost
            else:
                # Fallback if no buy found (shouldn't happen)
                pnl = 0.0
            
            self.cash += proceeds
            self.total_slippage_cost += slippage_cost
            
            self.trades.append({
                "date": date,
                "type": "sell",
                "price": float(price),  # Original price
                "execution_price": float(execution_price),  # Price with slippage
                "shares": float(self.shares),
                "pnl": float(pnl),
                "slippage_cost": float(slippage_cost),
                "slippage_pct": float(slippage_pct * 100),  # As percentage
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
        
        # Calculate total slippage cost
        total_slippage = round(float(self.total_slippage_cost), 2)
        slippage_impact_pct = (total_slippage / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0
        
        return {
            "pnl": round(float(pnl), 2),
            "returnPercent": round(float(return_percent), 2),
            "sharpeRatio": round(float(sharpe_ratio), 2),
            "winRate": round(float(win_rate), 3),
            "maxDrawdown": round(float(-max_drawdown), 4),  # Negative for convention
            "totalTrades": total_trades,
            "avgWin": round(float(avg_win), 2),
            "avgLoss": round(float(avg_loss), 2),
            "totalSlippage": total_slippage,
            "slippageImpactPercent": round(float(slippage_impact_pct), 4),
        }

