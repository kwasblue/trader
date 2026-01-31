"""
Vectorized Backtester Module

High-performance vectorized backtesting engine.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass

from strategies.strategy_registry import load_strategy
from core.backtest.validation import validate_ohlcv_data
from core.backtest.slippage_models import SlippageModel, FixedSlippage


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    position_sizing: str = 'fixed'
    position_size: float = 0.1
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.0
    initial_capital: float = 10000
    transaction_cost: float = 0.001


class VectorizedBacktester:
    """
    High-performance vectorized backtester.

    Uses vectorized signal generation for 10-100x speedup on large datasets.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        slippage_model: SlippageModel = None
    ):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage_model or FixedSlippage(0.0005)

        # Validate and prepare data
        result = validate_ohlcv_data(self.data, fix_issues=True)
        if not result.is_valid:
            raise ValueError(f"Invalid data: {result.errors}")
        if result.cleaned_data is not None:
            self.data = result.cleaned_data

    def run(
        self,
        strategy_name: str,
        strategy_params: Dict = None,
        position_sizing: str = 'fixed',  # 'fixed', 'risk_parity', 'volatility_scaled'
        position_size: float = 0.1,  # Fraction of capital per trade
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 3.0
    ) -> pd.DataFrame:
        """
        Run vectorized backtest.

        Args:
            strategy_name: Name of strategy
            strategy_params: Strategy parameters
            position_sizing: Position sizing method
            position_size: Base position size as fraction of capital
            stop_loss_atr: Stop loss in ATR multiples
            take_profit_atr: Take profit in ATR multiples

        Returns:
            DataFrame with backtest results
        """
        if strategy_params is None:
            strategy_params = {}

        # Load strategy and generate signals
        strategy = load_strategy(strategy_name, params=strategy_params)

        # Use vectorized signal generation if available
        if hasattr(strategy, 'generate_signals_vectorized'):
            signals = strategy.generate_signals_vectorized(self.data)
            if signals is not None:
                self.data['Signal'] = signals

        if 'Signal' not in self.data.columns:
            # Fall back to row-by-row
            result = strategy.generate_signal(self.data.copy())
            if isinstance(result, int):
                # Single signal - need to iterate
                signals = []
                for i in range(len(self.data)):
                    sig = strategy.generate_signal(self.data.iloc[:i+1])
                    signals.append(sig)
                self.data['Signal'] = signals
            elif isinstance(result, pd.DataFrame) and 'Signal' in result.columns:
                self.data['Signal'] = result['Signal'].values

        # Calculate ATR for position sizing
        if 'ATR' not in self.data.columns:
            high = self.data['High'].values
            low = self.data['Low'].values
            close = self.data['Close'].values

            tr = np.maximum(high - low,
                           np.maximum(np.abs(high - np.roll(close, 1)),
                                     np.abs(low - np.roll(close, 1))))
            tr[0] = high[0] - low[0]
            self.data['ATR'] = pd.Series(tr).rolling(14).mean().values

        # Vectorized position simulation
        close = self.data['Close'].values
        signals = self.data['Signal'].values
        atr = self.data['ATR'].values
        volume = self.data['Volume'].values if 'Volume' in self.data.columns else np.ones(len(close)) * 1000000

        n = len(close)
        position = np.zeros(n)
        cash = np.zeros(n)
        portfolio_value = np.zeros(n)
        trades = []

        cash[0] = self.initial_capital

        for i in range(1, n):
            cash[i] = cash[i-1]
            position[i] = position[i-1]

            current_price = close[i]
            current_atr = atr[i] if not np.isnan(atr[i]) else close[i] * 0.02
            signal = signals[i]

            # Position sizing
            if position_sizing == 'volatility_scaled':
                target_vol = 0.15  # 15% annual vol target
                daily_vol = current_atr / current_price
                size_multiplier = target_vol / (daily_vol * np.sqrt(252)) if daily_vol > 0 else 1
                size_multiplier = np.clip(size_multiplier, 0.1, 3.0)
            else:
                size_multiplier = 1.0

            trade_value = self.initial_capital * position_size * size_multiplier
            quantity = int(trade_value / current_price)

            # Execute trades
            if signal == 1 and position[i] <= 0 and quantity > 0:
                # Close short if any
                if position[i] < 0:
                    exec_price = self.slippage.calculate_slippage(
                        current_price, abs(int(position[i])), 'buy', volume[i], current_atr/current_price*100
                    )
                    cost = abs(position[i]) * exec_price * (1 + self.transaction_cost)
                    cash[i] -= cost
                    position[i] = 0

                # Open long
                exec_price = self.slippage.calculate_slippage(
                    current_price, quantity, 'buy', volume[i], current_atr/current_price*100
                )
                cost = quantity * exec_price * (1 + self.transaction_cost)
                if cost <= cash[i]:
                    cash[i] -= cost
                    position[i] = quantity
                    trades.append({
                        'idx': i,
                        'action': 'BUY',
                        'price': exec_price,
                        'quantity': quantity,
                        'stop_loss': current_price - stop_loss_atr * current_atr
                    })

            elif signal == -1 and position[i] >= 0 and quantity > 0:
                # Close long if any
                if position[i] > 0:
                    exec_price = self.slippage.calculate_slippage(
                        current_price, int(position[i]), 'sell', volume[i], current_atr/current_price*100
                    )
                    proceeds = position[i] * exec_price * (1 - self.transaction_cost)
                    cash[i] += proceeds
                    position[i] = 0
                    trades.append({
                        'idx': i,
                        'action': 'SELL',
                        'price': exec_price,
                        'quantity': int(position[i-1])
                    })

            # Check stop loss
            if len(trades) > 0 and position[i] > 0:
                last_trade = trades[-1]
                if 'stop_loss' in last_trade and current_price <= last_trade['stop_loss']:
                    exec_price = current_price * (1 - 0.001)  # Extra slippage on stop
                    proceeds = position[i] * exec_price * (1 - self.transaction_cost)
                    cash[i] += proceeds
                    position[i] = 0
                    trades.append({
                        'idx': i,
                        'action': 'STOP_LOSS',
                        'price': exec_price,
                        'quantity': int(position[i-1])
                    })

            portfolio_value[i] = cash[i] + position[i] * current_price

        # Handle initial value
        portfolio_value[0] = self.initial_capital

        # Build result DataFrame
        result = self.data.copy()
        result['Position'] = position
        result['Cash'] = cash
        result['Portfolio_Value'] = portfolio_value
        result['Strategy_Return'] = pd.Series(portfolio_value).pct_change().fillna(0)

        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_value)
        result['Drawdown'] = (peak - portfolio_value) / peak

        return result

    def get_metrics(self, result: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from backtest result."""
        returns = result['Strategy_Return'].values
        portfolio_value = result['Portfolio_Value'].values

        total_return = (portfolio_value[-1] - self.initial_capital) / self.initial_capital

        # Sharpe ratio
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(252)
        else:
            sortino = sharpe

        # Max drawdown
        peak = np.maximum.accumulate(portfolio_value)
        drawdown = (peak - portfolio_value) / peak
        max_drawdown = np.max(drawdown)

        # Win rate
        trade_returns = returns[returns != 0]
        if len(trade_returns) > 0:
            win_rate = np.sum(trade_returns > 0) / len(trade_returns)
        else:
            win_rate = 0

        # Profit factor
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        profit_factor = gains / losses if losses > 0 else float('inf')

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(trade_returns),
            'final_value': portfolio_value[-1]
        }
