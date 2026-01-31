#!/usr/bin/env python3
"""
Example 6: Custom Strategy Implementation

Demonstrates how to create and test a custom trading strategy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List

from core.base.base_strategy import BaseStrategy
from core.backtest_suite import VectorizedBacktester, validate_ohlcv_data


# =============================================================================
# CUSTOM STRATEGY DEFINITION
# =============================================================================

class DualMomentumStrategy(BaseStrategy):
    """
    Dual Momentum Strategy

    Combines absolute momentum (trend) with relative momentum (strength).

    Buy when:
    - Price is above its N-period moving average (absolute momentum)
    - Rate of change over M periods is positive (relative momentum)

    Sell when:
    - Price is below its N-period moving average
    - OR rate of change is negative

    Parameters:
        ma_period: Moving average period for trend (default: 50)
        roc_period: Rate of change period (default: 20)
        roc_threshold: Minimum ROC for buy signal (default: 0)
    """

    def __init__(
        self,
        ma_period: int = 50,
        roc_period: int = 20,
        roc_threshold: float = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ma_period = ma_period
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold

    def generate_signal(self, data: pd.DataFrame) -> int:
        """Generate signal for latest bar."""
        if len(data) < max(self.ma_period, self.roc_period):
            return 0

        close = data['Close'] if 'Close' in data.columns else data['close']

        # Calculate indicators
        ma = close.rolling(self.ma_period).mean().iloc[-1]
        roc = ((close.iloc[-1] / close.iloc[-self.roc_period - 1]) - 1) * 100

        current_price = close.iloc[-1]

        # Dual momentum conditions
        above_ma = current_price > ma
        positive_roc = roc > self.roc_threshold

        if above_ma and positive_roc:
            return 1   # Buy: both momentums positive
        elif not above_ma or roc < -self.roc_threshold:
            return -1  # Sell: either momentum negative
        return 0       # Hold

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        """Vectorized signal generation for fast backtesting."""
        close = data['Close'] if 'Close' in data.columns else data['close']
        n = len(close)

        # Calculate moving average
        ma = close.rolling(self.ma_period).mean()

        # Calculate rate of change
        roc = (close / close.shift(self.roc_period) - 1) * 100

        # Generate signals
        above_ma = close > ma
        positive_roc = roc > self.roc_threshold
        negative_roc = roc < -self.roc_threshold

        signals = np.where(
            above_ma & positive_roc, 1,
            np.where(~above_ma | negative_roc, -1, 0)
        )

        # No signal during warmup
        warmup = max(self.ma_period, self.roc_period)
        signals[:warmup] = 0

        return signals.tolist()


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility Breakout Strategy

    Buys when price breaks above the previous day's range scaled by ATR.
    Sells when price breaks below.

    Parameters:
        atr_period: ATR calculation period (default: 14)
        breakout_mult: ATR multiplier for breakout (default: 1.5)
    """

    def __init__(
        self,
        atr_period: int = 14,
        breakout_mult: float = 1.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.atr_period = atr_period
        self.breakout_mult = breakout_mult

    def generate_signal(self, data: pd.DataFrame) -> int:
        if len(data) < self.atr_period + 1:
            return 0

        high = data['High'] if 'High' in data.columns else data['high']
        low = data['Low'] if 'Low' in data.columns else data['low']
        close = data['Close'] if 'Close' in data.columns else data['close']

        # Calculate ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean().iloc[-1]

        # Previous day's close and current price
        prev_close = close.iloc[-2]
        current = close.iloc[-1]

        # Breakout levels
        upper_breakout = prev_close + (atr * self.breakout_mult)
        lower_breakout = prev_close - (atr * self.breakout_mult)

        if current > upper_breakout:
            return 1   # Bullish breakout
        elif current < lower_breakout:
            return -1  # Bearish breakout
        return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        high = data['High'] if 'High' in data.columns else data['high']
        low = data['Low'] if 'Low' in data.columns else data['low']
        close = data['Close'] if 'Close' in data.columns else data['close']

        # Calculate ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        # Breakout levels
        prev_close = close.shift(1)
        upper = prev_close + (atr * self.breakout_mult)
        lower = prev_close - (atr * self.breakout_mult)

        signals = np.where(close > upper, 1, np.where(close < lower, -1, 0))
        signals[:self.atr_period + 1] = 0

        return signals.tolist()


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

def generate_sample_data(days: int = 500) -> pd.DataFrame:
    """Generate synthetic price data with trends."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Create trending data
    trend = np.sin(np.linspace(0, 4 * np.pi, days)) * 20
    noise = np.cumsum(np.random.randn(days) * 0.5)
    prices = 100 + trend + noise

    data = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(days) * 0.5,
        'High': prices + np.abs(np.random.randn(days) * 1.5),
        'Low': prices - np.abs(np.random.randn(days) * 1.5),
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, days)
    })

    data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)

    return data


def test_strategy(data: pd.DataFrame, strategy: BaseStrategy, name: str) -> dict:
    """Run backtest on custom strategy."""
    # Generate signals
    signals = strategy.generate_signals_vectorized(data)
    if signals is None:
        signals = [strategy.generate_signal(data.iloc[:i+1]) for i in range(len(data))]

    data_copy = data.copy()
    data_copy['Signal'] = signals

    # Run backtest manually
    initial_capital = 10000
    cash = initial_capital
    position = 0
    portfolio_values = []

    for i, row in data_copy.iterrows():
        price = row['Close']
        signal = row['Signal']

        # Execute trades
        if signal == 1 and position == 0:
            shares = int(cash * 0.95 / price)
            if shares > 0:
                cost = shares * price * 1.001
                if cost <= cash:
                    cash -= cost
                    position = shares

        elif signal == -1 and position > 0:
            proceeds = position * price * 0.999
            cash += proceeds
            position = 0

        portfolio_values.append(cash + position * price)

    # Calculate metrics
    pv = np.array(portfolio_values)
    returns = np.diff(pv) / pv[:-1]

    total_return = (pv[-1] - initial_capital) / initial_capital
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    peak = np.maximum.accumulate(pv)
    drawdown = (peak - pv) / peak
    max_dd = np.max(drawdown)

    return {
        'name': name,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'final_value': pv[-1]
    }


def main():
    print("=" * 70)
    print("CUSTOM STRATEGY EXAMPLE")
    print("=" * 70)

    # Generate data
    print("\n1. Generating sample data...")
    data = generate_sample_data(500)
    validation = validate_ohlcv_data(data, fix_issues=True)
    data = validation.cleaned_data
    print(f"   Data: {len(data)} bars")

    # Define custom strategies
    print("\n2. Creating custom strategies...")

    strategies = [
        (DualMomentumStrategy(ma_period=50, roc_period=20), "Dual Momentum (50/20)"),
        (DualMomentumStrategy(ma_period=30, roc_period=10), "Dual Momentum (30/10)"),
        (VolatilityBreakoutStrategy(atr_period=14, breakout_mult=1.5), "Vol Breakout (1.5x)"),
        (VolatilityBreakoutStrategy(atr_period=14, breakout_mult=2.0), "Vol Breakout (2.0x)"),
    ]

    for strategy, name in strategies:
        print(f"   • {name}")

    # Test strategies
    print("\n3. Running backtests...")
    results = []

    for strategy, name in strategies:
        result = test_strategy(data.copy(), strategy, name)
        results.append(result)
        print(f"   ✓ {name}")

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True):
        print(f"{r['name']:<25} "
              f"{r['total_return']:>+9.2%} "
              f"{r['sharpe_ratio']:>10.2f} "
              f"{r['max_drawdown']:>9.2%}")

    # Show strategy code
    print("\n" + "=" * 70)
    print("STRATEGY CODE TEMPLATE")
    print("=" * 70)

    print("""
class MyCustomStrategy(BaseStrategy):
    def __init__(self, param1: int = 10, param2: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2

    def generate_signal(self, data: pd.DataFrame) -> int:
        '''Generate signal for latest bar.'''
        if len(data) < self.param1:
            return 0

        close = data['Close']

        # Your indicator logic here
        indicator = close.rolling(self.param1).mean().iloc[-1]

        if close.iloc[-1] > indicator * (1 + self.param2):
            return 1   # Buy
        elif close.iloc[-1] < indicator * (1 - self.param2):
            return -1  # Sell
        return 0       # Hold

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        '''Vectorized version for backtesting (10-100x faster).'''
        close = data['Close']
        indicator = close.rolling(self.param1).mean()

        signals = np.where(
            close > indicator * (1 + self.param2), 1,
            np.where(close < indicator * (1 - self.param2), -1, 0)
        )
        signals[:self.param1] = 0
        return signals.tolist()
""")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
