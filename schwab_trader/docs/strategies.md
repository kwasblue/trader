# Trading Strategies Guide

Schwab Trader includes 18 built-in trading strategies. This guide covers each strategy, its parameters, and usage.

## Strategy Overview

| Strategy | Type | Description |
|----------|------|-------------|
| SMA | Trend | Simple Moving Average crossover |
| EMA | Trend | Exponential Moving Average crossover |
| MACD | Trend/Momentum | Moving Average Convergence Divergence |
| RSI | Oscillator | Relative Strength Index |
| Bollinger | Volatility | Bollinger Bands mean reversion |
| Momentum | Trend | Price momentum comparison |
| Mean Reversion | Mean Reversion | Z-score based mean reversion |
| Breakout | Trend | Price breakout detection |
| ADX | Trend | Average Directional Index |
| Stochastic | Oscillator | Stochastic oscillator crossovers |
| Ichimoku | Trend | Ichimoku Cloud signals |
| PSAR | Trend | Parabolic SAR |
| VWAP | Trend | Volume Weighted Average Price |
| Donchian | Trend | Donchian channel breakouts |
| Combined | Ensemble | Multiple strategy voting |
| Logistic Regression | ML | Machine learning predictions |

---

## Using Strategies

### Loading a Strategy

```python
from strategies.strategy_registry import load_strategy, list_strategies

# See all available strategies
print(list_strategies())
# ['sma', 'ema', 'macd', 'rsi', 'bollinger', 'momentum', 'meanreversion', ...]

# Load with default parameters
strategy = load_strategy('sma')

# Load with custom parameters
strategy = load_strategy('sma', params={'fast': 5, 'slow': 20})
```

### Generating Signals

```python
import pandas as pd

# Your OHLCV data
data = pd.DataFrame({
    'Date': [...],
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
})

# Generate signal for latest bar
signal = strategy.generate_signal(data)
# Returns: 1 (buy), -1 (sell), or 0 (hold)

# Generate signals for all bars (vectorized, faster)
signals = strategy.generate_signals_vectorized(data)
# Returns: [0, 0, 1, 1, -1, -1, 0, ...]
```

---

## Strategy Details

### SMA Strategy (Simple Moving Average)

**Logic**: Buy when fast SMA crosses above slow SMA, sell when crosses below.

```python
strategy = load_strategy('sma', params={
    'fast': 10,    # Fast SMA period (default: 10)
    'slow': 30     # Slow SMA period (default: 30)
})
```

**Signals**:
- `1` (Buy): Fast SMA > Slow SMA
- `-1` (Sell): Fast SMA < Slow SMA
- `0` (Hold): Warmup period

**Best for**: Trending markets, longer timeframes

---

### EMA Strategy (Exponential Moving Average)

**Logic**: Same as SMA but uses exponential weighting for faster response.

```python
strategy = load_strategy('ema', params={
    'short_window': 12,   # Short EMA period (default: 20)
    'long_window': 26     # Long EMA period (default: 50)
})
```

**Signals**:
- `1` (Buy): Short EMA > Long EMA
- `-1` (Sell): Short EMA < Long EMA

**Best for**: Faster trend detection, responsive to recent price changes

---

### MACD Strategy

**Logic**: Buy when MACD line crosses above signal line, sell on cross below.

```python
strategy = load_strategy('macd', params={
    'fast_window': 12,     # Fast EMA period (default: 12)
    'slow_window': 26,     # Slow EMA period (default: 26)
    'signal_window': 9     # Signal line period (default: 9)
})
```

**Calculation**:
```
MACD Line = EMA(fast) - EMA(slow)
Signal Line = EMA(MACD Line, signal_window)
```

**Signals**:
- `1` (Buy): MACD > Signal Line
- `-1` (Sell): MACD < Signal Line

**Best for**: Trend confirmation, momentum measurement

---

### RSI Strategy (Relative Strength Index)

**Logic**: Buy when oversold (RSI < threshold), sell when overbought (RSI > threshold).

```python
strategy = load_strategy('rsi', params={
    'window': 14,         # RSI period (default: 14)
    'oversold': 30,       # Oversold threshold (default: 30)
    'overbought': 70      # Overbought threshold (default: 70)
})
```

**Calculation**:
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss
```

**Signals**:
- `1` (Buy): RSI < oversold
- `-1` (Sell): RSI > overbought
- `0` (Hold): RSI between thresholds

**Best for**: Range-bound markets, reversal detection

---

### Bollinger Bands Strategy

**Logic**: Buy when price below lower band (oversold), sell when above upper band.

```python
strategy = load_strategy('bollinger', params={
    'window': 20,      # Moving average period (default: 20)
    'num_std': 2       # Standard deviations (default: 2)
})
```

**Calculation**:
```
Middle Band = SMA(window)
Upper Band = Middle + (num_std × StdDev)
Lower Band = Middle - (num_std × StdDev)
```

**Signals**:
- `1` (Buy): Close < Lower Band
- `-1` (Sell): Close > Upper Band
- `0` (Hold): Price within bands

**Best for**: Mean reversion, volatility-based entries

---

### Momentum Strategy

**Logic**: Buy if current price > price N periods ago, sell otherwise.

```python
strategy = load_strategy('momentum', params={
    'lookback': 20     # Comparison period (default: 20)
})
```

**Signals**:
- `1` (Buy): Close > Close[lookback periods ago]
- `-1` (Sell): Close < Close[lookback periods ago]

**Best for**: Trend following, breakout confirmation

---

### Mean Reversion Strategy

**Logic**: Buy when price is significantly below mean (negative Z-score), sell when above.

```python
strategy = load_strategy('meanreversion', params={
    'window': 14,       # Lookback period (default: 14)
    'threshold': 1.0    # Z-score threshold (default: 1.0)
})
```

**Calculation**:
```
Z-score = (Price - Mean) / StdDev
```

**Signals**:
- `1` (Buy): Z-score < -threshold (oversold)
- `-1` (Sell): Z-score > threshold (overbought)
- `0` (Hold): Z-score within thresholds

**Best for**: Range-bound markets, pairs trading

---

### Breakout Strategy

**Logic**: Buy on breakout above rolling high, sell on breakdown below rolling low.

```python
strategy = load_strategy('breakout', params={
    'window': 20       # Lookback period (default: 20)
})
```

**Signals**:
- `1` (Buy): Close > Highest High[window]
- `-1` (Sell): Close < Lowest Low[window]
- `0` (Hold): Price within range

**Best for**: Trend initiation, momentum breakouts

---

### ADX Strategy (Average Directional Index)

**Logic**: Trade in direction of +DI/-DI when trend is strong (ADX > threshold).

```python
strategy = load_strategy('adx', params={
    'window': 14,       # ADX period (default: 14)
    'threshold': 25     # Minimum ADX for signal (default: 25)
})
```

**Signals**:
- `1` (Buy): ADX > threshold AND +DI > -DI
- `-1` (Sell): ADX > threshold AND -DI > +DI
- `0` (Hold): ADX < threshold (weak trend)

**Best for**: Trend strength confirmation, avoiding choppy markets

---

### Stochastic Strategy

**Logic**: Trade crossovers of %K and %D in oversold/overbought zones.

```python
strategy = load_strategy('stochastic', params={
    'k_window': 14,     # %K period (default: 14)
    'd_window': 3,      # %D smoothing (default: 3)
    'oversold': 20,     # Oversold level (default: 20)
    'overbought': 80    # Overbought level (default: 80)
})
```

**Signals**:
- `1` (Buy): %K crosses above %D while below oversold
- `-1` (Sell): %K crosses below %D while above overbought
- `0` (Hold): No crossover in extreme zones

**Best for**: Reversal detection, timing entries

---

### Ichimoku Strategy

**Logic**: Trade based on price position relative to the Ichimoku Cloud.

```python
strategy = load_strategy('ichimoku')
# Uses standard periods: 9, 26, 52
```

**Components**:
- Tenkan-sen (Conversion): (9-high + 9-low) / 2
- Kijun-sen (Base): (26-high + 26-low) / 2
- Senkou Span A: (Tenkan + Kijun) / 2, shifted 26 periods
- Senkou Span B: (52-high + 52-low) / 2, shifted 26 periods

**Signals**:
- `1` (Buy): Close > Senkou Span A
- `-1` (Sell): Close < Senkou Span B
- `0` (Hold): Price in cloud

**Best for**: Comprehensive trend analysis, support/resistance

---

### PSAR Strategy (Parabolic SAR)

**Logic**: Buy when price above PSAR, sell when below.

```python
strategy = load_strategy('psar')
# Uses ta library defaults
```

**Signals**:
- `1` (Buy): Close > PSAR
- `-1` (Sell): Close < PSAR

**Best for**: Trend following with trailing stops

---

### VWAP Strategy

**Logic**: Buy when price below VWAP (undervalued), sell when above.

```python
strategy = load_strategy('vwap')
```

**Calculation**:
```
VWAP = Cumulative(Price × Volume) / Cumulative(Volume)
```

**Signals**:
- `1` (Buy): Close < VWAP
- `-1` (Sell): Close > VWAP

**Best for**: Intraday trading, institutional-style entries

---

### Donchian Strategy

**Logic**: Trade breakouts from Donchian channels (similar to Breakout).

```python
strategy = load_strategy('donchian', params={
    'window': 20       # Channel period (default: 20)
})
```

**Signals**:
- `1` (Buy): Close > Donchian High (previous)
- `-1` (Sell): Close < Donchian Low (previous)

**Best for**: Trend following, turtle trading style

---

### Combined Strategy

**Logic**: Combine multiple strategies using voting or weighted average.

```python
from strategies.strategy_registry.sma_strategy import SMAStrategy
from strategies.strategy_registry.rsi_strategy import RSIStrategy

strategy = load_strategy('combined', params={
    'strategy_instances': [
        SMAStrategy(fast=10, slow=30),
        RSIStrategy(window=14)
    ],
    'combine_method': 'vote',   # 'vote' or 'weighted'
    'weights': [0.6, 0.4]       # For weighted method
})
```

**Best for**: Reducing false signals, ensemble approaches

---

### Logistic Regression Strategy (ML)

**Logic**: Use trained ML model to predict price direction.

```python
from sklearn.pipeline import Pipeline
# Requires trained model

strategy = load_strategy('logisticregression', params={
    'model': trained_pipeline,
    'buy_threshold': 0.52,
    'sell_threshold': 0.48
})
```

**Best for**: Pattern recognition, feature-based prediction

---

## Creating Custom Strategies

### Basic Template

```python
# strategies/strategy_registry/my_strategy.py
import numpy as np
import pandas as pd
from typing import Optional, List
from core.base.base_strategy import BaseStrategy


class MyStrategy(BaseStrategy):
    """My custom trading strategy."""

    def __init__(self, period: int = 14, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.threshold = threshold

    def generate_signal(self, data: pd.DataFrame) -> int:
        """Generate signal for the latest bar."""
        if len(data) < self.period:
            return 0

        close = data['Close'] if 'Close' in data.columns else data['close']

        # Your signal logic here
        indicator = close.rolling(self.period).mean().iloc[-1]
        current_price = close.iloc[-1]

        if current_price > indicator * (1 + self.threshold):
            return -1  # Sell
        elif current_price < indicator * (1 - self.threshold):
            return 1   # Buy
        return 0       # Hold

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        """Vectorized signal generation for backtesting (10-100x faster)."""
        close = data['Close'] if 'Close' in data.columns else data['close']

        indicator = close.rolling(self.period).mean()

        signals = np.where(
            close > indicator * (1 + self.threshold), -1,
            np.where(close < indicator * (1 - self.threshold), 1, 0)
        )

        # No signal during warmup
        signals[:self.period] = 0

        return signals.tolist()
```

### Registration

Strategies are auto-discovered when placed in `strategies/strategy_registry/`.

The strategy will be available as:
- `'my'` (class name without "Strategy", lowercase)
- `'mystrategy'`
- `'my_strategy'`

```python
strategy = load_strategy('my', params={'period': 20, 'threshold': 0.3})
```

---

## Strategy Selection Guide

| Market Condition | Recommended Strategies |
|-----------------|------------------------|
| Strong Trend | SMA, EMA, MACD, Momentum, ADX |
| Range-Bound | RSI, Bollinger, Mean Reversion, Stochastic |
| High Volatility | Bollinger, ADX (with high threshold) |
| Low Volatility | Breakout, Donchian |
| Uncertain | Combined (ensemble) |

---

## Performance Comparison

Run backtests to compare strategy performance:

```python
from core.backtest_suite import VectorizedBacktester

strategies = ['sma', 'ema', 'macd', 'rsi', 'momentum']
results = {}

for name in strategies:
    bt = VectorizedBacktester(data)
    portfolio = bt.run(name)
    results[name] = bt.get_metrics(portfolio)

# Compare
import pandas as pd
comparison = pd.DataFrame(results).T
print(comparison.sort_values('sharpe_ratio', ascending=False))
```
