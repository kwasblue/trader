# Code Examples

Practical examples for common use cases with Schwab Trader.

---

## Basic Examples

### 1. Simple Backtest

```python
import pandas as pd
from core.backtester import Backtester
from core.position_sizer import DynamicPositionSizer

# Load data
data = pd.read_csv('AAPL_daily.csv')

# Ensure Date column is datetime
data['Date'] = pd.to_datetime(data['Date'])

# Initialize backtester
bt = Backtester(data, initial_capital=10000)

# Create position sizer
sizer = DynamicPositionSizer(
    risk_per_trade=0.02,
    max_position_pct=0.20,
    capital=10000
)

# Run SMA crossover strategy
results = bt.run_backtest('sma', {'fast': 10, 'slow': 30}, sizer)

# Print results
print(f"Final Value: ${results['Portfolio_Value'].iloc[-1]:,.2f}")
print(f"Total Return: {(results['Portfolio_Value'].iloc[-1]/10000 - 1)*100:.2f}%")
```

### 2. Multiple Strategy Comparison

```python
from core.backtest_suite import VectorizedBacktester
import pandas as pd

data = pd.read_csv('AAPL_daily.csv')
data['Date'] = pd.to_datetime(data['Date'])

strategies = {
    'SMA': ('sma', {'fast': 10, 'slow': 30}),
    'EMA': ('ema', {'short_window': 12, 'long_window': 26}),
    'MACD': ('macd', {}),
    'RSI': ('rsi', {'window': 14}),
    'Momentum': ('momentum', {'lookback': 20})
}

results = {}

for name, (strategy, params) in strategies.items():
    bt = VectorizedBacktester(data)
    portfolio = bt.run(strategy, params)
    metrics = bt.get_metrics(portfolio)
    results[name] = metrics

# Create comparison table
comparison = pd.DataFrame(results).T
comparison = comparison.round(4)
print("\n=== Strategy Comparison ===")
print(comparison.sort_values('sharpe_ratio', ascending=False))
```

### 3. Custom Strategy Implementation

```python
# File: strategies/strategy_registry/custom_strategy.py
import numpy as np
import pandas as pd
from typing import Optional, List
from core.base.base_strategy import BaseStrategy


class DoubleMAStrategy(BaseStrategy):
    """
    Custom double moving average strategy.

    Buy when:
    - Fast MA > Medium MA > Slow MA (strong uptrend)

    Sell when:
    - Fast MA < Medium MA < Slow MA (strong downtrend)
    """

    def __init__(
        self,
        fast: int = 5,
        medium: int = 15,
        slow: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fast = fast
        self.medium = medium
        self.slow = slow

    def generate_signal(self, data: pd.DataFrame) -> int:
        if len(data) < self.slow:
            return 0

        close = data['Close'] if 'Close' in data.columns else data['close']

        ma_fast = close.rolling(self.fast).mean().iloc[-1]
        ma_medium = close.rolling(self.medium).mean().iloc[-1]
        ma_slow = close.rolling(self.slow).mean().iloc[-1]

        # Strong uptrend
        if ma_fast > ma_medium > ma_slow:
            return 1
        # Strong downtrend
        elif ma_fast < ma_medium < ma_slow:
            return -1
        return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        close = data['Close'] if 'Close' in data.columns else data['close']

        ma_fast = close.rolling(self.fast).mean()
        ma_medium = close.rolling(self.medium).mean()
        ma_slow = close.rolling(self.slow).mean()

        signals = np.where(
            (ma_fast > ma_medium) & (ma_medium > ma_slow), 1,
            np.where((ma_fast < ma_medium) & (ma_medium < ma_slow), -1, 0)
        )

        signals[:self.slow] = 0
        return signals.tolist()


# Usage
from strategies.strategy_registry import load_strategy

strategy = load_strategy('doublema', params={'fast': 5, 'medium': 15, 'slow': 30})
signal = strategy.generate_signal(data)
```

---

## Advanced Examples

### 4. Parameter Optimization with Walk-Forward

```python
from core.backtest_suite import grid_search, walk_forward_analysis
import pandas as pd

# Load data
data = pd.read_csv('SPY_daily.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Define parameter grid
param_grid = {
    'fast': [5, 10, 15, 20],
    'slow': [20, 30, 40, 50, 60]
}

# Step 1: Find best parameters on full dataset
print("=== Grid Search ===")
opt_result = grid_search(
    data=data,
    strategy_name='sma',
    param_grid=param_grid,
    metric='sharpe_ratio',
    verbose=True
)

print(f"\nBest params: {opt_result.best_params}")
print(f"Best Sharpe: {opt_result.best_metric:.4f}")

# Step 2: Validate with walk-forward analysis
print("\n=== Walk-Forward Analysis ===")
wf_result = walk_forward_analysis(
    data=data,
    strategy_name='sma',
    param_grid=param_grid,
    train_size=252,  # 1 year
    test_size=63,    # 3 months
    step_size=63,    # Roll quarterly
    verbose=True
)

print(f"\nOverall OOS Return: {wf_result.overall_return:.2%}")
print(f"Average OOS Sharpe: {wf_result.overall_sharpe:.4f}")

# Analyze parameter stability
params_used = pd.DataFrame(wf_result.in_sample_params)
print("\nParameter usage across windows:")
print(params_used.value_counts())
```

### 5. Monte Carlo Risk Analysis

```python
from core.backtest_suite import VectorizedBacktester, monte_carlo_simulation
import matplotlib.pyplot as plt
import numpy as np

# Run backtest
data = pd.read_csv('AAPL_daily.csv')
data['Date'] = pd.to_datetime(data['Date'])

bt = VectorizedBacktester(data, initial_capital=10000)
results = bt.run('ema', {'short_window': 12, 'long_window': 26})

# Create synthetic trades from returns
returns = results['Strategy_Return'].values
trades = [{'pnl': r * 10000} for r in returns if r != 0]

# Run Monte Carlo simulation
mc = monte_carlo_simulation(
    trades=trades,
    initial_capital=10000,
    n_simulations=5000,
    seed=42
)

# Print statistics
print("=== Monte Carlo Results (5000 simulations) ===")
print(f"Mean Return: {mc.mean_return:.2%}")
print(f"Median Return: {mc.median_return:.2%}")
print(f"Std Dev: {mc.std_return:.2%}")
print(f"95% Confidence Interval: [{mc.confidence_interval_95[0]:.2%}, "
      f"{mc.confidence_interval_95[1]:.2%}]")
print("\nPercentiles:")
for pct, val in mc.percentiles.items():
    print(f"  {pct}th: {val:.2%}")

# Plot distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Return distribution
axes[0].hist(mc.final_values, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(np.mean(mc.final_values), color='red', linestyle='--', label='Mean')
axes[0].axvline(np.median(mc.final_values), color='green', linestyle='--', label='Median')
axes[0].set_title('Final Portfolio Value Distribution')
axes[0].set_xlabel('Final Value ($)')
axes[0].legend()

# Drawdown distribution
axes[1].hist(mc.max_drawdowns, bins=50, edgecolor='black', alpha=0.7, color='red')
axes[1].set_title('Maximum Drawdown Distribution')
axes[1].set_xlabel('Max Drawdown')

plt.tight_layout()
plt.savefig('monte_carlo_analysis.png')
plt.show()
```

### 6. Multi-Asset Portfolio Backtest

```python
from core.backtest_suite import VectorizedBacktester, compare_to_benchmark
import pandas as pd
import numpy as np

# Load multiple assets
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
portfolio_value = 100000
allocation = 1.0 / len(symbols)  # Equal weight

results = {}
all_returns = pd.DataFrame()

for symbol in symbols:
    data = pd.read_csv(f'{symbol}_daily.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

    bt = VectorizedBacktester(
        data.reset_index(),
        initial_capital=portfolio_value * allocation
    )
    result = bt.run('ema', {'short_window': 12, 'long_window': 26})
    result = result.set_index('Date')

    results[symbol] = bt.get_metrics(result)
    all_returns[symbol] = result['Strategy_Return']

# Calculate portfolio returns (equal weighted)
portfolio_returns = all_returns.mean(axis=1)
portfolio_cumulative = (1 + portfolio_returns).cumprod()

# Calculate portfolio metrics
sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
max_dd = (portfolio_cumulative / portfolio_cumulative.cummax() - 1).min()

print("=== Portfolio Results ===")
print(f"Total Return: {(portfolio_cumulative.iloc[-1] - 1):.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")

print("\n=== Per-Asset Performance ===")
for symbol, metrics in results.items():
    print(f"{symbol}: Return={metrics['total_return']:.2%}, "
          f"Sharpe={metrics['sharpe_ratio']:.2f}")
```

### 7. Real-Time Strategy Testing

```python
import asyncio
from core.events import EventBus, Event
from strategies.strategy_registry import load_strategy
import pandas as pd

class RealtimeStrategyTester:
    """Test strategy with simulated real-time data."""

    def __init__(self, strategy_name: str, params: dict = None):
        self.strategy = load_strategy(strategy_name, params)
        self.bus = EventBus()
        self.data_buffer = pd.DataFrame()
        self.signals = []

    async def on_bar(self, event: Event):
        """Process new bar and generate signal."""
        bar = event.payload

        # Add bar to buffer
        new_row = pd.DataFrame([bar])
        self.data_buffer = pd.concat([self.data_buffer, new_row], ignore_index=True)

        # Keep last 100 bars
        if len(self.data_buffer) > 100:
            self.data_buffer = self.data_buffer.tail(100)

        # Generate signal
        if len(self.data_buffer) >= 30:  # Minimum for strategy
            signal = self.strategy.generate_signal(self.data_buffer)
            self.signals.append({
                'timestamp': bar['Date'],
                'price': bar['Close'],
                'signal': signal
            })

            if signal != 0:
                print(f"[{bar['Date']}] Signal: {'BUY' if signal == 1 else 'SELL'} "
                      f"at ${bar['Close']:.2f}")

    async def run_simulation(self, data: pd.DataFrame, delay: float = 0.1):
        """Simulate real-time bars."""
        await self.bus.subscribe('new_bar', self.on_bar)

        for _, row in data.iterrows():
            bar = row.to_dict()
            await self.bus.emit(Event('new_bar', bar))
            await asyncio.sleep(delay)

        print(f"\nTotal signals generated: {len([s for s in self.signals if s['signal'] != 0])}")


# Usage
async def main():
    data = pd.read_csv('AAPL_daily.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    tester = RealtimeStrategyTester('macd', {
        'fast_window': 12,
        'slow_window': 26,
        'signal_window': 9
    })

    await tester.run_simulation(data.tail(100), delay=0.01)

asyncio.run(main())
```

### 8. Strategy Performance Report

```python
from core.backtest_suite import (
    VectorizedBacktester,
    validate_ohlcv_data,
    compare_to_benchmark
)
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def generate_strategy_report(
    data: pd.DataFrame,
    strategy_name: str,
    strategy_params: dict,
    benchmark_data: pd.DataFrame = None
):
    """Generate comprehensive strategy performance report."""

    # Validate data
    validation = validate_ohlcv_data(data, fix_issues=True)
    if not validation.is_valid:
        raise ValueError(f"Data validation failed: {validation.errors}")
    data = validation.cleaned_data

    # Run backtest
    bt = VectorizedBacktester(data, initial_capital=10000)
    results = bt.run(strategy_name, strategy_params)
    metrics = bt.get_metrics(results)

    # Generate report
    report = f"""
================================================================================
                        STRATEGY PERFORMANCE REPORT
================================================================================

Strategy: {strategy_name.upper()}
Parameters: {strategy_params}
Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Period: {data['Date'].min()} to {data['Date'].max()}
Total Bars: {len(data)}

--------------------------------------------------------------------------------
                           PERFORMANCE METRICS
--------------------------------------------------------------------------------

Total Return:       {metrics['total_return']:>12.2%}
Sharpe Ratio:       {metrics['sharpe_ratio']:>12.2f}
Sortino Ratio:      {metrics['sortino_ratio']:>12.2f}
Max Drawdown:       {metrics['max_drawdown']:>12.2%}
Win Rate:           {metrics['win_rate']:>12.2%}
Profit Factor:      {metrics['profit_factor']:>12.2f}
Number of Trades:   {metrics['num_trades']:>12d}
Final Value:        ${metrics['final_value']:>11,.2f}

"""

    # Benchmark comparison
    if benchmark_data is not None:
        benchmark_returns = benchmark_data['Close'].pct_change().fillna(0)
        comparison = compare_to_benchmark(
            results['Strategy_Return'],
            benchmark_returns
        )

        report += f"""
--------------------------------------------------------------------------------
                         BENCHMARK COMPARISON
--------------------------------------------------------------------------------

Strategy Return:    {comparison.strategy_return:>12.2%}
Benchmark Return:   {comparison.benchmark_return:>12.2%}
Excess Return:      {comparison.excess_return:>12.2%}

Alpha (annual):     {comparison.alpha:>12.2%}
Beta:               {comparison.beta:>12.2f}
Information Ratio:  {comparison.information_ratio:>12.2f}
Tracking Error:     {comparison.tracking_error:>12.2%}

Up Capture:         {comparison.up_capture:>12.2f}
Down Capture:       {comparison.down_capture:>12.2f}

"""

    report += """
================================================================================
"""

    print(report)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Equity curve
    axes[0, 0].plot(results['Date'], results['Portfolio_Value'])
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Portfolio Value ($)')

    # Drawdown
    axes[0, 1].fill_between(results['Date'], results['Drawdown'], 0, color='red', alpha=0.3)
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Drawdown %')

    # Returns distribution
    axes[1, 0].hist(results['Strategy_Return'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Return Distribution')
    axes[1, 0].set_xlabel('Daily Return')
    axes[1, 0].set_ylabel('Frequency')

    # Rolling Sharpe
    rolling_sharpe = (
        results['Strategy_Return'].rolling(63).mean() /
        results['Strategy_Return'].rolling(63).std() * np.sqrt(252)
    )
    axes[1, 1].plot(results['Date'], rolling_sharpe)
    axes[1, 1].axhline(0, color='red', linestyle='--')
    axes[1, 1].set_title('Rolling 3-Month Sharpe Ratio')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Sharpe Ratio')

    plt.tight_layout()
    plt.savefig(f'{strategy_name}_report.png', dpi=150)
    plt.show()

    return metrics


# Usage
import numpy as np

data = pd.read_csv('AAPL_daily.csv')
data['Date'] = pd.to_datetime(data['Date'])

metrics = generate_strategy_report(
    data=data,
    strategy_name='ema',
    strategy_params={'short_window': 12, 'long_window': 26},
    benchmark_data=data  # Use same data as buy-and-hold benchmark
)
```

---

## Quick Reference

### Loading Strategies

```python
from strategies.strategy_registry import load_strategy, list_strategies

# List all
print(list_strategies())

# Load
strategy = load_strategy('sma', {'fast': 10, 'slow': 30})
signal = strategy.generate_signal(data)
```

### Running Backtests

```python
from core.backtest_suite import VectorizedBacktester

bt = VectorizedBacktester(data)
results = bt.run('ema', {'short_window': 12, 'long_window': 26})
metrics = bt.get_metrics(results)
```

### Optimization

```python
from core.backtest_suite import grid_search

result = grid_search(
    data, 'sma',
    {'fast': [5, 10, 15], 'slow': [20, 30, 40]},
    metric='sharpe_ratio'
)
print(result.best_params)
```

### Validation

```python
from core.backtest_suite import validate_ohlcv_data

result = validate_ohlcv_data(data, fix_issues=True)
clean_data = result.cleaned_data
```
