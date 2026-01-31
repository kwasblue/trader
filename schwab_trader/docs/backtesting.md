# Backtesting Guide

This guide covers the backtesting framework, including basic backtesting, advanced analysis, and optimization.

## Overview

The backtesting system provides:

- **Basic Backtester** - Traditional bar-by-bar simulation
- **Vectorized Backtester** - 10-100x faster for large datasets
- **Grid Search** - Parameter optimization
- **Walk-Forward Analysis** - Out-of-sample testing
- **Monte Carlo Simulation** - Confidence intervals
- **Benchmark Comparison** - Alpha, Beta, Information Ratio

---

## Basic Backtesting

### Using the Standard Backtester

```python
import pandas as pd
from core.backtester import Backtester
from core.position_sizer import DynamicPositionSizer

# Load data
data = pd.read_csv('AAPL_daily.csv')

# Initialize backtester
bt = Backtester(
    data=data,
    initial_capital=10000,
    transaction_cost=0.001,  # 0.1%
    risk_free_rate=0.02      # 2% annual
)

# Create position sizer
sizer = DynamicPositionSizer(
    risk_per_trade=0.02,     # 2% risk per trade
    max_position_pct=0.20,   # Max 20% in single position
    capital=10000
)

# Run backtest
results = bt.run_backtest(
    strategy_name='sma',
    strategy_params={'fast': 10, 'slow': 30},
    sizer=sizer
)

# Evaluate
performance = bt.evaluate_performance(results)
print(performance)
```

### Output Metrics

| Metric | Description |
|--------|-------------|
| Standard Deviation | Daily return volatility |
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Value at Risk (VaR) | 5% worst-case daily loss |

### Generating Reports

```python
# Generate PDF report with charts
bt.generate_report(
    portfolio_df=results,
    strategy_name='SMA Crossover',
    performance=performance,
    file_name='backtest_report.pdf'
)

# Plot results
bt.plot_results(results, 'SMA Crossover', save_path='strategy_plot.png')
```

---

## Vectorized Backtesting

For faster backtesting on large datasets:

```python
from core.backtest_suite import VectorizedBacktester, VolatilityAdjustedSlippage

# Create slippage model
slippage = VolatilityAdjustedSlippage(
    base_slippage=0.0005,
    volatility_multiplier=2.0
)

# Initialize
bt = VectorizedBacktester(
    data=data,
    initial_capital=10000,
    transaction_cost=0.001,
    slippage_model=slippage
)

# Run with options
results = bt.run(
    strategy_name='ema',
    strategy_params={'short_window': 12, 'long_window': 26},
    position_sizing='volatility_scaled',  # or 'fixed', 'risk_parity'
    position_size=0.1,        # 10% of capital per trade
    stop_loss_atr=2.0,        # 2 ATR stop loss
    take_profit_atr=3.0       # 3 ATR take profit
)

# Get comprehensive metrics
metrics = bt.get_metrics(results)
```

### Performance Comparison

```
Standard Backtester:  ~5-10 seconds for 10,000 bars
Vectorized Backtester: ~0.05-0.1 seconds for 10,000 bars
Speedup: 50-100x
```

---

## Data Validation

Always validate data before backtesting:

```python
from core.backtest_suite import validate_ohlcv_data

# Validate and auto-fix issues
result = validate_ohlcv_data(data, fix_issues=True)

if result.is_valid:
    print("Data valid!")
    clean_data = result.cleaned_data
else:
    print("Errors:", result.errors)

# Warnings about potential issues
for warning in result.warnings:
    print(f"Warning: {warning}")
```

### Validation Checks

| Check | Description |
|-------|-------------|
| Required columns | Date, Open, High, Low, Close exist |
| NaN values | Forward-fills NaN if fix_issues=True |
| OHLC consistency | High >= Low, High >= Open/Close |
| Positive prices | All prices > 0 |
| Date sorting | Dates in ascending order |
| Duplicates | Removes duplicate timestamps |
| Extreme moves | Warns if >50% price change in single bar |

---

## Slippage Models

Choose appropriate slippage model for realistic simulation:

### Fixed Slippage

```python
from core.backtest_suite import FixedSlippage

slippage = FixedSlippage(slippage_pct=0.001)  # 0.1%
```

### Random Slippage

```python
from core.backtest_suite import RandomSlippage

slippage = RandomSlippage(min_pct=-0.001, max_pct=0.001)
```

### Volume-Based Slippage

Larger orders have more market impact:

```python
from core.backtest_suite import VolumeBasedSlippage

slippage = VolumeBasedSlippage(
    base_slippage=0.0001,   # Base slippage
    volume_impact=0.1,       # Impact coefficient
    max_slippage=0.02        # Cap at 2%
)
```

### Volatility-Adjusted Slippage

Higher volatility = more slippage:

```python
from core.backtest_suite import VolatilityAdjustedSlippage

slippage = VolatilityAdjustedSlippage(
    base_slippage=0.0005,
    volatility_multiplier=2.0,
    max_slippage=0.03
)
```

---

## Parameter Optimization

### Grid Search

Find optimal strategy parameters:

```python
from core.backtest_suite import grid_search

# Define parameter grid
param_grid = {
    'fast': [5, 10, 15, 20],
    'slow': [20, 30, 40, 50]
}

# Run optimization
result = grid_search(
    data=data,
    strategy_name='sma',
    param_grid=param_grid,
    metric='sharpe_ratio',  # Optimize for Sharpe
    initial_capital=10000,
    n_jobs=1,               # Parallel jobs
    verbose=True
)

print(f"Best params: {result.best_params}")
print(f"Best Sharpe: {result.best_metric:.4f}")

# All results for analysis
import pandas as pd
all_results = pd.DataFrame([
    {**r['params'], 'sharpe': r['metric_value']}
    for r in result.all_results
])
print(all_results.sort_values('sharpe', ascending=False))
```

### Available Metrics

| Metric | Description |
|--------|-------------|
| `sharpe_ratio` | Sharpe ratio (default) |
| `sortino_ratio` | Sortino ratio |
| `total_return` | Total return |
| `max_drawdown` | Maximum drawdown (negated) |

---

## Walk-Forward Analysis

Test strategy robustness with rolling train/test windows:

```python
from core.backtest_suite import walk_forward_analysis

# Define parameter grid
param_grid = {
    'fast': [5, 10, 15],
    'slow': [20, 30, 40]
}

# Run walk-forward
result = walk_forward_analysis(
    data=data,
    strategy_name='sma',
    param_grid=param_grid,
    train_size=252,    # 1 year training
    test_size=63,      # 3 months testing
    step_size=63,      # Roll forward 3 months
    metric='sharpe_ratio',
    verbose=True
)

print(f"Overall return: {result.overall_return:.2%}")
print(f"Average OOS Sharpe: {result.overall_sharpe:.4f}")

# Analyze each window
for window in result.windows:
    print(f"Window {window['window']}: "
          f"IS={window['is_metric']:.2f}, "
          f"OOS={window['oos_return']:.2%}")
```

### Understanding Walk-Forward

```
Data: |----Train----|--Test--|----Train----|--Test--|...
       [Window 1 IS] [OOS 1] [Window 2 IS] [OOS 2]

1. Optimize on training window → find best params
2. Test on out-of-sample window → measure performance
3. Roll forward and repeat
4. Aggregate out-of-sample results
```

---

## Monte Carlo Simulation

Estimate confidence intervals by randomizing trade order:

```python
from core.backtest_suite import monte_carlo_simulation

# First, run backtest to get trades
bt = Backtester(data, initial_capital=10000)
results = bt.run_backtest('sma', {'fast': 10, 'slow': 30}, sizer)
trades = bt.trade_log  # List of trade dicts

# Run Monte Carlo
mc_result = monte_carlo_simulation(
    trades=trades,
    initial_capital=10000,
    n_simulations=1000,
    seed=42
)

print(f"Mean return: {mc_result.mean_return:.2%}")
print(f"Median return: {mc_result.median_return:.2%}")
print(f"Std return: {mc_result.std_return:.2%}")
print(f"95% CI: [{mc_result.confidence_interval_95[0]:.2%}, "
      f"{mc_result.confidence_interval_95[1]:.2%}]")

# Percentiles
for pct, val in mc_result.percentiles.items():
    print(f"{pct}th percentile: {val:.2%}")
```

### Interpretation

- If 95% CI includes negative returns: strategy may not be robust
- If median significantly different from mean: return distribution is skewed
- Wide CI: high variance in outcomes

---

## Benchmark Comparison

Compare strategy to benchmark (e.g., buy-and-hold, S&P 500):

```python
from core.backtest_suite import compare_to_benchmark
import pandas as pd

# Strategy returns
strategy_returns = results['Strategy_Return']

# Benchmark returns (e.g., buy-and-hold)
benchmark_returns = data['Close'].pct_change().fillna(0)

# Compare
comparison = compare_to_benchmark(
    strategy_returns=strategy_returns,
    benchmark_returns=benchmark_returns,
    risk_free_rate=0.02
)

print(f"Strategy Return: {comparison.strategy_return:.2%}")
print(f"Benchmark Return: {comparison.benchmark_return:.2%}")
print(f"Excess Return: {comparison.excess_return:.2%}")
print(f"Strategy Sharpe: {comparison.strategy_sharpe:.2f}")
print(f"Benchmark Sharpe: {comparison.benchmark_sharpe:.2f}")
print(f"Beta: {comparison.beta:.2f}")
print(f"Alpha: {comparison.alpha:.2%}")
print(f"Information Ratio: {comparison.information_ratio:.2f}")
print(f"Up Capture: {comparison.up_capture:.2f}")
print(f"Down Capture: {comparison.down_capture:.2f}")
```

### Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| Excess Return | Strategy - Benchmark | > 0 |
| Alpha | Risk-adjusted outperformance | > 0 |
| Beta | Market sensitivity | Depends on strategy |
| Information Ratio | Excess return / Tracking error | > 0.5 |
| Up Capture | % of benchmark gains captured | > 1.0 |
| Down Capture | % of benchmark losses captured | < 1.0 |

---

## Complete Backtesting Workflow

```python
import pandas as pd
from core.backtest_suite import (
    validate_ohlcv_data,
    VectorizedBacktester,
    VolumeBasedSlippage,
    grid_search,
    walk_forward_analysis,
    monte_carlo_simulation,
    compare_to_benchmark
)

# 1. Load and validate data
data = pd.read_csv('AAPL_daily.csv')
validation = validate_ohlcv_data(data, fix_issues=True)
if not validation.is_valid:
    raise ValueError(validation.errors)
data = validation.cleaned_data

# 2. Find optimal parameters
param_grid = {'fast': [5, 10, 15], 'slow': [20, 30, 40]}
opt_result = grid_search(data, 'sma', param_grid, metric='sharpe_ratio')
best_params = opt_result.best_params
print(f"Best params: {best_params}")

# 3. Walk-forward validation
wf_result = walk_forward_analysis(
    data, 'sma', param_grid,
    train_size=252, test_size=63
)
print(f"Walk-forward return: {wf_result.overall_return:.2%}")

# 4. Run final backtest with best params
bt = VectorizedBacktester(data, slippage_model=VolumeBasedSlippage())
results = bt.run('sma', best_params)
metrics = bt.get_metrics(results)

# 5. Monte Carlo confidence intervals
# (Extract trades from backtest first)
# mc_result = monte_carlo_simulation(trades, n_simulations=1000)

# 6. Compare to benchmark
benchmark_returns = data['Close'].pct_change().fillna(0)
comparison = compare_to_benchmark(
    results['Strategy_Return'],
    benchmark_returns
)

# 7. Generate report
print("\n=== Final Results ===")
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Alpha: {comparison.alpha:.2%}")
print(f"Information Ratio: {comparison.information_ratio:.2f}")
```

---

## Best Practices

### 1. Data Quality
- Always validate data before backtesting
- Check for survivorship bias
- Use adjusted prices for splits/dividends

### 2. Realistic Costs
- Include transaction costs (0.1-0.5%)
- Use appropriate slippage model
- Account for bid-ask spread

### 3. Avoid Overfitting
- Use walk-forward analysis
- Keep parameter space small
- Test on multiple instruments

### 4. Position Sizing
- Use risk-based sizing, not fixed quantities
- Account for volatility
- Respect position limits

### 5. Sample Size
- Use sufficient data (1000+ bars minimum)
- Ensure enough trades for statistical significance
- Consider different market regimes

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Look-ahead bias | Use `.shift()` properly, no future data |
| Survivorship bias | Include delisted stocks |
| Overfitting | Walk-forward validation |
| Ignoring costs | Include realistic transaction costs |
| Ignoring slippage | Use volume-based slippage model |
| Curve fitting | Test on out-of-sample data |
| Small sample | Use 3+ years of data |
