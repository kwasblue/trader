# Automated Parameter Tuning Guide

## Overview

The continuous adaptive sizing system has 10+ parameters that affect performance. This guide explains how to automatically find optimal parameter values through backtesting.

---

## Quick Start

### **1. Install Dependencies**

```bash
pip install scikit-optimize  # For Bayesian optimization
```

### **2. Run Quick Tuning (10 trials)**

```bash
cd /Users/kwasiaddo/projects/trader/amsterdam
python tools/tune_adaptive_sizing.py --symbol AAPL --n-trials 10
```

### **3. Run Full Tuning (100 trials)**

```bash
python tools/tune_adaptive_sizing.py --symbol AAPL --n-trials 100 --output results/tuning_AAPL.json
```

### **4. Apply Best Parameters**

The tuner saves results to JSON. Copy the `best_params` section to your `config/continuous_adaptation.json`.

---

## How It Works

### **Walk-Forward Optimization**

Prevents overfitting by testing on future data the optimizer never saw:

```
Data Split (3 windows example):
[====Window 1====][====Window 2====][====Window 3====]
[Train|Test     ][Train|Test     ][Train|Test     ]
 70%   30%        70%   30%        70%   30%

Process:
1. Window 1: Optimize on 70%, test on 30%
2. Window 2: Optimize on 70%, test on 30%
3. Window 3: Optimize on 70%, test on 30%
4. Select parameters with best AVERAGE test performance
```

**Why this matters:**
- Traditional optimization tests on same data it trained on (overfits)
- Walk-forward tests on future data (realistic)
- Averaging across windows ensures robustness

### **Bayesian Optimization**

Intelligently explores parameter space:

```
Trial 1: Random guess
Trial 2: Random guess
Trial 3: Random guess (build initial model)
Trial 4: Use model to pick promising area
Trial 5: Refine around best area
...
Trial 50: Converged to optimum
```

**vs Random Search:**
- Random: 100 trials, ~60% coverage
- Bayesian: 100 trials, ~90% coverage (smarter)

### **Objective Function**

Combines multiple metrics with penalties:

```python
objective = sharpe_ratio
           - (max_drawdown / 100) * 2.0     # Penalize large drawdowns
           - (30 - num_trades) * 0.05        # Penalize too few trades
           - regularization_penalty          # Penalize extreme parameters
```

**Example:**
- Sharpe 2.0, DD 30%: objective = 2.0 - 0.6 = 1.4
- Sharpe 1.8, DD 15%: objective = 1.8 - 0.3 = 1.5 ← Better!

---

## Parameters Being Tuned

| Parameter | Description | Search Range | Impact |
|-----------|-------------|--------------|--------|
| **Bayesian Shrinkage** | | | |
| `bayesian_prior_sharpe` | Expected baseline Sharpe | [0.25, 1.0] | Cold-start conservatism |
| `bayesian_prior_weight` | Prior strength (trades equivalent) | [3, 10] | Shrinkage intensity |
| **Smoothing** | | | |
| `sharpe_smoothing_alpha` | EMA weight on new data | [0.1, 0.5] | Responsiveness vs stability |
| **Sizing** | | | |
| `scaling_alpha` | Leverage multiplier | [1.0, 2.0] | Max upside potential |
| `min_multiplier` | Floor (worst conditions) | [0.20, 0.50] | Downside protection |
| `max_multiplier` | Ceiling (best conditions) | [1.0, 2.0] | Upside cap |
| `skip_trade_threshold` | Skip if raw score < this | [0.01, 0.15] | Skip frequency |
| **Cold Start** | | | |
| `ceiling_very_low_trades` | Max when N < 5 | [0.30, 0.70] | Warm-up caution |
| `ceiling_low_trades` | Max when 5 ≤ N < 10 | [0.50, 1.0] | Warm-up transition |
| **Metrics** | | | |
| `max_sharpe_for_scaling` | Sharpe normalization cap | [1.5, 3.0] | Performance scaling |

---

## Usage Examples

### **Example 1: Quick Single-Symbol Tuning**

```bash
# 20 trials, 3 walk-forward windows
python tools/tune_adaptive_sizing.py \
    --symbol AAPL \
    --n-trials 20 \
    --n-windows 3 \
    --output results/quick_tune.json
```

**Runtime:** ~5-10 minutes (depends on data size)

**Output:**
```json
{
  "best_params": {
    "bayesian_prior_sharpe": 0.42,
    "bayesian_prior_weight": 7,
    "sharpe_smoothing_alpha": 0.28,
    "scaling_alpha": 1.35,
    "min_multiplier": 0.32,
    "max_multiplier": 1.42,
    ...
  },
  "best_score": 1.87
}
```

### **Example 2: Full Multi-Symbol Tuning**

```bash
# Tune for multiple symbols
python tools/tune_adaptive_sizing.py \
    --symbols AAPL,TSLA,MSFT \
    --n-trials 100 \
    --n-windows 5 \
    --output results/multi_symbol_tune.json
```

**Runtime:** ~30-60 minutes

**Use case:** Find parameters that work well across different stocks

### **Example 3: Conservative Tuning (Prefer Stability)**

Modify the objective function in `tune_adaptive_sizing.py`:

```python
# Line ~200: Increase drawdown penalty
drawdown_penalty = max_drawdown / 100.0
score -= drawdown_penalty * 3.0  # Was 2.0, now 3.0 (more conservative)
```

Then run:
```bash
python tools/tune_adaptive_sizing.py --symbol AAPL --n-trials 50
```

This will favor parameters that reduce drawdowns, even at the cost of lower Sharpe.

### **Example 4: Random Search (No Dependencies)**

If `scikit-optimize` is not installed:

```bash
python tools/tune_adaptive_sizing.py \
    --symbol AAPL \
    --n-trials 100 \
    --method random
```

**Trade-off:** Random search needs ~2x more trials to get similar results as Bayesian.

---

## Interpreting Results

### **What to Look For**

**1. Out-of-Sample Sharpe Stability**

```
Window 1: Test Sharpe = 1.85
Window 2: Test Sharpe = 1.92
Window 3: Test Sharpe = 1.78
Average: 1.85 ← Good (consistent)
```

vs

```
Window 1: Test Sharpe = 3.20
Window 2: Test Sharpe = 0.45
Window 3: Test Sharpe = 2.10
Average: 1.92 ← Bad (unstable, likely overfit)
```

**2. Train vs Test Gap**

```
Train Sharpe: 2.10
Test Sharpe: 1.85
Gap: 0.25 ← Acceptable (<0.5)
```

vs

```
Train Sharpe: 3.50
Test Sharpe: 1.20
Gap: 2.30 ← Overfitting! Parameters too aggressive.
```

**3. Parameter Reasonableness**

**Good parameters:**
```json
{
  "scaling_alpha": 1.35,      // Moderate leverage
  "min_multiplier": 0.32,     // Reasonable floor
  "max_multiplier": 1.42,     // Reasonable ceiling
  "skip_trade_threshold": 0.06  // Not skipping too often
}
```

**Suspicious parameters (might be overfit):**
```json
{
  "scaling_alpha": 1.98,      // Max leverage (overfitting?)
  "min_multiplier": 0.21,     // Very aggressive floor
  "max_multiplier": 1.99,     // Max ceiling (overfitting?)
  "skip_trade_threshold": 0.14  // Skipping often (cherry-picking?)
}
```

---

## Best Practices

### **1. Use Enough Data**

- **Minimum:** 500 bars (days)
- **Recommended:** 750+ bars
- **Why:** Need enough data for multiple walk-forward windows

### **2. Use Multiple Windows**

- **Minimum:** 3 windows
- **Recommended:** 5 windows
- **Why:** Averages out market regime differences

### **3. Start with Quick Tuning**

```bash
# Quick run (10 trials, 10 minutes)
python tune_adaptive_sizing.py --symbol AAPL --n-trials 10

# Check results
# If promising, run full tuning (100 trials, 1 hour)
python tune_adaptive_sizing.py --symbol AAPL --n-trials 100
```

### **4. Validate on Holdout Data**

After tuning:
1. Take best parameters
2. Test on completely unseen data (e.g., most recent 6 months)
3. If performance drops significantly, parameters overfit

### **5. Re-tune Periodically**

- Market conditions change
- Re-tune every 6-12 months
- Compare new vs old parameters
- If drastically different, market regime may have shifted

### **6. Use Regularization**

The objective function already includes regularization penalties for extreme parameters. Don't disable these unless you have a good reason.

---

## Troubleshooting

### **Problem: All trials have similar scores**

**Cause:** Search space too narrow, or metric not sensitive

**Fix:**
```python
# Widen search ranges in _define_search_space()
Real(0.10, 1.50, name='bayesian_prior_sharpe'),  # Was [0.25, 1.0]
Real(0.8, 2.5, name='scaling_alpha'),             # Was [1.0, 2.0]
```

### **Problem: Best parameters are at search boundaries**

**Example:** `scaling_alpha = 2.0` (max boundary)

**Cause:** Optimal value might be outside search range

**Fix:**
```python
# Expand upper bound
Real(1.0, 2.5, name='scaling_alpha'),  # Was [1.0, 2.0]
```

### **Problem: Tuning takes too long**

**Solutions:**

1. **Reduce trials:**
   ```bash
   --n-trials 20  # Instead of 100
   ```

2. **Reduce windows:**
   ```bash
   --n-windows 2  # Instead of 5
   ```

3. **Use less data:**
   ```python
   bars = bars.tail(500)  # Instead of 750
   ```

4. **Use random search:**
   ```bash
   --method random  # Faster than Bayesian
   ```

### **Problem: Results are unstable across runs**

**Cause:** Not enough data or windows

**Fix:**
```bash
# Use more windows and data
python tune_adaptive_sizing.py \
    --symbol AAPL \
    --n-trials 100 \
    --n-windows 5  # Was 3
```

---

## Advanced: Custom Objective Functions

### **Example: Optimize for Sortino Instead of Sharpe**

Edit `tune_adaptive_sizing.py`, line ~190:

```python
def _calculate_objective(self, sharpe, max_drawdown, win_rate, num_trades, params):
    # Calculate Sortino ratio from returns
    # (Only penalize downside volatility)
    sortino = self._calculate_sortino(returns)  # You'd need to pass returns

    score = sortino  # Use Sortino instead of Sharpe
    score -= (max_drawdown / 100.0) * 2.0
    # ... rest of penalties
    return score
```

### **Example: Optimize for Calmar Ratio**

```python
def _calculate_objective(self, sharpe, max_drawdown, win_rate, num_trades, params):
    # Calmar = Annual Return / Max Drawdown
    annual_return = total_return  # Would need to pass this
    calmar = annual_return / max(max_drawdown, 1.0)

    score = calmar
    # ... penalties
    return score
```

### **Example: Multi-Objective (Pareto Front)**

For true multi-objective optimization (not just weighted sum), you'd need to:

1. Install `pymoo`: `pip install pymoo`
2. Define multiple objectives separately
3. Return Pareto-optimal solutions

This is advanced - single weighted objective is usually sufficient.

---

## Applying Tuned Parameters

### **1. Review Results**

```bash
cat results/tuning_AAPL.json
```

### **2. Copy Best Params to Config**

Edit `config/continuous_adaptation.json`:

```json
{
  "position_sizing": {
    "scaling_alpha": 1.35,  // From tuning results
    "limits": {
      "min_multiplier": 0.32,  // From tuning
      "max_multiplier": 1.42,  // From tuning
      "skip_trade_threshold": 0.06  // From tuning
    },
    "bayesian_shrinkage": {
      "bayesian_prior_sharpe": 0.42,  // From tuning
      "bayesian_prior_weight": 7  // From tuning
    },
    "smoothing": {
      "sharpe_smoothing_alpha": 0.28  // From tuning
    },
    "cold_start": {
      "ceiling_very_low_trades": 0.48,  // From tuning
      "ceiling_low_trades": 0.82  // From tuning
    }
  }
}
```

### **3. Validate on Recent Data**

```bash
# Test on last 200 bars (unseen during tuning)
python tools/backtest_adaptive_features.py --symbol AAPL --bars 200
```

### **4. Paper Trade**

Before live deployment:
1. Run in paper trading for 2-4 weeks
2. Monitor performance vs backtest expectations
3. If performance matches, deploy to live

---

## Summary

**Workflow:**
1. `pip install scikit-optimize`
2. `python tune_adaptive_sizing.py --symbol AAPL --n-trials 50`
3. Review results in JSON output
4. Copy best params to config
5. Validate on holdout data
6. Paper trade
7. Deploy to live

**Expected Improvements:**
- 10-30% better Sharpe ratio vs default parameters
- 10-20% lower max drawdown
- More stable performance across market conditions

**Re-tune:**
- Every 6-12 months
- After major market regime changes
- When adding new symbols/strategies
