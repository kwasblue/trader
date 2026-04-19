# How to Use Your Existing Optimizer with Timeframes

You already have a comprehensive optimization system! I've extended it to include timeframe optimization.

---

## Your Existing System

### 1. `core/backtest/optimization.py`
Parameter optimization (grid search, random search) for strategy parameters.

**Usage:**
```python
from core.backtest.optimization import grid_search

# Optimize RSI parameters
result = grid_search(
    data=bars,
    strategy_name='rsi',
    param_grid={
        'rsi_period': [10, 14, 20],
        'oversold': [20, 30, 40],
        'overbought': [60, 70, 80]
    },
    metric='sharpe_ratio'
)

print(f"Best params: {result.best_params}")
```

### 2. `core/backtest/regime_backtest.py`
Regime-aware backtesting - tests strategies within each market regime.

**Usage:**
```python
from core.backtest.regime_backtest import RegimeBacktester

# Test strategies across regimes
tester = RegimeBacktester(data, symbol="AAPL")
result = tester.run_regime_analysis()

# Shows best strategy per regime
# low_volatility: sma
# normal: momentum
# high_volatility: meanreversion
```

### 3. `tools/optimize_routing.py`
Runs regime backtests on all symbols and generates `strategy_routing.json`.

**Usage:**
```bash
# Optimize all symbols
python tools/optimize_routing.py

# Specific symbols
python tools/optimize_routing.py -s AAPL,TSLA,MSFT

# Output:
# AAPL     low_vol=sma    normal=momentum    high_vol=rsi
# TSLA     low_vol=ema    normal=bollinger   high_vol=meanreversion
# ...
```

---

## What Was Missing: Timeframes

Your optimizer tests:
- ✅ Symbols (AAPL, TSLA, MSFT)
- ✅ Regimes (low_volatility, normal, high_volatility)
- ✅ Strategies (rsi, sma, meanreversion, etc.)
- ❌ Timeframes (5min, 15min, 30min, 1hour) ← **This was missing!**

---

## New: Multi-Timeframe Optimizer

I've created `tools/optimize_routing_multitf.py` which extends your existing optimizer.

### Step 1: Fetch Data at Multiple Timeframes

```bash
# Fetch historical data at multiple timeframes
python -m core.unified_data_pipeline \
    --symbols AAPL TSLA MSFT \
    --timeframes 5min 15min 30min 1hour \
    --days 750
```

**Result:**
```
data/data_storage/proc_data/
├── proc_AAPL_5min.json
├── proc_AAPL_15min.json
├── proc_AAPL_30min.json
├── proc_AAPL_1hour.json
├── proc_TSLA_5min.json
...
```

### Step 2: Run Multi-Timeframe Optimization

```bash
# Optimize with timeframes
python tools/optimize_routing_multitf.py \
    --symbols AAPL,TSLA,MSFT \
    --timeframes 5min,15min,30min,1hour \
    --strategies rsi,sma,meanreversion,bollinger \
    --days 750
```

**What it does:**
- For each symbol:
  - For each timeframe:
    - For each regime:
      - For each strategy:
        - Run backtest
        - Measure performance

- Finds best (strategy, timeframe) for each (symbol, regime)
- Generates optimal `strategy_routing.json`

**Output:**
```
Symbol   Low Volatility        Normal               High Volatility      Hybrid
--------------------------------------------------------------------------------
AAPL     meanrev    @5min     rsi        @15min    bollinger  @30min   NO
TSLA     sma        @15min    bollinger  @30min    rsi        @1hour   YES
MSFT     meanrev    @5min     momentum   @15min    rsi        @30min   NO
```

**Generated Config:**
```json
{
  "AAPL": {
    "low_volatility": {
      "strategy": "meanreversion",
      "timeframe": "5min"
    },
    "normal": {
      "strategy": "rsi",
      "timeframe": "15min"
    },
    "high_volatility": {
      "strategy": "bollinger",
      "timeframe": "30min"
    },
    "use_hybrid": false
  }
}
```

### Step 3: Use the Optimized Config

```bash
# Review results
cat config/strategy_routing.json

# Start trading with optimized timeframes
trader start --broker schwab_multitf
```

---

## Comparison: Before vs After

### Before (Strategy-Only Optimization)

```bash
python tools/optimize_routing.py -s AAPL
```

**Result:**
```json
{
  "AAPL": {
    "low_volatility": "meanreversion",
    "normal": "rsi",
    "high_volatility": "bollinger"
  }
}
```
**Problem:** What timeframe? You'd have to guess (maybe 5min? 15min?).

### After (Strategy + Timeframe Optimization)

```bash
python tools/optimize_routing_multitf.py -s AAPL --timeframes 5min,15min,30min
```

**Result:**
```json
{
  "AAPL": {
    "low_volatility": {
      "strategy": "meanreversion",
      "timeframe": "5min"
    },
    "normal": {
      "strategy": "rsi",
      "timeframe": "15min"
    },
    "high_volatility": {
      "strategy": "bollinger",
      "timeframe": "30min"
    }
  }
}
```
**Solution:** Optimal timeframe determined through backtesting!

---

## How It Integrates

The new optimizer:
- ✅ Uses your existing `RegimeBacktester` (no changes needed)
- ✅ Uses your existing `UnifiedDataPipeline` (already supports multi-timeframe)
- ✅ Follows your existing patterns (same CLI args, same output format)
- ✅ Reuses your hybrid sizing logic
- ✅ Generates compatible `strategy_routing.json`

**Zero changes to existing code!** Just an additional tool that adds the timeframe dimension.

---

## Quick Examples

### Example 1: Quick Test (3 symbols, 2 timeframes)
```bash
python tools/optimize_routing_multitf.py \
    -s AAPL,TSLA,MSFT \
    --timeframes 15min,30min \
    --days 365
```

### Example 2: Full Optimization (all timeframes)
```bash
python tools/optimize_routing_multitf.py \
    -s AAPL,TSLA,MSFT,NVDA,AMD \
    --timeframes 5min,15min,30min,1hour \
    --days 750
```

### Example 3: Specific Strategies
```bash
python tools/optimize_routing_multitf.py \
    -s AAPL \
    --timeframes 5min,15min,30min \
    --strategies rsi,meanreversion,bollinger
```

### Example 4: Dry Run (don't save)
```bash
python tools/optimize_routing_multitf.py \
    -s AAPL \
    --timeframes 15min,30min \
    --dry-run
```

### Example 5: With Auto Data Fetch
```bash
python tools/optimize_routing_multitf.py \
    -s AAPL,TSLA \
    --timeframes 5min,15min,30min \
    --fetch-data  # Automatically fetches missing data
```

---

## Workflow

### 1. Initial Setup
```bash
# Fetch historical data at multiple timeframes (one time)
python -m core.unified_data_pipeline \
    --symbols AAPL TSLA MSFT NVDA AMD \
    --timeframes 5min 15min 30min 1hour \
    --days 750
```

### 2. Run Optimization
```bash
# Find optimal (strategy, timeframe) for each (symbol, regime)
python tools/optimize_routing_multitf.py \
    --symbols AAPL,TSLA,MSFT,NVDA,AMD \
    --timeframes 5min,15min,30min,1hour \
    --days 750
```

### 3. Review and Apply
```bash
# Results saved to config/strategy_routing.json
cat config/strategy_routing.json

# Start trading
trader start --broker schwab_multitf
```

### 4. Re-optimize Periodically
```bash
# Re-run quarterly or when market conditions change
python tools/optimize_routing_multitf.py \
    --symbols AAPL,TSLA,MSFT,NVDA,AMD \
    --timeframes 5min,15min,30min,1hour
```

---

## Benefits

1. **Leverages Existing Infrastructure**
   - No need to rewrite your backtesting engine
   - Uses proven RegimeBacktester
   - Compatible with existing configs

2. **Data-Driven Timeframe Selection**
   - No guessing
   - Tests all combinations
   - Uses same metrics (Sharpe, returns, win rate)

3. **Flexible**
   - Test any timeframes
   - Test any strategies
   - Use any metric for ranking

4. **Production Ready**
   - Generates config for `schwab_multitf` broker
   - Works with existing multi-timeframe system
   - Already integrated with BarAggregator

---

## Summary

**Before:**
```bash
python tools/optimize_routing.py  # Strategy only
# → Generates: {"AAPL": {"normal": "rsi"}}
# → Question: What timeframe?
```

**After:**
```bash
python tools/optimize_routing_multitf.py --timeframes 5min,15min,30min
# → Generates: {"AAPL": {"normal": {"strategy": "rsi", "timeframe": "15min"}}}
# → Answer: 15min is optimal for AAPL/RSI in normal regime!
```

**Simple!** 🎯

---

## Files

- ✅ `tools/optimize_routing.py` - Your existing optimizer (strategy only)
- ✅ `tools/optimize_routing_multitf.py` - New multi-timeframe optimizer
- ✅ `core/backtest/regime_backtest.py` - Unchanged (works with both)
- ✅ `core/unified_data_pipeline.py` - Already supports multi-timeframe
- ✅ `core/bar_aggregator.py` - Already handles real-time aggregation
- ✅ `core/schwab_runner_multitf.py` - Already uses timeframes from config

**Everything fits together!** 🚀
