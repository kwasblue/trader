# How to Optimize Timeframes - Find the Best Config Values

**The Problem:** How do you know what timeframes to put in `strategy_routing.json`?

**The Solution:** Backtest all combinations and use data to decide!

---

## Quick Start (3 Steps)

### Step 1: Fetch Historical Data at Multiple Timeframes

```bash
# Fetch 2+ years of data at multiple timeframes
python -m core.unified_data_pipeline \
    --symbols AAPL TSLA MSFT NVDA \
    --timeframes 5min 15min 30min 1hour \
    --days 750 \
    --source alpaca
```

**This will:**
- Download data at 5min, 15min, 30min, and 1hour timeframes
- Store as: `proc_AAPL_5min.json`, `proc_AAPL_15min.json`, etc.
- Take ~15-30 minutes depending on symbols

### Step 2: Run the Optimizer

```bash
python -m core.backtest.timeframe_optimizer \
    --symbols AAPL TSLA MSFT \
    --timeframes 5min 15min 30min 1hour \
    --strategies rsi sma meanreversion bollinger \
    --days 750 \
    --output config/strategy_routing_optimized.json
```

**This will:**
- Test every combination: 3 symbols × 4 timeframes × 4 strategies = **48 backtests**
- Measure: Sharpe ratio, returns, win rate, drawdown
- Generate optimal config with best timeframe per symbol
- Save results to `config/strategy_routing_optimized.json`

### Step 3: Use the Optimized Config

```bash
# Review the results
cat config/strategy_routing_optimized.json

# If it looks good, use it
cp config/strategy_routing_optimized.json config/strategy_routing.json

# Restart trader with optimized config
trader restart --broker schwab_multitf
```

Done! Now you're using data-driven optimal timeframes! 🎯

---

## What the Optimizer Does

### The Process

```
For each symbol (AAPL, TSLA, MSFT):
  For each timeframe (5min, 15min, 30min, 1hour):
    For each strategy (RSI, SMA, MeanRev, Bollinger):

      1. Load historical data at that timeframe
      2. Run backtest with that strategy
      3. Measure performance (Sharpe, return, etc.)
      4. Save results

  Find best performing combinations
  Generate config with optimal timeframes
```

### Example Output

```
AAPL:
  1. meanreversion @ 5min  - Sharpe: 1.85, Return: 45.2%, Score: 2.14
  2. rsi          @ 15min - Sharpe: 1.62, Return: 38.7%, Score: 1.93
  3. bollinger    @ 30min - Sharpe: 1.43, Return: 32.1%, Score: 1.71

TSLA:
  1. bollinger    @ 30min - Sharpe: 1.92, Return: 52.3%, Score: 2.28
  2. momentum     @ 15min - Sharpe: 1.71, Return: 48.1%, Score: 2.09
  3. rsi          @ 1hour - Sharpe: 1.55, Return: 41.2%, Score: 1.88
```

### Generated Config

Based on these results, it generates:

```json
{
  "AAPL": {
    "low_volatility": {
      "strategy": "meanreversion",
      "timeframe": "5min"
    },
    "normal": {
      "strategy": "meanreversion",
      "timeframe": "5min"
    },
    "high_volatility": {
      "strategy": "bollinger",
      "timeframe": "30min"
    }
  },
  "TSLA": {
    "normal": {
      "strategy": "bollinger",
      "timeframe": "30min"
    },
    "high_volatility": {
      "strategy": "rsi",
      "timeframe": "1hour"
    }
  }
}
```

---

## Advanced Usage

### Test Specific Combinations

```bash
# Just test AAPL with RSI on different timeframes
python -m core.backtest.timeframe_optimizer \
    --symbols AAPL \
    --timeframes 5min 15min 30min \
    --strategies rsi
```

### Optimize for More Symbols

```bash
# Test your entire watchlist
python -m core.backtest.timeframe_optimizer \
    --symbols AAPL TSLA MSFT NVDA AMD GOOGL META \
    --timeframes 5min 15min 30min 1hour \
    --strategies rsi sma meanreversion bollinger macd ema
```

### Use Python API

```python
from core.backtest.timeframe_optimizer import TimeframeOptimizer

# Create optimizer
optimizer = TimeframeOptimizer(
    symbols=['AAPL', 'TSLA'],
    timeframes=['5min', '15min', '30min'],
    strategies=['rsi', 'sma', 'meanreversion']
)

# Run optimization
results = await optimizer.run_optimization(days=750)

# View results
optimizer.print_summary()

# Find best timeframe for a specific combination
best_tf = optimizer.find_best_timeframe('AAPL', 'rsi')
print(f"Best timeframe for AAPL with RSI: {best_tf}")

# Generate config
config = optimizer.generate_routing_config(results)
optimizer.save_config(config, 'config/strategy_routing_optimized.json')
```

---

## Understanding the Results

### Metrics Explained

The optimizer measures these metrics for each combination:

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Total Return** | Overall profit | > 20% |
| **Win Rate** | % of winning trades | > 50% |
| **Max Drawdown** | Largest peak-to-trough loss | < 20% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |
| **Calmar Ratio** | Return / Max drawdown | > 1.0 |

### Composite Score

The optimizer creates a weighted score:
```
Score = (Sharpe × 40%) + (Return × 30%) + (Win Rate × 20%) + (Calmar × 10%)
```

**Higher score = Better overall performance**

### Example Result

```json
{
  "symbol": "AAPL",
  "strategy": "meanreversion",
  "timeframe": "5min",
  "sharpe_ratio": 1.85,
  "total_return": 0.452,
  "win_rate": 0.62,
  "max_drawdown": 0.12,
  "profit_factor": 2.1,
  "num_trades": 234,
  "score": 2.14  ← Composite score
}
```

---

## Interpreting Results

### What Timeframe for What Strategy?

Based on typical backtest results:

**Mean Reversion** → Best on **5-15min**
- Needs quick reversals
- Shorter timeframes capture mean reversion better
- Example: `"meanreversion": "5min"`

**RSI/Stochastic** → Best on **15-30min**
- Needs time for oscillators to develop
- Too short = too noisy
- Example: `"rsi": "15min"`

**Momentum** → Best on **15-30min**
- Needs sustained moves
- Medium timeframes balance signal and noise
- Example: `"momentum": "15min"`

**Bollinger Bands** → Best on **30min-1hour**
- Needs volatility to develop
- Longer timeframes = clearer signals
- Example: `"bollinger": "30min"`

**SMA Crossovers** → Best on **30min-1hour**
- Trend-following needs time
- Longer timeframes reduce whipsaws
- Example: `"sma": "30min"`

### Regime-Specific Patterns

**Low Volatility:**
- Shorter timeframes work better (5-15min)
- More trades, smaller moves
- Mean reversion strategies excel

**Normal Markets:**
- Medium timeframes (15-30min)
- Balanced approach
- Most strategies work

**High Volatility:**
- Longer timeframes (30min-1hour)
- Reduces noise, filters false signals
- Trend-following strategies excel

---

## Real-World Example

Let's optimize AAPL step-by-step:

### 1. Fetch Data

```bash
python -m core.unified_data_pipeline \
    --symbols AAPL \
    --timeframes 5min 15min 30min 1hour \
    --days 750
```

### 2. Run Tests

```bash
python -m core.backtest.timeframe_optimizer \
    --symbols AAPL \
    --timeframes 5min 15min 30min 1hour \
    --strategies rsi sma meanreversion bollinger
```

### 3. Results

```
AAPL:
  1. meanreversion @ 5min  - Sharpe: 1.85, Return: 45.2%, Score: 2.14
  2. rsi          @ 15min - Sharpe: 1.62, Return: 38.7%, Score: 1.93
  3. bollinger    @ 30min - Sharpe: 1.43, Return: 32.1%, Score: 1.71
  4. sma          @ 1hour - Sharpe: 1.21, Return: 28.5%, Score: 1.52
```

### 4. Decision

**Best overall:** Mean reversion @ 5min (Score: 2.14)

**Config:**
```json
{
  "AAPL": {
    "low_volatility": {
      "strategy": "meanreversion",
      "timeframe": "5min"
    },
    "normal": {
      "strategy": "meanreversion",
      "timeframe": "5min"
    },
    "high_volatility": {
      "strategy": "bollinger",
      "timeframe": "30min"
    }
  }
}
```

**Rationale:**
- Use best performer (meanrev @ 5min) for low vol and normal
- Use longer timeframe (bollinger @ 30min) for high vol to reduce noise

---

## Tips for Better Optimization

### 1. Use Enough Data
```bash
# Minimum 1 year, prefer 2+ years
--days 750  # ~2 years
```

### 2. Test Relevant Timeframes
```bash
# Skip 1min (too noisy, huge data)
# Focus on: 5min, 15min, 30min, 1hour
--timeframes 5min 15min 30min 1hour
```

### 3. Include Multiple Strategies
```bash
# Test different strategy types
--strategies rsi sma meanreversion bollinger momentum macd
```

### 4. Review Results Manually
```bash
# Don't blindly accept top result
# Look for consistent performance across metrics
# Check if results make sense
```

### 5. Consider Transaction Costs
```bash
# Shorter timeframes = more trades = more fees
# Factor in your broker's commission structure
```

---

## Troubleshooting

### "No data available"

**Cause:** Haven't fetched data for that timeframe

**Fix:**
```bash
python -m core.unified_data_pipeline \
    --symbols AAPL \
    --timeframes 15min \
    --days 750
```

### "Insufficient bars for backtest"

**Cause:** Not enough historical data (< 100 bars)

**Fix:** Increase `--days` or use shorter timeframe:
```bash
--days 750  # More data
```

### Optimizer is slow

**Cause:** Testing many combinations

**Tips:**
- Start with fewer symbols: `--symbols AAPL TSLA`
- Test fewer strategies: `--strategies rsi sma`
- Reduce timeframes: `--timeframes 15min 30min`
- Run overnight for full optimization

---

## Next Steps

1. **Run optimization on your symbols:**
   ```bash
   python -m core.backtest.timeframe_optimizer \
       --symbols YOUR_SYMBOLS \
       --timeframes 5min 15min 30min \
       --strategies rsi sma meanreversion
   ```

2. **Review results:**
   ```bash
   cat results/optimization_results.json
   ```

3. **Apply optimal config:**
   ```bash
   cp config/strategy_routing_optimized.json config/strategy_routing.json
   ```

4. **Test in paper trading:**
   ```bash
   trader start --broker schwab_multitf --dry-run
   ```

5. **Monitor performance:**
   - Track actual vs. backtested results
   - Re-optimize periodically (quarterly/annually)
   - Adjust if market conditions change

---

## Summary

✅ **The optimizer solves the problem:**
- No more guessing timeframes
- Data-driven decisions
- Finds optimal combination per symbol
- Generates ready-to-use config

✅ **Simple workflow:**
1. Fetch data → 2. Run optimizer → 3. Use results

✅ **Now you know** what to put in `strategy_routing.json`!

---

**Run it now:**
```bash
python -m core.backtest.timeframe_optimizer --symbols AAPL TSLA MSFT
```

Then use the optimized config! 🚀
