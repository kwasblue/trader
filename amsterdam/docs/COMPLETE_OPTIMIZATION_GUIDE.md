# Complete Optimization Guide

## Overview

This guide walks you through optimizing **all parameters** in your trading system.

---

## Quick Start - Full Optimization

Run everything at once:

```bash
# Complete optimization of all 19 symbols (takes 1-2 hours)
python tools/complete_optimization.py --save
```

This runs:
1. ✅ Strategy/timeframe selection (DONE - you already ran this)
2. ✅ Strategy parameter optimization (NEW)
3. ✅ Hybrid sizing test (NEW)

---

## Individual Optimization Steps

If you want to run steps separately:

### **Step 1: Strategy & Timeframe Optimization** ✅ DONE

You already completed this! It optimized:
- Which strategy per symbol/regime
- Which timeframe per symbol/regime

```bash
# Already done, but to re-run:
python tools/optimize_all_symbols.py --save
```

---

### **Step 2: Strategy Parameter Optimization** 🆕

Optimizes the specific parameters for each strategy (e.g., SMA periods, RSI thresholds).

```bash
# Optimize all symbols
python tools/optimize_strategy_params.py --save

# Optimize specific symbols
python tools/optimize_strategy_params.py --symbols AAPL TSLA NVDA --save

# Preview first
python tools/optimize_strategy_params.py --dry-run
```

**What it optimizes:**

| Strategy | Parameters | Example |
|----------|-----------|---------|
| SMA | fast, slow periods | {fast: 10, slow: 30} |
| RSI | window, oversold, overbought | {window: 14, oversold: 30, overbought: 70} |
| MACD | fast, slow, signal | {fast: 12, slow: 26, signal: 9} |
| Bollinger | window, num_std | {window: 20, num_std: 2.0} |
| PSAR | af_start, af_max | {af_start: 0.02, af_max: 0.2} |

**Output:** `config/strategy_params.json`

**Time:** ~30-60 minutes for 19 symbols

---

### **Step 3: Hybrid Sizing Test** 🆕

Tests whether enabling hybrid position sizing improves performance.

```bash
# Test all symbols
python tools/test_hybrid_sizing.py --save

# Test specific symbols
python tools/test_hybrid_sizing.py --symbols AAPL TSLA --save

# Compare single symbol
python tools/test_hybrid_sizing.py AAPL --compare
```

**What it does:**
- Compares performance with `use_hybrid: true` vs `use_hybrid: false`
- Recommends setting per symbol
- Updates `use_hybrid` flag in `strategy_routing.json`

**Output:** Updated `use_hybrid` flags in `config/strategy_routing.json`

**Time:** ~10-20 minutes for 19 symbols

---

## Optimization Strategies

### **Fast Track** (30 minutes)
Skip parameter optimization, just test hybrid sizing:

```bash
python tools/complete_optimization.py --skip-params --save
```

### **Thorough** (1-2 hours)
Full optimization of everything:

```bash
python tools/complete_optimization.py --save
```

### **Specific Symbols** (10-15 minutes)
Optimize only your key positions:

```bash
python tools/complete_optimization.py --symbols AAPL NVDA GOOGL --save
```

---

## What Gets Optimized

### ✅ Already Done (from optimize_all_symbols.py)

**File:** `config/strategy_routing.json`

```json
{
  "AAPL": {
    "low_volatility": {
      "strategy": "stochastic",  // ✅ Optimized
      "timeframe": "1hour"        // ✅ Optimized
    },
    "use_hybrid": false           // ⚠️ Not optimized yet
  }
}
```

### 🆕 Step 2: Strategy Parameters

**File:** `config/strategy_params.json`

```json
{
  "AAPL": {
    "low_volatility": {
      "params": {
        "k_window": 14,     // 🆕 Will be optimized
        "d_window": 3,      // 🆕 Will be optimized
        "oversold": 20,     // 🆕 Will be optimized
        "overbought": 80    // 🆕 Will be optimized
      },
      "strategy": "stochastic",
      "_optimized_sharpe": 1.17
    }
  }
}
```

### 🆕 Step 3: Hybrid Sizing Flag

**File:** `config/strategy_routing.json` (updated)

```json
{
  "AAPL": {
    "use_hybrid": true,  // 🆕 Will be set based on test results
    "_hybrid_test_sharpe_improvement": 0.15
  }
}
```

---

## Understanding the Results

### Parameter Optimization Output

```
AAPL:
  low_volatility    : stochastic      params={'k_window': 14, 'd_window': 3} (Sharpe: 1.25)
  normal            : psar            params={'af_start': 0.02, 'af_max': 0.2} (Sharpe: 0.95)
  high_volatility   : psar            params={'af_start': 0.03, 'af_max': 0.3} (Sharpe: 1.05)
```

This tells you the best parameters for each strategy in each regime.

### Hybrid Sizing Output

```
AAPL:
  No Hybrid: Sharpe 0.89, Return +4.9%
  Hybrid:    Sharpe 1.02, Return +5.2%
  Improvement: Sharpe +0.13, Return +0.3%
  ✓ RECOMMEND: Enable hybrid - Sharpe improves by 0.13
```

This tells you whether to set `use_hybrid: true` or `false`.

---

## Performance Expectations

### Before Full Optimization
- Using default parameters
- No hybrid sizing
- Baseline: Avg Sharpe ~0.8-1.2

### After Full Optimization
- Optimized parameters per strategy
- Hybrid sizing where beneficial
- Expected: Avg Sharpe ~1.0-1.5 (+20-30% improvement)

---

## Configuration Files - Final State

After complete optimization, you'll have:

### 1. `strategy_routing.json` (Updated)
```json
{
  "AAPL": {
    "low_volatility": {"strategy": "stochastic", "timeframe": "1hour"},
    "normal": {"strategy": "psar", "timeframe": "30min"},
    "high_volatility": {"strategy": "psar", "timeframe": "15min"},
    "use_hybrid": true,  // ← Updated by hybrid test
    "_hybrid_test_sharpe_improvement": 0.15
  }
}
```

### 2. `strategy_params.json` (New/Updated)
```json
{
  "AAPL": {
    "low_volatility": {
      "params": {"k_window": 14, "d_window": 3, "oversold": 20, "overbought": 80},
      "strategy": "stochastic",
      "_optimized_sharpe": 1.25
    },
    "normal": {
      "params": {"af_start": 0.02, "af_max": 0.2},
      "strategy": "psar",
      "_optimized_sharpe": 0.95
    },
    "high_volatility": {
      "params": {"af_start": 0.03, "af_max": 0.3},
      "strategy": "psar",
      "_optimized_sharpe": 1.05
    }
  }
}
```

### 3. `trading_config.json` (Manual - optimize later)
Risk parameters, stop loss/take profit multipliers, etc.

---

## Recommended Workflow

### **First Time (Full Optimization)**

```bash
# Step 1: Full optimization (1-2 hours)
python tools/complete_optimization.py --save

# Step 2: Review results
cat config/strategy_routing.json
cat config/strategy_params.json

# Step 3: Start trading
amsterdam start

# Step 4: Monitor for 1-2 weeks
```

### **Subsequent Optimizations (Every 3-6 months)**

```bash
# Quick re-optimization (30 mins)
python tools/complete_optimization.py --skip-params --save

# Or full re-optimization if market changed significantly
python tools/complete_optimization.py --save
```

---

## Troubleshooting

### "No data for symbol"
```bash
# Update historical data first
amsterdam data update -s AAPL,TSLA,NVDA -t day
```

### "Strategy routing not found"
```bash
# Run strategy optimization first
python tools/optimize_all_symbols.py --save
```

### Too slow?
```bash
# Optimize fewer symbols
python tools/complete_optimization.py --symbols AAPL NVDA GOOGL --save

# Or skip parameter optimization
python tools/complete_optimization.py --skip-params --save
```

---

## Next Steps After Optimization

1. **Review the configs:**
   ```bash
   cat config/strategy_routing.json
   cat config/strategy_params.json
   ```

2. **Backup current configs** (optional):
   ```bash
   cp config/strategy_routing.json config/strategy_routing_backup.json
   ```

3. **Start trading with optimized settings:**
   ```bash
   amsterdam start
   ```

4. **Monitor performance:**
   - Track actual Sharpe ratios vs. backtested
   - Note which regimes occur most
   - Compare to non-optimized baseline

5. **Re-optimize periodically:**
   - Every 3-6 months
   - After major market regime changes
   - When performance degrades

---

## Summary

**You now have 3 optimization tools:**

1. ✅ **optimize_all_symbols.py** - Strategy/timeframe per regime (DONE)
2. 🆕 **optimize_strategy_params.py** - Strategy parameters per regime
3. 🆕 **test_hybrid_sizing.py** - Hybrid sizing recommendation per symbol

**Or run all at once:**
```bash
python tools/complete_optimization.py --save
```

**Ready to complete the optimization?**
```bash
# Full optimization (recommended first time)
python tools/complete_optimization.py --save

# Or fast track (if time constrained)
python tools/complete_optimization.py --skip-params --save
```

---

**Author:** Claude Code
**Date:** March 15, 2026
**Version:** 1.0
