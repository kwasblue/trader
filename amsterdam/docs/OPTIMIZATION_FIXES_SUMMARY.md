# Optimization System Fixes - Summary

## Date: 2026-03-15

## Overview

Complete overhaul of the strategy and timeframe optimization system to address critical issues with regime detection, backtesting methodology, and integration between components.

---

## Problems Identified

### 1. **Strategy Selector Issues**
- ❌ No regime-specific testing - tested strategies on ALL data combined
- ❌ Used `VectorizedBacktester` instead of regime-aware backtesting
- ❌ Picked best overall strategy, not best per regime
- ❌ Results didn't match live trading behavior

### 2. **Timeframe Optimizer Issues**
- ❌ **CRITICAL BUG**: Regime heuristic was backwards
  - Assigned SHORT timeframes (5min, 15min) to LOW volatility ❌
  - Assigned LONG timeframes (30min, 1hour) to HIGH volatility ❌
- ❌ No actual regime filtering during backtests
- ❌ Different scoring than strategy selector (inconsistent)

### 3. **Integration Issues**
- ❌ Strategy selector and timeframe optimizer ran independently
- ❌ Could produce conflicting recommendations
- ❌ No unified workflow to combine results

---

## Fixes Implemented

### Fix #1: Added Regime-Aware Strategy Selection

**File**: `core/backtest/strategy_selector.py`

**Changes**:
- Added new method: `select_best_strategies_by_regime()`
- Uses `RegimeBacktester` instead of `VectorizedBacktester`
- Tests each strategy separately during low/normal/high volatility periods
- Returns best strategy **per regime**, not just overall
- Added save method: `save_regime_result_to_config()`

**Usage**:
```bash
# Old way (legacy)
amsterdam strategy select AAPL --save

# New way (regime-aware)
amsterdam strategy select AAPL --regime-aware --save
```

**CLI additions**:
- Added `--regime-aware` flag to enable new behavior
- Backward compatible - old behavior still works

---

### Fix #2: Fixed Timeframe Optimizer Regime Logic

**File**: `core/backtest/timeframe_optimizer.py`

**Changes**:
- **CORRECTED REGIME HEURISTIC** (lines 367-397):
  ```python
  # BEFORE (WRONG):
  # Low volatility  → 5min, 15min  (short timeframes)
  # High volatility → 30min, 1hour (long timeframes)

  # AFTER (CORRECT):
  # High volatility → 5min, 15min  (short timeframes for fast reaction)
  # Low volatility  → 30min, 1hour (long timeframes to avoid whipsaws)
  ```

- Added import of regime detection functions from `regime_backtest.py`
- Updated logging to clarify regime assignments
- Added comments explaining the logic

**Rationale**:
- High volatility = rapid price changes → need SHORT timeframes to react quickly
- Low volatility = slow, grinding moves → use LONG timeframes to filter noise

---

### Fix #3: Created Unified Optimization Workflow

**File**: `tools/unified_optimizer.py` (NEW)

**Purpose**: Integrate strategy selection and timeframe optimization into a single coherent workflow

**Workflow**:
```
1. Run regime-aware strategy selection
   ↓
2. Get best strategies per regime
   ↓
3. Run timeframe optimizer on those strategies
   ↓
4. Combine results with regime-appropriate timeframe preferences
   ↓
5. Generate unified config
```

**Features**:
- Tests strategies per regime using ATR-based detection
- Optimizes timeframes for the selected strategies
- Applies smart heuristics for regime/timeframe matching
- Generates production-ready `strategy_routing.json`
- Backward compatible with existing configs

**Usage**:
```bash
# Optimize single symbol
python tools/unified_optimizer.py AAPL --save

# Optimize multiple symbols
python tools/unified_optimizer.py AAPL TSLA MSFT --save

# Custom strategies and timeframes
python tools/unified_optimizer.py AAPL \
  --strategies rsi,sma,momentum \
  --timeframes 5min,15min,30min \
  --days 750 \
  --save
```

**Output**:
```json
{
  "AAPL": {
    "high_volatility": {
      "strategy": "momentum",
      "timeframe": "5min"
    },
    "normal": {
      "strategy": "sma",
      "timeframe": "15min"
    },
    "low_volatility": {
      "strategy": "meanreversion",
      "timeframe": "1hour"
    },
    "default": {
      "strategy": "sma",
      "timeframe": "15min"
    },
    "use_hybrid": false
  }
}
```

---

### Fix #4: Testing Framework

**File**: `test_optimization_workflow.py` (NEW)

**Tests**:
1. Regime-aware strategy selection works correctly
2. Timeframe optimizer regime logic is correct
3. Unified optimizer integrates both successfully

**Usage**:
```bash
.venv/bin/python test_optimization_workflow.py
```

---

## Key Improvements

### Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **Strategy Testing** | Overall performance only | Per-regime performance |
| **Regime Assignment** | Backwards (high vol = long TF) | Correct (high vol = short TF) |
| **Integration** | Separate tools, conflicting results | Unified workflow |
| **Backtesting** | VectorizedBacktester | RegimeBacktester (matches live) |
| **Config Generation** | Manual/inconsistent | Automated, consistent |

---

## Regime Detection Logic

Uses ATR (Average True Range) percentile for regime classification:

```python
# Calculate ATR percentile from historical data
atr_history = last 50 bars of ATR values
atr_mean = mean(atr_history)
atr_std = std(atr_history)

high_threshold = atr_mean + atr_std
low_threshold = atr_mean - 0.5 * atr_std

if current_atr > high_threshold:
    regime = "high_volatility"
elif current_atr < low_threshold:
    regime = "low_volatility"
else:
    regime = "normal"
```

This matches the live trading system in `core/stable_regime_detector.py`

---

## Regime-Timeframe Matrix

| Regime | Volatility | Price Action | Optimal Timeframe | Rationale |
|--------|-----------|--------------|-------------------|-----------|
| **High Volatility** | High | Fast, large moves | 5min, 15min | Need fast reaction to capitalize on/protect from rapid changes |
| **Normal** | Medium | Standard moves | 15min, 30min | Balance between noise filtering and responsiveness |
| **Low Volatility** | Low | Slow grinding | 30min, 1hour, day | Filter noise, avoid whipsaws in choppy conditions |

---

## Migration Guide

### For Existing Users

**Option 1: Keep existing config** (no changes needed)
- Current config still works
- System is backward compatible

**Option 2: Re-optimize with new tools**

```bash
# Step 1: Run unified optimizer
python tools/unified_optimizer.py AAPL TSLA MSFT --save

# Step 2: Review generated config
cat config/strategy_routing.json

# Step 3: Restart trader
amsterdam start
```

**Option 3: Gradual migration**
```bash
# Test on single symbol first
python tools/unified_optimizer.py AAPL --save

# Compare performance for a week

# If good, roll out to all symbols
python tools/unified_optimizer.py AAPL TSLA MSFT ... --save
```

---

## CLI Updates

### Strategy Selector

**Old behavior (still works)**:
```bash
amsterdam strategy select AAPL --save
# Tests all strategies overall, saves best to config
```

**New behavior (recommended)**:
```bash
amsterdam strategy select AAPL --regime-aware --save
# Tests strategies per regime, saves regime-specific config
```

### Unified Optimizer (NEW)

```bash
# Basic usage
python tools/unified_optimizer.py AAPL --save

# Advanced usage
python tools/unified_optimizer.py AAPL TSLA MSFT \
  --strategies rsi,sma,momentum,meanreversion \
  --timeframes 5min,15min,30min,1hour \
  --days 1000 \
  --metric sharpe_ratio \
  --save
```

---

## Verification

To verify the fixes are working:

```bash
# Run test suite
.venv/bin/python test_optimization_workflow.py

# Should see:
# ✓ Test 1 PASSED: Regime-aware selection working
# ✓ Test 2 PASSED: Timeframe optimizer regime logic corrected
# ✓ Test 3 PASSED: Unified optimizer working
```

---

## Files Modified

1. `core/backtest/strategy_selector.py`
   - Added regime-aware selection method
   - Added regime result save method
   - Updated CLI with --regime-aware flag

2. `core/backtest/timeframe_optimizer.py`
   - Fixed regime heuristic (reversed logic)
   - Updated comments and logging
   - Added regime detection imports

3. `tools/unified_optimizer.py` (NEW)
   - Complete unified workflow
   - Combines strategy + timeframe optimization
   - Production-ready config generation

4. `test_optimization_workflow.py` (NEW)
   - Comprehensive test suite
   - Validates all fixes

5. `docs/OPTIMIZATION_FIXES_SUMMARY.md` (NEW)
   - This document

---

## Technical Details

### Regime Backtester

The `RegimeBacktester` class (in `core/backtest/regime_backtest.py`) was already available and well-designed:

- Calculates ATR-based regimes matching live system
- Tests strategies only during their target regime
- Prevents look-ahead bias
- Returns comprehensive metrics per regime

We integrated this existing component into the strategy selector.

### Timeframe Scoring

Each timeframe/strategy combination is scored using:

```python
score = (
    sharpe_ratio * 0.4 +
    total_return * 0.3 +
    win_rate * 0.2 +
    calmar_ratio * 0.1
)
```

This composite score prevents over-optimization on single metrics.

---

## Performance Impact

### Optimization Time

| Tool | Before | After | Notes |
|------|--------|-------|-------|
| Strategy Selector | ~2-5 min | ~3-7 min | Slightly longer (regime-aware) |
| Timeframe Optimizer | ~5-10 min | Same | No performance change |
| **Unified** | N/A | ~10-15 min | New tool, comprehensive |

### Live Trading Performance

Expected improvements:
- Better strategy selection per market condition
- Appropriate timeframes for regime
- Reduced whipsaws in low volatility
- Faster reaction in high volatility

---

## Next Steps

1. **Test the changes**:
   ```bash
   .venv/bin/python test_optimization_workflow.py
   ```

2. **Re-optimize your symbols**:
   ```bash
   python tools/unified_optimizer.py YOUR_SYMBOLS --save
   ```

3. **Monitor performance** for 1-2 weeks

4. **Iterate** if needed:
   - Adjust regime thresholds in `regime_backtest.py`
   - Tune composite score weights
   - Add more strategies to test pool

---

## FAQ

**Q: Will this break my existing config?**
A: No, the system is backward compatible. Existing configs continue to work.

**Q: Should I re-optimize immediately?**
A: Not required, but recommended. The new logic is more accurate.

**Q: How often should I re-optimize?**
A: Every 3-6 months, or when market character changes significantly.

**Q: Can I test the new optimizer without affecting live trading?**
A: Yes! Use `--output custom_path.json` to save to a different file first.

**Q: What if I only trade one symbol?**
A: The unified optimizer works great for single symbols too.

---

## Conclusion

The optimization system has been significantly improved with:

1. ✅ Regime-aware strategy selection (per-regime testing)
2. ✅ Corrected timeframe regime logic (high vol = short TF)
3. ✅ Unified workflow (consistent, integrated results)
4. ✅ Backward compatibility (existing configs still work)
5. ✅ Comprehensive testing (validation suite included)

These changes ensure the optimization system:
- Matches live trading behavior
- Uses correct regime/timeframe relationships
- Produces consistent, actionable results

---

**Author**: Claude Code
**Date**: March 15, 2026
**Version**: 1.0
