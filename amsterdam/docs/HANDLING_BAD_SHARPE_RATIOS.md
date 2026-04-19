# Handling Bad Sharpe Ratios - Complete Guide

## Problem

After optimization, some symbol/regime combinations have poor Sharpe ratios:

**Negative Sharpe (losing strategies):**
- AAPL normal: -0.39
- COST normal: -0.64
- COST high_volatility: -0.30
- HD high_volatility: -0.33

**Near-zero Sharpe (no edge):**
- COST low_volatility: 0.10
- PG low_volatility: 0.00
- UNH low_volatility: 0.00
- UNH high_volatility: 0.15

## Solutions

### ✅ Solution 1: Sharpe Quality Filter (Recommended & Implemented)

**What it does:** Automatically blocks trades from strategies with poor backtested performance.

**How it works:**
1. Reads optimized Sharpe ratios from `strategy_params.json`
2. Blocks signals when Sharpe < threshold
3. Returns NoOp strategy (always signals 0 = hold)

**Configuration:**

Edit `config/trading_config.json`:

```json
{
  "strategy_quality_filter": {
    "enabled": true,    // Set to false to disable
    "min_sharpe": 0.5   // Adjust threshold (see below)
  }
}
```

**Threshold Guide:**

| min_sharpe | Blocks | Effect |
|------------|--------|--------|
| 0.0 | 4 regimes (7%) | Only block losing strategies |
| 0.3 | 9 regimes (16%) | Block poor performers |
| **0.5** | **13 regimes (23%)** | **Block marginal strategies (recommended)** |
| 0.7 | 20 regimes (35%) | Only trade good strategies |
| 1.0 | 31 regimes (54%) | Only trade excellent strategies |

**Analyze impact:**

```bash
# See what gets blocked at different thresholds
python tools/analyze_sharpe_filter.py

# Check specific threshold
python tools/analyze_sharpe_filter.py --min-sharpe 0.5
```

**What gets blocked at 0.5:**

```
COST   : All 3 regimes (terrible across all volatility levels)
AAPL   : normal regime only (low/high vol are good)
HD     : high_volatility only
NVDA   : low/normal regimes
PG     : low/high regimes
UNH    : low/high regimes
XOM    : high_volatility only
```

### 🔧 Solution 2: Re-optimize Problem Strategies

Some strategies may perform better with different parameters or larger search space.

**Expand parameter grid:**

Edit `tools/optimize_strategy_params.py`:

```python
# Example: Expand PSAR parameter grid
"psar": {
    "af_start": [0.01, 0.02, 0.03, 0.04, 0.05],  # More options
    "af_max": [0.1, 0.15, 0.2, 0.25, 0.3]        # More options
}
```

**Re-run optimization:**

```bash
# Re-optimize specific symbols
python tools/optimize_strategy_params.py --symbols COST HD UNH --save
```

### 🎯 Solution 3: Try Alternative Strategies

If a strategy consistently underperforms in a regime, try a different strategy.

**Manual override:**

Edit `config/strategy_routing.json`:

```json
{
  "COST": {
    "low_volatility": {
      "strategy": "bollinger",    // Try different strategy
      "timeframe": "1hour"
    },
    "normal": {
      "strategy": "sma",          // Try different strategy
      "timeframe": "30min"
    }
  }
}
```

**Re-run strategy selection:**

```bash
# Let optimizer find better strategies
python tools/optimize_all_symbols.py --symbols COST --save
```

### 📊 Solution 4: Accept Portfolio-Level Performance

Individual regimes may have low Sharpe, but portfolio-level performance may still be good.

**Why this works:**
- Regimes don't occur equally (e.g., AAPL's "normal" regime may be rare)
- High Sharpe in other regimes compensates
- Diversification across symbols/regimes reduces overall risk

**Check regime frequency:**

```bash
# Backtest to see actual regime distribution
python tools/backtest_adaptive_features.py --symbol AAPL --days 365
```

### 🚫 Solution 5: Don't Trade Certain Symbols/Regimes

If a symbol consistently underperforms, consider removing it from the trade list.

**Remove from trade list:**

```bash
amsterdam list remove COST  # If all regimes are bad
```

**Or keep in watchlist but don't auto-trade:**

Move to watch list instead of trade list.

---

## Recommendations by Symbol

### COST (All regimes bad: 0.10, -0.64, -0.30)
✅ **Recommended:** Remove from trade list or re-optimize with different strategies

### AAPL (Normal regime: -0.39, others good: 1.99, 0.54)
✅ **Recommended:** Keep with Sharpe filter (blocks normal regime only)

### HD (High volatility: -0.33, others okay: 0.73, 1.53)
✅ **Recommended:** Keep with Sharpe filter (blocks high vol only)

### UNH (Low/high bad: 0.00, 0.15; normal good: 1.59)
✅ **Recommended:** Keep with Sharpe filter (blocks low/high vol, trades normal)

### NVDA (All mediocre: 0.46, 0.24, 0.77)
⚠️ **Consider:** Re-optimize or try different strategies

### PG (Low/high bad: 0.00, 0.42; normal good: 1.34)
✅ **Recommended:** Keep with Sharpe filter (blocks low/high vol)

---

## Quick Start

**Immediate solution (already implemented):**

1. Sharpe filter is enabled with min_sharpe=0.5
2. System will automatically block 13 poor-performing regimes
3. You'll still trade 77% of regimes (the good ones)

**To adjust:**

```bash
# Edit config/trading_config.json
{
  "strategy_quality_filter": {
    "enabled": true,
    "min_sharpe": 0.5  // Change this value
  }
}

# Restart trader
amsterdam start
```

**To disable:**

```bash
# Edit config/trading_config.json
{
  "strategy_quality_filter": {
    "enabled": false
  }
}
```

---

## Monitoring

**Check filter activity in logs:**

```bash
tail -f logs/strategy_routing.log | grep "Sharpe filter"
```

**You'll see:**
```
Sharpe filter blocked COST/normal (Sharpe: -0.64 < 0.50)
Sharpe filter blocked AAPL/normal (Sharpe: -0.39 < 0.50)
```

**Performance monitoring:**

After running for a while, compare:
- Actual portfolio Sharpe vs. backtested
- Win rate per symbol/regime
- Adjust min_sharpe if needed

---

## Summary

✅ **Sharpe filter is now active** with min_sharpe=0.5 (recommended)

**Result:**
- Blocks 13/57 regimes (23%) with poor performance
- Trades 44/57 regimes (77%) with decent-to-excellent performance
- No code changes needed - works automatically

**Files:**
- Configuration: `config/trading_config.json`
- Filter logic: `core/logic/sharpe_filter.py`
- Integration: `core/logic/strategy_routing_manager.py`
- Analysis tool: `tools/analyze_sharpe_filter.py`
