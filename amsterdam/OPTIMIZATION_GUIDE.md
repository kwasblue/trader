# Strategy Optimization Guide

This guide explains the two optimization approaches available in Amsterdam.

## Two Versions

### Version 1: Daily Optimization (Simple)
**File:** `core/backtest/daily_optimize.py`
**Script:** `bin/daily-optimize`
**Schedule:** Daily at 9:00 AM ET

**Approach:**
- Uses 90 days of data
- Simple regime-based backtesting
- Picks best strategy per regime by Sharpe ratio
- Fast (~5 seconds)

**Pros:**
- Simple and fast
- Adapts quickly to market changes
- Easy to understand

**Cons:**
- ⚠️ High overfitting risk (only 90 days)
- ⚠️ No validation - might pick lucky winners
- ⚠️ Daily updates may cause excessive strategy switching
- ⚠️ Unreliable Sharpe ratios on short periods

**Recommendation:** Use for rapid experimentation, but be aware of overfitting risks.

---

### Version 2: Walk-Forward Optimization (Robust) ⭐ RECOMMENDED
**File:** `core/backtest/daily_optimize_v2.py`
**Script:** `bin/weekly-optimize`
**Schedule:** Weekly (Sunday 2:00 AM)

**Approach:**
- Uses 365 days of data (1 year)
- Walk-forward validation:
  - Train on 80% of data (292 days)
  - Validate on 20% of data (73 days)
- Confidence scoring system
- Only updates if new strategy is significantly better
- Requires minimum trade counts for statistical significance

**Improvements:**
- ✓ 4x more data (365 vs 90 days)
- ✓ Out-of-sample validation prevents overfitting
- ✓ Confidence scoring (0.0-1.0) based on:
  - Number of trades
  - Train/validation consistency
  - Regime data coverage
- ✓ Requires minimum 5 trades per regime
- ✓ Only updates if Sharpe improves by at least 0.3
- ✓ Preserves existing routing for low-confidence results

**Validation Process:**
1. Split data: 292 days training, 73 days validation
2. Find best strategy on training data
3. Test that strategy on validation data
4. Calculate confidence score
5. Only update if:
   - Confidence ≥ 0.6
   - Minimum 5 trades per regime
   - Sharpe improvement ≥ 0.3 vs existing strategy

**Recommendation:** ⭐ Use this for live trading. More reliable, less overfitting.

---

## Configuration Parameters

### Version 2 Key Settings
In `daily_optimize_v2.py`:

```python
TRAIN_DAYS = 365  # Use 1 year of data
VALIDATION_SPLIT = 0.8  # 80% train, 20% validate
MIN_TRADES_PER_REGIME = 5  # Minimum trades for significance
MIN_SHARPE_IMPROVEMENT = 0.3  # Minimum improvement to switch
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to update
```

### Adjusting for Your Needs

**More conservative (less frequent updates):**
```python
MIN_SHARPE_IMPROVEMENT = 0.5  # Higher bar for switching
CONFIDENCE_THRESHOLD = 0.7  # Higher confidence required
MIN_TRADES_PER_REGIME = 10  # More trades required
```

**More aggressive (more updates):**
```python
MIN_SHARPE_IMPROVEMENT = 0.1  # Lower bar for switching
CONFIDENCE_THRESHOLD = 0.5  # Lower confidence required
MIN_TRADES_PER_REGIME = 3  # Fewer trades required
```

---

## Scheduling

### Current Setup (Raspberry Pi)

**Daily optimization (v1):**
```bash
0 9 * * 1-5 /home/kwasi/trader/amsterdam/bin/daily-optimize
```

**Switch to weekly optimization (v2) - RECOMMENDED:**
```bash
# Remove daily optimization from cron:
crontab -e
# Comment out or remove the daily-optimize line

# Add weekly optimization:
0 2 * * 0 /home/kwasi/trader/amsterdam/bin/weekly-optimize
```

This runs every Sunday at 2:00 AM, giving fresh optimized strategies for the week ahead.

### Monthly Schedule (Even More Conservative)
```bash
# First Sunday of each month at 2:00 AM
0 2 1-7 * 0 /home/kwasi/trader/amsterdam/bin/weekly-optimize
```

---

## Manual Testing

### Test walk-forward optimization:
```bash
cd ~/trader/amsterdam
source venv/bin/activate
python core/backtest/daily_optimize_v2.py
```

### Compare both approaches:
```bash
# Run v1 (simple)
python core/backtest/daily_optimize.py > /tmp/v1_results.txt

# Run v2 (walk-forward)
python core/backtest/daily_optimize_v2.py > /tmp/v2_results.txt

# Compare confidence levels and strategy selections
diff /tmp/v1_results.txt /tmp/v2_results.txt
```

---

## Understanding Confidence Scores

Confidence scores range from 0.0 (no confidence) to 1.0 (high confidence).

**Score breakdown:**
- **0.8-1.0:** High confidence - many trades, good validation consistency
- **0.6-0.8:** Medium confidence - acceptable for updates
- **0.4-0.6:** Low confidence - preserves existing routing
- **0.0-0.4:** Very low confidence - insufficient data

**Factors affecting confidence:**
1. **Trade count** (40% weight): More trades = higher confidence
2. **Validation consistency** (40% weight): Train/validation Sharpe agreement
3. **Regime coverage** (20% weight): % of bars in this regime

---

## Interpreting Results

### Example Output (v2):

```
[1/19] AAPL...
  Training: 292 bars, Validation: 73 bars
  low_volatility: ema (conf=0.78, train_sharpe=1.45, val_sharpe=1.32)
  normal: meanreversion (conf=0.82, Sharpe +0.45 vs bollinger)
  high_volatility: sma (conf=0.91, train_sharpe=2.10, val_sharpe=1.98)
```

**What this means:**
- All regimes have high confidence (>0.75)
- Validation Sharpe is close to training Sharpe (good consistency)
- Normal regime switched from "bollinger" to "meanreversion" (Sharpe improved by 0.45)
- High volatility has highest confidence (0.91) with strong train/val agreement

### Red Flags:

```
  low_volatility: Insufficient trades (2) - preserving existing
  normal: Low confidence (0.45) - using bollinger
```

This means:
- Not enough trades to be confident in the result
- Existing routing is preserved (safer)

---

## Best Practices

1. **Start with weekly optimization** - Good balance of adaptation and stability
2. **Monitor confidence scores** - If consistently low, consider:
   - Increasing data window (500+ days)
   - Adjusting regime thresholds
   - Using fewer strategies to test

3. **Review backups** - Optimization creates timestamped backups:
   ```
   config/strategy_routing.backup.20260314_020000.json
   ```

4. **Check logs regularly:**
   ```bash
   tail -100 ~/trader/amsterdam/logs/weekly_optimize.log
   ```

5. **Don't over-optimize** - If you're tweaking parameters daily based on results, you're overfitting

---

## Migration Path

**Currently using daily v1 → Switch to weekly v2:**

1. Install v2 files on Pi:
   ```bash
   scp daily_optimize_v2.py raspi:~/trader/amsterdam/
   scp bin/weekly-optimize raspi:~/trader/amsterdam/bin/
   ssh raspi "chmod +x ~/trader/amsterdam/bin/weekly-optimize"
   ```

2. Test v2 manually:
   ```bash
   ssh raspi
   cd ~/trader/amsterdam
   source venv/bin/activate
   python daily_optimize_v2.py
   ```

3. Update cron schedule:
   ```bash
   ssh raspi
   crontab -e
   # Replace daily-optimize with weekly-optimize
   # Change from: 0 9 * * 1-5
   # To: 0 2 * * 0
   ```

4. Monitor for a few weeks, compare results

---

## Advanced: Custom Metrics

You can modify the ranking metric in both versions:

```python
# In run_regime_analysis call:
result = tester.run_regime_analysis(
    metric="sharpe_ratio",  # or "total_return", "win_rate", "profit_factor"
    verbose=False
)
```

**Sharpe ratio** (default) balances returns and risk - generally best for strategy selection.

---

## Questions?

- Check logs: `~/trader/amsterdam/logs/weekly_optimize.log`
- Review backups: `~/trader/amsterdam/config/strategy_routing.backup.*.json`
- Monitor live performance: `http://100.101.141.79:8080` (dashboard)
