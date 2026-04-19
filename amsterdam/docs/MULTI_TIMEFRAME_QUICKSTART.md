# Multi-Timeframe System - Quick Start Guide

**TL;DR:** Your trading system now supports different timeframes per symbol. Here's how to use it.

---

## 🚀 Quick Start (3 Steps)

### 1. Fetch Historical Data

```bash
cd /Users/kwasiaddo/projects/trader/amsterdam

# Fetch multiple timeframes for backtesting
python -m core.unified_data_pipeline \
    --symbols AAPL TSLA MSFT \
    --timeframes 5min 15min 30min \
    --days 750
```

### 2. Update Strategy Routing Config

Edit `config/strategy_routing.json`:

```json
{
  "AAPL": {
    "trending": {
      "strategy": "momentum",
      "timeframe": "15min"
    },
    "ranging": {
      "strategy": "meanreversion",
      "timeframe": "5min"
    },
    "use_hybrid": false
  }
}
```

### 3. Integrate BarAggregator

In your main trading file (`autoamsterdam.py`):

```python
from core.bar_aggregator import BarAggregator, Bar
from core.logic.strategy_routing_manager import StrategyRoutingManager

# Initialize
aggregator = BarAggregator()
router = StrategyRoutingManager("config/strategy_routing.json")

# Configure timeframes
for symbol in trading_symbols:
    regime = regime_detector.get_current_regime(symbol)
    routing = router.get_routing(symbol, regime)
    aggregator.set_timeframe(symbol, routing['timeframe'])

# Register callback
def on_bar(bar: Bar):
    strategy = router.get_strategy(bar.symbol, current_regime)
    signal = strategy.generate_signal(bar)
    if signal:
        execute_trade(signal)

aggregator.register_callback(on_bar)

# Feed websocket data
def on_websocket_bar(raw_bar):
    bar = convert_to_bar(raw_bar)  # Your conversion logic
    aggregator.process_bar(bar)

schwab_client.set_quote_callback(on_websocket_bar)
```

---

## 📊 Common Use Cases

### Load Historical Data for Backtesting

```python
from core.unified_data_pipeline import UnifiedDataPipeline

pipeline = UnifiedDataPipeline()

# Load 5-minute bars
bars_5min = pipeline.get_data('AAPL', timeframe='5min')

# Load 15-minute bars
bars_15min = pipeline.get_data('AAPL', timeframe='15min')

# Test strategy
sharpe_5min = backtest(strategy, bars_5min)
sharpe_15min = backtest(strategy, bars_15min)
```

### Handle Regime Changes

```python
# When regime changes
old_regime = current_regime
new_regime = regime_detector.detect_regime("AAPL")

if new_regime != old_regime:
    # Get new routing
    routing = router.get_routing("AAPL", new_regime)

    # Update timeframe (completes partial window automatically)
    aggregator.set_timeframe("AAPL", routing['timeframe'])

    current_regime = new_regime
```

### Market Close Handling

```python
# At 4:00 PM ET
def on_market_close():
    # Complete all partial windows
    aggregator.force_complete_all()
```

---

## 🔍 Quick Verification

### Check Available Timeframes

```python
from core.unified_data_pipeline import UnifiedDataPipeline

p = UnifiedDataPipeline()

# List timeframes for a symbol
print(p.list_available_timeframes('AAPL'))
# Output: ['5min', '15min', '30min', 'day']

# List symbols with 15min data
print(p.list_available_symbols(source='file', timeframe='15min'))
# Output: ['AAPL', 'TSLA', 'MSFT']
```

### Check Aggregator Status

```python
# Get statistics
stats = aggregator.get_stats()
print(stats)
# Output: {
#   'bars_received': 1000,
#   'bars_emitted': 200,
#   'active_symbols': 3,
#   'timeframes': {'AAPL': '5min', 'TSLA': '15min'}
# }

# Get timeframe for a symbol
print(aggregator.get_timeframe('AAPL'))
# Output: '5min'
```

---

## 🧪 Testing

### Run Unit Tests

```bash
# All tests
python -m pytest tests/test_bar_aggregator.py tests/test_unified_pipeline_multitf.py -v

# Quick smoke test
python -m core.bar_aggregator  # Runs built-in demo
```

### Test Integration

```bash
python examples/multi_timeframe_integration.py
```

---

## 📁 File Locations

### Code Files

- **Historical Pipeline:** `core/unified_data_pipeline.py`
- **Streaming Aggregator:** `core/bar_aggregator.py`
- **Strategy Routing:** `core/logic/strategy_routing_manager.py`

### Configuration

- **Routing Config:** `config/strategy_routing.json`

### Data Files

- **Processed Data:** `data/data_storage/proc_data/proc_AAPL_5min.json`
- **Raw Data:** `data/data_storage/raw_data/raw_AAPL_5min.json`

### Documentation

- **Full Guide:** `docs/MULTI_TIMEFRAME_GUIDE.md`
- **Implementation Summary:** `MULTI_TIMEFRAME_IMPLEMENTATION.md`
- **This Quick Start:** `MULTI_TIMEFRAME_QUICKSTART.md`

---

## 🎯 Supported Timeframes

| Timeframe | Description | Best For |
|-----------|-------------|----------|
| `1min` | 1-minute bars | Scalping (use sparingly - large data) |
| `5min` | 5-minute bars | **Intraday mean reversion** ⭐ |
| `15min` | 15-minute bars | **Intraday momentum** ⭐ |
| `30min` | 30-minute bars | **Swing trading** ⭐ |
| `1hour` | 1-hour bars | Position trading |
| `day` | Daily bars | Long-term trends (default) |

**Recommendation:** Start with 5min, 15min, 30min for most strategies.

---

## ⚙️ Configuration Template

Copy this to `config/strategy_routing.json`:

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
      "strategy": "defensive",
      "timeframe": "30min"
    },
    "default": {
      "strategy": "sma",
      "timeframe": "5min"
    },
    "use_hybrid": false
  },
  "TSLA": {
    "trending": {
      "strategy": "momentum",
      "timeframe": "15min"
    },
    "ranging": {
      "strategy": "sma",
      "timeframe": "30min"
    },
    "use_hybrid": true
  },
  "default": {
    "default": {
      "strategy": "sma",
      "timeframe": "5min"
    }
  }
}
```

---

## 🐛 Troubleshooting

### No bars being emitted?

```python
# Check if symbol is configured
print(aggregator.get_configured_symbols())

# Check stats
print(aggregator.get_stats())

# Force complete for testing
aggregator.force_complete_all()
```

### Can't find data for timeframe?

```bash
# Fetch the timeframe
python -m core.unified_data_pipeline \
    --symbols AAPL \
    --timeframes 15min \
    --days 30
```

### Timeframe not changing on regime switch?

```python
# Make sure you're calling this:
routing = router.get_routing(symbol, new_regime)
aggregator.set_timeframe(symbol, routing['timeframe'])
```

---

## 💡 Pro Tips

1. **Start with 5min and 15min** - Good balance between resolution and data size
2. **Avoid 1min for long periods** - Generates huge data files
3. **Test in paper trading first** - Verify aggregation works correctly
4. **Monitor aggregator stats** - Check bars_received vs bars_emitted ratio
5. **Use force_complete at market close** - Ensures no data loss

---

## 📞 Need Help?

1. **Read the full guide:** `docs/MULTI_TIMEFRAME_GUIDE.md`
2. **Check examples:** `examples/multi_timeframe_integration.py`
3. **Review tests:** `tests/test_bar_aggregator.py` for usage patterns
4. **Check logs:** `logs/bar_aggregator.log`, `logs/unified_pipeline.log`

---

**That's it! You're ready to trade on multiple timeframes.** 🎉

For detailed information, see `docs/MULTI_TIMEFRAME_GUIDE.md`.
