# Multi-Timeframe Trading System - Implementation Guide

**Version:** 1.0
**Date:** March 2026
**Status:** Implemented

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Usage Examples](#usage-examples)
5. [Integration Guide](#integration-guide)
6. [Configuration](#configuration)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The multi-timeframe trading system enables the Amsterdam trading platform to operate on different timeframes per symbol/regime combination. This allows strategies to adapt their bar resolution based on market conditions.

### Key Features

- **Multiple Timeframes:** Support for 1min, 5min, 15min, 30min, 1hour, and daily bars
- **Per-Symbol Configuration:** Each symbol can use a different timeframe
- **Regime-Adaptive:** Timeframe changes automatically with regime transitions
- **Dual Pipeline:** Separate historical and streaming data pipelines
- **Backward Compatible:** Works with existing code using default timeframes

### Use Cases

1. **Strategy Optimization:** Test which timeframe performs best during backtesting
2. **Regime Adaptation:** Use shorter timeframes in volatile regimes, longer in calm markets
3. **Strategy-Specific Timeframes:** Mean reversion on 5min, momentum on 15min
4. **Multi-Symbol Trading:** Different symbols operate on optimal timeframes simultaneously

---

## Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    HISTORICAL DATA PIPELINE                      │
│                        (Backtesting/Optimization)                │
└─────────────────────────────────────────────────────────────────┘

Schwab/Alpaca API
    ↓ (fetch 1min, 5min, 15min, 30min, 1hour, day)
UnifiedDataPipeline
    ↓ (store: proc_AAPL_5min.json, proc_AAPL_15min.json, etc.)
Historical Bar Loader
    ↓ (load specific timeframe)
Backtesting Engine
    ↓ (test strategy on each timeframe)
Optimization
    ↓ (select best timeframe per symbol/regime)


┌─────────────────────────────────────────────────────────────────┐
│                     STREAMING DATA PIPELINE                      │
│                          (Live Trading)                          │
└─────────────────────────────────────────────────────────────────┘

Schwab Websocket Stream
    ↓ (high-frequency bars: seconds/sub-minute)
BarAggregator
    ↓ (aggregated bars per configured timeframe)
    ├─ AAPL @ 5min  → Strategy (Mean Reversion)
    ├─ TSLA @ 15min → Strategy (Momentum)
    └─ MSFT @ 1min  → Strategy (Scalping)
        ↓
    Signal Generation
        ↓
    Order Execution
```

### Component Relationships

```
┌──────────────────────┐
│ Strategy Routing     │
│ Manager              │◄────── regime changes
│                      │
│ - get_routing()      │
│   returns:           │
│   • strategy name    │
│   • timeframe        │
│   • use_hybrid       │
└──────────┬───────────┘
           │
           │ routing decision
           ↓
┌──────────────────────┐     ┌──────────────────────┐
│ BarAggregator        │     │ UnifiedDataPipeline  │
│                      │     │                      │
│ - set_timeframe()    │     │ - update_symbols()   │
│ - process_bar()      │     │ - get_data()         │
│ - force_complete()   │     │   (timeframe param)  │
└──────────────────────┘     └──────────────────────┘
           │                            │
           │ aggregated bars            │ historical data
           ↓                            ↓
┌──────────────────────────────────────────────────┐
│              Strategy Logic                      │
│                                                  │
│ - Receives bars at configured timeframe          │
│ - Generates signals                              │
│ - Executes trades                                │
└──────────────────────────────────────────────────┘
```

---

## Components

### 1. UnifiedDataPipeline (Historical Data)

**File:** `core/unified_data_pipeline.py`

**Purpose:** Fetch and store historical data at multiple timeframes for backtesting.

#### Key Methods

```python
# Fetch multiple timeframes
await pipeline.update_symbols(
    symbols=['AAPL', 'TSLA'],
    timeframes=['1min', '5min', '15min', '30min', '1hour'],
    days=750  # ~3 years of data
)

# Load specific timeframe
bars_5min = pipeline.get_data('AAPL', timeframe='5min')
bars_15min = pipeline.get_data('AAPL', timeframe='15min')

# List available timeframes
timeframes = pipeline.list_available_timeframes('AAPL')
# Returns: ['1min', '5min', '15min', '30min', '1hour', 'day']
```

#### Storage Format

**Files:**
```
data/data_storage/
├── proc_data/
│   ├── proc_AAPL_1min.json    # 1-minute bars
│   ├── proc_AAPL_5min.json    # 5-minute bars
│   ├── proc_AAPL_15min.json   # 15-minute bars
│   ├── proc_AAPL_30min.json   # 30-minute bars
│   ├── proc_AAPL_1hour.json   # 1-hour bars
│   ├── proc_AAPL_file.json    # Daily bars (backward compatibility)
│   └── ...
└── raw_data/
    ├── raw_AAPL_5min.json
    └── ...
```

**Database:**
- Table: `stock_table`
- Columns: `symbol`, `Date`, `timeframe`, `open`, `high`, `low`, `close`, `volume`, ...
- Unique key: `(symbol, Date, timeframe)`

#### Supported Timeframes

| Timeframe | Description | Use Case |
|-----------|-------------|----------|
| `1min` | 1-minute bars | High-frequency trading, scalping |
| `5min` | 5-minute bars | Intraday mean reversion |
| `15min` | 15-minute bars | Intraday momentum |
| `30min` | 30-minute bars | Swing trading |
| `1hour` | 1-hour bars | Position trading |
| `day` | Daily bars | Long-term trends (default) |

---

### 2. BarAggregator (Streaming Data)

**File:** `core/bar_aggregator.py`

**Purpose:** Aggregate high-frequency streaming bars to target timeframes in real-time.

#### Key Classes

**Bar:**
```python
@dataclass
class Bar:
    timestamp: datetime    # Aligned to interval start
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str
    timeframe: str        # e.g., "5min"
```

**BarAggregator:**
```python
aggregator = BarAggregator()

# Configure timeframes
aggregator.set_timeframe("AAPL", "5min")
aggregator.set_timeframe("TSLA", "15min")

# Register callback for completed bars
def on_bar(bar: Bar):
    strategy.process_bar(bar)

aggregator.register_callback(on_bar)

# Process streaming bars
for raw_bar in websocket_stream:
    completed_bars = aggregator.process_bar(raw_bar)

# Force complete at market close
aggregator.force_complete_all()
```

#### Features

1. **Window Alignment:** Automatically aligns bars to interval boundaries
   - 5min: 9:30, 9:35, 9:40, ...
   - 15min: 9:30, 9:45, 10:00, ...

2. **Aggregation Logic:**
   - Open: First bar's open
   - High: Maximum high across all bars
   - Low: Minimum low across all bars
   - Close: Last bar's close
   - Volume: Sum of all volumes

3. **Seamless Timeframe Switching:**
   - Completes partial window before switching
   - No data loss on regime changes

4. **Statistics Tracking:**
   ```python
   stats = aggregator.get_stats()
   # Returns: {
   #   'bars_received': 1000,
   #   'bars_emitted': 200,
   #   'timeframe_changes': 3,
   #   'active_symbols': 5,
   #   'timeframes': {'AAPL': '5min', 'TSLA': '15min', ...}
   # }
   ```

---

### 3. StrategyRoutingManager (Routing Logic)

**File:** `core/logic/strategy_routing_manager.py`

**Purpose:** Route (symbol, regime) pairs to strategies with timeframe configuration.

#### Configuration Format

**Simple (Backward Compatible):**
```json
{
  "AAPL": {
    "trending": "momentum_strategy",
    "ranging": "mean_reversion_strategy",
    "default": "sma_strategy"
  }
}
```

**Extended (With Timeframes):**
```json
{
  "AAPL": {
    "low_volatility": {
      "strategy": "meanreversion",
      "timeframe": "5min"
    },
    "normal": {
      "strategy": "rsi",
      "timeframe": "1min"
    },
    "high_volatility": {
      "strategy": "rsi",
      "timeframe": "15min"
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
  }
}
```

#### API Methods

```python
router = StrategyRoutingManager("config/strategy_routing.json")

# Get full routing decision
routing = router.get_routing("AAPL", "trending")
# Returns: {
#   'strategy': 'momentum_strategy',
#   'timeframe': '15min',
#   'use_hybrid': False
# }

# Get just strategy name
strategy_name = router.get_strategy_name("AAPL", "trending")

# Get strategy instance
strategy = router.get_strategy("AAPL", "trending")

# Update routing
router.set_strategy(
    "AAPL",
    "trending",
    "momentum",
    timeframe="15min",
    persist=True
)
```

#### Resolution Order

1. Symbol-specific regime mapping
2. Symbol default
3. Global regime mapping
4. Global default
5. Hardcoded fallback (`'momentum_strategy'`, `'5min'`)

---

## Usage Examples

### Example 1: Historical Data Fetching

```python
from core.unified_data_pipeline import UnifiedDataPipeline

# Initialize pipeline
pipeline = UnifiedDataPipeline()

# Fetch multiple timeframes for backtesting
await pipeline.update_symbols(
    symbols=['AAPL', 'TSLA', 'MSFT'],
    timeframes=['1min', '5min', '15min', '30min', '1hour'],
    days=750,  # ~3 years
    source='alpaca'  # or 'schwab'
)

# Load data for backtesting
for symbol in ['AAPL', 'TSLA', 'MSFT']:
    for timeframe in ['5min', '15min', '30min']:
        bars = pipeline.get_data(symbol, timeframe=timeframe)

        # Run backtest
        metrics = backtest_strategy(strategy, bars)
        print(f"{symbol} @ {timeframe}: Sharpe = {metrics['sharpe_ratio']:.2f}")

# Find best timeframe
best_timeframe = max(results, key=lambda x: x['sharpe'])
print(f"Best timeframe for {symbol}: {best_timeframe}")
```

### Example 2: Live Trading with BarAggregator

```python
from core.bar_aggregator import BarAggregator, Bar
from core.logic.strategy_routing_manager import StrategyRoutingManager
from data.streaming.schwab_stream import SchwabStreamingClient

# Initialize components
aggregator = BarAggregator()
router = StrategyRoutingManager("config/strategy_routing.json")

# Configure aggregator from routing config
def configure_timeframes(symbols):
    """Set up BarAggregator timeframes from strategy routing."""
    regime_detector = RegimeDetector()

    for symbol in symbols:
        # Get current regime
        regime = regime_detector.get_current_regime(symbol)

        # Get routing decision
        routing = router.get_routing(symbol, regime)

        # Configure aggregator
        aggregator.set_timeframe(symbol, routing['timeframe'])

        print(f"[{symbol}] Regime: {regime}, Strategy: {routing['strategy']}, "
              f"Timeframe: {routing['timeframe']}")

# Configure for trading symbols
configure_timeframes(['AAPL', 'TSLA', 'MSFT'])

# Register bar handler
def on_aggregated_bar(bar: Bar):
    """Handle completed aggregated bars."""
    # Get strategy for this symbol
    regime = regime_detector.get_current_regime(bar.symbol)
    strategy = router.get_strategy(bar.symbol, regime)

    # Generate signal
    signal = strategy.generate_signal(bar)

    if signal:
        print(f"[{bar.symbol}] {signal.action} @ {bar.timestamp}")
        # Execute trade
        execute_trade(signal)

aggregator.register_callback(on_aggregated_bar)

# Connect to Schwab websocket
schwab_client = SchwabStreamingClient()

def on_websocket_bar(raw_data):
    """Handle incoming websocket data."""
    # Convert to Bar object
    bar = Bar(
        timestamp=datetime.fromtimestamp(raw_data['timestamp'] / 1000),
        open=raw_data['open'],
        high=raw_data['high'],
        low=raw_data['low'],
        close=raw_data['close'],
        volume=raw_data['volume'],
        symbol=raw_data['symbol'],
        timeframe='raw'
    )

    # Aggregate
    aggregator.process_bar(bar)

schwab_client.set_quote_callback(on_websocket_bar)
schwab_client.connect()

# Market close handler
def on_market_close():
    """Complete all windows at market close."""
    aggregator.force_complete_all()
```

### Example 3: Regime Change with Timeframe Switch

```python
# Detect regime change
old_regime = current_regime
new_regime = regime_detector.detect_regime("AAPL")

if new_regime != old_regime:
    print(f"[AAPL] Regime change: {old_regime} → {new_regime}")

    # Get new routing
    new_routing = router.get_routing("AAPL", new_regime)

    # Update aggregator (will complete partial window)
    aggregator.set_timeframe("AAPL", new_routing['timeframe'])

    print(f"[AAPL] Timeframe changed to {new_routing['timeframe']}")

    # Update current regime
    current_regime = new_regime
```

### Example 4: Backtesting Across Timeframes

```python
from core.backtest.unified_backtest_runner import UnifiedBacktestRunner

# Test strategy on multiple timeframes
results = {}

for timeframe in ['1min', '5min', '15min', '30min', '1hour']:
    # Load data
    bars = pipeline.get_data('AAPL', timeframe=timeframe)

    # Run backtest
    runner = UnifiedBacktestRunner(
        strategy_name='rsi',
        data=bars,
        config={
            'initial_capital': 10000,
            'position_sizing': 'volatility_scaled'
        }
    )

    metrics = runner.run()
    results[timeframe] = metrics

    print(f"{timeframe:6s}: Sharpe={metrics['sharpe_ratio']:.2f}, "
          f"Return={metrics['total_return']:.2%}")

# Find optimal timeframe
optimal = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
print(f"\nOptimal timeframe: {optimal[0]} (Sharpe: {optimal[1]['sharpe_ratio']:.2f})")

# Update strategy routing config
router.set_strategy(
    'AAPL',
    regime='normal',
    strategy_name='rsi',
    timeframe=optimal[0],
    persist=True
)
```

---

## Integration Guide

### Step 1: Update Historical Data

```bash
# Fetch multiple timeframes for all trading symbols
python -m core.unified_data_pipeline \
    --symbols AAPL TSLA MSFT NVDA AMD \
    --timeframes 1min 5min 15min 30min 1hour \
    --days 750 \
    --source alpaca
```

### Step 2: Run Backtesting Optimization

```python
# Find optimal timeframes per symbol/regime
from core.backtest.strategy_selector import StrategySelector

selector = StrategySelector(
    symbols=['AAPL', 'TSLA', 'MSFT'],
    timeframes=['1min', '5min', '15min', '30min', '1hour'],
    strategies=['rsi', 'sma', 'momentum'],
    regimes=['trending', 'ranging', 'volatile']
)

# Run optimization
results = selector.optimize_all()

# Generate routing config
config = selector.generate_routing_config(results)

# Save to file
with open('config/strategy_routing.json', 'w') as f:
    json.dump(config, f, indent=2)
```

### Step 3: Integrate BarAggregator into Live Trading

**File:** `autoamsterdam.py` (or main trading loop)

```python
# Initialize aggregator
bar_aggregator = BarAggregator()

# Configure from strategy routing
def configure_aggregator():
    """Set up aggregator timeframes from routing config."""
    routing_manager = StrategyRoutingManager('config/strategy_routing.json')
    regime_detector = RegimeDetector()

    for symbol in trading_symbols:
        regime = regime_detector.get_current_regime(symbol)
        routing = routing_manager.get_routing(symbol, regime)
        bar_aggregator.set_timeframe(symbol, routing['timeframe'])

configure_aggregator()

# Connect to Schwab stream
schwab_stream = SchwabStreamingClient()

def on_bar(bar_data):
    bar = convert_to_bar(bar_data)
    bar_aggregator.process_bar(bar)

schwab_stream.set_quote_callback(on_bar)

# Register strategy callback
def on_aggregated_bar(bar: Bar):
    # Get strategy for this symbol/regime
    regime = regime_detector.get_current_regime(bar.symbol)
    strategy = routing_manager.get_strategy(bar.symbol, regime)

    # Process bar
    signal = strategy.generate_signal(bar)
    if signal:
        execute_trade(signal)

bar_aggregator.register_callback(on_aggregated_bar)
```

### Step 4: Handle Regime Changes

```python
# In regime monitoring loop
def check_regime_changes():
    """Check for regime changes and update timeframes."""
    for symbol in trading_symbols:
        old_regime = current_regimes.get(symbol)
        new_regime = regime_detector.detect_regime(symbol)

        if new_regime != old_regime:
            # Get new routing
            new_routing = routing_manager.get_routing(symbol, new_regime)

            # Update aggregator (completes partial window)
            bar_aggregator.set_timeframe(symbol, new_routing['timeframe'])

            # Log change
            logger.info(
                f"[{symbol}] Regime: {old_regime} → {new_regime}, "
                f"Timeframe: {new_routing['timeframe']}"
            )

            # Update tracking
            current_regimes[symbol] = new_regime

# Run periodically (e.g., every 5 minutes)
schedule.every(5).minutes.do(check_regime_changes)
```

---

## Configuration

### Strategy Routing Config

**File:** `config/strategy_routing.json`

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
    "volatile": {
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
      "timeframe": "30min"
    },
    "ranging": {
      "strategy": "sma",
      "timeframe": "15min"
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

### Environment Variables

No additional environment variables required. Uses existing Alpaca/Schwab credentials.

---

## Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test suites
python -m pytest tests/test_bar_aggregator.py -v
python -m pytest tests/test_unified_pipeline_multitf.py -v

# Run with coverage
python -m pytest tests/ --cov=core --cov-report=html
```

### Integration Tests

```python
# Test end-to-end flow
python -m core.bar_aggregator  # Runs built-in demo

# Test with mock data
python tests/test_integration_multitf.py
```

### Manual Testing

```python
# Test BarAggregator standalone
from core.bar_aggregator import BarAggregator, Bar
from datetime import datetime, timedelta

aggregator = BarAggregator()
aggregator.set_timeframe("TEST", "5min")

def on_bar(bar):
    print(f"Completed bar: {bar.timestamp} OHLCV({bar.open}, {bar.high}, {bar.low}, {bar.close}, {bar.volume})")

aggregator.register_callback(on_bar)

# Send test bars
base_time = datetime(2026, 3, 14, 9, 30, 0)
for i in range(10):
    bar = Bar(
        timestamp=base_time + timedelta(minutes=i),
        open=150.0 + i * 0.1,
        high=151.0 + i * 0.1,
        low=149.0 + i * 0.1,
        close=150.5 + i * 0.1,
        volume=1000,
        symbol="TEST",
        timeframe="1min"
    )
    aggregator.process_bar(bar)
```

---

## Troubleshooting

### Common Issues

#### 1. "Unsupported timeframe" Error

**Cause:** Using invalid timeframe string.

**Solution:**
```python
# Valid timeframes
valid = ['1min', '5min', '15min', '30min', '1hour', 'day']

# Check before using
if timeframe in pipeline.SUPPORTED_TIMEFRAMES:
    pipeline.get_data(symbol, timeframe=timeframe)
```

#### 2. No Data for Timeframe

**Cause:** Data not fetched for that timeframe.

**Solution:**
```bash
# Fetch missing timeframe
python -m core.unified_data_pipeline \
    --symbols AAPL \
    --timeframes 15min \
    --days 750
```

#### 3. Aggregator Not Emitting Bars

**Cause:** Window not complete yet, or symbol not configured.

**Debug:**
```python
# Check configuration
print(aggregator.get_configured_symbols())
print(aggregator.get_timeframe("AAPL"))

# Check stats
stats = aggregator.get_stats()
print(f"Received: {stats['bars_received']}, Emitted: {stats['bars_emitted']}")

# Force complete for testing
aggregator.force_complete_all()
```

#### 4. Timeframe Not Changing on Regime Switch

**Cause:** Aggregator not being updated when regime changes.

**Solution:**
```python
# Ensure this is called when regime changes
new_routing = router.get_routing(symbol, new_regime)
aggregator.set_timeframe(symbol, new_routing['timeframe'])
```

#### 5. Memory Usage High

**Cause:** Too many symbols or very small timeframes (1min) with long lookback.

**Solution:**
```python
# Limit data retention
pipeline.update_symbols(
    symbols=['AAPL'],
    timeframes=['5min'],  # Avoid 1min for long periods
    days=30  # Reduce lookback
)

# For aggregator, memory is minimal (only current windows)
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check aggregator stats
stats = aggregator.get_stats()
print(json.dumps(stats, indent=2))

# Verify pipeline files
print(pipeline.list_available_timeframes('AAPL'))
```

---

## Performance Considerations

### Disk Space

**Estimates:**
- 1 symbol, 1 timeframe, 1 year ≈ 5 MB (1min bars)
- 20 symbols × 5 timeframes × 2 years ≈ 1 GB

### Memory Usage

**BarAggregator:**
- ~60 bars per symbol (max for 1min aggregation)
- 20 symbols × 60 bars × 200 bytes ≈ 240 KB (negligible)

**Pipeline:**
- Processes data in chunks
- Peak usage: ~100 MB for processing

### CPU/Latency

- Bar aggregation: ~1ms per bar
- Window completion: O(n) where n = bars in window (typically 5-60)
- Negligible impact on strategy execution

---

## Next Steps

1. **Run Backtesting Optimization** (on Mac with more resources)
   - Test all strategy-timeframe combinations
   - Find optimal configurations per symbol/regime
   - Generate production routing config

2. **Deploy to Pi**
   - Update `strategy_routing.json` with optimized timeframes
   - Ensure all required historical data is fetched
   - Monitor performance in paper trading

3. **Monitor and Tune**
   - Track Sharpe ratios per symbol/timeframe
   - A/B test timeframe changes
   - Adjust based on live performance

4. **Future Enhancements**
   - Dynamic timeframe selection based on realized volatility
   - Multi-timeframe signal combination (e.g., trend on 1hr, entry on 5min)
   - Adaptive timeframe windows (shrink/expand based on volume)

---

## Support

For questions or issues:
1. Check this documentation
2. Review unit tests for usage examples
3. Check logs: `logs/bar_aggregator.log`, `logs/unified_pipeline.log`
4. Review implementation plan in project docs

---

**End of Guide**
