# How Timeframe Selection Works - Visual Guide

## The Decision Tree

```
Symbol: AAPL
    ↓
What's the current regime?
    ↓
┌───────────────────┬──────────────┬────────────────────┐
│ low_volatility    │    normal    │  high_volatility   │
└───────────────────┴──────────────┴────────────────────┘
         │                 │                  │
         ↓                 ↓                  ↓
    5min bars         15min bars         30min bars
         │                 │                  │
         ↓                 ↓                  ↓
  meanreversion         rsi                rsi
   strategy          strategy           strategy
```

## Step-by-Step Breakdown

### Step 1: System Startup (9:30 AM)

```python
# In MultiTimeframeManager.__init__()
for symbol in ['AAPL', 'TSLA', 'MSFT']:
    # 1. Determine initial regime (default: "normal")
    regime = "normal"

    # 2. Look up routing for (symbol, regime)
    routing = routing_manager.get_routing(symbol, regime)
    # Returns: {'strategy': 'rsi', 'timeframe': '15min', 'use_hybrid': False}

    # 3. Configure aggregator with that timeframe
    aggregator.set_timeframe(symbol, routing['timeframe'])
    # AAPL is now configured for 15min bars
```

**Result:**
```
AAPL  → regime: normal     → timeframe: 15min → strategy: rsi
TSLA  → regime: normal     → timeframe: 30min → strategy: bollinger
MSFT  → regime: normal     → timeframe: 15min → strategy: meanreversion
```

### Step 2: Lookup in Config (How it Works)

When `routing_manager.get_routing("AAPL", "normal")` is called:

**1. Opens config file:**
```json
{
  "AAPL": {
    "low_volatility": {
      "strategy": "meanreversion",
      "timeframe": "5min"
    },
    "normal": {
      "strategy": "rsi",
      "timeframe": "15min"  ← Found it!
    },
    "high_volatility": {
      "strategy": "rsi",
      "timeframe": "30min"
    }
  }
}
```

**2. Finds the match:**
- Symbol: "AAPL" ✓
- Regime: "normal" ✓
- Returns: `{"strategy": "rsi", "timeframe": "15min", "use_hybrid": false}`

**3. Configures BarAggregator:**
```python
# Now AAPL will aggregate bars every 15 minutes
aggregator.set_timeframe("AAPL", "15min")
```

### Step 3: Real-Time Bar Aggregation

```
9:30 AM - Quote arrives for AAPL
     ↓
BarAggregator checks: "What timeframe for AAPL?"
     ↓
Answer: "15min" (from config lookup)
     ↓
Aggregator: "I'll collect quotes until 9:45, then emit a bar"
     ↓
9:31 AM - Another quote arrives
     ↓
Aggregator: "Still collecting... window ends at 9:45"
     ↓
9:45 AM - Window complete!
     ↓
Aggregator emits 15-minute bar:
  OHLCV(150.00, 151.50, 149.50, 150.75, 250000)
     ↓
RSI strategy processes this 15-minute bar
```

### Step 4: Regime Changes (11:00 AM)

```
11:00 AM - Market becomes volatile
     ↓
RegimeDetector detects: AAPL regime changed
  OLD: "normal"
  NEW: "high_volatility"
     ↓
MultiTimeframeManager.update_regime("AAPL", "high_volatility")
     ↓
Looks up new routing:
  routing_manager.get_routing("AAPL", "high_volatility")
  Returns: {"strategy": "rsi", "timeframe": "30min"}
     ↓
BarAggregator.set_timeframe("AAPL", "30min")
     ↓
Aggregator:
  1. Completes partial 15min bar (if any)
  2. Switches to 30min windows
  3. Next bar will be emitted at 11:30
```

## Live Example with Current Config

Let's trace AAPL through a trading day:

### Morning (Low Volatility)
```
9:30 AM - Market opens
Regime: low_volatility
Config lookup:
  AAPL + low_volatility → timeframe: "5min"

Bars emitted at:
  9:35, 9:40, 9:45, 9:50, 9:55, 10:00... (every 5 minutes)

Strategy: meanreversion (fast bars, good for mean reversion)
```

### Midday (Normal Conditions)
```
10:30 AM - Volatility normalizes
Regime: normal
Config lookup:
  AAPL + normal → timeframe: "15min"

System:
  1. Completes partial 5min bar
  2. Switches to 15min bars

Bars emitted at:
  10:45, 11:00, 11:15, 11:30... (every 15 minutes)

Strategy: rsi (balanced timeframe for RSI signals)
```

### Afternoon (High Volatility)
```
2:00 PM - Major news, market volatile
Regime: high_volatility
Config lookup:
  AAPL + high_volatility → timeframe: "30min"

System:
  1. Completes partial 15min bar
  2. Switches to 30min bars

Bars emitted at:
  2:30, 3:00, 3:30, 4:00 (every 30 minutes)

Strategy: rsi (longer bars reduce noise in volatile markets)
```

## The Code Path

### 1. Initial Configuration
```python
# In MultiTimeframeManager._configure_initial_timeframes()

for symbol in symbols:
    # Default to "normal" regime at startup
    regime = "normal"

    # Get routing decision
    routing = self.routing_manager.get_routing(symbol, regime)
    # For AAPL: {'strategy': 'rsi', 'timeframe': '15min', 'use_hybrid': False}

    # Tell aggregator to use this timeframe for this symbol
    self.aggregator.set_timeframe(symbol, routing['timeframe'])
    # BarAggregator now knows: AAPL → 15min windows
```

### 2. During Trading
```python
# In MultiTimeframeManager.process_websocket_data()

def process_websocket_data(raw_data):
    # 1. Convert websocket quote to Bar object
    bar = Bar(
        symbol='AAPL',
        timestamp=now,
        open=150.00,
        high=150.50,
        low=149.50,
        close=150.25,
        volume=1000
    )

    # 2. Process through aggregator
    completed_bars = self.aggregator.process_bar(bar)

    # Aggregator internally:
    # - Checks: "What timeframe for AAPL?" → "15min"
    # - Adds this quote to current 15min window
    # - If window complete, returns aggregated bar
    # - Otherwise returns empty list

    return completed_bars
```

### 3. Regime Change
```python
# In MultiTimeframeManager.update_regime()

def update_regime(symbol, new_regime):
    # 1. Get new routing for this regime
    new_routing = self.routing_manager.get_routing(symbol, new_regime)
    # AAPL + high_volatility → {'strategy': 'rsi', 'timeframe': '30min'}

    # 2. Update aggregator
    self.aggregator.set_timeframe(symbol, new_routing['timeframe'])
    # Aggregator:
    #   - Completes partial 15min bar
    #   - Switches to 30min windows

    # 3. Track new regime
    self.current_regimes[symbol] = new_regime
```

## Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│                   Configuration File                        │
│            (config/strategy_routing.json)                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
            ┌───────────────────────────┐
            │  StrategyRoutingManager   │
            │  get_routing(symbol,      │
            │              regime)      │
            └───────────────────────────┘
                            │
                            ↓ Returns timeframe
            ┌───────────────────────────┐
            │     BarAggregator         │
            │  set_timeframe(symbol,    │
            │                timeframe) │
            └───────────────────────────┘
                            │
                            ↓ Aggregates to timeframe
            ┌───────────────────────────┐
            │   Completed Bars          │
            │   (at correct timeframe)  │
            └───────────────────────────┘
```

## Quick Test

See it in action:

```bash
python3 -c "
from core.logic.strategy_routing_manager import StrategyRoutingManager

router = StrategyRoutingManager('config/strategy_routing.json')

# Simulate different regimes for AAPL
for regime in ['low_volatility', 'normal', 'high_volatility']:
    routing = router.get_routing('AAPL', regime)
    print(f'AAPL in {regime:15s} → {routing[\"timeframe\"]:6s} bars using {routing[\"strategy\"]}')
"
```

**Output:**
```
AAPL in low_volatility  → 5min   bars using meanreversion
AAPL in normal          → 15min  bars using rsi
AAPL in high_volatility → 30min  bars using rsi
```

## Summary

**The timeframe is determined by:**

1. **Symbol** (e.g., AAPL)
2. **Current Regime** (e.g., normal, high_volatility)
3. **Config Lookup** (symbol + regime → timeframe)

**The config is the source of truth:**
```json
"AAPL": {
  "normal": {
    "strategy": "rsi",
    "timeframe": "15min"  ← THIS determines the timeframe
  }
}
```

**The aggregator follows the config:**
- AAPL in normal regime → 15min bars
- AAPL in high_volatility → 30min bars
- TSLA in normal regime → 30min bars

**Simple!** 🎯
