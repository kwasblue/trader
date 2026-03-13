# Event System

The event system is the backbone of communication between all components in Schwab Trader. It enables loose coupling between the trading engine, GUI, and other subsystems.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EVENT BUS ARCHITECTURE                            │
│                                                                              │
│  ┌──────────────────┐                           ┌──────────────────┐        │
│  │  SimulationRunner│                           │   AlpacaRunner   │        │
│  │  SchwabRunner    │──┐                    ┌───│   (Live Trading) │        │
│  │  (Event Sources) │  │                    │   └──────────────────┘        │
│  └──────────────────┘  │                    │                               │
│                        ▼                    ▼                               │
│              ┌─────────────────────────────────────────────┐                │
│              │           EventHandler (Singleton)           │                │
│              │                                              │                │
│              │  - Async event emission                      │                │
│              │  - Sync & async subscriber support           │                │
│              │  - Schema validation (TypedDict)             │                │
│              │  - Thread pool for sync callbacks            │                │
│              │  - Semaphore-limited concurrency             │                │
│              └─────────────────────┬───────────────────────┘                │
│                                    │                                         │
│              ┌─────────────────────┼─────────────────────┐                  │
│              ▼                     ▼                     ▼                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   DataFeeder     │  │ ExecutionEngine  │  │   TradeLogger    │          │
│  │   (GUI Bridge)   │  │ (Signal Handler) │  │   (Logging)      │          │
│  └────────┬─────────┘  └──────────────────┘  └──────────────────┘          │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                       │
│  │   Qt Signals     │ ◄── Thread-safe signal emission                       │
│  │   (MainWindow)   │     via QTimer.singleShot(0)                          │
│  └──────────────────┘                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### EventHandler (Singleton)

The central event bus that all components use for communication.

**Location:** `core/events/eventhandler.py`

```python
from core.events.eventhandler import get_event_handler

# Get the global singleton
bus = get_event_handler()

# Subscribe to events (sync - immediate)
bus.subscribe_sync("PNL_UPDATE", my_handler)

# Subscribe to events (async)
await bus.subscribe("NEW_BAR", my_async_handler)

# Emit events
await bus.emit("NEW_BAR", {
    "symbol": "AAPL",
    "timestamp": "2024-01-15T10:00:00Z",
    "open": 150.0,
    "high": 151.0,
    "low": 149.0,
    "close": 150.5,
    "volume": 1000
})
```

### Event Types

All event types are defined in `core/contracts/events.py`:

| Event | Description | Payload Type |
|-------|-------------|--------------|
| `EVENT_NEW_BAR` | New price bar received | `BarPayload` |
| `EVENT_PNL_UPDATE` | Portfolio P&L update | `PnLPayload` |
| `EVENT_NEW_TRADE` | Trade executed | `TradePayload` |
| `EVENT_ORDER_STATUS` | Order status change | `OrderStatusPayload` |
| `EVENT_POSITION_UPDATE` | Position change | `PositionPayload` |
| `EVENT_STRATEGY_SIGNAL` | Strategy signal generated | `StrategySignalPayload` |
| `EVENT_REGIME_UPDATE` | Market regime change | `RegimePayload` |
| `EVENT_HEALTH_UPDATE` | System health status | `HealthPayload` |
| `EVENT_ALERT` | Alert/notification | `AlertPayload` |
| `EVENT_GUARDRAIL_TRIGGERED` | Risk guardrail hit | `GuardrailPayload` |

### Payload Schemas

Payloads are validated against TypedDict schemas:

```python
from core.contracts.events import BarPayload, PnLPayload

# BarPayload (all required)
bar: BarPayload = {
    "symbol": "AAPL",
    "timestamp": "2024-01-15T10:00:00Z",
    "open": 150.0,
    "high": 151.0,
    "low": 149.0,
    "close": 150.5,
    "volume": 1000
}

# PnLPayload (total=False, all optional)
pnl: PnLPayload = {
    "portfolio_value": 100000.0,
    "equity_curve": [99000, 99500, 100000],
    "unrealized": 500.0,
    "realized": 200.0,
    "drawdown": 0.01,
    "timestamp": "2024-01-15T10:00:00Z"
}
```

## DataFeeder (GUI Bridge)

Bridges the EventHandler to Qt signals for the GUI.

**Location:** `monitoring/feeds/feeder.py`

```
EventBus Events → DataFeeder → Qt Signals → MainWindow
```

### How It Works

1. **Synchronous Subscription:** DataFeeder subscribes to events in `__init__` using `subscribe_sync()` to ensure subscriptions are in place before any events are emitted.

2. **Async Event Handlers:** Event handlers are async functions that receive events from the EventHandler.

3. **Qt Signal Emission:** Handlers emit Qt signals using `QTimer.singleShot(0)` for thread safety.

```python
class DataFeeder(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.s = FeedSignals()  # Qt signals
        self.bus = get_event_handler()

        # Subscribe synchronously to ensure we don't miss events
        self.bus.subscribe_sync("PNL_UPDATE", self._handle_pnl)
        self.bus.subscribe_sync("NEW_BAR", self._handle_bar)

    async def _handle_pnl(self, event):
        data = event.payload
        # Thread-safe Qt signal emission
        QtCore.QTimer.singleShot(0, lambda: self.s.pnl_update.emit(data))
```

## Event Flow Example

### Simulation → GUI

```
1. SimulationRunner generates bar
   └── await self.events.emit(EVENT_NEW_BAR, bar_payload)

2. EventHandler dispatches to all subscribers
   └── Calls DataFeeder._handle_bar(event)

3. DataFeeder emits Qt signal
   └── QTimer.singleShot(0, lambda: self.s.bar_update.emit(symbol, data))

4. MainWindow receives signal
   └── self.feeder.s.bar_update.connect(self._gui_on_bar)
   └── _gui_on_bar() updates the chart
```

### Live Trading → GUI

```
1. AlpacaRunner receives websocket bar
   └── await self.event_handler.emit(EVENT_NEW_BAR, bar_payload)

2. EventHandler dispatches to:
   - DataFeeder → GUI update
   - ExecutionEngine → Signal processing
   - TradeLogger → Logging

3. If ExecutionEngine generates trade:
   └── await self.event_handler.emit(EVENT_NEW_TRADE, trade_payload)
   └── DataFeeder → Trade notification in GUI
```

## Best Practices

### 1. Use Synchronous Subscription for Critical Handlers

```python
# Good: Guaranteed to be subscribed before events start
bus.subscribe_sync("NEW_BAR", critical_handler)

# Less reliable for critical paths
await bus.subscribe("NEW_BAR", handler)
```

### 2. Always Use the Singleton

```python
# Good: Everyone shares the same bus
from core.events.eventhandler import get_event_handler
bus = get_event_handler()

# Bad: Creates separate instances
bus = EventHandler()  # Don't do this directly
```

### 3. Handle Exceptions in Handlers

```python
async def my_handler(event):
    try:
        process(event.payload)
    except Exception as e:
        logger.error(f"Handler error: {e}")
        # Don't re-raise - let other handlers continue
```

### 4. Use TypedDict for Payloads

```python
from core.contracts.events import BarPayload

# Good: Type-checked payload
payload: BarPayload = {
    "symbol": "AAPL",
    "timestamp": now.isoformat(),
    ...
}

# Validation happens automatically on emit
await bus.emit(EVENT_NEW_BAR, payload)
```

## Debugging

### Check Subscriber Count

```python
bus = get_event_handler()
print(f"PNL_UPDATE listeners: {len(bus.listeners.get('PNL_UPDATE', []))}")
print(f"NEW_BAR listeners: {len(bus.listeners.get('NEW_BAR', []))}")
```

### Verify Singleton Identity

```python
bus1 = get_event_handler()
bus2 = get_event_handler()
print(f"Same instance: {id(bus1) == id(bus2)}")  # Should be True
```

### Enable Debug Logging

The EventHandler logs subscription and emission events:

```
DEBUG:EventHandler:[EventHandler] Subscribed to 'PNL_UPDATE' -> on_pnl (total: 1)
DEBUG:EventHandler:[EventHandler] Emit 'NEW_BAR' to 3 listener(s)
```

## Graceful Shutdown

```python
# Wait for pending tasks and close thread pool
await bus.shutdown()
```

## Related Documentation

- [Architecture](architecture.md) - System design overview
- [Data Flow](data-flow.md) - Complete data flow diagram
- [Monitoring](monitoring.md) - GUI architecture
