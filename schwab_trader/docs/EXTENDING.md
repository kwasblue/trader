# Extending the Trading System

This guide explains how to extend the trading system with new components using the established factory/registry patterns.

## Architecture Overview

The system uses **factory/registry patterns** for extensibility:

| Component | Factory | Base Class |
|-----------|---------|------------|
| Live Trading Runners | `RunnerFactory` | `BaseLiveRunner` |
| Position Sizers | `PositionSizerFactory` | `PositionSizerBase` |
| Brokers | Direct inheritance | `BaseBrokerInterface` |

---

## Adding a New Broker

### Step 1: Create the Broker Class

Create a new file `core/broker/your_broker.py`:

```python
from core.base.base_broker_interface import BaseBrokerInterface
from core.contracts.types import OrderResult, PositionView, BrokerSnapshot
from core.enums import OrderSide


class YourBroker(BaseBrokerInterface):
    """Your broker implementation."""

    def __init__(self, api_key: str, api_secret: str, **kwargs):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        # Initialize your broker client here

    def connect_sync(self) -> None:
        """Connect to the broker API."""
        # Implement connection logic
        pass

    def disconnect(self) -> None:
        """Disconnect from the broker API."""
        pass

    def place_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        order_type: str = "market",
        **kwargs
    ) -> OrderResult:
        """Place an order."""
        # Implement order placement
        return OrderResult(
            order_id="...",
            symbol=symbol,
            side=side.value,
            qty=qty,
            status="submitted"
        )

    def get_position(self, symbol: str) -> PositionView:
        """Get position for a symbol."""
        # Implement position lookup
        return PositionView(symbol=symbol, qty=0)

    def get_positions(self) -> dict[str, PositionView]:
        """Get all positions."""
        return {}

    def get_account_snapshot(self) -> BrokerSnapshot:
        """Get account snapshot."""
        return BrokerSnapshot(
            cash=0.0,
            equity=0.0,
            buying_power=0.0
        )
```

### Step 2: Create the Live Runner

Create `core/your_runner.py`:

```python
from core.base.base_live_runner import BaseLiveRunner
from core.broker.your_broker import YourBroker


class YourLiveRunner(BaseLiveRunner):
    """Live trading runner for Your Broker."""

    BROKER_NAME = "YourBroker"
    LOG_FILE_KEY = "YourBrokerLive"
    TRADE_LOG_FILE = "your_broker_live_trades.csv"

    def _create_broker(self) -> YourBroker:
        """Create broker instance."""
        return YourBroker(
            api_key=os.getenv("YOUR_API_KEY"),
            api_secret=os.getenv("YOUR_API_SECRET"),
        )

    def _canonicalize_bar(self, raw_bar) -> dict:
        """Convert broker-specific bar format to canonical format."""
        return {
            "symbol": raw_bar.symbol,
            "open": float(raw_bar.open),
            "high": float(raw_bar.high),
            "low": float(raw_bar.low),
            "close": float(raw_bar.close),
            "volume": float(raw_bar.volume),
            "timestamp": raw_bar.timestamp,
        }

    async def _connect_broker(self) -> None:
        """Connect to broker."""
        self.broker.connect_sync()

    async def _start_streaming(self) -> asyncio.Task:
        """Start data streaming."""
        return asyncio.create_task(self.broker.start_stream())

    async def _disconnect_broker(self) -> None:
        """Disconnect from broker."""
        self.broker.disconnect()

    def _subscribe_to_data(self) -> None:
        """Subscribe to market data."""
        for symbol in self.symbols:
            self.broker.subscribe_bars(self._on_bar, symbol)
```

### Step 3: Register the Runner

Add to `core/runner_factory.py`:

```python
# In _ensure_loaded():
try:
    from core.your_runner import YourLiveRunner
    cls._registry["yourbroker"] = YourLiveRunner
except ImportError:
    pass
```

Or register dynamically:

```python
from core.runner_factory import RunnerFactory
from core.your_runner import YourLiveRunner

RunnerFactory.register("yourbroker", YourLiveRunner)
```

### Step 4: Add Config Section (Optional)

Add to `core/config_loader.py`:

```python
@dataclass
class YourBrokerConfig:
    enabled: bool = True
    paper: bool = True
    # Add your config fields
```

---

## Adding a New Position Sizer

### Step 1: Create the Sizer Class

```python
from core.base.position_sizer_base import PositionSizerBase


class MyCustomSizer(PositionSizerBase):
    """Custom position sizing strategy."""

    def __init__(self, risk_percentage: float, **kwargs):
        super().__init__()
        self.risk_percentage = risk_percentage
        # Initialize custom parameters

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        account_value: float,
        signal_strength: float = 1.0,
        atr: float | None = None,
        stop_loss_price: float | None = None,
        **kwargs
    ) -> int:
        """Calculate position size."""
        # Implement your sizing logic
        return 0
```

### Step 2: Register the Sizer

```python
from core.position_sizer_factory import PositionSizerFactory
from my_sizers import MyCustomSizer

PositionSizerFactory.register("custom", MyCustomSizer)

# Now use via config:
# config.position_sizer.type = "custom"
```

---

## Adding a New Event Type

### Step 1: Define the Event

Add to `core/contracts/events.py`:

```python
# Add event name constant
EVENT_MY_CUSTOM = "MY_CUSTOM"

# Add TypedDict payload
class MyCustomPayload(TypedDict):
    field1: str
    field2: float
    timestamp: str

# Register in EVENT_SCHEMA_MAP
EVENT_SCHEMA_MAP[EVENT_MY_CUSTOM] = MyCustomPayload
```

### Step 2: Emit the Event

```python
from core.events.eventhandler import get_event_handler
from core.contracts.events import EVENT_MY_CUSTOM, MyCustomPayload

async def emit_custom_event():
    handler = get_event_handler()
    payload: MyCustomPayload = {
        "field1": "value",
        "field2": 123.45,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await handler.emit(EVENT_MY_CUSTOM, payload)
```

### Step 3: Subscribe to the Event

```python
from core.events.eventhandler import get_event_handler
from core.contracts.events import EVENT_MY_CUSTOM

def on_custom_event(event):
    print(f"Received: {event.payload}")

handler = get_event_handler()
handler.subscribe_sync(EVENT_MY_CUSTOM, on_custom_event)
```

---

## Adding a New Strategy

### Step 1: Create the Strategy

```python
from core.base.base_strategy import BaseStrategy


class MyStrategy(BaseStrategy):
    """Custom trading strategy."""

    def __init__(self, **params):
        super().__init__()
        self.params = params

    def generate_signal(self, df, symbol: str) -> int:
        """Generate trading signal: -1 (sell), 0 (hold), 1 (buy)."""
        # Implement your strategy logic
        return 0

    def get_regime(self, df) -> str:
        """Classify market regime."""
        return "normal"
```

### Step 2: Register with Strategy Router

```python
from core.logic.strategy_routing_manager import StrategyRoutingManager

router = StrategyRoutingManager()
router.register_strategy("my_strategy", MyStrategy)
router.assign_symbol("AAPL", "my_strategy")
```

---

## Configuration-Driven Extension

Most extensions can be configured via `config/trading_config.json`:

```json
{
  "position_sizer": {
    "type": "kelly",  // or "simple", "custom"
    "risk_percentage": 0.02
  },
  "autotrader": {
    "default_broker": "alpaca"  // or "schwab", "yourbroker"
  }
}
```

Environment variable overrides:

```bash
TRADING__POSITION_SIZER__TYPE=custom
TRADING__AUTOTRADER__DEFAULT_BROKER=yourbroker
```

---

## Best Practices

1. **Use Type Hints**: All public methods should have type hints
2. **Use TypedDicts for Events**: Define payloads in `core/contracts/events.py`
3. **Use OrderSide Enum**: Import from `core.enums`, not strings
4. **Register in Factories**: Use factory patterns for discoverability
5. **Add Config Sections**: Allow config-driven behavior
6. **Write Tests**: Add tests in `tests/test_*.py`
