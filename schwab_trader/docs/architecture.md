# Architecture Overview

This document describes the system architecture of Schwab Trader.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Schwab Trader                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Data      │    │  Strategy   │    │  Execution  │    │  Monitoring │  │
│  │   Layer     │───▶│   Layer     │───▶│   Layer     │───▶│   Layer     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │           │
│        ▼                  ▼                  ▼                  ▼           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Event Bus (Async)                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Layers

### 1. Data Layer

Handles all data acquisition, storage, and preprocessing.

```
data/
├── streaming/
│   ├── schwab_client.py      # Schwab API client
│   ├── authenticator.py      # OAuth authentication
│   └── streamer.py           # WebSocket streaming
├── datastorage.py            # SQLite database
├── aggregate.py              # Data aggregation
└── datautils.py              # Utility functions
```

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `SchwabClient` | REST API for quotes, orders, account info |
| `Authenticator` | OAuth 2.0 token management |
| `DataStore` | Thread-safe SQLite operations |
| `Aggregator` | Real-time bar aggregation |

**Data Flow:**
```
External API → Streamer → Aggregator → DataStore
                  │
                  └─────→ Event Bus (EVENT_NEW_BAR)
```

---

### 2. Strategy Layer

Contains all trading strategies and signal generation logic.

```
strategies/
└── strategy_registry/
    ├── __init__.py           # Strategy loader
    ├── strategy_registry.py  # Auto-discovery
    ├── sma_strategy.py       # SMA crossover
    ├── ema_strategy.py       # EMA crossover
    ├── macd_strategy.py      # MACD
    ├── rsi_strategy.py       # RSI oscillator
    ├── bollinger_strategy.py # Bollinger Bands
    ├── momentum_strategy.py  # Price momentum
    ├── mean_reversion_strategy.py
    ├── breakout_strategy.py
    ├── adx_strategy.py
    ├── stochastic_strategy.py
    ├── ichimoku_strategy.py
    ├── psar_strategy.py
    ├── vwap_strategy.py
    ├── donchian_strategy.py
    ├── combined_strategy.py
    └── logistic_regression_strategy.py
```

**Base Strategy Interface:**

```python
class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> int:
        """Generate trading signal.

        Returns:
            1: Buy signal
           -1: Sell signal
            0: Hold/No signal
        """
        pass

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        """Vectorized signal generation for backtesting (optional)."""
        return None
```

**Strategy Loading:**
```python
from strategies.strategy_registry import load_strategy, list_strategies

# List available strategies
strategies = list_strategies()  # ['sma', 'ema', 'macd', ...]

# Load with parameters
strategy = load_strategy('rsi', params={'window': 14, 'oversold': 30})
signal = strategy.generate_signal(data)
```

---

### 3. Execution Layer

Handles order execution, position management, and risk control.

```
core/
├── executor.py              # Main trade executor
├── position_sizer.py        # Position sizing algorithms
├── drawdown_monitor.py      # Drawdown tracking
├── logic/
│   ├── trade_logic_manager.py
│   ├── default_trade_logic.py
│   ├── portfolio_state.py   # Portfolio tracking
│   ├── symbol_state.py      # Per-symbol state
│   └── mock_execution_engine.py
└── broker/
    ├── base_broker.py       # Broker interface
    ├── schwab_broker.py     # Schwab implementation
    ├── alpaca_broker.py     # Alpaca implementation
    └── mock_broker.py       # Paper trading
```

**Execution Flow:**
```
Strategy Signal
      │
      ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Trade      │────▶│  Position   │────▶│  Drawdown   │
│  Logic      │     │  Sizer      │     │  Monitor    │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      │                   │                   │
      ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────┐
│                     Executor                         │
├─────────────────────────────────────────────────────┤
│  • Validate signal                                   │
│  • Calculate position size                           │
│  • Check risk limits                                 │
│  • Submit order to broker                            │
│  • Track fills and P&L                               │
└─────────────────────────────────────────────────────┘
      │
      ▼
   Broker API
```

**Position Sizing:**

```python
class DynamicPositionSizer:
    """Risk-based position sizing with volatility adjustment."""

    def calculate_position_size(
        self,
        stock_price: float,
        stop_loss_price: float,
        current_cash: float,
        market_conditions: str,  # 'low_volatility', 'normal', 'high_volatility'
        signal: int
    ) -> int:
        # Risk per share
        risk_per_share = abs(stock_price - stop_loss_price)

        # Adjust risk based on market conditions
        risk_multiplier = {
            'low_volatility': 1.2,
            'normal': 1.0,
            'high_volatility': 0.7
        }.get(market_conditions, 1.0)

        # Calculate position size
        risk_amount = current_cash * self.risk_per_trade * risk_multiplier
        position_size = int(risk_amount / risk_per_share)

        # Apply position limits
        max_shares = int(current_cash * self.max_position_pct / stock_price)
        return min(position_size, max_shares)
```

---

### 4. Monitoring Layer

Real-time GUI for monitoring trades, positions, and performance.

```
monitoring/
├── app.py                   # Application entry
├── theme.py                 # Dark theme styling
├── bus.py                   # Control bridge
├── models.py                # Table models
├── views/
│   └── main_window.py       # Main GUI window
├── feeds/
│   ├── feeder.py            # Event → Qt bridge
│   └── state_aggregator.py  # State compilation
├── dialogs/
│   └── manual_order.py      # Order dialog
└── widgets/
    └── candles.py           # Candlestick chart
```

**GUI Architecture:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                          MainWindow                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Tabs:                                                               │
│  ┌─────────┬─────────┬───────────┬───────────┬────────┬──────────┐ │
│  │Dashboard│ Market  │Performance│ Execution │ Alerts │Strategies│ │
│  └─────────┴─────────┴───────────┴───────────┴────────┴──────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    DataFeeder (Async)                         │  │
│  │  Subscribes to EventBus events, emits Qt signals              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   StateAggregator                             │  │
│  │  Compiles state from multiple sources, emits unified snapshot │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    GUI Update Slots                           │  │
│  │  _update_from_snapshot(), _update_price_chart(), etc.         │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Event System

The system uses an async event bus for inter-component communication.

**Event Types:**

| Event | Description | Payload |
|-------|-------------|---------|
| `EVENT_NEW_BAR` | New OHLCV bar | symbol, ohlcv data |
| `EVENT_STRATEGY_SIGNAL` | Strategy generated signal | symbol, signal, timestamp |
| `EVENT_ORDER_SUBMITTED` | Order sent to broker | order details |
| `EVENT_ORDER_FILLED` | Order executed | fill details |
| `EVENT_POSITION_UPDATE` | Position changed | position data |
| `EVENT_PNL_UPDATE` | P&L changed | portfolio value, unrealized, realized |
| `EVENT_DRAWDOWN_ALERT` | Drawdown threshold hit | symbol, drawdown % |
| `EVENT_REGIME_UPDATE` | Market regime changed | volatility, trend |
| `EVENT_HALT_STATE` | Trading halted/resumed | halted boolean |

**Event Flow Example:**
```
1. Streamer receives new price data
2. Aggregator builds new bar → EVENT_NEW_BAR
3. Strategy processes bar → EVENT_STRATEGY_SIGNAL (if signal)
4. Executor validates and sizes → EVENT_ORDER_SUBMITTED
5. Broker fills order → EVENT_ORDER_FILLED
6. Portfolio updates → EVENT_POSITION_UPDATE, EVENT_PNL_UPDATE
7. Drawdown monitor checks → EVENT_DRAWDOWN_ALERT (if threshold)
8. GUI updates all displays
```

---

## Backtesting Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Backtester                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐   │
│  │   Data      │────▶│   Strategy  │────▶│   Position Sizer    │   │
│  │ Validation  │     │   Signals   │     │   (Risk-based)      │   │
│  └─────────────┘     └─────────────┘     └─────────────────────┘   │
│                                                 │                    │
│                                                 ▼                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Trade Simulation                          │   │
│  │  • Slippage modeling (Fixed, Volume, Volatility)             │   │
│  │  • Transaction costs                                          │   │
│  │  • Stop-loss enforcement                                      │   │
│  │  • Position tracking                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Performance Metrics                         │   │
│  │  Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

                    Advanced Analysis Tools
                    ─────────────────────────
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Grid Search │  │Walk-Forward │  │Monte Carlo  │  │ Benchmark   │
│Optimization │  │  Analysis   │  │ Simulation  │  │ Comparison  │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

---

## Database Schema

SQLite database for persistent storage:

```sql
-- Price data
CREATE TABLE ohlcv (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    UNIQUE(symbol, timestamp)
);

-- Trade log
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    timestamp INTEGER,
    symbol TEXT,
    side TEXT,
    quantity INTEGER,
    price REAL,
    fees REAL,
    pnl REAL
);

-- Positions
CREATE TABLE positions (
    symbol TEXT PRIMARY KEY,
    quantity INTEGER,
    avg_price REAL,
    unrealized REAL,
    realized REAL
);
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `.env` | API credentials, secrets |
| `config/strategy_routing.json` | Symbol → Strategy mapping |
| `config/trade_logic_routing.json` | Trade logic rules |
| `config/ml_config.json` | ML model settings |

---

## Thread Safety

- **Database**: Uses `RLock` for thread-safe writes
- **Event Bus**: Async with proper awaiting
- **GUI**: Qt signals bridge async to main thread
- **State**: Immutable snapshots for thread-safe reads

---

## Extension Points

1. **New Strategies**: Inherit from `BaseStrategy`, place in `strategy_registry/`
2. **New Brokers**: Inherit from `BaseBroker`, implement required methods
3. **New Indicators**: Add to `indicators/` directory
4. **Custom Slippage**: Inherit from `SlippageModel`
5. **Custom Position Sizing**: Inherit from `DynamicPositionSizer`
