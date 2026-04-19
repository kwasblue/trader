# Schwab Trader: End-to-End System Walkthrough

This document explains how the entire system operates, from market data coming in to trades being executed and monitored.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│   MARKET DATA          STRATEGY            EXECUTION           MONITORING       │
│   ───────────          ────────            ─────────           ──────────       │
│                                                                                  │
│   ┌─────────┐       ┌───────────┐       ┌───────────┐       ┌───────────┐      │
│   │ Schwab  │──────▶│ Strategy  │──────▶│ Position  │──────▶│    GUI    │      │
│   │   API   │       │  Engine   │       │  Sizer    │       │ Dashboard │      │
│   └─────────┘       └───────────┘       └───────────┘       └───────────┘      │
│        │                 │                   │                    ▲             │
│        ▼                 ▼                   ▼                    │             │
│   ┌─────────┐       ┌───────────┐       ┌───────────┐            │             │
│   │ Streamer│──────▶│  Signal   │──────▶│ Drawdown  │────────────┤             │
│   │WebSocket│       │ Generator │       │  Monitor  │            │             │
│   └─────────┘       └───────────┘       └───────────┘            │             │
│        │                 │                   │                    │             │
│        ▼                 ▼                   ▼                    │             │
│   ┌─────────┐       ┌───────────┐       ┌───────────┐            │             │
│   │  Data   │       │   Trade   │──────▶│  Broker   │────────────┘             │
│   │  Store  │       │   Logic   │       │    API    │                          │
│   └─────────┘       └───────────┘       └───────────┘                          │
│                                                                                  │
│                          ┌─────────────────────────┐                            │
│                          │       EVENT BUS         │                            │
│                          │  (Async Communication)  │                            │
│                          └─────────────────────────┘                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Ingestion

### Live Trading Mode

```
Schwab/Alpaca API
       │
       ▼
┌─────────────────┐
│  Authenticator  │  ← OAuth 2.0 token management
│  (OAuth Flow)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SchwabClient   │  ← REST API for quotes, orders, account
│  (REST API)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Streamer     │  ← WebSocket for real-time prices
│  (WebSocket)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Aggregator    │  ← Builds OHLCV bars from ticks
│  (Bar Builder)  │
└────────┬────────┘
         │
         ├──────────────────────┐
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│   DataStore     │    │    EventBus     │
│   (SQLite)      │    │  EVENT_NEW_BAR  │
└─────────────────┘    └─────────────────┘
```

**What happens:**

1. **Authentication** (`data/streaming/authenticator.py`)
   - Loads credentials from `.env`
   - Handles OAuth 2.0 flow with Schwab
   - Refreshes tokens automatically before expiry

2. **REST Client** (`data/streaming/schwab_client.py`)
   - Gets account info, positions, order status
   - Submits orders to broker
   - Fetches historical data for warmup

3. **WebSocket Streamer** (`data/streaming/streamer.py`)
   - Connects to real-time price feed
   - Receives tick-by-tick quotes
   - Handles reconnection on disconnect

4. **Aggregator** (`data/aggregate.py`)
   - Collects ticks into time-based bars
   - Builds OHLCV candles (1-min, 5-min, etc.)
   - Emits `EVENT_NEW_BAR` when bar completes

5. **Storage** (`data/datastorage.py`)
   - Persists bars to SQLite database
   - Thread-safe with RLock
   - Supports upsert for updates

---

## Phase 2: Strategy Signal Generation

```
EVENT_NEW_BAR
      │
      ▼
┌──────────────────────────────────────────────────────┐
│                   STRATEGY ENGINE                     │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────┐    ┌─────────────────────────────┐  │
│  │   Symbol    │───▶│    Strategy Router          │  │
│  │   State     │    │  (strategy_routing.json)    │  │
│  └─────────────┘    └──────────────┬──────────────┘  │
│                                    │                  │
│         ┌──────────────────────────┼───────────┐     │
│         ▼                          ▼           ▼     │
│  ┌─────────────┐          ┌─────────────┐ ┌───────┐  │
│  │ SMAStrategy │          │ EMAStrategy │ │ MACD  │  │
│  └──────┬──────┘          └──────┬──────┘ └───┬───┘  │
│         │                        │            │      │
│         └────────────────────────┴────────────┘      │
│                          │                           │
│                          ▼                           │
│                 ┌─────────────────┐                  │
│                 │ generate_signal │                  │
│                 │   Returns:      │                  │
│                 │   1 = BUY       │                  │
│                 │  -1 = SELL      │                  │
│                 │   0 = HOLD      │                  │
│                 └────────┬────────┘                  │
│                          │                           │
└──────────────────────────┼───────────────────────────┘
                           │
                           ▼
                  EVENT_STRATEGY_SIGNAL
```

**What happens:**

1. **Symbol State** (`core/logic/symbol_state.py`)
   - Maintains rolling window of bars per symbol
   - Tracks current position, entry price, P&L
   - Stores ATR, regime, and other indicators

2. **Strategy Routing** (`config/strategy_routing.json`)
   - Maps symbols to specific strategies
   - `AAPL` → `ema`, `TSLA` → `momentum`, etc.
   - Falls back to default strategy if not mapped

3. **Strategy Execution** (`strategies/strategy_registry/`)
   - Loads appropriate strategy class
   - Calculates indicators (SMA, RSI, MACD, etc.)
   - Generates signal: `1` (buy), `-1` (sell), `0` (hold)

4. **Vectorized Path** (for backtesting)
   - Uses `generate_signals_vectorized()` for 100x speedup
   - Processes all bars at once using NumPy
   - Same logic, just vectorized

**Example - SMA Strategy Signal:**
```python
def generate_signal(self, data: pd.DataFrame) -> int:
    close = data["Close"]
    sma_fast = close.rolling(self.fast).mean()
    sma_slow = close.rolling(self.slow).mean()

    if sma_fast.iloc[-1] > sma_slow.iloc[-1]:
        return 1   # Fast above slow = uptrend = BUY
    elif sma_fast.iloc[-1] < sma_slow.iloc[-1]:
        return -1  # Fast below slow = downtrend = SELL
    return 0       # Equal = no signal
```

---

## Phase 3: Trade Logic & Risk Management

```
EVENT_STRATEGY_SIGNAL (signal=1, symbol="AAPL")
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│                       TRADE LOGIC LAYER                        │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  Step 1: VALIDATE SIGNAL                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ • Is trading halted? (cooldown/panic)                   │  │
│  │ • Do we already have max positions?                     │  │
│  │ • Is this symbol already at max position?               │  │
│  │ • Has minimum holding period passed?                    │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼                                     │
│  Step 2: CHECK RISK LIMITS                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ DrawdownMonitor:                                        │  │
│  │ • Global drawdown < 15%? ✓                              │  │
│  │ • Daily drawdown < 5%? ✓                                │  │
│  │ • Per-symbol drawdown < 3%? ✓                           │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼                                     │
│  Step 3: SIZE POSITION                                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ DynamicPositionSizer:                                   │  │
│  │ • Calculate stop loss (2 × ATR below entry)             │  │
│  │ • Risk per share = entry - stop_loss                    │  │
│  │ • Risk amount = capital × 2% × volatility_adj           │  │
│  │ • Shares = risk_amount / risk_per_share                 │  │
│  │ • Apply max position cap (20% of capital)               │  │
│  └─────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          ▼                                     │
│  Step 4: CREATE ORDER                                          │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Order Details:                                          │  │
│  │ • Symbol: AAPL                                          │  │
│  │ • Side: BUY                                             │  │
│  │ • Quantity: 45 shares                                   │  │
│  │ • Type: MARKET                                          │  │
│  │ • Stop Loss: $148.50                                    │  │
│  │ • Take Profit: $162.00 (optional)                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
                    EVENT_ORDER_SUBMITTED
```

**Position Sizing Example:**

```python
# Current state
stock_price = 150.00
atr = 2.50
capital = 10000
risk_per_trade = 0.02  # 2%

# Calculate stop loss
stop_loss = stock_price - (2 * atr)  # $145.00

# Risk per share
risk_per_share = stock_price - stop_loss  # $5.00

# Position size
risk_amount = capital * risk_per_trade  # $200
shares = risk_amount / risk_per_share   # 40 shares

# Apply cap (max 20% of capital)
max_shares = int((capital * 0.20) / stock_price)  # 13 shares
final_shares = min(shares, max_shares)  # 13 shares
```

---

## Phase 4: Order Execution

```
EVENT_ORDER_SUBMITTED
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                      EXECUTOR                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐                                         │
│  │  Apply         │  • Volume-based slippage                │
│  │  Slippage      │  • Volatility adjustment                │
│  │                │  • Execution price = $150.05            │
│  └───────┬────────┘                                         │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────┐                                         │
│  │  Submit to     │  • Schwab/Alpaca/Mock Broker            │
│  │  Broker API    │  • Wait for acknowledgment              │
│  └───────┬────────┘                                         │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────┐                                         │
│  │  Track Order   │  • PENDING → SUBMITTED → FILLED         │
│  │  Status        │  • Handle rejects, partial fills        │
│  └───────┬────────┘                                         │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────┐                                         │
│  │  Update        │  • Add to positions dict                │
│  │  Positions     │  • Set entry price, stop loss           │
│  └───────┬────────┘                                         │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────┐                                         │
│  │  Calculate     │  • Transaction cost: $0.15              │
│  │  Fees          │  • Total cost: $1950.65 + $0.15         │
│  └────────────────┘                                         │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ├─── EVENT_ORDER_FILLED
                           ├─── EVENT_POSITION_UPDATE
                           └─── EVENT_PNL_UPDATE
```

---

## Phase 5: Position Management & P&L Tracking

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PORTFOLIO STATE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Portfolio:                                                          │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │ Cash:           $8,049.35                                  │     │
│  │ Positions:      $1,950.65 (13 AAPL @ $150.05)             │     │
│  │ Total Value:    $10,000.00                                 │     │
│  │ Unrealized P&L: $0.00                                      │     │
│  │ Realized P&L:   $0.00                                      │     │
│  │ Day P&L:        $0.00                                      │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  Positions:                                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Symbol │ Qty │ Entry  │ Current │ Unreal │ Stop   │ Regime  │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │ AAPL   │ 13  │ 150.05 │ 151.20  │ +14.95 │ 145.00 │ Normal  │   │
│  │ GOOGL  │ 5   │ 142.30 │ 141.80  │ -2.50  │ 138.00 │ Low Vol │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  On each price update:                                               │
│  1. Update current price                                             │
│  2. Recalculate unrealized P&L                                       │
│  3. Check stop loss (price <= stop → close position)                 │
│  4. Check take profit (price >= target → close position)             │
│  5. Emit EVENT_PNL_UPDATE                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Continuous Monitoring Loop:**

```python
async def on_new_bar(self, event):
    symbol = event.payload['symbol']
    current_price = event.payload['close']

    # Update position value
    if symbol in self.positions:
        position = self.positions[symbol]
        position.current_price = current_price
        position.unrealized = (current_price - position.entry) * position.qty

        # Check stop loss
        if current_price <= position.stop_loss:
            await self.close_position(symbol, reason="STOP_LOSS")

        # Check take profit
        elif position.take_profit and current_price >= position.take_profit:
            await self.close_position(symbol, reason="TAKE_PROFIT")

    # Update portfolio totals
    self.portfolio.update_values()
    await self.bus.emit(Event(EVENT_PNL_UPDATE, self.portfolio.snapshot()))
```

---

## Phase 6: Monitoring & GUI Updates

```
                    EVENT_PNL_UPDATE
                    EVENT_POSITION_UPDATE
                    EVENT_ORDER_FILLED
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        DATA FEEDER                                │
│              (Async EventBus → Qt Signals Bridge)                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   async def _on_pnl_update(self, event):                         │
│       data = event.payload                                        │
│       # Emit Qt signals (thread-safe)                            │
│       self.s.equity_point.emit(ts, portfolio_value)              │
│       self.s.risk_stats.emit(unrealized, realized, drawdown)     │
│                                                                   │
└──────────────────────────┬────────────────────────────────────────┘
                           │ Qt Signals
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      STATE AGGREGATOR                             │
│                (Compiles unified snapshots)                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Every 1 second, emit snapshot:                                  │
│   {                                                               │
│     'timestamp': '2024-01-15T10:30:00',                          │
│     'portfolio': {                                                │
│       'value': 10150.00,                                          │
│       'unrealized': 150.00,                                       │
│       'realized': 0.00,                                           │
│       'drawdown': 0.0                                             │
│     },                                                            │
│     'positions': [...],                                           │
│     'orders': [...],                                              │
│     'alerts': [...]                                               │
│   }                                                               │
│                                                                   │
└──────────────────────────┬────────────────────────────────────────┘
                           │ snapshot_ready signal
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        MAIN WINDOW                                │
│                    (Qt GUI Updates)                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   def _update_from_snapshot(self, snapshot):                     │
│       # Update KPI labels                                         │
│       self.portfolio_value.setText(...)                          │
│       self.unrealized_pnl.setText(...)                           │
│                                                                   │
│       # Update positions table                                    │
│       self.pos_model.replace_rows(snapshot['positions'])         │
│                                                                   │
│       # Update equity curve                                       │
│       self.equity_curve.append(...)                              │
│                                                                   │
│       # Update drawdown chart                                     │
│       self.drawdown_chart.update(...)                            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**GUI Tab Updates:**

| Tab | Data Source | Update Frequency |
|-----|-------------|------------------|
| Dashboard | Portfolio snapshot | 1 second |
| Market | OHLC bars | On each bar |
| Performance | Equity curve, metrics | 1 second |
| Execution | Orders, trades | On event |
| Alerts | Alert events | On event |
| Strategies | Signal history | On signal |

---

## Phase 7: Risk Events & Circuit Breakers

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DRAWDOWN MONITOR                                 │
│                  (Continuous Risk Checks)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   On every P&L update:                                               │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Check 1: Global Drawdown                                    │   │
│   │                                                             │   │
│   │   peak_value = max(historical_portfolio_values)             │   │
│   │   current_dd = (peak - current) / peak                      │   │
│   │                                                             │   │
│   │   if current_dd > 0.15:  # 15% max drawdown                │   │
│   │       HALT ALL TRADING                                      │   │
│   │       emit EVENT_HALT_STATE                                 │   │
│   │       emit EVENT_DRAWDOWN_ALERT                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Check 2: Daily Drawdown                                     │   │
│   │                                                             │   │
│   │   day_start = portfolio_value_at_open                       │   │
│   │   daily_dd = (day_start - current) / day_start              │   │
│   │                                                             │   │
│   │   if daily_dd > 0.05:  # 5% daily limit                    │   │
│   │       HALT TRADING FOR TODAY                                │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Check 3: Per-Symbol Drawdown                                │   │
│   │                                                             │   │
│   │   for symbol in positions:                                  │   │
│   │       symbol_dd = unrealized_loss / entry_value             │   │
│   │       if symbol_dd > 0.03:  # 3% per-symbol limit          │   │
│   │           CLOSE POSITION                                    │   │
│   │           LOCK SYMBOL (cooldown period)                     │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   Recovery:                                                          │
│   - After drawdown recovers, wait cooldown_period (5 min)           │
│   - Then unlock trading                                              │
│   - Emit EVENT_COOLDOWN_STATE(false)                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Backtesting Mode vs Live Trading

The same components work in both modes with different data sources:

```
                    BACKTESTING                      LIVE TRADING
                    ───────────                      ────────────

Data Source:        CSV/DataFrame                    WebSocket Stream
                         │                                │
                         ▼                                ▼
                    ┌─────────┐                     ┌─────────┐
                    │  Loop   │                     │  Async  │
                    │  over   │                     │  Event  │
                    │  bars   │                     │  Loop   │
                    └────┬────┘                     └────┬────┘
                         │                               │
                         └───────────┬───────────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   SAME COMPONENTS:    │
                         │   • Strategy Engine   │
                         │   • Position Sizer    │
                         │   • Trade Logic       │
                         │   • Risk Monitor      │
                         └───────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
              ┌───────────┐                    ┌───────────┐
              │   Mock    │                    │  Broker   │
              │  Executor │                    │    API    │
              └───────────┘                    └───────────┘
                    │                                │
                    ▼                                ▼
              ┌───────────┐                    ┌───────────┐
              │  Results  │                    │   Real    │
              │ DataFrame │                    │  Trades   │
              └───────────┘                    └───────────┘
```

**Key Differences:**

| Aspect | Backtesting | Live Trading |
|--------|-------------|--------------|
| Data | Historical DataFrame | Real-time stream |
| Execution | Instant, simulated | Async, real broker |
| Slippage | Modeled | Actual market |
| Speed | 10,000 bars/second | Real-time |
| Risk | Paper only | Real money |

---

## Complete Event Flow Example

Let's trace a complete trade from signal to fill:

```
T+0.000s: New bar arrives (AAPL closes at $152.00)
          └── EVENT_NEW_BAR emitted

T+0.001s: Strategy engine receives bar
          └── SMA crossover detected (fast > slow)
          └── generate_signal() returns 1 (BUY)
          └── EVENT_STRATEGY_SIGNAL emitted

T+0.002s: Trade logic validates signal
          └── Check: Not halted ✓
          └── Check: < max positions ✓
          └── Check: Drawdown OK ✓

T+0.003s: Position sizer calculates size
          └── ATR = $2.50
          └── Stop loss = $147.00
          └── Risk per share = $5.00
          └── Risk amount = $200 (2% of $10k)
          └── Shares = 40

T+0.004s: Order created and submitted
          └── BUY 40 AAPL @ MARKET
          └── Stop loss: $147.00
          └── EVENT_ORDER_SUBMITTED emitted

T+0.050s: Broker acknowledges order
          └── Order status: SUBMITTED

T+0.150s: Order filled
          └── Filled: 40 AAPL @ $152.03 (slippage)
          └── Fee: $0.15
          └── EVENT_ORDER_FILLED emitted

T+0.151s: Position updated
          └── positions['AAPL'] = {qty: 40, entry: 152.03, stop: 147.00}
          └── EVENT_POSITION_UPDATE emitted

T+0.152s: Portfolio recalculated
          └── Cash: $3,918.65
          └── Positions: $6,081.20
          └── Total: $9,999.85
          └── EVENT_PNL_UPDATE emitted

T+0.153s: GUI updated
          └── Positions table shows AAPL
          └── Portfolio value updated
          └── Trade logged to Execution tab

T+1.000s: StateAggregator emits snapshot
          └── All GUI components refresh
```

---

## Event Types Reference

| Event | Trigger | Payload |
|-------|---------|---------|
| `EVENT_NEW_BAR` | Bar completes | symbol, OHLCV data |
| `EVENT_STRATEGY_SIGNAL` | Strategy generates signal | symbol, signal, timestamp |
| `EVENT_ORDER_SUBMITTED` | Order sent to broker | order details |
| `EVENT_ORDER_FILLED` | Order executed | fill details |
| `EVENT_ORDER_CANCELED` | Order canceled | order id, reason |
| `EVENT_ORDER_REJECTED` | Broker rejects order | order id, reason |
| `EVENT_POSITION_UPDATE` | Position changes | position data |
| `EVENT_PNL_UPDATE` | P&L recalculated | portfolio snapshot |
| `EVENT_DRAWDOWN_ALERT` | Drawdown threshold hit | symbol, drawdown % |
| `EVENT_HALT_STATE` | Trading halted/resumed | halted boolean |
| `EVENT_COOLDOWN_STATE` | Cooldown started/ended | cooldown boolean |
| `EVENT_REGIME_UPDATE` | Market regime changes | volatility, trend |
| `EVENT_LOG` | Log message | message string |
| `EVENT_HEALTH_UPDATE` | System health check | component statuses |

---

## Summary

The system operates as a **pipeline**:

1. **Data In** → Streaming prices aggregated into bars
2. **Signal Generation** → Strategy analyzes bars, emits signals
3. **Risk Check** → Drawdown limits, position limits validated
4. **Position Sizing** → Risk-based sizing with volatility adjustment
5. **Execution** → Order submitted to broker with slippage
6. **Tracking** → Positions and P&L continuously updated
7. **Monitoring** → GUI displays real-time state
8. **Protection** → Circuit breakers halt trading if limits hit

All components communicate via the **EventBus**, making the system:
- **Decoupled** - Components don't know about each other
- **Testable** - Each component can be tested independently
- **Extensible** - Add new strategies/brokers without changing core

---

## File Reference

| Component | Primary Files |
|-----------|---------------|
| Data Ingestion | `data/streaming/schwab_client.py`, `data/streaming/streamer.py`, `data/aggregate.py` |
| Storage | `data/datastorage.py` |
| Strategies | `strategies/strategy_registry/*.py` |
| Trade Logic | `core/logic/trade_logic_manager.py`, `core/logic/default_trade_logic.py` |
| Position Sizing | `core/position_sizer.py` |
| Risk Management | `core/drawdown_monitor.py` |
| Execution | `core/executor.py`, `core/broker/*.py` |
| Portfolio State | `core/logic/portfolio_state.py`, `core/logic/symbol_state.py` |
| Backtesting | `core/backtester.py`, `core/backtest_suite.py` |
| Monitoring | `monitoring/views/main_window.py`, `monitoring/feeds/feeder.py`, `monitoring/feeds/state_aggregator.py` |
| Events | `core/events.py` |
