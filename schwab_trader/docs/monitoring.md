# Monitoring Dashboard Guide

The real-time monitoring dashboard provides live visibility into trading activity, positions, and performance.

## Overview

The monitoring GUI is built with PySide6 (Qt) and provides:

- Real-time price charts
- Position tracking
- P&L monitoring
- Order management
- Alert notifications
- Strategy performance

---

## Launching the Dashboard

```bash
# From project root
python run_live.py
```

Or programmatically:

```python
from monitoring.app import run_app

if __name__ == '__main__':
    run_app()
```

---

## Dashboard Tabs

### 1. Dashboard Tab

Main overview with key metrics:

| Widget | Description |
|--------|-------------|
| Portfolio Value | Current total portfolio value |
| Day P&L | Today's profit/loss |
| Unrealized P&L | Open position P&L |
| Realized P&L | Closed trade P&L |
| Position Count | Number of open positions |
| Pending Orders | Orders awaiting fill |

### 2. Market Tab

Live price data and charts:

- Real-time price chart (last 500 points)
- Current price display
- Volume indicators
- Price change percentage

### 3. Performance Tab

Portfolio performance metrics:

- Equity curve chart
- Drawdown visualization
- Sharpe ratio
- Win rate
- Profit factor

### 4. Execution Tab

Order and trade management:

- Open orders table
- Recent trades table
- Order status tracking
- Fill notifications

### 5. Alerts Tab

System alerts and notifications:

- Drawdown alerts
- Price alerts
- Risk limit warnings
- System health messages

### 6. Strategies Tab

Strategy performance breakdown:

- Per-strategy P&L
- Signal history
- Win rate by strategy
- Regime performance

### 7. Ops Tab

Operational controls:

- Mode selection (Simulation, Live)
- Symbol configuration
- Start/Stop controls
- Log output

### 8. Replay Tab

Historical replay functionality:

- Load historical data
- Replay at variable speed
- Analyze past trades

---

## Control Buttons

| Button | Function |
|--------|----------|
| Start | Begin trading/simulation |
| Stop | Stop trading |
| HALT | Emergency stop all trading |
| Flatten | Close all positions |
| Cancel All | Cancel all pending orders |
| New Order | Open manual order dialog |

---

## Manual Order Entry

Click "New Order" to open the order dialog:

```
┌─────────────────────────────────┐
│         Manual Order            │
├─────────────────────────────────┤
│ Symbol:     [AAPL          ]   │
│ Side:       [BUY ▼]            │
│ Quantity:   [100           ]   │
│ Order Type: [MARKET ▼]         │
│ Price:      [___.___ ] (limit) │
│ Stop Loss:  [___.___ ]         │
│ Take Profit:[___.___ ]         │
├─────────────────────────────────┤
│      [Cancel]    [Submit]       │
└─────────────────────────────────┘
```

---

## Table Models

### Positions Table (SymbolsTableModel)

| Column | Description |
|--------|-------------|
| Symbol | Stock symbol |
| Side | Long/Short |
| Qty | Position quantity |
| Avg | Average entry price |
| Last | Current price |
| Unreal | Unrealized P&L |
| Realized | Realized P&L |
| Total | Total P&L |
| PnL % | Percentage gain/loss |
| ATR | Current ATR |
| Regime | Market regime |
| Risk | Risk score (0-100) |

**Color coding:**
- Green: Positive P&L
- Red: Negative P&L
- Background: Risk level indication

### Orders Table (OrdersTableModel)

| Column | Description |
|--------|-------------|
| Time | Order timestamp |
| Symbol | Stock symbol |
| Side | Buy/Sell |
| Qty | Order quantity |
| Price | Limit price |
| Status | Order status |

**Status values:** PENDING, SUBMITTED, FILLED, CANCELED, REJECTED

### Trades Table (TradesTableModel)

| Column | Description |
|--------|-------------|
| Time | Execution timestamp |
| Symbol | Stock symbol |
| Side | Buy/Sell |
| Qty | Filled quantity |
| Price | Execution price |
| Fee | Transaction fee |
| SL | Stop loss price |
| TP | Take profit price |

---

## Event Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  EventBus    │────▶│  DataFeeder  │────▶│ MainWindow   │
│  (Async)     │     │ (Qt Signals) │     │ (GUI Update) │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   State      │
                     │  Aggregator  │
                     └──────────────┘
```

### Event Types Handled

| Event | GUI Update |
|-------|------------|
| EVENT_NEW_BAR | Price chart |
| EVENT_POSITION_UPDATE | Positions table |
| EVENT_ORDER_UPDATE | Orders table |
| EVENT_TRADE | Trades table |
| EVENT_PNL_UPDATE | P&L displays, equity curve |
| EVENT_DRAWDOWN_ALERT | Alerts list |
| EVENT_HALT_STATE | Panic button state |
| EVENT_REGIME_UPDATE | Strategy tab |
| EVENT_LOG | Ops tab log |

---

## Configuration

### Mode Selection

```python
# Available modes
MODES = ['Simulation', 'Paper', 'Live']
```

- **Simulation**: Uses mock executor, no real orders
- **Paper**: Connects to paper trading API
- **Live**: Real trading (requires broker credentials)

### Symbol Configuration

Enter symbols comma-separated:
```
AAPL, GOOGL, MSFT, TSLA
```

---

## State Aggregator

The `StateAggregator` compiles data from multiple sources into unified snapshots:

```python
class StateAggregator(QObject):
    """Aggregates state from DataFeeder into unified snapshots."""

    snapshot_ready = Signal(dict)  # Emitted every 1 second

    def __init__(self, feeder: DataFeeder):
        self.feeder = feeder
        # Subscribe to feeder signals
        feeder.s.symbols.connect(self._on_positions)
        feeder.s.orders.connect(self._on_orders)
        feeder.s.pnl_update.connect(self._on_pnl)
        # ...
```

### Snapshot Structure

```python
snapshot = {
    'timestamp': datetime,
    'portfolio': {
        'value': float,
        'cash': float,
        'unrealized': float,
        'realized': float,
        'drawdown': float
    },
    'positions': [
        {'symbol': str, 'qty': int, 'avg': float, ...},
        ...
    ],
    'orders': [
        {'id': str, 'symbol': str, 'status': str, ...},
        ...
    ],
    'alerts': [
        {'type': str, 'message': str, 'timestamp': str},
        ...
    ]
}
```

---

## Customization

### Theme

The dashboard uses a dark theme defined in `monitoring/theme.py`:

```python
def get_dark_palette():
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    # ...
    return palette
```

### Adding Custom Widgets

1. Create widget in `monitoring/widgets/`
2. Add to `MainWindow._build_*_tab()` method
3. Connect to appropriate signals

Example:

```python
# monitoring/widgets/custom_chart.py
from PySide6 import QtWidgets
import pyqtgraph as pg


class CustomChart(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.plot)

    def update_data(self, x, y):
        self.plot.clear()
        self.plot.plot(x, y, pen='g')
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+S | Start trading |
| Ctrl+Q | Stop trading |
| Ctrl+H | Toggle halt |
| Ctrl+F | Flatten all |
| Ctrl+N | New order |
| F5 | Refresh data |

---

## Troubleshooting

### GUI Not Updating

1. Check EventBus is running
2. Verify DataFeeder started successfully
3. Check for errors in Ops tab log

### Performance Issues

1. Reduce chart history (default 500 points)
2. Lower update frequency
3. Disable unused tabs

### Connection Errors

1. Verify API credentials in `.env`
2. Check network connectivity
3. Confirm broker API status

---

## Architecture Details

### Thread Safety

- EventBus runs in async context
- DataFeeder bridges to Qt signals
- GUI updates only on main thread
- State snapshots are immutable

### Memory Management

- Charts keep last 500 points
- Old alerts pruned after 1000 entries
- Closed positions removed after session

### Error Handling

```python
def _emit_safe(self, signal, *args):
    """Safely emit Qt signal with error handling."""
    try:
        signal.emit(*args)
    except Exception as e:
        self._logger.error(f"Signal emit failed: {e}")
```
