# Getting Started

This guide will help you set up Schwab Trader and run your first backtest.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd schwab_trader
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
ta>=0.10.0
matplotlib>=3.7.0
PySide6>=6.5.0
pyqtgraph>=0.13.0
aiohttp>=3.8.0
websockets>=11.0
python-dotenv>=1.0.0
requests>=2.28.0
fpdf>=1.7.2
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
# Schwab API credentials
SCHWAB_CLIENT_ID=your_client_id
SCHWAB_CLIENT_SECRET=your_client_secret
SCHWAB_REDIRECT_URI=https://localhost:8080/callback

# Alpaca API credentials (optional)
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Trading settings
DEFAULT_RISK_PER_TRADE=0.02
MAX_POSITION_PCT=0.20
```

---

## Quick Start: Running a Backtest

### Basic Backtest

```python
import pandas as pd
from core.backtester import Backtester
from core.position_sizer import DynamicPositionSizer

# Load historical data
data = pd.read_csv('data/AAPL_daily.csv')

# Initialize backtester
backtester = Backtester(
    data=data,
    initial_capital=10000,
    transaction_cost=0.001,  # 0.1% per trade
    risk_free_rate=0.02      # 2% annual
)

# Create position sizer
sizer = DynamicPositionSizer(
    risk_per_trade=0.02,     # Risk 2% per trade
    max_position_pct=0.20,   # Max 20% in one position
    capital=10000
)

# Run backtest with SMA crossover strategy
results = backtester.run_backtest(
    strategy_name='sma',
    strategy_params={'fast': 10, 'slow': 30},
    sizer=sizer
)

# Evaluate performance
performance = backtester.evaluate_performance(results)
print(performance)
```

### Using the Vectorized Backtester

For larger datasets, use the vectorized backtester for 10-100x speedup:

```python
from core.backtest_suite import VectorizedBacktester

# Initialize with data
bt = VectorizedBacktester(
    data=data,
    initial_capital=10000,
    transaction_cost=0.001
)

# Run backtest
results = bt.run(
    strategy_name='ema',
    strategy_params={'short_window': 12, 'long_window': 26},
    position_sizing='volatility_scaled',
    stop_loss_atr=2.0
)

# Get metrics
metrics = bt.get_metrics(results)
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

---

## Quick Start: Running the Monitoring GUI

```bash
# Using CLI (recommended)
trader gui

# Or directly
python run_trading.py
```

This launches the real-time monitoring dashboard with:
- Live price charts
- Position tracking
- P&L monitoring
- Order management
- Alert notifications

---

## Data Format

Historical data should be a pandas DataFrame with these columns:

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Bar timestamp |
| Open | float | Opening price |
| High | float | High price |
| Low | float | Low price |
| Close | float | Closing price |
| Volume | int | Trading volume (optional) |

**Example:**
```python
import pandas as pd

data = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
    'Open': [100.0] * 100,
    'High': [102.0] * 100,
    'Low': [99.0] * 100,
    'Close': [101.0] * 100,
    'Volume': [1000000] * 100
})
```

---

## Available Strategies

List all available strategies:

```python
from strategies.strategy_registry import list_strategies

print(list_strategies())
# Output: ['sma', 'ema', 'macd', 'rsi', 'bollinger', 'momentum', ...]
```

Load and use a strategy:

```python
from strategies.strategy_registry import load_strategy

# Load with parameters
strategy = load_strategy('macd', params={
    'fast_window': 12,
    'slow_window': 26,
    'signal_window': 9
})

# Generate signal for latest bar
signal = strategy.generate_signal(data)
# Returns: 1 (buy), -1 (sell), or 0 (hold)
```

---

## Project Configuration

Configuration files are in the `config/` directory:

| File | Purpose |
|------|---------|
| `strategy_routing.json` | Map symbols to strategies |
| `trade_logic_routing.json` | Trade logic configuration |
| `ml_config.json` | Machine learning settings |

See [Configuration Guide](configuration.md) for details.

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_strategy.py -v

# Run with coverage
python -m pytest tests/ --cov=core --cov=strategies
```

---

## Next Steps

- [Architecture Overview](architecture.md) - Understand system design
- [Strategies Guide](strategies.md) - Learn about all 18 strategies
- [Backtesting Guide](backtesting.md) - Advanced backtesting features
- [Examples](examples.md) - More code examples
