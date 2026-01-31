# Schwab Trader Documentation

A comprehensive algorithmic trading platform with strategy backtesting, live trading, and real-time monitoring capabilities.

## Table of Contents

1. [Getting Started](getting-started.md) - Installation and quick start guide
2. [System Overview](system-overview.md) - End-to-end system walkthrough
3. [Architecture](architecture.md) - System design and component overview
4. [Configuration](configuration.md) - Configuration files and settings
5. [Strategies](strategies.md) - Available strategies and creating custom ones
6. [Backtesting](backtesting.md) - Backtesting framework and analysis tools
7. [Monitoring](monitoring.md) - Real-time GUI monitoring dashboard
8. [API Reference](api-reference.md) - Detailed API documentation
9. [Examples](examples.md) - Code examples and tutorials

---

## Overview

Schwab Trader is a modular algorithmic trading system that supports:

- **18 Built-in Strategies** - From simple moving averages to machine learning
- **Multiple Brokers** - Schwab, Alpaca, Coinbase integrations
- **Advanced Backtesting** - Walk-forward analysis, Monte Carlo simulation, parameter optimization
- **Real-time Monitoring** - Qt-based GUI with live charts, P&L tracking, and alerts
- **Risk Management** - Drawdown monitoring, position sizing, stop-loss enforcement

## Quick Start

```bash
# Clone and setup
cd schwab_trader
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run backtest
python -c "
from core.backtester import Backtester
from core.position_sizer import DynamicPositionSizer
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')
bt = Backtester(data, initial_capital=10000)
sizer = DynamicPositionSizer(risk_per_trade=0.02, capital=10000)
results = bt.run_backtest('sma', {'fast': 10, 'slow': 30}, sizer)
print(bt.evaluate_performance(results))
"

# Run monitoring GUI
python run_live.py
```

## Project Structure

```
schwab_trader/
├── core/                    # Core trading engine
│   ├── backtester.py       # Main backtesting engine
│   ├── backtest_suite.py   # Advanced backtesting tools
│   ├── executor.py         # Trade execution
│   ├── position_sizer.py   # Position sizing algorithms
│   └── broker/             # Broker integrations
├── strategies/
│   └── strategy_registry/  # 18 trading strategies
├── indicators/             # Technical indicators
├── monitoring/             # Real-time GUI
│   ├── views/             # Qt widgets
│   └── feeds/             # Data feeds
├── data/                   # Data management
│   ├── streaming/         # Live data streams
│   └── datastorage.py     # SQLite storage
├── tests/                  # Test suite (212 tests)
├── config/                 # Configuration files
└── docs/                   # Documentation
```

## Key Features

### Strategies
- SMA, EMA, MACD, RSI, Bollinger Bands
- Momentum, Mean Reversion, Breakout
- ADX, Stochastic, Ichimoku, PSAR
- VWAP, Donchian Channels
- Combined strategies with voting
- ML-based (Logistic Regression)

### Backtesting
- Vectorized signal generation (100x+ speedup)
- Walk-forward analysis
- Monte Carlo simulation
- Grid search optimization
- Multiple slippage models
- Benchmark comparison

### Risk Management
- Dynamic position sizing
- Per-symbol drawdown limits
- Daily loss limits
- Stop-loss enforcement
- Cooldown periods

### Monitoring
- Real-time equity curves
- Position tracking
- Order management
- Alert system
- Strategy performance

## Requirements

- Python 3.10+
- pandas, numpy, scipy
- PySide6 (for GUI)
- ta (technical analysis)
- Additional: requests, aiohttp, websockets

## License

MIT License - See LICENSE file for details.
