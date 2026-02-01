# Schwab Trader Documentation

A comprehensive algorithmic trading platform with strategy backtesting, live trading, and real-time monitoring capabilities.

## Table of Contents

1. [Quick Reference](quick-reference.md) - Command cheatsheet
2. [Getting Started](getting-started.md) - Installation and quick start guide
3. [System Overview](system-overview.md) - End-to-end system walkthrough
4. [Architecture](architecture.md) - System design and component overview
5. [Configuration](configuration.md) - Configuration files and settings
6. [Strategies](strategies.md) - Available strategies and creating custom ones
7. [Backtesting](backtesting.md) - Backtesting framework and analysis tools
8. [Monitoring](monitoring.md) - Real-time GUI monitoring dashboard
9. [AutoTrader](autotrader.md) - Autonomous trading daemon
10. [Pre-Flight Checks](preflight.md) - System validation before trading
11. [Data Pipeline](data-pipeline.md) - Historical data management
12. [API Reference](api-reference.md) - Detailed API documentation
13. [Examples](examples.md) - Code examples and tutorials

---

## Overview

Schwab Trader is a modular algorithmic trading system that supports:

- **Autonomous Trading** - AutoTrader daemon manages complete daily trading cycle
- **18 Built-in Strategies** - From simple moving averages to machine learning
- **Multiple Brokers** - Schwab, Alpaca integrations with paper/live modes
- **Advanced Backtesting** - Walk-forward analysis, Monte Carlo simulation, parameter optimization
- **Real-time Monitoring** - Qt-based GUI with live charts, P&L tracking, and alerts
- **Risk Management** - Drawdown monitoring, position sizing, stop-loss enforcement
- **Pre-Flight Validation** - Automated checks before each trading session

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
├── autotrader.py           # Autonomous trading daemon
├── autotrader_ctl.py       # Daemon control script
├── preflight.py            # Pre-flight validation
├── run_trading.py          # Manual trading entry point
├── run_tests.py            # Test runner
├── core/                   # Core trading engine
│   ├── alpaca_runner.py    # Alpaca live trading
│   ├── schwab_runner.py    # Schwab live trading
│   ├── backtest_suite.py   # Backtesting framework
│   ├── position_sizer.py   # Position sizing
│   ├── drawdown_monitor.py # Risk monitoring
│   ├── credential_validator.py # Credential validation
│   ├── unified_data_pipeline.py # Data management
│   ├── broker/             # Broker adapters
│   ├── logic/              # Trade logic and state
│   └── events/             # Event system
├── strategies/
│   └── strategy_registry/  # Trading strategies
├── indicators/             # Technical indicators
├── monitoring/             # Real-time GUI
│   ├── views/              # Qt widgets
│   └── feeds/              # Data feeds
├── data/                   # Data management
│   ├── streaming/          # Live data streams
│   └── datastorage.py      # SQLite storage
├── tests/                  # Test suite (500+ tests)
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

### AutoTrader
- Fully autonomous daily trading cycle
- Market hours awareness (9:30 AM - 4:00 PM ET)
- US holiday calendar integration
- Automatic pre-flight validation
- Post-market data updates
- Daemon mode with launchd support

## Requirements

- Python 3.10+
- pandas, numpy, scipy
- PySide6 (for GUI)
- ta (technical analysis)
- Additional: requests, aiohttp, websockets

## License

MIT License - See LICENSE file for details.
