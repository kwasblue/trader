# Schwab Trader

A comprehensive algorithmic trading platform with support for Schwab and Alpaca brokers, featuring real-time streaming, backtesting, and a professional monitoring GUI.

## Features

- **Multi-Broker Support**: Trade with Schwab or Alpaca (paper/live)
- **Autonomous Trading (AutoTrader)**: Fully automated trading daemon with market hours awareness
- **Real-Time Streaming**: WebSocket-based price feeds with automatic reconnection
- **Strategy Framework**: 18+ pluggable strategies with regime-based routing
- **Risk Management**: Drawdown monitoring, position sizing, trade gates
- **Professional GUI**: PySide6-based monitoring dashboard with real-time charts
- **Comprehensive Backtesting**: Vectorized backtester with walk-forward analysis and Monte Carlo simulation
- **Event-Driven Architecture**: Async event bus for decoupled components
- **Pre-Flight Checks**: Automated validation before trading sessions
- **Historical Data Management**: Unified data pipeline with automatic source selection

## Quick Start

```bash
# Setup
cd schwab_trader
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure credentials in .env
cp .env.example .env
# Edit .env with your API keys

# Run pre-flight checks
trader preflight -v

# Start simulation mode (no real trades)
trader gui

# Start autonomous trading (waits for market open)
trader start --symbols AAPL,MSFT
```

## Common Commands

| Task | Command |
|------|---------|
| Start trading | `trader start` |
| Stop trading | `trader stop` |
| Check status | `trader status` |
| Launch GUI | `trader gui` |
| Pre-flight check | `trader preflight -v` |
| Check tokens | `trader token status` |
| Refresh tokens | `trader token refresh` |
| Select strategies | `trader strategy select AAPL --save` |
| Show routing | `trader strategy show` |
| Run tests | `trader test` |
| View logs | `trader logs -f` |

## Configuration

All trading settings are centralized in `config/trading_config.json`:

```python
from core.config_loader import get_config, create_position_sizer

cfg = get_config()
sizer = create_position_sizer(cfg)  # Pre-configured from config
```

Key configurable parameters:
- **Position Sizing**: risk_percentage, max_trade_pct, max_holding_pct
- **Trade Logic**: cooldown_bars, swing_mode, min_bars_to_hold, SL/TP multipliers
- **Drawdown Monitor**: enabled, max_portfolio_drawdown, cooldown_seconds
- **Indicators**: ATR period, SMA periods, RSI settings

See [Configuration Guide](docs/configuration.md) for full details.

## Documentation

Full documentation is available in the [docs/](docs/) folder:

- [Quick Reference](docs/quick-reference.md) - Command cheatsheet
- [Commands Reference](docs/commands.md) - Complete command reference
- [Configuration](docs/configuration.md) - Configuration guide
- [Getting Started](docs/getting-started.md) - Installation guide
- [Architecture](docs/architecture.md) - System design
- [Event System](docs/event-system.md) - Event bus architecture
- [Data Flow](docs/data-flow.md) - Data flow diagram
- [Strategies](docs/strategies.md) - Available strategies
- [Backtesting](docs/backtesting.md) - Backtesting guide
- [Data Pipeline](docs/data-pipeline.md) - Historical data management
- [ML Training Workflow](docs/ml-training-workflow.md) - Training meta-models from trade data

See [docs/index.md](docs/index.md) for the complete documentation index.

## Project Structure

```
schwab_trader/
├── autotrader.py           # Autonomous trading daemon
├── autotrader_ctl.py       # Daemon control script
├── preflight.py            # Pre-flight validation
├── run_trading.py          # GUI trading entry point
├── core/                   # Core trading engine
│   ├── alpaca_runner.py    # Alpaca live trading
│   ├── schwab_runner.py    # Schwab live trading
│   ├── events/             # Event system
│   ├── logic/              # Trade logic
│   └── broker/             # Broker adapters
├── strategies/             # Trading strategies
├── monitoring/             # Real-time GUI
├── data/                   # Data management
├── config/                 # Configuration files
├── docs/                   # Documentation
└── tests/                  # Test suite
```

## Requirements

- Python 3.10+
- PySide6 (GUI)
- pandas, numpy, scipy
- alpaca-py (Alpaca trading)
- See requirements.txt for full list

## License

MIT License
