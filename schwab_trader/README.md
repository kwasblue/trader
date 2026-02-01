# Schwab Trader

A comprehensive algorithmic trading platform with support for Schwab and Alpaca brokers, featuring real-time streaming, backtesting, and a professional monitoring GUI.

## Features

- **Multi-Broker Support**: Trade with Schwab or Alpaca (paper/live)
- **Autonomous Trading (AutoTrader)**: Fully automated trading daemon with market hours awareness
- **Real-Time Streaming**: WebSocket-based price feeds with automatic reconnection
- **Strategy Framework**: Pluggable strategies with regime-based routing
- **Risk Management**: Drawdown monitoring, position sizing, trade gates
- **Professional GUI**: PySide6-based monitoring dashboard with real-time charts
- **Comprehensive Backtesting**: Vectorized backtester with walk-forward analysis and Monte Carlo simulation
- **Event-Driven Architecture**: Async event bus for decoupled components
- **Pre-Flight Checks**: Automated validation before trading sessions
- **Historical Data Management**: Unified data pipeline with multiple sources

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/schwab_trader.git
cd schwab_trader

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
# Required for Schwab: SCHWAB_API_KEY, SCHWAB_SECRET
# Required for Alpaca: ALPACA_API_KEY, ALPACA_SECRET_KEY
```

### 3. Run the Application

```bash
# Run with simulation (default)
python run_trading.py

# Run with Alpaca paper trading
python run_trading.py --mode alpaca --paper

# Run with Schwab live trading
python run_trading.py --mode schwab --symbols AAPL,MSFT

# Run with custom symbols
python run_trading.py --symbols AAPL,MSFT,GOOGL --mode simulation
```

## Architecture

```
schwab_trader/
├── core/                      # Core trading engine
│   ├── broker/                # Broker adapters (Schwab, Alpaca)
│   ├── logic/                 # Trade logic and state management
│   ├── events/                # Event system
│   ├── alpaca_runner.py       # Alpaca live trading runner
│   ├── schwab_runner.py       # Schwab live trading runner
│   └── backtest_suite.py      # Backtesting framework
├── data/                      # Data pipeline
│   └── streaming/             # WebSocket streaming clients
├── strategies/                # Trading strategies
├── monitoring/                # GUI and monitoring
│   ├── views/                 # Qt windows and widgets
│   └── feeds/                 # Data feeders
├── config/                    # Configuration files
├── tests/                     # Unit tests
└── run_trading.py             # Main entry point
```

### Event Flow

```
Market Data → Broker Adapter → Live Runner → Strategy → Execution Engine → Orders
                    ↓              ↓            ↓             ↓
                Event Bus ←───────────────────────────────────┘
                    ↓
                GUI Dashboard
```

## Configuration

### Symbol Configuration

The unified `config/symbol_configuration.json` defines:
- Strategy routing per symbol and regime
- Strategy parameters
- Trade logic configuration
- Risk overrides

Example:
```json
{
  "symbols": {
    "AAPL": {
      "regimes": {
        "low_volatility": {
          "strategy": "sma_strategy",
          "strategy_params": {"fast": 10, "slow": 30},
          "trade_logic": "default"
        }
      }
    }
  }
}
```

### Trading Configuration

`config/trading_config.json` contains:
- General settings (symbols, mode)
- Broker-specific settings
- Risk parameters
- Indicator periods

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_schwab_broker.py -v

# Run with coverage
pytest tests/ --cov=schwab_trader --cov-report=html

# Run async tests
pytest tests/ -v --asyncio-mode=auto
```

## Strategies

### Built-in Strategies

- **SMA Strategy**: Simple moving average crossover
- **Momentum Strategy**: Price momentum based signals
- **Mean Reversion Strategy**: Bollinger band reversion

### Adding a New Strategy

1. Create strategy in `strategies/`:
```python
from strategies.strategy_registry import register_strategy

@register_strategy("my_strategy")
class MyStrategy:
    def __init__(self, params=None):
        self.params = params or {}

    def generate_signal(self, data):
        # Return: 1 (buy), 0 (hold), -1 (sell)
        return 0
```

2. Add to `config/symbol_configuration.json`:
```json
{
  "AAPL": {
    "regimes": {
      "normal": {
        "strategy": "my_strategy",
        "strategy_params": {"param1": "value1"}
      }
    }
  }
}
```

## Backtesting

```python
from core.backtest_suite import VectorizedBacktester, grid_search, walk_forward_analysis

# Basic backtest
bt = VectorizedBacktester(data, initial_capital=10000)
result = bt.run("sma_strategy", {"fast": 10, "slow": 30})
metrics = bt.get_metrics(result)

# Parameter optimization
opt_result = grid_search(
    data, "sma_strategy",
    {"fast": [5, 10, 15], "slow": [20, 30, 40]},
    metric="sharpe_ratio"
)

# Walk-forward analysis
wf_result = walk_forward_analysis(
    data, "sma_strategy",
    {"fast": [5, 10, 15], "slow": [20, 30, 40]},
    train_size=252, test_size=63
)
```

## GUI Dashboard

The monitoring dashboard provides:
- Real-time equity curve
- Position tracking
- P&L monitoring
- Trade alerts
- Manual order entry
- Simulation controls

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Shift+Esc | Toggle HALT |
| Ctrl+M | Manual Order |
| Ctrl+L | Flatten All |
| Ctrl+K | Cancel All |

## AutoTrader (Autonomous Trading)

The AutoTrader is a fully autonomous trading daemon that manages the complete daily trading cycle.

### Features

- **Market Hours Awareness**: Automatically waits for market open and stops at close
- **Pre-Flight Checks**: Validates credentials, data freshness, and system config before trading
- **Historical Data Updates**: Updates data after market close for next session
- **Holiday Handling**: Automatically skips US market holidays
- **Graceful Shutdown**: Proper cleanup on SIGTERM/SIGINT signals
- **State Machine**: Clear state transitions (WAITING → PRE_FLIGHT → TRADING → POST_MARKET → SLEEPING)

### Quick Start

```bash
# Run AutoTrader (foreground)
python autotrader.py --symbols AAPL MSFT --broker alpaca

# Dry run mode (no actual trades)
python autotrader.py --dry-run

# Run as background daemon
nohup python autotrader.py > logs/autotrader_stdout.log 2>&1 &
```

### Control Script

```bash
# Start the AutoTrader daemon
python autotrader_ctl.py start

# Check status
python autotrader_ctl.py status

# Stop gracefully
python autotrader_ctl.py stop

# View logs
python autotrader_ctl.py logs
python autotrader_ctl.py logs --tail 100
```

### macOS Auto-Start (launchd)

```bash
# Install the launch agent
cp com.schwabtrader.autotrader.plist ~/Library/LaunchAgents/

# Load (start on login)
launchctl load ~/Library/LaunchAgents/com.schwabtrader.autotrader.plist

# Unload
launchctl unload ~/Library/LaunchAgents/com.schwabtrader.autotrader.plist
```

### Configuration

AutoTrader settings in `config/trading_config.json`:

```json
{
  "autotrader": {
    "enabled": true,
    "default_broker": "alpaca",
    "pre_market_buffer_minutes": 15,
    "post_market_delay_minutes": 5,
    "data_update_days": 5,
    "dry_run": false
  }
}
```

### Daily Cycle

1. **WAITING_FOR_MARKET**: Sleeps until 15 minutes before market open
2. **PRE_FLIGHT**: Runs credential and data validation checks
3. **TRADING**: Executes strategies during market hours (9:30 AM - 4:00 PM ET)
4. **POST_MARKET**: Waits 5 minutes after close, updates historical data
5. **SLEEPING**: Calculates next trading day and sleeps until then

See [docs/autotrader.md](docs/autotrader.md) for complete documentation.

## Pre-Flight Checks

Run validation before trading:

```bash
# Quick check
python preflight.py

# Verbose output
python preflight.py -v

# Update stale data
python preflight.py --update-data

# Re-authenticate Schwab tokens
python preflight.py --reauth-schwab
```

## Deployment

### Production Checklist

- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Configure proper logging level
- [ ] Set up credential management (not in code)
- [ ] Configure appropriate risk limits
- [ ] Test with paper trading first
- [ ] Set up monitoring and alerts

### Docker (Future)

```bash
docker build -t schwab_trader .
docker run -e SCHWAB_API_KEY=xxx schwab_trader
```

## Security

- Never commit `.env` files or credentials
- Use environment variables for sensitive data
- The `utils/cdp_api_key.json` is in `.gitignore`
- Review `.gitignore` before committing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## License

[Your License Here]

## Support

- Issues: [GitHub Issues](https://github.com/your-repo/schwab_trader/issues)
- Documentation: [docs/](docs/)
