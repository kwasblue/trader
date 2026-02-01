# Quick Reference

A cheatsheet for common Schwab Trader operations.

## Starting the System

### Autonomous Mode (Recommended)

```bash
# Start AutoTrader daemon
python autotrader_ctl.py start --symbols AAPL MSFT

# Check status
python autotrader_ctl.py status

# View logs
python autotrader_ctl.py logs --follow

# Stop
python autotrader_ctl.py stop
```

### Manual Trading

```bash
# Simulation mode
python run_trading.py --mode simulation

# Alpaca paper trading
python run_trading.py --mode alpaca --paper --symbols AAPL MSFT

# Schwab live trading
python run_trading.py --mode schwab --symbols AAPL
```

### GUI Only

```bash
python run_live.py
```

## Pre-Trading Checks

```bash
# Quick validation
python preflight.py

# Verbose output
python preflight.py -v

# Update stale data
python preflight.py --update-data

# Refresh Schwab tokens
python preflight.py --reauth-schwab
```

## Data Management

### Update Historical Data

```bash
# Via command line
python -m core.unified_data_pipeline --symbols AAPL MSFT --days 30

# Via pre-flight
python preflight.py --update-data
```

### Python API

```python
from core.unified_data_pipeline import UnifiedDataPipeline

pipeline = UnifiedDataPipeline()
await pipeline.update_symbols(['AAPL', 'MSFT'], days=30)

# Get data
df = pipeline.get_data('AAPL')
```

## Testing

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_autotrader.py -v

# With coverage
pytest tests/ --cov=core --cov-report=html

# Quick run
python run_tests.py
```

## Configuration Files

| File | Purpose |
|------|---------|
| `.env` | API credentials |
| `config/trading_config.json` | General settings |
| `config/symbol_configuration.json` | Symbol-specific settings |

### Key Environment Variables

```bash
# Alpaca
ALPACA_API_KEY=xxx
ALPACA_SECRET_KEY=xxx

# Schwab
SCHWAB_API_KEY=xxx
SCHWAB_SECRET=xxx
```

## Common Operations

### Check Credentials

```python
from core.credential_validator import can_use_alpaca, can_use_schwab

print(await can_use_alpaca())  # True/False
print(await can_use_schwab())  # True/False
```

### Run Backtest

```python
from core.backtest_suite import VectorizedBacktester

bt = VectorizedBacktester(data, initial_capital=10000)
result = bt.run("sma_strategy", {"fast": 10, "slow": 30})
print(bt.get_metrics(result))
```

### Get Market Status

```python
from autotrader import MarketScheduler

scheduler = MarketScheduler()
print(scheduler.is_market_open())
print(scheduler.is_trading_day())
```

## Keyboard Shortcuts (GUI)

| Shortcut | Action |
|----------|--------|
| `Shift+Esc` | Toggle HALT |
| `Ctrl+M` | Manual Order |
| `Ctrl+L` | Flatten All |
| `Ctrl+K` | Cancel All |

## Logging

### Log Files

| File | Content |
|------|---------|
| `logs/app.log` | Main application log |
| `logs/autotrader.log` | AutoTrader specific |
| `logs/preflight.log` | Pre-flight checks |
| `logs/trading.log` | Trade execution |

### View Logs

```bash
# Recent entries
tail -100 logs/app.log

# Follow in real-time
tail -f logs/app.log

# AutoTrader logs
python autotrader_ctl.py logs --follow
```

## Troubleshooting

### AutoTrader won't start

```bash
# Check if already running
python autotrader_ctl.py status

# Check logs
python autotrader_ctl.py logs

# Run pre-flight
python preflight.py -v
```

### No data

```bash
# Update data
python preflight.py --update-data

# Check sources
python -c "
from core.credential_validator import check_credentials
import asyncio
print(asyncio.run(check_credentials()))
"
```

### Tests failing

```bash
# Run with verbose output
pytest tests/ -v --tb=long

# Run single test
pytest tests/test_autotrader.py::TestAutoTrader::test_init -v
```

## File Locations

```
schwab_trader/
├── .env                    # Credentials (DO NOT COMMIT)
├── autotrader.py           # Main daemon
├── preflight.py            # Pre-trading checks
├── run_trading.py          # Manual trading
├── config/
│   └── trading_config.json # Settings
├── data/data_storage/
│   ├── proc_data/          # Processed data
│   └── raw_data/           # Raw API data
├── logs/                   # Log files
└── tokens/                 # OAuth tokens
```
