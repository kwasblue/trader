# Operations Guide

Complete reference for running and managing the Schwab Trader system.

## Quick Start

```bash
cd /Users/kwasiaddo/projects/trader/schwab_trader

# 1. Run pre-flight checks
python preflight.py -v

# 2. Start the autotrader
python autotrader_ctl.py start --symbols AAPL MSFT NVDA

# 3. Check status
python autotrader_ctl.py status

# 4. View logs
python autotrader_ctl.py logs --follow
```

## Command Reference

### autotrader_ctl.py - Daemon Management

```bash
# Start autotrader as background daemon
python autotrader_ctl.py start
python autotrader_ctl.py start --symbols AAPL MSFT NVDA
python autotrader_ctl.py start --broker alpaca
python autotrader_ctl.py start --broker schwab
python autotrader_ctl.py start --dry-run          # Simulation mode

# Check if running
python autotrader_ctl.py status

# Stop gracefully
python autotrader_ctl.py stop
python autotrader_ctl.py stop --force             # Force kill (SIGKILL)

# View logs
python autotrader_ctl.py logs                     # Recent logs
python autotrader_ctl.py logs --tail 100          # Last 100 lines
python autotrader_ctl.py logs --follow            # Real-time tail
```

### autotrader.py - Direct Execution

```bash
# Run in foreground (for debugging)
python autotrader.py
python autotrader.py --symbols AAPL MSFT
python autotrader.py --broker alpaca
python autotrader.py --broker schwab
python autotrader.py --dry-run
python autotrader.py -v                           # Verbose logging
```

### preflight.py - Pre-Trading Validation

```bash
# Quick validation
python preflight.py

# Verbose output
python preflight.py -v

# Update stale historical data
python preflight.py --update-data

# Refresh Schwab tokens
python preflight.py --reauth-schwab
```

### refresh_schwab_token.py - Token Management

```bash
# Check token status and refresh if needed
python refresh_schwab_token.py

# Force full re-authentication (browser login)
python refresh_schwab_token.py --force
```

### token_keeper.py - Background Token Service

```bash
# Run in foreground
python token_keeper.py

# Custom check interval (seconds)
python token_keeper.py --interval 300             # Every 5 minutes

# Run as daemon
python token_keeper.py --daemon
```

### run_trading.py - Manual Trading

```bash
# Simulation mode
python run_trading.py --mode simulation

# Alpaca paper trading
python run_trading.py --mode alpaca --paper --symbols AAPL MSFT

# Alpaca live trading
python run_trading.py --mode alpaca --symbols AAPL MSFT

# Schwab live trading
python run_trading.py --mode schwab --symbols AAPL
```

## macOS Services (launchd)

### AutoTrader Service

Starts automatically on login and runs the trading daemon.

```bash
# Install
cp com.schwabtrader.autotrader.plist ~/Library/LaunchAgents/

# Load (enable)
launchctl load ~/Library/LaunchAgents/com.schwabtrader.autotrader.plist

# Check status
launchctl list | grep autotrader

# Unload (disable)
launchctl unload ~/Library/LaunchAgents/com.schwabtrader.autotrader.plist
```

### Token Keeper Service

Keeps Schwab tokens fresh by checking every 5 minutes.

```bash
# Install
cp com.schwabtrader.tokenkeeper.plist ~/Library/LaunchAgents/

# Load (enable)
launchctl load ~/Library/LaunchAgents/com.schwabtrader.tokenkeeper.plist

# Check status
launchctl list | grep tokenkeeper

# Unload (disable)
launchctl unload ~/Library/LaunchAgents/com.schwabtrader.tokenkeeper.plist
```

### Service Management Commands

```bash
# List all schwabtrader services
launchctl list | grep schwab

# Restart a service
launchctl unload ~/Library/LaunchAgents/com.schwabtrader.autotrader.plist
launchctl load ~/Library/LaunchAgents/com.schwabtrader.autotrader.plist

# View service errors
cat ~/Library/LaunchAgents/com.schwabtrader.*.plist
```

## Token Management

### How Tokens Work

Schwab uses OAuth2 with two token types:

| Token | Lifetime | Refresh Method |
|-------|----------|----------------|
| Access Token | ~30 minutes | Automatic (uses refresh token) |
| Refresh Token | ~7 days | Manual (browser login required) |

### Automatic Token Renewal

The `token_keeper` service handles automatic renewal:

1. Checks tokens every 5 minutes
2. Automatically renews access tokens using the refresh token
3. Warns when refresh token is expiring (2 days before)
4. Alerts when manual re-authentication is needed

### Manual Token Refresh

When refresh token expires:

```bash
# Check status and refresh
python refresh_schwab_token.py

# Force re-authentication
python refresh_schwab_token.py --force
```

You will need to:
1. Open the printed URL in your browser
2. Log in to Schwab
3. Copy the redirect URL and paste it back

### Token Files

Tokens are stored in: `tokens/token_file.json`

## Log Files

### Core Logs
| File | Purpose |
|------|---------|
| `logs/app.log` | Main application log (aggregates all) |
| `logs/autotrader.log` | AutoTrader daemon state changes |
| `logs/execution_engine.log` | Trade decisions, order flow |
| `logs/state_reconciler.log` | Broker sync, position matching |

### Trading Logs
| File | Purpose |
|------|---------|
| `logs/live_trades.csv` | Trade records (CSV format) |
| `logs/portfolio_state.log` | Portfolio changes, P&L |
| `logs/position_sizer.log` | Position sizing calculations |
| `logs/drawdown_monitor.log` | Risk limit monitoring |

### Routing Logs
| File | Purpose |
|------|---------|
| `logs/strategy_routing.log` | Strategy-to-symbol mapping |
| `logs/trade_logic_router.log` | Trade logic routing |
| `logs/trade_logic_manager.log` | Trade logic decisions |
| `logs/trade_logic.log` | Individual logic evaluations |

### Data & Infrastructure
| File | Purpose |
|------|---------|
| `logs/unified_pipeline.log` | Data fetching, processing |
| `logs/historical_updater.log` | Historical data updates |
| `logs/alpaca.log` | Alpaca broker operations |
| `logs/schwab_broker.log` | Schwab broker operations |
| `logs/token_keeper.log` | Token refresh service |
| `logs/credential_validator.log` | Credential checks |
| `logs/preflight.log` | Pre-flight validation |
| `logs/event_handler.log` | Event bus activity |

### Viewing Logs

```bash
# Recent entries
tail -100 logs/app.log

# Follow in real-time
tail -f logs/app.log

# Multiple logs
tail -f logs/app.log logs/autotrader.log

# Search for errors
grep -i error logs/app.log

# Today's errors
grep "$(date +%Y-%m-%d)" logs/app.log | grep -i error
```

## Configuration Files

| File | Purpose |
|------|---------|
| `.env` | API credentials (never commit!) |
| `config/trading_config.json` | General trading settings |
| `config/symbol_configuration.json` | Per-symbol settings |
| `config/strategy_routing.json` | Strategy-to-symbol mapping |
| `config/trade_logic_routing.json` | Trade logic routing |
| `config/strategy_params.json` | Strategy parameters |

### Key Environment Variables

```bash
# Alpaca (Paper Trading)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true

# Schwab
SCHWAB_API_KEY=your_key
SCHWAB_SECRET=your_secret
SCHWAB_REDIRECT_URL=https://127.0.0.1
```

## Daily Operations Checklist

### Before Market Open

1. Check token status:
   ```bash
   python refresh_schwab_token.py
   ```

2. Run pre-flight:
   ```bash
   python preflight.py -v
   ```

3. Verify autotrader is running:
   ```bash
   python autotrader_ctl.py status
   ```

### During Trading

1. Monitor logs:
   ```bash
   python autotrader_ctl.py logs --follow
   ```

2. Check for errors:
   ```bash
   grep -i error logs/app.log | tail -20
   ```

### After Market Close

1. Check data updates completed:
   ```bash
   grep "data update" logs/autotrader.log | tail -5
   ```

2. Review trading summary in logs

## Troubleshooting

### AutoTrader Won't Start

```bash
# Check if already running
python autotrader_ctl.py status
ps aux | grep autotrader

# Check for errors
python autotrader_ctl.py logs

# Run pre-flight
python preflight.py -v

# Try running in foreground
python autotrader.py -v
```

### Token Errors

```bash
# Check token status
python refresh_schwab_token.py

# Force re-authentication
python refresh_schwab_token.py --force

# Restart token keeper
launchctl unload ~/Library/LaunchAgents/com.schwabtrader.tokenkeeper.plist
launchctl load ~/Library/LaunchAgents/com.schwabtrader.tokenkeeper.plist
```

### Connection Issues

```bash
# Check credentials
python -c "
from core.credential_validator import check_credentials
import asyncio
print(asyncio.run(check_credentials()))
"

# Test Alpaca connection
python -c "
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
load_dotenv()
client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
print(client.get_account())
"
```

### Data Issues

```bash
# Update historical data
python preflight.py --update-data

# Check data freshness
ls -la data/data_storage/proc_data/

# Force data refresh
python -c "
from core.unified_data_pipeline import UnifiedDataPipeline
import asyncio
pipeline = UnifiedDataPipeline()
asyncio.run(pipeline.update_symbols(['AAPL', 'MSFT', 'NVDA'], days=30))
"
```

### Too Many WebSocket Connections

```bash
# Kill zombie processes
pkill -f "autotrader.py"
pkill -f "alpaca"

# Check for running processes
ps aux | grep -E "autotrader|alpaca|schwab"

# Restart cleanly
python autotrader_ctl.py stop --force
sleep 5
python autotrader_ctl.py start
```

### Strategy/Logic Errors

Check the routing configuration:

```bash
# View strategy routing
cat config/strategy_routing.json

# View trade logic routing
cat config/trade_logic_routing.json

# Check logs for routing decisions
grep -i "routing\|logic\|strategy" logs/app.log | tail -20
```

## File Locations

```
schwab_trader/
├── .env                          # Credentials (DO NOT COMMIT)
├── autotrader.py                 # Main daemon
├── autotrader_ctl.py             # Daemon control script
├── preflight.py                  # Pre-trading checks
├── refresh_schwab_token.py       # Manual token refresh
├── token_keeper.py               # Background token service
├── run_trading.py                # GUI trading application
├── com.schwabtrader.autotrader.plist   # macOS service (autotrader)
├── com.schwabtrader.tokenkeeper.plist  # macOS service (tokens)
├── config/
│   ├── trading_config.json       # Main settings
│   ├── strategy_routing.json     # Strategy mapping
│   └── trade_logic_routing.json  # Trade logic mapping
├── data/data_storage/
│   ├── proc_data/                # Processed data (JSON)
│   └── raw_data/                 # Raw API data
├── logs/                         # Log files
└── tokens/
    └── token_file.json           # OAuth tokens
```

## Emergency Procedures

### Stop All Trading Immediately

```bash
# Stop autotrader
python autotrader_ctl.py stop --force

# Kill all related processes
pkill -9 -f "autotrader"
pkill -9 -f "run_trading"
pkill -9 -f "alpaca"
```

### Flatten All Positions

```bash
# Via GUI: Press Ctrl+L or Shift+Esc (HALT)

# Via API (Alpaca):
python -c "
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
load_dotenv()
client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
client.close_all_positions(cancel_orders=True)
print('All positions closed')
"
```

### Cancel All Orders

```bash
# Via GUI: Press Ctrl+K

# Via API (Alpaca):
python -c "
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
load_dotenv()
client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
client.cancel_orders()
print('All orders cancelled')
"
```
