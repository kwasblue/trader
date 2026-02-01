# AutoTrader - Autonomous Trading Daemon

The AutoTrader is a fully autonomous trading daemon that manages the complete daily trading cycle without manual intervention.

## Overview

AutoTrader automates the entire trading workflow:

1. Waits for market open
2. Runs pre-flight validation checks
3. Executes trading strategies during market hours
4. Stops trading at market close
5. Updates historical data
6. Sleeps until the next trading day

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AutoTrader                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Market     │───▶│  PreFlight   │───▶│   Trading    │       │
│  │  Scheduler   │    │   Checker    │    │   Session    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Holiday     │    │  Credential  │    │ AlpacaRunner │       │
│  │  Calendar    │    │  Validator   │    │ SchwabRunner │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Historical Data Pipeline                 │       │
│  │  (Updates data after market close for next session)  │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## State Machine

AutoTrader uses a state machine to manage its lifecycle:

```
INITIALIZING
     │
     ▼
WAITING_FOR_MARKET ◀─────────────────┐
     │                                │
     ▼                                │
PRE_FLIGHT                            │
     │                                │
     ├──(success)──▶ TRADING          │
     │                   │            │
     └──(failure)──▶ ERROR           │
                         │            │
                         ▼            │
                   POST_MARKET        │
                         │            │
                         ▼            │
                   UPDATING_DATA      │
                         │            │
                         ▼            │
                     SLEEPING ────────┘
```

### States

| State | Description |
|-------|-------------|
| `INITIALIZING` | Startup, loading configuration |
| `WAITING_FOR_MARKET` | Sleeping until market open |
| `PRE_FLIGHT` | Running validation checks |
| `TRADING` | Active trading session |
| `POST_MARKET` | Market closed, preparing for data update |
| `UPDATING_DATA` | Fetching and processing historical data |
| `SLEEPING` | Waiting for next trading day |
| `STOPPED` | Graceful shutdown complete |
| `ERROR` | Error state (recoverable) |

## Usage

### Command Line

```bash
# Basic usage
python autotrader.py

# With specific symbols
python autotrader.py --symbols AAPL MSFT GOOGL

# With specific broker
python autotrader.py --broker alpaca
python autotrader.py --broker schwab

# Dry run (simulation only)
python autotrader.py --dry-run

# Verbose logging
python autotrader.py -v
```

### Control Script

The `autotrader_ctl.py` script provides daemon management:

```bash
# Start as background daemon
python autotrader_ctl.py start

# Start with options
python autotrader_ctl.py start --symbols AAPL MSFT --dry-run

# Check status
python autotrader_ctl.py status

# Stop gracefully (SIGTERM)
python autotrader_ctl.py stop

# Force stop (SIGKILL)
python autotrader_ctl.py stop --force

# View recent logs
python autotrader_ctl.py logs

# Tail logs
python autotrader_ctl.py logs --tail 50

# Follow logs in real-time
python autotrader_ctl.py logs --follow
```

### macOS Launch Agent

For automatic startup on macOS:

```bash
# Copy plist file
cp com.schwabtrader.autotrader.plist ~/Library/LaunchAgents/

# Edit paths in the plist file if needed
nano ~/Library/LaunchAgents/com.schwabtrader.autotrader.plist

# Load the agent (starts on login)
launchctl load ~/Library/LaunchAgents/com.schwabtrader.autotrader.plist

# Start immediately
launchctl start com.schwabtrader.autotrader

# Check status
launchctl list | grep schwabtrader

# Unload
launchctl unload ~/Library/LaunchAgents/com.schwabtrader.autotrader.plist
```

## Configuration

### trading_config.json

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

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Enable/disable autotrader |
| `default_broker` | `"alpaca"` | Default broker (`alpaca` or `schwab`) |
| `pre_market_buffer_minutes` | `15` | Minutes before market open to start pre-flight |
| `post_market_delay_minutes` | `5` | Minutes after close before data update |
| `data_update_days` | `5` | Days of history to update |
| `dry_run` | `false` | Run without placing actual orders |

### Environment Variables

Ensure these are set in `.env`:

```bash
# Alpaca
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret

# Schwab
SCHWAB_API_KEY=your_key
SCHWAB_SECRET=your_secret
```

## Market Schedule

### Trading Hours

- **Market Open**: 9:30 AM Eastern Time
- **Market Close**: 4:00 PM Eastern Time
- **Pre-Market Buffer**: Starts pre-flight checks 15 minutes before open

### Holidays

AutoTrader automatically skips US market holidays:

- New Year's Day
- Martin Luther King Jr. Day
- Presidents Day
- Good Friday
- Memorial Day
- Juneteenth
- Independence Day
- Labor Day
- Thanksgiving
- Christmas

### Weekend Handling

When current time is:
- **Friday after close**: Sleeps until Monday 9:15 AM ET
- **Saturday/Sunday**: Sleeps until Monday 9:15 AM ET

## Pre-Flight Checks

Before trading begins, AutoTrader validates:

1. **Credentials**: Broker API keys and tokens
2. **Token Expiry**: Schwab OAuth token validity
3. **Data Freshness**: Historical data is up-to-date
4. **Configuration**: Required config files exist
5. **Connectivity**: Can reach broker APIs

If pre-flight fails, AutoTrader enters ERROR state but continues to retry on the next trading day.

## Data Updates

After market close, AutoTrader:

1. Waits `post_market_delay_minutes`
2. Fetches latest bars for all symbols
3. Processes through ML pipeline (indicators, features)
4. Saves to JSON files and SQLite database
5. Updates cache metadata

## Logging

Logs are written to:

- **Main log**: `logs/autotrader.log`
- **Console**: stdout (when running in foreground)
- **Stdout file**: `logs/autotrader_stdout.log` (when daemonized)

Log format:
```
2024-01-15 09:30:00 INFO [AutoTrader] State: TRADING
2024-01-15 09:30:01 INFO [AutoTrader] Trading session started for ['AAPL', 'MSFT']
```

## Monitoring

### Get Status

```python
from autotrader import AutoTrader

trader = AutoTrader(symbols=['AAPL'])
status = trader.get_status()

print(status)
# {
#     'state': 'TRADING',
#     'running': True,
#     'symbols': ['AAPL'],
#     'broker': 'alpaca',
#     'dry_run': False,
#     'stats': {
#         'sessions_run': 5,
#         'last_session_start': '2024-01-15T09:30:00Z',
#         'last_data_update': '2024-01-14T16:15:00Z'
#     }
# }
```

### Integration with GUI

The AutoTrader can run alongside or instead of the GUI:

```bash
# Run AutoTrader without GUI
python autotrader.py --no-gui

# Run with GUI monitoring (default)
python autotrader.py
```

## Error Handling

### Recoverable Errors

- Network timeouts: Retries with exponential backoff
- API rate limits: Waits and retries
- Stale data: Updates before trading

### Fatal Errors

- Invalid credentials: Stops and logs error
- Missing configuration: Stops and logs error
- Broker connection failure after retries: Enters ERROR state

### Graceful Shutdown

AutoTrader handles signals properly:

- **SIGTERM**: Graceful shutdown, closes positions if configured
- **SIGINT**: Same as SIGTERM (Ctrl+C)
- **SIGKILL**: Immediate termination (not recommended)

## Best Practices

1. **Always test with dry-run first**
   ```bash
   python autotrader.py --dry-run --symbols AAPL
   ```

2. **Monitor logs during first few sessions**
   ```bash
   python autotrader_ctl.py logs --follow
   ```

3. **Set up alerts** for ERROR state transitions

4. **Keep credentials secure** - use environment variables

5. **Regular data validation**
   ```bash
   python preflight.py -v
   ```

## Troubleshooting

### AutoTrader won't start

1. Check logs: `python autotrader_ctl.py logs`
2. Verify credentials: `python preflight.py -v`
3. Check if another instance is running: `python autotrader_ctl.py status`

### Pre-flight keeps failing

1. Validate credentials manually
2. Check token expiry for Schwab
3. Verify network connectivity
4. Check data freshness

### Data not updating

1. Check `post_market_delay_minutes` setting
2. Verify data source availability
3. Check disk space for data files

### High CPU usage

1. Reduce logging verbosity
2. Check for infinite loops in strategies
3. Monitor event bus queue size

## API Reference

### AutoTrader Class

```python
class AutoTrader:
    def __init__(
        self,
        symbols: List[str],
        broker: str = "alpaca",
        dry_run: bool = False,
        config_path: Optional[str] = None
    )

    async def run(self) -> None:
        """Main run loop - runs until stopped"""

    def stop(self) -> None:
        """Signal graceful shutdown"""

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
```

### MarketScheduler Class

```python
class MarketScheduler:
    def is_trading_day(self, dt: datetime) -> bool:
        """Check if date is a trading day"""

    def is_market_open(self, dt: datetime) -> bool:
        """Check if market is currently open"""

    def get_next_market_open(self, dt: datetime) -> datetime:
        """Get next market open time"""

    def get_market_close_today(self, dt: datetime) -> datetime:
        """Get today's market close time"""

    def seconds_until_market_open(self, dt: datetime) -> int:
        """Seconds until next market open"""
```

## Related Documentation

- [Pre-Flight Checks](preflight.md)
- [Configuration Guide](configuration.md)
- [Broker Adapters](architecture.md#brokers)
- [Data Pipeline](architecture.md#data-pipeline)
