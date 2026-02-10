# Quick Reference

A cheatsheet for common Schwab Trader operations using the `trader` CLI.

## Getting Started

```bash
# Install the CLI
cd schwab_trader
source .venv/bin/activate
pip install -e .

# Verify installation
trader --help
```

## Starting the System

### Autonomous Mode (Recommended)

```bash
# Start trading daemon
trader start --symbols AAPL,MSFT

# Check status
trader status

# View logs in real-time
trader logs -f

# Stop
trader stop
```

### GUI Trading

```bash
# Simulation mode (no real trades)
trader gui

# Alpaca paper trading
trader gui --mode alpaca

# Schwab live trading
trader gui --mode schwab

# Custom symbols
trader gui --symbols AAPL,GOOGL,TSLA
```

## Pre-Trading Checks

```bash
# Quick validation
trader preflight

# Verbose output
trader preflight -v

# Update stale data
trader preflight --update-data
```

## Token Management

```bash
# Check token status
trader token status

# Refresh tokens
trader token refresh

# Force re-authentication (browser)
trader token refresh --force

# Run token keeper service
trader token keeper
```

## Data Management

```bash
# Update historical data
trader data update --symbols AAPL,MSFT

# Check data freshness
trader data status
```

## Symbol Management

```bash
# List all symbols
trader symbols list

# Add to trade list
trader symbols add TSLA --trade

# Add to watch list
trader symbols add AMD --watch

# Remove symbol
trader symbols remove AAPL
```

## Testing

```bash
# Run all tests
trader test

# With coverage
trader test --coverage

# Specific file
trader test tests/test_autotrader.py
```

## Viewing Logs

```bash
# Recent app logs
trader logs

# Follow in real-time
trader logs -f

# Trade execution logs
trader logs --file trades

# More lines
trader logs -n 100
```

## Configuration

| File | Purpose |
|------|---------|
| `.env` | API credentials |
| `config/trading_config.json` | Trading settings |
| `config/symbols.json` | Symbol lists |

### Key Environment Variables

```bash
ALPACA_API_KEY=xxx
ALPACA_SECRET_KEY=xxx
SCHWAB_API_KEY=xxx
SCHWAB_SECRET=xxx
```

## Keyboard Shortcuts (GUI)

| Shortcut | Action |
|----------|--------|
| `Shift+Esc` | Toggle HALT |
| `Ctrl+M` | Manual Order |
| `Ctrl+L` | Flatten All |
| `Ctrl+K` | Cancel All |

## Common Workflows

### Morning Startup

```bash
trader preflight -v          # Check system
trader token status          # Verify tokens
trader start --symbols AAPL  # Start trading
```

### Evening Shutdown

```bash
trader stop                  # Stop daemon
trader logs --file trades    # Review trades
```

### Troubleshooting

```bash
trader status                # Is daemon running?
trader token status          # Token issues?
trader preflight -v          # Full system check
trader logs -f               # Watch for errors
```

## More Information

- [Commands Reference](commands.md) - Complete CLI documentation
- [Operations Guide](operations.md) - Daily operations procedures
