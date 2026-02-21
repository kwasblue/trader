# Commands Reference

The `trader` CLI provides a unified interface for all trading system operations.

## Installation

After cloning the repository:

```bash
cd schwab_trader
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the `trader` command globally in your virtual environment.

---

## Quick Reference

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

---

## Daily Startup Workflow

Recommended sequence to start trading each day:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run preflight checks (also starts token keeper)
trader preflight -v

# 3. Check token status
trader token status

# 4. Update historical data if needed
trader data update

# 5. Launch GUI for simulation/monitoring
trader gui

# Or start autonomous trading
trader start --broker alpaca
```

---

## Trading Commands

### Start Daemon

```bash
# Start with defaults (Alpaca broker)
trader start

# Specific symbols
trader start --symbols AAPL,MSFT,TSLA

# Use Schwab broker
trader start --broker schwab

# Dry run (no real trades)
trader start --dry-run

# Run as background daemon
trader start --daemon
```

### Stop Daemon

```bash
trader stop
```

### Check Status

```bash
trader status
```

---

## GUI Trading

```bash
# Simulation mode (default) - no real trades
trader gui

# Alpaca paper trading
trader gui --mode alpaca

# Schwab live trading
trader gui --mode schwab

# Custom symbols
trader gui --symbols AAPL,GOOGL,TSLA

# Slower simulation (1 second per bar)
trader gui --speed 1.0
```

---

## Pre-Flight Checks

The preflight command validates system readiness before trading. When Schwab credentials are valid, it automatically starts the token keeper daemon to maintain token freshness.

```bash
# Quick check
trader preflight

# Verbose output
trader preflight -v

# Update stale historical data
trader preflight --update-data

# Force Schwab re-authentication
trader preflight --reauth-schwab

# Full check with data update
trader preflight -v --update-data
```

**What preflight checks:**
- Environment variables (API keys)
- Broker credentials (Alpaca, Schwab)
- Token expiry status
- Historical data freshness
- Configuration files
- Auto-starts token keeper daemon if Schwab tokens are valid

---

## Token Management

### Check Token Status

```bash
trader token status
```

Output shows status for both Schwab and Alpaca:
- **VALID** - Token is valid and ready
- **EXPIRING_SOON** - Token expires within 24 hours
- **EXPIRED** - Token has expired, needs refresh
- **MISSING** - Credentials not configured

### Refresh Tokens

```bash
# Refresh if needed
trader token refresh

# Force full re-authentication (opens browser)
trader token refresh --force
```

### Token Keeper Service

Keeps tokens fresh by periodically checking and renewing:

```bash
# Run in foreground
trader token keeper

# Run as background daemon
trader token keeper --daemon

# Custom check interval (every 5 minutes)
trader token keeper --interval 300
```

---

## Symbol Management

### List Symbols

```bash
# Show all symbols
trader symbols list

# Show trade list only
trader symbols list --trade

# Show watch list only
trader symbols list --watch
```

### Add Symbols

```bash
# Add to trade list (default)
trader symbols add TSLA

# Add to trade list explicitly
trader symbols add NVDA --trade

# Add to watch list
trader symbols add AMD --watch
```

### Remove Symbols

```bash
trader symbols remove AAPL
```

---

## Strategy Management

Strategy selection combines backtesting and walk-forward validation to find the best strategies for each symbol.

### Select Best Strategies

```bash
# Evaluate all strategies for a symbol
trader strategy select AAPL

# Select and save to config
trader strategy select AAPL --save

# Use 180 days of historical data
trader strategy select MSFT --days 180 --save

# Optimize for specific market regime
trader strategy select TSLA --regime high_volatility --save

# Select top 5 strategies instead of 3
trader strategy select AAPL --top 5 --save

# Use specific ranking metric
trader strategy select AAPL --metric sharpe_ratio --save

# Skip walk-forward validation (faster but may overfit)
trader strategy select AAPL --no-walk-forward --save
```

Output includes:
- Composite score (weighted combination of metrics)
- Sharpe and Sortino ratios
- Total return and max drawdown
- Win rate and profit factor
- Walk-forward Sharpe (out-of-sample validation)
- Overfit warning if in-sample >> out-of-sample

### List Available Strategies

```bash
trader strategy list
```

Shows all 18+ available strategies (SMA, EMA, MACD, RSI, Bollinger, etc.)

### Show Current Routing

```bash
# Show all routing configuration
trader strategy show

# Show routing for specific symbol
trader strategy show --symbol AAPL
```

### Refresh Routing (Hot Reload)

After editing `config/strategy_routing.json` manually:

```bash
trader strategy refresh
```

### Workflow Example

```bash
# 1. Select best strategies for your symbols
trader strategy select AAPL --days 365 --save
trader strategy select MSFT --days 365 --save
trader strategy select TSLA --regime high_volatility --save

# 2. Verify routing configuration
trader strategy show

# 3. Run simulation with selected strategies
trader gui --symbols AAPL,MSFT,TSLA

# 4. Start live trading when ready
trader start --symbols AAPL,MSFT,TSLA
```

---

## Data Management

### Update Historical Data

```bash
# Update all configured symbols
trader data update

# Specific symbols
trader data update --symbols AAPL,MSFT

# More history (60 days)
trader data update --days 60

# Force specific source
trader data update --source alpaca
```

### Check Data Freshness

```bash
# Check all available symbols
trader data status

# Check specific symbols
trader data status --symbols AAPL,MSFT
```

---

## Testing

```bash
# Run all tests
trader test

# Verbose output
trader test -v

# With coverage report
trader test --coverage

# Specific test file
trader test tests/test_autotrader.py

# Unit tests only
trader test --unit
```

---

## Viewing Logs

```bash
# Last 50 lines of app.log
trader logs

# Follow in real-time
trader logs -f

# More lines
trader logs -n 100

# Specific log file
trader logs --file trades      # Trade execution log
trader logs --file autotrader  # Daemon operations
trader logs --file preflight   # Pre-flight checks
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `.env` | API keys and secrets |
| `config/trading_config.json` | General trading settings |
| `config/symbols.json` | Trade and watch lists |
| `config/strategy_routing.json` | Strategy-to-symbol mapping |
| `config/trade_logic_routing.json` | Trade logic routing rules |

### Environment Variables

```bash
# Alpaca credentials
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret

# Schwab credentials
SCHWAB_API_KEY=your_key
SCHWAB_SECRET=your_secret
```

---

## Log Files

All logs are in the `logs/` directory:

| File | Purpose |
|------|---------|
| `app.log` | Main application log (aggregated) |
| `trades.log` | Trade execution log |
| `autotrader.log` | Daemon operations |
| `preflight.log` | Pre-flight check results |
| `credential_validator.log` | Token status checks |
| `unified_pipeline.log` | Data pipeline operations |

---

## Legacy Scripts

The original standalone scripts are still available for compatibility:

| Script | Equivalent CLI Command |
|--------|----------------------|
| `python autotrader.py` | `trader start` |
| `python autotrader_ctl.py status` | `trader status` |
| `python run_trading.py` | `trader gui` |
| `python preflight.py` | `trader preflight` |
| `python refresh_schwab_token.py` | `trader token refresh` |
| `python token_keeper.py` | `trader token keeper` |
| `python run_tests.py` | `trader test` |

---

## Global Options

```bash
# Show version
trader --version

# Show help for any command
trader --help
trader start --help
trader token --help
```

---

## Troubleshooting

### Environment Variables Not Loading

If `trader token status` shows MISSING for credentials that are set in `.env`:

```bash
# Verify .env exists in project root
ls -la .env

# Check .env contents (redacted)
head -5 .env

# Ensure you're in the virtual environment
which python  # Should show .venv/bin/python
```

### Token Keeper Not Starting

```bash
# Check if already running
pgrep -f token_keeper.py

# View token keeper logs
tail -f logs/token_keeper.log

# Start manually
trader token keeper --daemon
```

### Strategy Selection Fails

```bash
# Ensure historical data exists
trader data status --symbols AAPL

# Update data if needed
trader data update --symbols AAPL --days 365

# Then retry
trader strategy select AAPL --save
```

---

## See Also

- [Quick Reference](quick-reference.md) - Condensed cheatsheet
- [Architecture](architecture.md) - System design overview
- [Event System](event-system.md) - Event-driven architecture
- [Strategies](strategies.md) - Available trading strategies
