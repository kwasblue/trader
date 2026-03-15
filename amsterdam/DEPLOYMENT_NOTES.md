# Deployment Notes - Script Reorganization

## Background Services

### Token Keeper Service
The `token_keeper.py` script runs as a background daemon to automatically renew Schwab OAuth tokens:
- **Access tokens** expire every 30 minutes - automatically renewed
- **Refresh tokens** expire after 7 days - warning logged when approaching expiry
- Checks token status every 60 seconds
- Logs to `data/streaming/token_keeper.log`
- Runs via systemd service `amsterdam-token-keeper`

This prevents trading interruptions due to expired authentication tokens.

## Scripts Moved

### Monitoring Scripts
- `daily_summary.py` → `monitoring/scripts/daily_summary.py`
- `dashboard.py` → `monitoring/scripts/dashboard.py`
- `event_monitor.py` → `monitoring/scripts/event_monitor.py`

### Optimization Scripts
- `daily_optimize.py` → `core/backtest/daily_optimize.py` (NOT RECOMMENDED FOR PRODUCTION)
- `daily_optimize_v2.py` → `core/backtest/daily_optimize_v2.py` (RECOMMENDED - Run weekly or monthly)

### Authentication Scripts
- `refresh_schwab_token.py` → `data/streaming/refresh_schwab_token.py`
- `token_keeper.py` → `data/streaming/token_keeper.py`

## Updates Needed on Raspberry Pi

### 1. Update Systemd Services

The dashboard service file needs to be updated:

```bash
ssh raspi
sudo systemctl stop amsterdam-dashboard

# Update the service file
sudo vim /etc/systemd/system/amsterdam-dashboard.service
# Change ExecStart line to:
# ExecStart=/home/kwasi/amsterdam/venv/bin/python /home/kwasi/amsterdam/monitoring/scripts/dashboard.py

sudo systemctl daemon-reload
sudo systemctl start amsterdam-dashboard
sudo systemctl status amsterdam-dashboard
```

**New: Install Token Keeper Service**

The token keeper service automatically renews Schwab OAuth tokens in the background:

```bash
ssh raspi
cd ~/amsterdam

# Copy service file to systemd
sudo cp amsterdam-token-keeper.service /etc/systemd/system/

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable amsterdam-token-keeper
sudo systemctl start amsterdam-token-keeper
sudo systemctl status amsterdam-token-keeper

# View logs
journalctl -u amsterdam-token-keeper -f
```

### 2. Update Cron Jobs

Current cron references old paths. Update with:

```bash
ssh raspi
crontab -e
```

**Replace these lines:**
```cron
# OLD:
5 16 * * 1-5 cd /home/kwasi/trader/amsterdam && /home/kwasi/trader/amsterdam/venv/bin/python /home/kwasi/trader/amsterdam/daily_summary.py
* * * * * cd /home/kwasi/trader/amsterdam && /home/kwasi/trader/amsterdam/venv/bin/python /home/kwasi/trader/amsterdam/event_monitor.py
```

**With these:**
```cron
# Monitoring scripts (keep these)
5 16 * * 1-5 cd /home/kwasi/trader/amsterdam && /home/kwasi/trader/amsterdam/venv/bin/python /home/kwasi/trader/amsterdam/monitoring/scripts/daily_summary.py
* * * * * cd /home/kwasi/trader/amsterdam && /home/kwasi/trader/amsterdam/venv/bin/python /home/kwasi/trader/amsterdam/monitoring/scripts/event_monitor.py

# Strategy optimization - Choose ONE:

# Option A: Weekly optimization (RECOMMENDED)
0 2 * * 0 cd /home/kwasi/trader/amsterdam && /home/kwasi/trader/amsterdam/bin/weekly-optimize

# Option B: Monthly optimization (more conservative)
# 0 2 1-7 * 0 cd /home/kwasi/trader/amsterdam && /home/kwasi/trader/amsterdam/bin/weekly-optimize
```

### 3. Pull Latest Changes

```bash
ssh raspi
cd ~/trader/amsterdam
git fetch origin
git pull origin efficiency_improvements
```

### 4. Verify Everything Works

```bash
# Test monitoring scripts
cd ~/trader/amsterdam
source venv/bin/activate

python monitoring/scripts/daily_summary.py
python monitoring/scripts/event_monitor.py
python monitoring/scripts/dashboard.py &  # Should start web server

# Test optimization script (weekly/monthly version)
python core/backtest/daily_optimize_v2.py --dry-run

# Test auth scripts
python data/streaming/refresh_schwab_token.py --help
```

## Final Directory Structure

```
amsterdam/
├── autoamsterdam.py                   # Main trading entry point
├── run_trading.py                     # Alternative entry point
├── preflight.py                       # Pre-flight checks
├── amsterdam_ctl.py                   # Control script
├── amsterdam-dashboard.service        # Systemd service for dashboard
├── amsterdam-token-keeper.service     # Systemd service for token renewal
│
├── core/
│   └── backtest/
│       ├── daily_optimize.py      # V1 - NOT for production use
│       ├── daily_optimize_v2.py   # V2 - PRODUCTION (run weekly/monthly)
│       ├── regime_backtest.py
│       └── ...
│
├── monitoring/
│   └── scripts/
│       ├── daily_summary.py       # Daily Slack summary
│       ├── dashboard.py           # Web dashboard (port 8080)
│       └── event_monitor.py       # Critical event alerts
│
├── data/
│   └── streaming/
│       ├── authenticator.py
│       ├── refresh_schwab_token.py   # Manual token refresh script
│       └── token_keeper.py           # Background token renewal daemon
│
├── tools/
│   ├── optimize_routing.py
│   ├── analyze_trades.py
│   └── ...
│
└── bin/
    ├── daily-optimize        # Wrapper for v1 (testing only)
    └── weekly-optimize       # Wrapper for v2 (production - use weekly/monthly)
```

## Deployment Checklist

- [ ] Pull latest code on Pi
- [ ] Update dashboard systemd service file
- [ ] Install token keeper systemd service
- [ ] Update cron jobs
- [ ] Restart dashboard service
- [ ] Start token keeper service
- [ ] Test all scripts
- [ ] Verify cron jobs run successfully
- [ ] Check dashboard at http://100.101.141.79:8080
- [ ] Verify token keeper is running: `sudo systemctl status amsterdam-token-keeper`
