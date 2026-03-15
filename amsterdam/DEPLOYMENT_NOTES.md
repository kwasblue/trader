# Deployment Notes - Script Reorganization

## Scripts Moved

### Monitoring Scripts
- `daily_summary.py` → `monitoring/scripts/daily_summary.py`
- `dashboard.py` → `monitoring/scripts/dashboard.py`
- `event_monitor.py` → `monitoring/scripts/event_monitor.py`

### Optimization Scripts
- `daily_optimize.py` → `core/backtest/daily_optimize.py`
- `daily_optimize_v2.py` → `core/backtest/daily_optimize_v2.py`

### Authentication Scripts
- `refresh_schwab_token.py` → `data/streaming/refresh_schwab_token.py`
- `token_keeper.py` → `data/streaming/token_keeper.py`

## Updates Needed on Raspberry Pi

### 1. Update Systemd Service

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
# NEW:
5 16 * * 1-5 cd /home/kwasi/trader/amsterdam && /home/kwasi/trader/amsterdam/venv/bin/python /home/kwasi/trader/amsterdam/monitoring/scripts/daily_summary.py
* * * * * cd /home/kwasi/trader/amsterdam && /home/kwasi/trader/amsterdam/venv/bin/python /home/kwasi/trader/amsterdam/monitoring/scripts/event_monitor.py
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

# Test optimization scripts
python core/backtest/daily_optimize.py --dry-run
python core/backtest/daily_optimize_v2.py --dry-run

# Test auth scripts
python data/streaming/refresh_schwab_token.py --help
```

## Final Directory Structure

```
amsterdam/
├── autoamsterdam.py          # Main trading entry point
├── run_trading.py            # Alternative entry point
├── preflight.py              # Pre-flight checks
├── amsterdam_ctl.py          # Control script
│
├── core/
│   └── backtest/
│       ├── daily_optimize.py      # V1 optimization
│       ├── daily_optimize_v2.py   # V2 walk-forward optimization
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
│       ├── refresh_schwab_token.py
│       └── token_keeper.py
│
├── tools/
│   ├── optimize_routing.py
│   ├── analyze_trades.py
│   └── ...
│
└── bin/
    ├── daily-optimize        # Wrapper for daily_optimize.py
    └── weekly-optimize       # Wrapper for daily_optimize_v2.py
```

## Deployment Checklist

- [ ] Pull latest code on Pi
- [ ] Update systemd service file
- [ ] Update cron jobs
- [ ] Restart dashboard service
- [ ] Test all scripts
- [ ] Verify cron jobs run successfully
- [ ] Check dashboard at http://100.101.141.79:8080
