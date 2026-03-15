# ML Training Data Collection Guide

This guide explains how to continuously collect trade data on the Raspberry Pi for ML model training.

## Overview

The system automatically logs every trade (entry + exit) to JSONL files that can be converted to Parquet for model training.

**Data Flow:**
```
Trading Bot → MetaTradeLogger → logs/meta_trades_live.jsonl
                                        ↓
                                 convert_meta_trades.py
                                        ↓
                              data/trades_multi.parquet
                                        ↓
                              train_trade_model.py
```

## Setup on Raspberry Pi

### 1. Install the Trading Service

```bash
ssh raspi
cd ~/amsterdam

# Copy service file to systemd
sudo cp amsterdam-trader.service /etc/systemd/system/

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable amsterdam-trader
sudo systemctl start amsterdam-trader

# Check status
sudo systemctl status amsterdam-trader

# View live logs
journalctl -u amsterdam-trader -f
```

### 2. Verify Data Collection

Check that trade data is being logged:

```bash
ssh raspi
cd ~/amsterdam

# Check if meta trades log exists and is growing
ls -lh logs/meta_trades_live.jsonl

# View recent trades (last 5 entries)
tail -n 5 logs/meta_trades_live.jsonl | jq .

# Count total trades logged
wc -l logs/meta_trades_live.jsonl
```

Expected output format:
```json
{"event": "entry", "trade_id": "20260314_AAPL_001", "timestamp": "2026-03-14T09:35:12Z", ...}
{"event": "exit", "trade_id": "20260314_AAPL_001", "timestamp": "2026-03-14T10:42:18Z", ...}
```

### 3. Monitor Progress

```bash
# Quick stats on logged trades
ssh raspi "cd ~/amsterdam && python3 -c '
import json
from pathlib import Path

log_file = Path(\"logs/meta_trades_live.jsonl\")
if log_file.exists():
    entries = 0
    exits = 0
    with open(log_file) as f:
        for line in f:
            event = json.loads(line)
            if event[\"event\"] == \"entry\":
                entries += 1
            elif event[\"event\"] == \"exit\":
                exits += 1
    print(f\"Entries: {entries}, Exits: {exits}, Matched: {min(entries, exits)}\")
else:
    print(\"No log file yet\")
'"
```

## Syncing Data from Pi to Mac

### Manual Sync

```bash
# On Mac - sync logs from Pi to local machine
cd ~/projects/trader/amsterdam

# Copy JSONL log file
scp raspi:~/amsterdam/logs/meta_trades_live.jsonl logs/

# Or copy existing Parquet files
scp raspi:~/amsterdam/data/trades_*.parquet data/
```

### Automatic Sync (Optional)

Create a cron job on your Mac to sync daily:

```bash
# Edit Mac crontab
crontab -e

# Add this line to sync at 6 PM daily
0 18 * * * scp raspi:~/amsterdam/logs/meta_trades_live.jsonl ~/projects/trader/amsterdam/logs/ 2>&1 | logger -t amsterdam-sync
```

## Converting JSONL to Parquet

After syncing logs from the Pi:

```bash
cd ~/projects/trader/amsterdam

# Convert JSONL to Parquet
python tools/convert_meta_trades.py

# This creates/updates: data/trades_multi.parquet

# Check how many trades you have
python -c "
import pandas as pd
df = pd.read_parquet('data/trades_multi.parquet')
wins = (df['out_pnl_dollars'] > 0).sum()
losses = (df['out_pnl_dollars'] <= 0).sum()
print(f'Total trades: {len(df)}')
print(f'Wins: {wins} ({wins/len(df)*100:.1f}%)')
print(f'Losses: {losses} ({losses/len(df)*100:.1f}%)')
print(f'Ready for training: {\"Yes\" if wins >= 200 else \"No (need \" + str(200-wins) + \" more wins)\"}')
"
```

## Training the Model

Once you have enough data (200+ wins recommended):

```bash
# Install dependencies if needed
pip install imbalanced-learn

# Train the model
python tools/train_trade_model.py

# The script will tell you if you have enough data:
# ✓  Good: 523 winning trades (sufficient for training)
# ✓  ROC-AUC 0.7-0.8: Model has acceptable predictive power
```

## Data Collection Checklist

- [ ] Trading service is running: `sudo systemctl status amsterdam-trader`
- [ ] Token keeper is running: `sudo systemctl status amsterdam-token-keeper`
- [ ] Dashboard is accessible: http://100.101.141.79:8080
- [ ] Meta trades log exists: `ls -lh logs/meta_trades_live.jsonl`
- [ ] Trades are being logged: `tail logs/meta_trades_live.jsonl`
- [ ] Periodic sync set up (manual or cron)

## Troubleshooting

### No trades being logged

```bash
# Check if meta logging is enabled
ssh raspi
cd ~/amsterdam
grep -A 3 '"ml_training"' config/trading_config.json

# Should show: "meta_logging_enabled": true

# Check trading bot logs
journalctl -u amsterdam-trader -n 100

# Check for errors in meta trade logger
tail -50 logs/MetaTradeLogger.log
```

### Service won't start

```bash
# Check service logs
sudo journalctl -u amsterdam-trader -n 50

# Common issues:
# 1. Virtual environment not found
# 2. Missing dependencies
# 3. Config file errors

# Try running manually to see errors
cd ~/amsterdam
source venv/bin/activate
python autoamsterdam.py --broker alpaca
```

### JSONL to Parquet conversion fails

```bash
# Make sure pandas and pyarrow are installed
pip install pandas pyarrow

# Check JSONL file is valid JSON
head -1 logs/meta_trades_live.jsonl | jq .

# If that fails, the JSONL is corrupted - check for partial writes
```

## Data Quality Targets

For reliable ML model training:

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Total Trades | 500 | 1,000 | 2,000+ |
| Winning Trades | 200 | 500 | 1,000+ |
| Win Rate | 10% | 20% | 30%+ |
| Data Collection Period | 1 month | 3 months | 6+ months |

**Note:** The model training script will automatically warn you if you don't have enough data.

## Monitoring Dashboard

The web dashboard shows live trading status:
- URL: http://100.101.141.79:8080
- Shows: Active positions, recent trades, performance stats
- Logs are also visible in the dashboard

## Next Steps

Once you have 200+ winning trades:
1. Sync logs from Pi
2. Convert to Parquet
3. Train the model
4. Evaluate metrics (ROC-AUC, F1-score)
5. Deploy if metrics are good (ROC-AUC > 0.7)

The improved training script will guide you through the process and tell you exactly when you have enough data.
