#!/bin/bash
# Sync trades from Alpaca every hour
cd /home/kwasi/amsterdam
source venv/bin/activate
python3 sync_trades_from_alpaca.py >> logs/alpaca_sync_cron.log 2>&1
