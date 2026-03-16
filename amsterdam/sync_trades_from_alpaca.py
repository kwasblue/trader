#!/usr/bin/env python3
"""
Sync trades from Alpaca to local logs
Runs periodically to ensure logs match broker reality
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, '/home/kwasi/amsterdam')
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from loggers.logger import Logger

load_dotenv('/home/kwasi/amsterdam/.env')

# Setup logging
logger = Logger(
    log_file='alpaca_sync.log',
    logger_name='AlpacaSync',
    log_dir='/home/kwasi/amsterdam/logs'
).get_logger()

LOG_FILE = Path('/home/kwasi/amsterdam/logs/meta_trades_live.jsonl')

def get_alpaca_client():
    """Get authenticated Alpaca client"""
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        logger.error("Alpaca credentials not found")
        return None

    return TradingClient(api_key, secret_key, paper=True)

def load_existing_trades():
    """Load existing trades from log file"""
    if not LOG_FILE.exists():
        return {}, []

    trade_ids = set()
    entries = {}
    all_events = []

    try:
        with open(LOG_FILE, 'r') as f:
            for line in f:
                event = json.loads(line.strip())
                all_events.append(event)

                if event['event'] == 'entry':
                    entries[event['trade_id']] = event
                    trade_ids.add(event['trade_id'])
                elif event['event'] == 'exit':
                    trade_ids.add(event['trade_id'])

        logger.info(f"Loaded {len(all_events)} events ({len(entries)} entries)")
        return entries, all_events

    except Exception as e:
        logger.error(f"Failed to load existing trades: {e}")
        return {}, []

def fetch_recent_alpaca_orders(since_hours=24):
    """Fetch recent filled orders from Alpaca"""
    client = get_alpaca_client()
    if not client:
        return []

    try:
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=500,
            nested=True
        )

        orders = client.get_orders(request)

        # Filter to filled orders from last N hours
        cutoff = datetime.now().astimezone() - timedelta(hours=since_hours)
        recent_filled = [
            o for o in orders
            if o.status == 'filled' and o.filled_at > cutoff
        ]

        logger.info(f"Fetched {len(recent_filled)} filled orders from last {since_hours}h")
        return recent_filled

    except Exception as e:
        logger.error(f"Failed to fetch Alpaca orders: {e}")
        return []

def match_orders_to_trades(orders):
    """Match buy/sell orders into trade pairs"""
    orders.sort(key=lambda x: x.filled_at)

    new_trades = []
    positions = defaultdict(list)

    for order in orders:
        symbol = order.symbol
        side = order.side.value
        qty = float(order.filled_qty)
        price = float(order.filled_avg_price)
        filled_at = order.filled_at

        trade_id = f"{filled_at.strftime('%Y%m%d_%H%M%S')}_{symbol}_{order.id[:8]}"

        if side == 'buy':
            # Entry
            entry = {
                'event': 'entry',
                'trade_id': trade_id,
                'timestamp': filled_at.isoformat(),
                'symbol': symbol,
                'side': 'buy',
                'qty': int(qty),
                'price': price,
                'features': {
                    'strategy': 'AlpacaBroker',
                    'source': 'alpaca_sync',
                    'order_id': order.id
                }
            }

            positions[symbol].append({
                'trade_id': trade_id,
                'entry': entry,
                'entry_price': price,
                'qty': qty,
                'entry_time': filled_at
            })

        elif side == 'sell':
            # Exit
            remaining_qty = qty

            while remaining_qty > 0 and positions[symbol]:
                position = positions[symbol][0]
                close_qty = min(remaining_qty, position['qty'])

                pnl = (price - position['entry_price']) * close_qty
                pnl_pct = ((price - position['entry_price']) / position['entry_price'])
                hold_time = (filled_at - position['entry_time']).total_seconds() / 3600

                exit_event = {
                    'event': 'exit',
                    'trade_id': position['trade_id'],
                    'timestamp': filled_at.isoformat(),
                    'price': price,
                    'outcome': {
                        'pnl_dollars': pnl,
                        'pnl_percent': pnl_pct,
                        'hold_time_hours': hold_time,
                        'exit_reason': 'broker_confirmed',
                        'source': 'alpaca_sync',
                        'order_id': order.id
                    }
                }

                # Add completed trade
                new_trades.append({
                    'entry': position['entry'],
                    'exit': exit_event
                })

                position['qty'] -= close_qty
                remaining_qty -= close_qty

                if position['qty'] <= 0:
                    positions[symbol].pop(0)

    logger.info(f"Matched {len(new_trades)} new completed trades")
    return new_trades

def append_new_trades(new_trades):
    """Append new trades to log file"""
    if not new_trades:
        logger.info("No new trades to append")
        return 0

    added = 0
    try:
        with open(LOG_FILE, 'a') as f:
            for trade in new_trades:
                # Write entry
                f.write(json.dumps(trade['entry']) + '\n')
                # Write exit
                f.write(json.dumps(trade['exit']) + '\n')
                added += 1

        logger.info(f"Appended {added} new trades to log")
        return added

    except Exception as e:
        logger.error(f"Failed to append trades: {e}")
        return 0

def main():
    """Main sync process"""
    logger.info("=" * 60)
    logger.info("ALPACA TRADE SYNC - Starting")
    logger.info("=" * 60)

    # Load existing trades
    existing_entries, all_events = load_existing_trades()
    existing_trade_ids = set(e['trade_id'] for e in all_events)

    # Fetch recent Alpaca orders
    orders = fetch_recent_alpaca_orders(since_hours=48)  # Last 48 hours

    if not orders:
        logger.info("No recent orders to sync")
        return

    # Match into trades
    new_trades = match_orders_to_trades(orders)

    # Filter out trades we already have
    fresh_trades = []
    for trade in new_trades:
        if trade['entry']['trade_id'] not in existing_trade_ids:
            fresh_trades.append(trade)
            logger.info(
                f"New trade: {trade['entry']['symbol']} "
                f"${trade['exit']['outcome']['pnl_dollars']:.2f}"
            )

    # Append new trades
    added = append_new_trades(fresh_trades)

    logger.info("=" * 60)
    logger.info(f"SYNC COMPLETE - Added {added} new trades")
    logger.info("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        sys.exit(1)
