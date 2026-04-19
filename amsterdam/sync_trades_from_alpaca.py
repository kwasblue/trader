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
            if str(o.status.value).lower() == 'filled' and o.filled_at and o.filled_at > cutoff
        ]

        logger.info(f"Fetched {len(recent_filled)} filled orders from last {since_hours}h")
        return recent_filled

    except Exception as e:
        logger.error(f"Failed to fetch Alpaca orders: {e}")
        return []

def match_orders_to_existing_positions(orders, existing_entries, exited_trade_ids):
    """Match orders against existing open positions from log file"""
    orders.sort(key=lambda x: x.filled_at)

    # Build open positions from existing entries (entries without exits)
    open_positions = defaultdict(list)
    for trade_id, entry in existing_entries.items():
        if trade_id not in exited_trade_ids:
            symbol = entry['symbol']
            entry_side = entry['side']
            open_positions[symbol].append({
                'trade_id': trade_id,
                'entry': entry,
                'entry_price': entry['price'],
                'qty': entry['qty'],
                'entry_side': entry_side,
                'entry_time': datetime.fromisoformat(entry['timestamp'])
            })

    logger.info(f"Found {sum(len(v) for v in open_positions.values())} open positions")

    exit_events = []

    for order in orders:
        symbol = order.symbol
        side = str(order.side.value).lower()
        qty = float(order.filled_qty)
        price = float(order.filled_avg_price)
        filled_at = order.filled_at
        order_id_str = str(order.id)

        # Check if this order closes an existing position
        # Long positions (entry=buy) are closed by sells
        # Short positions (entry=sell) are closed by buys
        remaining_qty = qty

        positions = open_positions.get(symbol, [])
        i = 0
        while remaining_qty > 0 and i < len(positions):
            position = positions[i]
            entry_side = position['entry_side']

            # Check if this order closes this position
            closes_long = entry_side == 'buy' and side == 'sell'
            closes_short = entry_side == 'sell' and side == 'buy'

            if closes_long or closes_short:
                close_qty = min(remaining_qty, position['qty'])

                # Calculate PnL (positive = profit)
                if closes_long:
                    pnl = (price - position['entry_price']) * close_qty
                else:  # closes_short
                    pnl = (position['entry_price'] - price) * close_qty

                pnl_pct = pnl / (position['entry_price'] * close_qty)
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
                        'order_id': order_id_str
                    }
                }

                exit_events.append(exit_event)
                logger.info(f"Matched exit: {symbol} {position['trade_id']} PnL=${pnl:.2f}")

                position['qty'] -= close_qty
                remaining_qty -= close_qty

                if position['qty'] <= 0:
                    positions.pop(i)
                else:
                    i += 1
            else:
                i += 1

    logger.info(f"Generated {len(exit_events)} exit events")
    return exit_events

def append_exit_events(exit_events):
    """Append exit events to log file"""
    if not exit_events:
        logger.info("No exit events to append")
        return 0

    added = 0
    try:
        with open(LOG_FILE, 'a') as f:
            for event in exit_events:
                f.write(json.dumps(event) + '\n')
                added += 1

        logger.info(f"Appended {added} exit events to log")
        return added

    except Exception as e:
        logger.error(f"Failed to append events: {e}")
        return 0

def main():
    """Main sync process"""
    logger.info("=" * 60)
    logger.info("ALPACA TRADE SYNC - Starting")
    logger.info("=" * 60)

    # Load existing trades
    existing_entries, all_events = load_existing_trades()

    # Find which trade_ids already have exits
    exited_trade_ids = set()
    for event in all_events:
        if event['event'] == 'exit':
            exited_trade_ids.add(event['trade_id'])

    # Fetch recent Alpaca orders
    orders = fetch_recent_alpaca_orders(since_hours=48)  # Last 48 hours

    if not orders:
        logger.info("No recent orders to sync")
        return

    # Match orders to existing open positions
    exit_events = match_orders_to_existing_positions(orders, existing_entries, exited_trade_ids)

    # Filter out exits we already have
    fresh_exits = []
    for event in exit_events:
        if event['trade_id'] not in exited_trade_ids:
            fresh_exits.append(event)

    # Append new exit events
    added = append_exit_events(fresh_exits)

    logger.info("=" * 60)
    logger.info(f"SYNC COMPLETE - Added {added} exit events")
    logger.info("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        sys.exit(1)
