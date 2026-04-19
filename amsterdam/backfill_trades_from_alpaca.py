#!/usr/bin/env python3
"""
Backfill trade logs from Alpaca order history
This replaces the incorrect log data with real broker data
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project to path
sys.path.insert(0, '/home/kwasi/amsterdam')
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv('/home/kwasi/amsterdam/.env')

def fetch_all_alpaca_orders():
    """Fetch ALL orders from Alpaca with pagination"""
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        print("ERROR: Alpaca credentials not found")
        return []

    client = TradingClient(api_key, secret_key, paper=True)

    all_orders = []
    page = 1
    limit = 500

    print("Fetching orders from Alpaca...")

    while True:
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=limit,
            nested=True
        )

        orders = client.get_orders(request)
        filled = [o for o in orders if o.status == 'filled']

        if not filled:
            break

        all_orders.extend(filled)
        print(f"  Page {page}: {len(filled)} filled orders (total: {len(all_orders)})")

        if len(filled) < limit:
            break

        page += 1

    print(f"\nTotal filled orders retrieved: {len(all_orders)}")
    return all_orders

def match_trades(orders):
    """Match buy/sell orders into completed trades using FIFO"""
    orders.sort(key=lambda x: x.filled_at)

    completed_trades = []
    open_positions = defaultdict(list)

    for order in orders:
        symbol = order.symbol
        side = order.side.value
        qty = float(order.filled_qty)
        price = float(order.filled_avg_price)
        filled_at = order.filled_at

        if side == 'buy':
            # Entry trade
            entry = {
                'event': 'entry',
                'trade_id': f"{filled_at.strftime('%Y%m%d_%H%M%S')}_{symbol}",
                'timestamp': filled_at.isoformat(),
                'symbol': symbol,
                'side': 'buy',
                'qty': int(qty),
                'price': price,
                'features': {
                    'strategy': 'AlpacaBroker',
                    'source': 'alpaca_backfill'
                }
            }

            open_positions[symbol].append({
                'entry': entry,
                'entry_price': price,
                'qty': qty,
                'entry_time': filled_at
            })

        elif side == 'sell':
            # Exit trade - match with oldest position (FIFO)
            remaining_qty = qty

            while remaining_qty > 0 and open_positions[symbol]:
                position = open_positions[symbol][0]
                close_qty = min(remaining_qty, position['qty'])

                pnl = (price - position['entry_price']) * close_qty
                pnl_pct = ((price - position['entry_price']) / position['entry_price']) * 100

                # Create exit trade
                exit_trade = {
                    'event': 'exit',
                    'trade_id': position['entry']['trade_id'],
                    'timestamp': filled_at.isoformat(),
                    'price': price,
                    'outcome': {
                        'pnl_dollars': pnl,
                        'pnl_percent': pnl_pct / 100,
                        'hold_time': (filled_at - position['entry_time']).total_seconds() / 3600,  # hours
                        'exit_reason': 'broker_confirmed',
                        'source': 'alpaca_backfill'
                    }
                }

                # Store completed trade pair
                completed_trades.append({
                    'entry': position['entry'],
                    'exit': exit_trade
                })

                # Update position
                position['qty'] -= close_qty
                remaining_qty -= close_qty

                if position['qty'] <= 0:
                    open_positions[symbol].pop(0)

    print(f"\nMatched {len(completed_trades)} completed trades")
    print(f"Open positions remaining: {sum(len(p) for p in open_positions.values())}")

    return completed_trades, open_positions

def write_to_log_file(completed_trades, output_file):
    """Write trades to JSONL file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing file if it exists
    if output_path.exists():
        backup_path = output_path.with_suffix('.jsonl.backup')
        print(f"\nBacking up existing file to: {backup_path}")
        output_path.rename(backup_path)

    print(f"\nWriting {len(completed_trades)} trades to: {output_file}")

    with open(output_path, 'w') as f:
        for trade_pair in sorted(completed_trades, key=lambda x: x['entry']['timestamp']):
            # Write entry
            f.write(json.dumps(trade_pair['entry']) + '\n')
            # Write exit
            f.write(json.dumps(trade_pair['exit']) + '\n')

    print(f"✓ Successfully wrote {len(completed_trades) * 2} lines (entry + exit pairs)")

def main():
    print("=" * 60)
    print("BACKFILL TRADES FROM ALPACA")
    print("=" * 60)
    print()

    # Fetch all orders
    orders = fetch_all_alpaca_orders()

    if not orders:
        print("No orders found!")
        return

    # Match trades
    completed_trades, open_positions = match_trades(orders)

    # Calculate stats
    wins = sum(1 for t in completed_trades if t['exit']['outcome']['pnl_dollars'] > 0)
    losses = len(completed_trades) - wins
    total_pnl = sum(t['exit']['outcome']['pnl_dollars'] for t in completed_trades)

    print(f"\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Completed Trades: {len(completed_trades)}")
    print(f"Wins: {wins} ({wins/len(completed_trades)*100:.1f}%)")
    print(f"Losses: {losses}")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print()

    # Write to file
    output_file = '/home/kwasi/amsterdam/logs/meta_trades_alpaca_backfill.jsonl'
    write_to_log_file(completed_trades, output_file)

    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"1. Review the backfilled data: {output_file}")
    print(f"2. If correct, replace meta_trades_live.jsonl:")
    print(f"   cp {output_file} /home/kwasi/amsterdam/logs/meta_trades_live.jsonl")
    print()
    print("✓ Backfill complete!")

if __name__ == '__main__':
    main()
