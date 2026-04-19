#!/usr/bin/env python3
"""
Fetch ALL orders from Alpaca (paper or live account)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


def fetch_all_orders(paper: bool = True):
    """Fetch ALL orders from Alpaca with proper pagination"""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("ERROR: Alpaca credentials not found in environment")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return []

    client = TradingClient(api_key, secret_key, paper=paper)
    account_type = "PAPER" if paper else "LIVE"
    print(f"Fetching orders from Alpaca ({account_type} account)...")

    all_orders = []

    # Fetch ALL orders (any status)
    for status in [QueryOrderStatus.ALL]:
        page = 1
        until = None  # Use 'until' to paginate backwards in time

        while True:
            request = GetOrdersRequest(status=status, limit=500, nested=True, until=until)

            orders = client.get_orders(request)

            if not orders:
                break

            all_orders.extend(orders)
            print(f"  Fetched page {page}: {len(orders)} orders (total: {len(all_orders)})")

            if len(orders) < 500:
                break

            # Use the oldest order's created_at for pagination (go further back in time)
            oldest = min(orders, key=lambda x: x.created_at if x.created_at else datetime.max)
            until = oldest.created_at
            page += 1

    # Sort by created_at (newest first)
    all_orders.sort(key=lambda x: x.created_at if x.created_at else datetime.min, reverse=True)

    return all_orders


def display_orders(orders):
    """Display orders in a readable format"""
    print(f"\n{'=' * 80}")
    print(f"TOTAL ORDERS: {len(orders)}")
    print(f"{'=' * 80}\n")

    # Count by status
    status_counts = {}
    for o in orders:
        s = str(o.status)
        status_counts[s] = status_counts.get(s, 0) + 1

    print("Orders by status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print()

    # Display each order
    print(
        f"{'Date':<20} {'Symbol':<8} {'Side':<5} {'Qty':<8} {'Status':<12} {'Type':<10} {'Fill Price':<12} {'Order ID'}"
    )
    print("-" * 120)

    for o in orders:
        created = o.created_at.strftime("%Y-%m-%d %H:%M") if o.created_at else "N/A"
        fill_price = f"${float(o.filled_avg_price):.2f}" if o.filled_avg_price else "-"
        qty = str(o.qty) if o.qty else str(o.notional)

        print(
            f"{created:<20} {o.symbol:<8} {o.side.value:<5} {qty:<8} {str(o.status):<12} {str(o.type):<10} {fill_price:<12} {o.id}"
        )


def export_to_json(orders, output_file: str):
    """Export orders to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    orders_data = []
    for o in orders:
        orders_data.append(
            {
                "id": str(o.id),
                "client_order_id": o.client_order_id,
                "symbol": o.symbol,
                "side": o.side.value,
                "qty": str(o.qty) if o.qty else None,
                "notional": str(o.notional) if o.notional else None,
                "filled_qty": str(o.filled_qty) if o.filled_qty else None,
                "type": str(o.type),
                "status": str(o.status),
                "limit_price": str(o.limit_price) if o.limit_price else None,
                "stop_price": str(o.stop_price) if o.stop_price else None,
                "filled_avg_price": str(o.filled_avg_price) if o.filled_avg_price else None,
                "time_in_force": str(o.time_in_force),
                "created_at": o.created_at.isoformat() if o.created_at else None,
                "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
                "filled_at": o.filled_at.isoformat() if o.filled_at else None,
                "canceled_at": o.canceled_at.isoformat() if o.canceled_at else None,
            }
        )

    with open(output_path, "w") as f:
        json.dump(orders_data, f, indent=2)

    print(f"\nExported {len(orders)} orders to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch all Alpaca orders")
    parser.add_argument("--live", action="store_true", help="Use live account (default: paper)")
    parser.add_argument("--export", type=str, help="Export to JSON file")
    parser.add_argument("--no-display", action="store_true", help="Skip displaying orders")
    args = parser.parse_args()

    paper = not args.live
    orders = fetch_all_orders(paper=paper)

    if not orders:
        print("No orders found!")
        return

    if not args.no_display:
        display_orders(orders)

    if args.export:
        export_to_json(orders, args.export)
    else:
        # Default export location
        default_export = Path(__file__).parent.parent / "logs" / "all_alpaca_orders.json"
        export_to_json(orders, str(default_export))


if __name__ == "__main__":
    main()
