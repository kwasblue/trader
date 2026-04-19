#!/usr/bin/env python3
"""
Convert Alpaca orders to ML-ready training data format.

Fetches all orders from Alpaca, matches buy/sell pairs into complete trades,
calculates features from timestamps and price data, and outputs in the format
expected by the ML pipeline (train_trade_model.py).

Features that can be derived from Alpaca data:
- Time features (hour, day, minutes_since_open)
- hours_since_last_trade
- pnl_dollars, pnl_percent
- hold_time

Features that require historical data (fetched if possible):
- atr_percentile (requires bar data)

Features set to neutral defaults (no data available):
- drawdown_portfolio_pct, drawdown_symbol_pct (0.0 - no drawdown)
- position_size_pct (estimated from typical portfolio)
- bars_in_regime (0 - unknown)
- signal_strength (1 for buys - assumed bullish)

Usage:
    python alpaca_orders_to_ml.py
    python alpaca_orders_to_ml.py --live --output data/alpaca_trades.parquet
    python alpaca_orders_to_ml.py --fetch-atr  # Fetch ATR from historical data
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: pandas and numpy required. Install with: pip install pandas numpy")
    sys.exit(1)

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

# Optional: for fetching ATR data
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    HAS_DATA_CLIENT = True
except ImportError:
    HAS_DATA_CLIENT = False

# Load environment
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

ET = ZoneInfo("America/New_York")
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30


def fetch_all_orders(paper: bool = True) -> List[Any]:
    """Fetch ALL orders from Alpaca with pagination."""
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        print("ERROR: Alpaca credentials not found")
        return []

    client = TradingClient(api_key, secret_key, paper=paper)
    account_type = "PAPER" if paper else "LIVE"
    print(f"Fetching orders from Alpaca ({account_type})...")

    all_orders = []
    page = 1
    until = None

    while True:
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=500,
            nested=True,
            until=until
        )
        orders = client.get_orders(request)

        if not orders:
            break

        all_orders.extend(orders)
        print(f"  Page {page}: {len(orders)} orders (total: {len(all_orders)})")

        if len(orders) < 500:
            break

        oldest = min(orders, key=lambda x: x.created_at if x.created_at else datetime.max)
        until = oldest.created_at
        page += 1

    return all_orders


def calculate_atr(symbol: str, at_time: datetime, data_client, lookback_days: int = 50) -> Optional[Tuple[float, float]]:
    """
    Calculate ATR and ATR percentile for a symbol at a given time.

    Returns:
        Tuple of (atr_value, atr_percentile) or None if data unavailable
    """
    if not HAS_DATA_CLIENT or data_client is None:
        return None

    try:
        # Get bars for ATR calculation
        start = at_time - timedelta(days=lookback_days)
        end = at_time

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )

        bars = data_client.get_stock_bars(request)

        # Access via bars[symbol] - may raise KeyError if no data
        try:
            bar_list = bars[symbol]
        except (KeyError, TypeError):
            return None

        if len(bar_list) < 14:
            return None

        df = pd.DataFrame([{
            'high': b.high,
            'low': b.low,
            'close': b.close
        } for b in bar_list])

        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = df.apply(
            lambda x: max(
                x['high'] - x['low'],
                abs(x['high'] - x['prev_close']) if pd.notna(x['prev_close']) else 0,
                abs(x['low'] - x['prev_close']) if pd.notna(x['prev_close']) else 0
            ), axis=1
        )

        # Calculate ATR (14-period)
        df['atr'] = df['tr'].rolling(window=14).mean()

        current_atr = df['atr'].iloc[-1]
        if pd.isna(current_atr):
            return None

        # Calculate percentile
        historical_atrs = df['atr'].dropna().tolist()
        atr_percentile = sum(1 for a in historical_atrs if a <= current_atr) / len(historical_atrs)

        return (current_atr, atr_percentile)

    except Exception as e:
        return None


def calculate_time_features(timestamp: datetime) -> Dict[str, Any]:
    """Calculate time-based features from a timestamp."""
    # Convert to Eastern Time
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC"))
    et_time = timestamp.astimezone(ET)

    hour = et_time.hour
    day_of_week = et_time.weekday()  # 0=Monday

    # Minutes since market open (9:30 ET)
    market_open = et_time.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0)
    if et_time < market_open:
        minutes_since_open = 0
    else:
        minutes_since_open = int((et_time - market_open).total_seconds() / 60)
        minutes_since_open = min(max(minutes_since_open, 0), 390)  # Cap at market close

    return {
        'hour_of_day': hour,
        'day_of_week': day_of_week,
        'minutes_since_open': minutes_since_open
    }


def match_orders_to_trades(orders: List[Any], fetch_atr: bool = False) -> List[Dict[str, Any]]:
    """
    Match buy/sell orders into complete trades using FIFO.

    Returns:
        List of trade dictionaries in ML-ready format
    """
    # Filter to filled orders only
    filled = [o for o in orders if str(o.status) == 'OrderStatus.FILLED' and o.filled_at]
    filled.sort(key=lambda x: x.filled_at)

    print(f"Processing {len(filled)} filled orders...")

    # Initialize data client for ATR if needed
    data_client = None
    atr_cache: Dict[str, Tuple[float, float]] = {}  # Cache by symbol+date

    if fetch_atr and HAS_DATA_CLIENT:
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        data_client = StockHistoricalDataClient(api_key, secret_key)
        print("ATR enrichment enabled - fetching historical data...")

    completed_trades = []
    open_positions: Dict[str, List[Dict]] = defaultdict(list)
    last_trade_time: Dict[str, datetime] = {}

    # Track all trade times for hours_since_last calculation
    all_trade_times: List[datetime] = []
    buy_count = 0

    for order in filled:
        symbol = order.symbol
        side = order.side.value  # 'buy' or 'sell'
        qty = float(order.filled_qty)
        price = float(order.filled_avg_price)
        filled_at = order.filled_at

        # Calculate hours since last trade (any symbol)
        hours_since_last = 0.0
        if all_trade_times:
            last_any = max(all_trade_times)
            hours_since_last = (filled_at - last_any).total_seconds() / 3600
        all_trade_times.append(filled_at)

        if side == 'buy':
            # Entry trade
            buy_count += 1
            time_features = calculate_time_features(filled_at)

            # Calculate ATR if requested
            atr_value = None
            atr_percentile = 0.5  # Default to median
            if fetch_atr and data_client:
                # Use cache key of symbol + date (not time, since we use daily bars)
                cache_key = f"{symbol}_{filled_at.date()}"
                if cache_key in atr_cache:
                    atr_value, atr_percentile = atr_cache[cache_key]
                else:
                    atr_result = calculate_atr(symbol, filled_at, data_client)
                    if atr_result:
                        atr_value, atr_percentile = atr_result
                        atr_cache[cache_key] = atr_result
                        if buy_count % 50 == 0:
                            print(f"  Processed {buy_count} buys, ATR cache size: {len(atr_cache)}")

            entry = {
                'trade_id': f"{filled_at.strftime('%Y%m%d_%H%M%S')}_{symbol}_{str(order.id)[:8]}",
                'entry_timestamp': filled_at.isoformat(),
                'symbol': symbol,
                'side': 'buy',
                'qty': int(qty),
                'entry_price': price,
                'order_id': str(order.id),

                # Features we can calculate
                'feat_hour_of_day': time_features['hour_of_day'],
                'feat_day_of_week': time_features['day_of_week'],
                'feat_minutes_since_open': time_features['minutes_since_open'],
                'feat_hours_since_last_trade': round(hours_since_last, 2),
                'feat_atr_percentile': round(atr_percentile, 4),

                # Features with neutral defaults (no portfolio state available)
                'feat_drawdown_portfolio_pct': 0.0,  # Assume no drawdown
                'feat_drawdown_symbol_pct': 0.0,
                'feat_position_size_pct': 0.05,  # Assume ~5% position size
                'feat_bars_in_regime': 0,  # Unknown
                'feat_signal_strength': 1,  # Bullish (bought)

                # Store for matching
                '_filled_at': filled_at,
                '_qty_remaining': qty,
            }

            open_positions[symbol].append(entry)

        elif side == 'sell':
            # Exit trade - match with oldest position (FIFO)
            remaining_qty = qty

            while remaining_qty > 0 and open_positions[symbol]:
                position = open_positions[symbol][0]
                close_qty = min(remaining_qty, position['_qty_remaining'])

                # Calculate P&L
                pnl_dollars = (price - position['entry_price']) * close_qty
                pnl_percent = (price - position['entry_price']) / position['entry_price']

                # Calculate hold time
                hold_time_hours = (filled_at - position['_filled_at']).total_seconds() / 3600

                # Build complete trade record
                trade = {
                    'trade_id': position['trade_id'],
                    'entry_timestamp': position['entry_timestamp'],
                    'exit_timestamp': filled_at.isoformat(),
                    'symbol': symbol,
                    'side': 'buy',
                    'qty': int(close_qty),
                    'entry_price': position['entry_price'],
                    'exit_price': price,
                    'is_complete': True,

                    # Entry features
                    'feat_atr_percentile': position['feat_atr_percentile'],
                    'feat_drawdown_portfolio_pct': position['feat_drawdown_portfolio_pct'],
                    'feat_drawdown_symbol_pct': position['feat_drawdown_symbol_pct'],
                    'feat_position_size_pct': position['feat_position_size_pct'],
                    'feat_hour_of_day': position['feat_hour_of_day'],
                    'feat_day_of_week': position['feat_day_of_week'],
                    'feat_minutes_since_open': position['feat_minutes_since_open'],
                    'feat_bars_in_regime': position['feat_bars_in_regime'],
                    'feat_hours_since_last_trade': position['feat_hours_since_last_trade'],
                    'feat_signal_strength': position['feat_signal_strength'],

                    # Outcome metrics
                    'out_pnl_dollars': round(pnl_dollars, 2),
                    'out_pnl_percent': round(pnl_percent, 6),
                    'out_hold_time_hours': round(hold_time_hours, 2),
                    'out_hold_bars': None,  # Would need bar data
                    'out_mae_percent': None,  # Would need intrabar data
                    'out_mfe_percent': None,
                    'out_exit_reason': 'broker_matched',
                }

                completed_trades.append(trade)

                # Update remaining quantities
                position['_qty_remaining'] -= close_qty
                remaining_qty -= close_qty

                if position['_qty_remaining'] <= 0:
                    open_positions[symbol].pop(0)

    # Report open positions
    total_open = sum(len(p) for p in open_positions.values())
    if total_open > 0:
        print(f"\nOpen positions (no matching sell): {total_open}")
        for symbol, positions in open_positions.items():
            if positions:
                print(f"  {symbol}: {len(positions)} position(s)")

    return completed_trades


def to_dataframe(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert trade list to DataFrame with proper types."""
    df = pd.DataFrame(trades)

    if df.empty:
        return df

    # Convert timestamps
    df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
    df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])

    # Ensure numeric types
    numeric_cols = [
        'entry_price', 'exit_price',
        'feat_atr_percentile', 'feat_drawdown_portfolio_pct',
        'feat_drawdown_symbol_pct', 'feat_position_size_pct',
        'feat_hours_since_last_trade',
        'out_pnl_dollars', 'out_pnl_percent', 'out_hold_time_hours',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Integer columns
    int_cols = [
        'qty', 'feat_hour_of_day', 'feat_day_of_week',
        'feat_minutes_since_open', 'feat_bars_in_regime',
        'feat_signal_strength',
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # Categorical
    df['symbol'] = df['symbol'].astype('category')
    df['side'] = df['side'].astype('category')

    return df


def print_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("TRADE STATISTICS")
    print(f"{'='*60}")

    print(f"Total complete trades: {len(df)}")
    print(f"Unique symbols: {df['symbol'].nunique()}")

    if 'out_pnl_dollars' in df.columns:
        pnl = df['out_pnl_dollars'].dropna()
        wins = (pnl > 0).sum()
        losses = (pnl <= 0).sum()

        print(f"\nP&L Summary:")
        print(f"  Total P&L: ${pnl.sum():,.2f}")
        print(f"  Avg P&L: ${pnl.mean():,.2f}")
        print(f"  Wins: {wins} ({wins/len(pnl)*100:.1f}%)")
        print(f"  Losses: {losses}")

        if pnl[pnl < 0].sum() != 0:
            pf = abs(pnl[pnl > 0].sum() / pnl[pnl < 0].sum())
            print(f"  Profit Factor: {pf:.2f}")

    if 'out_hold_time_hours' in df.columns:
        hold = df['out_hold_time_hours'].dropna()
        print(f"\nHold Time:")
        print(f"  Avg: {hold.mean():.1f} hours")
        print(f"  Median: {hold.median():.1f} hours")

    # Date range
    print(f"\nDate Range:")
    print(f"  First: {df['entry_timestamp'].min()}")
    print(f"  Last: {df['entry_timestamp'].max()}")

    # Feature summary
    print(f"\nFeature Ranges:")
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    for col in feat_cols[:5]:  # Show first 5
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"  {col}: [{vals.min():.2f}, {vals.max():.2f}]")


def main():
    parser = argparse.ArgumentParser(description='Convert Alpaca orders to ML training data')
    parser.add_argument('--live', action='store_true', help='Use live account (default: paper)')
    parser.add_argument('--output', '-o', type=str, default='data/alpaca_trades_ml.parquet',
                        help='Output file path (supports .parquet, .csv, .jsonl)')
    parser.add_argument('--fetch-atr', action='store_true',
                        help='Fetch ATR from historical data (slower but more accurate)')
    parser.add_argument('--stats', action='store_true', default=True,
                        help='Print statistics')
    args = parser.parse_args()

    paper = not args.live

    # Fetch orders
    orders = fetch_all_orders(paper=paper)

    if not orders:
        print("No orders found!")
        return

    # Match into trades
    trades = match_orders_to_trades(orders, fetch_atr=args.fetch_atr)

    if not trades:
        print("No complete trades found (need matching buy/sell pairs)")
        return

    print(f"\nMatched {len(trades)} complete trades")

    # Convert to DataFrame
    df = to_dataframe(trades)

    # Determine output format and save
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix == '.parquet':
        df.to_parquet(output_path, index=False)
    elif suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif suffix in ('.jsonl', '.json'):
        # JSONL format compatible with convert_meta_trades.py
        with open(output_path, 'w') as f:
            for _, row in df.iterrows():
                record = row.to_dict()
                # Convert timestamps to strings
                for k, v in record.items():
                    if pd.isna(v):
                        record[k] = None
                    elif hasattr(v, 'isoformat'):
                        record[k] = v.isoformat()
                f.write(json.dumps(record) + '\n')
    else:
        print(f"Unknown format: {suffix}, defaulting to parquet")
        output_path = output_path.with_suffix('.parquet')
        df.to_parquet(output_path, index=False)

    print(f"\nSaved to: {output_path}")

    # Print stats
    if args.stats:
        print_stats(df)

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print(f"Train model with:")
    print(f"  python tools/train_trade_model.py --input {output_path}")
    print()


if __name__ == '__main__':
    main()
