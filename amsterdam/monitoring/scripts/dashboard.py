#!/usr/bin/env python3
"""
Amsterdam Trading Dashboard - v3
Uses Alpaca as source of truth for all trade data
"""

from flask import Flask, render_template_string, jsonify
import os
import json
import psutil
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import re

app = Flask(__name__)

# Load environment variables
load_dotenv(Path("/home/kwasi/amsterdam/.env"))

# Paths
BASE_DIR = Path("/home/kwasi/amsterdam")
LOG_FILE = BASE_DIR / "logs" / "autotrader.log"
PID_FILE = BASE_DIR / "logs" / "autotrader.pid"
TRADES_LOG_FILE = BASE_DIR / "logs" / "meta_trades_live.jsonl"

def get_status():
    """Get Amsterdam running status"""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return "Running"
        except (ProcessLookupError, ValueError):
            return "Stopped"
    return "Unknown"

def get_current_state():
    """Parse current state from logs"""
    if not LOG_FILE.exists():
        return "Unknown"

    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines[-50:]):
                # Check for waiting state
                if "Waiting" in line and "until pre-market" in line:
                    return "Waiting for Market"
                # Check for state changes
                elif "State change:" in line or "state:" in line.lower():
                    if "TRADING" in line or "trading" in line:
                        return "Trading"
                    elif "PRE_FLIGHT" in line or "pre-flight" in line.lower():
                        return "Pre-flight Checks"
                    elif "POST_MARKET" in line or "post-market" in line.lower():
                        return "Post-Market"
                    elif "UPDATING_DATA" in line or "updating" in line.lower():
                        return "Updating Data"
    except Exception:
        pass

    return "Unknown"

def get_next_market_open():
    """Get next market open time from logs"""
    if not LOG_FILE.exists():
        return "Unknown"

    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines[-20:]):
                if "Market opens at" in line:
                    match = re.search(r'Market opens at ([\d-]+ [\d:]+ \w+)', line)
                    if match:
                        return match.group(1)
                if r"Waiting (\d+)h (\d+)m" in line:
                    match = re.search(r'Waiting (\d+h \d+m)', line)
                    if match:
                        return f"in {match.group(1)}"
    except Exception:
        pass

    return "Unknown"

def get_recent_logs(n=10):
    """Get recent log entries"""
    if not LOG_FILE.exists():
        return []

    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-n:]]
    except Exception:
        return []

def parse_log_line(line):
    """Parse log line into structured data"""
    try:
        parts = line.split(' - ', 3)
        if len(parts) >= 4:
            return {
                'timestamp': parts[0],
                'level': parts[2],
                'message': parts[3]
            }
    except Exception:
        pass
    return {'timestamp': '', 'level': '', 'message': line}

def get_system_health():
    """Get system health metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = shutil.disk_usage("/")

        return {
            'cpu_percent': round(cpu_percent, 1),
            'memory_percent': round(memory.percent, 1),
            'memory_used_gb': round(memory.used / (1024**3), 1),
            'memory_total_gb': round(memory.total / (1024**3), 1),
            'disk_percent': round((disk.used / disk.total) * 100, 1),
            'disk_used_gb': round(disk.used / (1024**3), 1),
            'disk_total_gb': round(disk.total / (1024**3), 1)
        }
    except Exception:
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'memory_used_gb': 0,
            'memory_total_gb': 0,
            'disk_percent': 0,
            'disk_used_gb': 0,
            'disk_total_gb': 0
        }

def get_broker_info():
    """Get broker account and position info"""
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            return None

        from alpaca.trading.client import TradingClient

        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        positions = client.get_all_positions()

        position_list = []
        total_unrealized_pnl = 0.0

        for pos in positions:
            pnl = float(pos.unrealized_pl)
            pnl_pct = float(pos.unrealized_plpc) * 100
            total_unrealized_pnl += pnl

            position_list.append({
                'symbol': pos.symbol,
                'qty': int(pos.qty),
                'side': 'LONG' if float(pos.qty) > 0 else 'SHORT',
                'entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pnl': pnl,
                'unrealized_pnl_pct': pnl_pct
            })

        return {
            'equity': float(account.equity),
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'positions': position_list,
            'total_positions': len(position_list),
            'total_unrealized_pnl': total_unrealized_pnl
        }
    except Exception:
        return None

def get_active_symbols():
    """Get list of symbols being monitored"""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines[-100:]):
                if "Symbols:" in line:
                    match = re.search(r'Symbols: \[(.*?)\]', line)
                    if match:
                        symbols = [s.strip().strip("'") for s in match.group(1).split(',')]
                        return symbols
    except Exception:
        pass
    return []

def get_trade_stats_from_log():
    """Get trade statistics from backfilled log file"""
    if not TRADES_LOG_FILE.exists():
        return None

    try:
        entries = {}
        exits = {}

        # Read all trades from log
        with open(TRADES_LOG_FILE, 'r') as f:
            for line in f:
                event = json.loads(line.strip())
                if event['event'] == 'entry':
                    entries[event['trade_id']] = event
                elif event['event'] == 'exit':
                    exits[event['trade_id']] = event

        # Match entries with exits
        completed_trades = []
        for trade_id, exit_event in exits.items():
            if trade_id in entries:
                entry = entries[trade_id]
                completed_trades.append({
                    'symbol': entry['symbol'],
                    'side': entry['side'],
                    'entry_price': entry['price'],
                    'exit_price': exit_event['price'],
                    'qty': entry['qty'],
                    'pnl': exit_event['outcome']['pnl_dollars'],
                    'pnl_pct': exit_event['outcome']['pnl_percent'] * 100,
                    'entry_time': datetime.fromisoformat(entry['timestamp']),
                    'exit_time': datetime.fromisoformat(exit_event['timestamp'])
                })

        if not completed_trades:
            return None

        # Calculate statistics
        wins = [t for t in completed_trades if t['pnl'] > 0]
        losses = [t for t in completed_trades if t['pnl'] <= 0]

        total_pnl = sum(t['pnl'] for t in completed_trades)
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0

        best_trade = max(completed_trades, key=lambda x: x['pnl']) if completed_trades else None
        worst_trade = min(completed_trades, key=lambda x: x['pnl']) if completed_trades else None

        return {
            'total_trades': len(entries),
            'completed_trades': len(completed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(completed_trades) * 100) if completed_trades else 0,
            'total_pnl_from_trades': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'recent_trades': sorted(completed_trades, key=lambda x: x['exit_time'], reverse=True)[:10],
            'data_source': 'log_file'
        }
    except Exception as e:
        return None

def get_alpaca_trade_stats():
    """Get real trade statistics from Alpaca orders"""
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            return None

        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        client = TradingClient(api_key, secret_key, paper=True)

        # Get filled orders
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=500,
            nested=True
        )

        orders = client.get_orders(request)
        filled_orders = [o for o in orders if o.status == 'filled']
        filled_orders.sort(key=lambda x: x.filled_at)

        # Match trades using FIFO
        completed_trades = []
        positions = defaultdict(list)

        for order in filled_orders:
            symbol = order.symbol
            side = order.side.value
            qty = float(order.filled_qty)
            price = float(order.filled_avg_price)
            filled_at = order.filled_at

            if side == 'buy':
                positions[symbol].append({
                    'entry_price': price,
                    'qty': qty,
                    'entry_time': filled_at,
                    'side': 'long'
                })
            elif side == 'sell':
                remaining_qty = qty

                while remaining_qty > 0 and positions[symbol]:
                    position = positions[symbol][0]

                    if position['side'] == 'long':
                        close_qty = min(remaining_qty, position['qty'])

                        pnl = (price - position['entry_price']) * close_qty
                        pnl_pct = ((price - position['entry_price']) / position['entry_price']) * 100

                        completed_trades.append({
                            'symbol': symbol,
                            'side': 'long',
                            'entry_price': position['entry_price'],
                            'exit_price': price,
                            'qty': close_qty,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'entry_time': position['entry_time'],
                            'exit_time': filled_at
                        })

                        position['qty'] -= close_qty
                        remaining_qty -= close_qty

                        if position['qty'] <= 0:
                            positions[symbol].pop(0)
                    else:
                        break

                if remaining_qty > 0:
                    positions[symbol].append({
                        'entry_price': price,
                        'qty': remaining_qty,
                        'entry_time': filled_at,
                        'side': 'short'
                    })

        # Calculate statistics
        if not completed_trades:
            return None

        wins = [t for t in completed_trades if t['pnl'] > 0]
        losses = [t for t in completed_trades if t['pnl'] <= 0]

        total_pnl_trades = sum(t['pnl'] for t in completed_trades)
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0

        best_trade = max(completed_trades, key=lambda x: x['pnl']) if completed_trades else None
        worst_trade = min(completed_trades, key=lambda x: x['pnl']) if completed_trades else None

        return {
            'total_trades': len(filled_orders),
            'completed_trades': len(completed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': (len(wins) / len(completed_trades) * 100) if completed_trades else 0,
            'total_pnl_from_trades': total_pnl_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'recent_trades': sorted(completed_trades, key=lambda x: x['exit_time'], reverse=True)[:10]
        }
    except Exception as e:
        return None

def get_equity_curve():
    """Get equity curve from Alpaca portfolio history"""
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            return {'dates': [], 'equity': []}

        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        from datetime import datetime

        client = TradingClient(api_key, secret_key, paper=True)

        request = GetPortfolioHistoryRequest(
            period='1M',
            timeframe='1D'
        )

        history = client.get_portfolio_history(request)

        if not history.timestamp:
            return {'dates': [], 'equity': []}

        dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in history.timestamp]
        equity_points = history.equity

        return {
            'dates': dates,
            'equity': equity_points,
            'starting_equity': equity_points[0] if equity_points else 100000,
            'current_equity': equity_points[-1] if equity_points else 0,
            'total_pnl': history.profit_loss[-1] if history.profit_loss else 0
        }
    except Exception:
        return {'dates': [], 'equity': []}


def get_symbol_performance():
    """Get performance metrics grouped by symbol"""
    if not TRADES_LOG_FILE.exists():
        return []

    try:
        entries = {}
        exits = {}

        with open(TRADES_LOG_FILE, 'r') as f:
            for line in f:
                event = json.loads(line.strip())
                if event['event'] == 'entry':
                    entries[event['trade_id']] = event
                elif event['event'] == 'exit':
                    exits[event['trade_id']] = event

        # Group trades by symbol
        symbol_trades = defaultdict(list)
        for trade_id, exit_event in exits.items():
            if trade_id in entries:
                entry = entries[trade_id]
                symbol_trades[entry['symbol']].append({
                    'pnl': exit_event['outcome']['pnl_dollars'],
                    'pnl_pct': exit_event['outcome']['pnl_percent'] * 100
                })

        # Calculate metrics per symbol
        performance = []
        for symbol, trades in symbol_trades.items():
            wins = [t for t in trades if t['pnl'] > 0]
            total_pnl = sum(t['pnl'] for t in trades)
            win_rate = (len(wins) / len(trades) * 100) if trades else 0
            avg_pnl = total_pnl / len(trades) if trades else 0

            performance.append({
                'symbol': symbol,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl
            })

        # Sort by total P&L descending
        performance.sort(key=lambda x: x['total_pnl'], reverse=True)
        return performance

    except Exception:
        return []

def get_daily_pnl():
    """Get daily P&L for bar chart"""
    if not TRADES_LOG_FILE.exists():
        return {'dates': [], 'pnl': []}

    try:
        entries = {}
        exits = {}

        with open(TRADES_LOG_FILE, 'r') as f:
            for line in f:
                event = json.loads(line.strip())
                if event['event'] == 'entry':
                    entries[event['trade_id']] = event
                elif event['event'] == 'exit':
                    exits[event['trade_id']] = event

        # Group by exit date
        daily_pnl = defaultdict(float)
        for trade_id, exit_event in exits.items():
            if trade_id in entries:
                exit_date = datetime.fromisoformat(exit_event['timestamp']).date()
                daily_pnl[exit_date] += exit_event['outcome']['pnl_dollars']

        # Sort by date and prepare chart data
        sorted_dates = sorted(daily_pnl.keys())

        # Take last 30 days
        recent_dates = sorted_dates[-30:] if len(sorted_dates) > 30 else sorted_dates

        return {
            'dates': [d.strftime('%Y-%m-%d') for d in recent_dates],
            'pnl': [daily_pnl[d] for d in recent_dates]
        }

    except Exception:
        return {'dates': [], 'pnl': []}

def get_recent_errors(n=10):
    """Get recent ERROR and WARNING log entries"""
    if not LOG_FILE.exists():
        return []

    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()

        errors = []
        for line in reversed(lines[-200:]):  # Check last 200 lines
            parsed = parse_log_line(line)
            if parsed['level'] in ['ERROR', 'WARNING']:
                errors.append(parsed)
                if len(errors) >= n:
                    break

        return errors
    except Exception:
        return []

def get_market_status():
    """Determine current market status"""
    try:
        from datetime import datetime
        import pytz

        # Get current time in ET
        et_tz = pytz.timezone('America/New_York')
        now_et = datetime.now(et_tz)

        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        weekday = now_et.weekday()  # 0=Monday, 6=Sunday

        if weekday >= 5:  # Weekend
            return 'closed'

        current_time = now_et.time()

        # Pre-market: 4:00 AM - 9:30 AM
        from datetime import time
        if time(4, 0) <= current_time < time(9, 30):
            return 'pre-market'

        # Market open: 9:30 AM - 4:00 PM
        elif time(9, 30) <= current_time < time(16, 0):
            return 'open'

        # After-hours: 4:00 PM - 8:00 PM
        elif time(16, 0) <= current_time < time(20, 0):
            return 'after-hours'

        # Closed
        else:
            return 'closed'

    except Exception:
        return 'unknown'

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Amsterdam Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 {
            font-size: 28px;
            margin-bottom: 8px;
            color: #fff;
        }
        .subtitle {
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 20px;
        }
        .card h2 {
            font-size: 14px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        .status-value {
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .status-running { color: #10b981; }
        .status-stopped { color: #ef4444; }
        .status-unknown { color: #888; }
        .state-value {
            font-size: 24px;
            color: #3b82f6;
        }
        .next-market {
            font-size: 18px;
            color: #fbbf24;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .metric-label { color: #888; }
        .metric-value { color: #fff; font-weight: 500; }
        .big-stat {
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .stat-label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .progress-bar {
            height: 8px;
            background: #2a2a2a;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #06b6d4);
            transition: width 0.3s ease;
        }
        .progress-fill.warning { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
        .progress-fill.danger { background: linear-gradient(90deg, #ef4444, #dc2626); }
        .positions-table, .trades-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 12px;
        }
        .positions-table th, .trades-table th {
            text-align: left;
            color: #888;
            font-weight: 500;
            padding: 8px 4px;
            border-bottom: 1px solid #2a2a2a;
        }
        .positions-table td, .trades-table td {
            padding: 8px 4px;
            border-bottom: 1px solid #2a2a2a;
        }
        .pnl-positive { color: #10b981; }
        .pnl-negative { color: #ef4444; }
        .symbols-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(70px, 1fr));
            gap: 8px;
            margin-top: 12px;
        }
        .symbol-badge {
            background: #2a2a2a;
            padding: 6px;
            border-radius: 6px;
            text-align: center;
            font-size: 12px;
            font-weight: 500;
        }
        .log-container {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }
        .log-entry {
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            padding: 8px;
            border-bottom: 1px solid #2a2a2a;
            line-height: 1.5;
        }
        .log-entry:last-child { border-bottom: none; }
        .log-timestamp { color: #666; }
        .log-info { color: #3b82f6; }
        .log-warning { color: #fbbf24; }
        .log-error { color: #ef4444; }
        .log-message { color: #e0e0e0; }
        .updated {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 20px;
        }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .data-source {
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }
        @media (max-width: 1200px) {
            .wide-card { grid-column: span 1; }
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 900px) {
            .best-worst-grid { grid-template-columns: 1fr !important; }
        }
        .market-status {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .market-open { background: #10b981; color: white; }
        .market-closed { background: #6b7280; color: white; }
        .market-pre { background: #3b82f6; color: white; }
        .market-after { background: #8b5cf6; color: white; }
        .performance-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 12px;
        }
        .performance-table th {
            text-align: left;
            color: #888;
            font-weight: 500;
            padding: 8px 4px;
            border-bottom: 1px solid #2a2a2a;
        }
        .performance-table td {
            padding: 8px 4px;
            border-bottom: 1px solid #2a2a2a;
        }
        .error-entry {
            padding: 10px;
            margin-bottom: 8px;
            background: #2a2a2a;
            border-radius: 6px;
            border-left: 3px solid #ef4444;
            font-size: 12px;
        }
        .warning-entry {
            padding: 10px;
            margin-bottom: 8px;
            background: #2a2a2a;
            border-radius: 6px;
            border-left: 3px solid #fbbf24;
            font-size: 12px;
        }
    </style>
    <script>
        setTimeout(() => location.reload(), 10000);
    </script>
</head>
<body>
    <div class="container">
        <h1>Amsterdam Trading Dashboard</h1>
        <div class="subtitle">Live monitoring • Auto-refresh every 10s • Data from Alpaca API</div>

        <div class="grid">
            <div class="card">
                <h2>Status</h2>
                <div class="status-value status-{{ status_class }}">{{ status }}</div>
            </div>

            <div class="card">
                <h2>Current State</h2>
                <div class="state-value">{{ state }}</div>
            </div>

            <div class="card">
                <h2>Next Market Open</h2>
                <div class="next-market">{{ next_open }}</div>
            </div>

            {% if stats %}
            <div class="card">
                <h2>Completed Trades</h2>
                <div class="big-stat">{{ stats.completed_trades }}</div>
                <div class="stat-label">From Alpaca API</div>
                <div class="data-source">Last 500 orders analyzed</div>
            </div>

            <div class="card">
                <h2>Win Rate</h2>
                <div class="big-stat {% if stats.win_rate >= 50 %}pnl-positive{% else %}pnl-negative{% endif %}">
                    {{ "%.1f"|format(stats.win_rate) }}%
                </div>
                <div class="stat-label">{{ stats.wins }}W / {{ stats.losses }}L</div>
                <div class="data-source">Real broker data</div>
            </div>
            {% endif %}

            {% if broker %}
            <div class="card">
                <h2>Total P&L</h2>
                <div class="big-stat {% if (broker.equity - 100000) >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
                    ${{ "%.2f"|format(broker.equity - 100000) }}
                </div>
                <div class="stat-label">{{ "%.2f"|format((broker.equity - 100000) / 100000 * 100) }}% Return</div>
                <div class="data-source">From $100k start</div>
            </div>
            {% endif %}

            <div class="card">
                <h2>System Health</h2>
                <div class="metric-row">
                    <span class="metric-label">CPU</span>
                    <span class="metric-value">{{ health.cpu_percent }}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill {% if health.cpu_percent > 80 %}danger{% elif health.cpu_percent > 60 %}warning{% endif %}"
                         style="width: {{ health.cpu_percent }}%"></div>
                </div>

                <div class="metric-row" style="margin-top: 16px;">
                    <span class="metric-label">Memory</span>
                    <span class="metric-value">{{ health.memory_used_gb }}GB / {{ health.memory_total_gb }}GB</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill {% if health.memory_percent > 80 %}danger{% elif health.memory_percent > 60 %}warning{% endif %}"
                         style="width: {{ health.memory_percent }}%"></div>
                </div>

                <div class="metric-row" style="margin-top: 16px;">
                    <span class="metric-label">Disk</span>
                    <span class="metric-value">{{ health.disk_used_gb }}GB / {{ health.disk_total_gb }}GB</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill {% if health.disk_percent > 80 %}danger{% elif health.disk_percent > 60 %}warning{% endif %}"
                         style="width: {{ health.disk_percent }}%"></div>
                </div>
            </div>

            {% if broker %}
            <div class="card">
                <h2>Account</h2>
                <div class="metric-row">
                    <span class="metric-label">Equity</span>
                    <span class="metric-value">${{ "%.2f"|format(broker.equity) }}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Cash</span>
                    <span class="metric-value">${{ "%.2f"|format(broker.cash) }}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Buying Power</span>
                    <span class="metric-value">${{ "%.2f"|format(broker.buying_power) }}</span>
                </div>
                <div class="metric-row" style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #2a2a2a;">
                    <span class="metric-label">Unrealized P&L</span>
                    <span class="metric-value {% if broker.total_unrealized_pnl >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
                        ${{ "%.2f"|format(broker.total_unrealized_pnl) }}
                    </span>
                </div>
            </div>
            {% endif %}

            <div class="card">
                <h2>Active Symbols ({{ symbols|length }})</h2>
                <div class="symbols-grid">
                    {% for symbol in symbols %}
                    <div class="symbol-badge">{{ symbol }}</div>
                    {% endfor %}
                </div>
            </div>

            <div class="card">
                <h2>Market Status</h2>
                <div style="margin-top: 12px;">
                    <span class="market-status market-{{ market_status }}">
                        {% if market_status == 'open' %}🟢 OPEN
                        {% elif market_status == 'pre-market' %}🔵 PRE-MARKET
                        {% elif market_status == 'after-hours' %}🟣 AFTER-HOURS
                        {% elif market_status == 'closed' %}⚫ CLOSED
                        {% else %}❓ UNKNOWN
                        {% endif %}
                    </span>
                </div>
            </div>

            {% if stats and stats.profit_factor %}
            <div class="card">
                <h2>Profit Factor</h2>
                <div class="big-stat {% if stats.profit_factor >= 1.5 %}pnl-positive{% elif stats.profit_factor >= 1.0 %}{% else %}pnl-negative{% endif %}">
                    {{ "%.2f"|format(stats.profit_factor) }}
                </div>
                <div class="stat-label">Avg Win / Avg Loss</div>
                <div class="data-source">{{ "%.0f"|format(stats.profit_factor >= 1.5 and 5 or (stats.profit_factor >= 1.0 and 3 or 1)) }}/5 rating</div>
            </div>
            {% endif %}
        </div>

        

        {% if symbol_performance %}
        <div class="card" style="margin-bottom: 30px;">
            <h2>Symbol Performance (All-Time)</h2>
            <table class="performance-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>Total P&L</th>
                        <th>Avg P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {% for perf in symbol_performance %}
                    <tr>
                        <td><strong>{{ perf.symbol }}</strong></td>
                        <td>{{ perf.total_trades }}</td>
                        <td class="{% if perf.win_rate >= 50 %}pnl-positive{% else %}pnl-negative{% endif %}">
                            {{ "%.1f"|format(perf.win_rate) }}%
                        </td>
                        <td class="{% if perf.total_pnl >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
                            ${{ "%.2f"|format(perf.total_pnl) }}
                        </td>
                        <td class="{% if perf.avg_pnl >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
                            ${{ "%.2f"|format(perf.avg_pnl) }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        {% if daily_pnl and daily_pnl.dates %}
        <div class="card" style="margin-bottom: 30px;">
            <h2>Daily P&L (Last 30 Days)</h2>
            <div class="chart-container">
                <canvas id="dailyPnlChart"></canvas>
            </div>
        </div>
        {% endif %}

        <!-- Emergency Controls -->
        <div class="card" style="margin: 30px 0; border: 2px solid #ef4444;">
            <h2 style="color: #ef4444;">⚠️ Emergency Controls</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px;">
                <button onclick="emergencyStopBot()" style="background: #ef4444; color: white; padding: 15px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 14px;">
                    🛑 STOP BOT
                </button>
                <button onclick="closeAllPositions()" style="background: #f59e0b; color: white; padding: 15px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 14px;">
                    📤 CLOSE ALL POSITIONS
                </button>
            </div>
            <div id="emergency-status" style="margin-top: 10px; padding: 10px; background: #2a2a2a; border-radius: 6px; display: none;"></div>
        </div>

        {% if stats and stats.best_trade and stats.worst_trade %}
        <!-- Best/Worst Trades -->
        <div class="best-worst-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 0 0 30px 0;">
            <div class="card" style="border: 2px solid #10b981;">
                <h2 style="color: #10b981;">🏆 Best Trade</h2>
                <div style="margin-top: 15px;">
                    <div style="font-size: 24px; font-weight: 700; color: #10b981; margin-bottom: 8px;">
                        ${{ "%.2f"|format(stats.best_trade.pnl) }}
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Symbol</span>
                        <span class="metric-value">{{ stats.best_trade.symbol }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Return</span>
                        <span class="metric-value">{{ "%.2f"|format(stats.best_trade.pnl_pct) }}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Entry</span>
                        <span class="metric-value">${{ "%.2f"|format(stats.best_trade.entry_price) }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Exit</span>
                        <span class="metric-value">${{ "%.2f"|format(stats.best_trade.exit_price) }}</span>
                    </div>
                </div>
            </div>

            <div class="card" style="border: 2px solid #ef4444;">
                <h2 style="color: #ef4444;">📉 Worst Trade</h2>
                <div style="margin-top: 15px;">
                    <div style="font-size: 24px; font-weight: 700; color: #ef4444; margin-bottom: 8px;">
                        ${{ "%.2f"|format(stats.worst_trade.pnl) }}
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Symbol</span>
                        <span class="metric-value">{{ stats.worst_trade.symbol }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Return</span>
                        <span class="metric-value">{{ "%.2f"|format(stats.worst_trade.pnl_pct) }}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Entry</span>
                        <span class="metric-value">${{ "%.2f"|format(stats.worst_trade.entry_price) }}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Exit</span>
                        <span class="metric-value">${{ "%.2f"|format(stats.worst_trade.exit_price) }}</span>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        {% if equity_curve and equity_curve.dates %}
        <div class="card" style="margin-bottom: 30px;">
            <h2>Equity Curve (Last 30 Days)</h2>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>
        {% endif %}

        {% if broker and broker.positions %}
        <div class="card" style="margin-bottom: 30px;">
            <h2>Open Positions ({{ broker.total_positions }})</h2>
            <table class="positions-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Qty</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>Value</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                    </tr>
                </thead>
                <tbody>
                    {% for pos in broker.positions %}
                    <tr>
                        <td><strong>{{ pos.symbol }}</strong></td>
                        <td>{{ pos.side }}</td>
                        <td>{{ pos.qty }}</td>
                        <td>${{ "%.2f"|format(pos.entry_price) }}</td>
                        <td>${{ "%.2f"|format(pos.current_price) }}</td>
                        <td>${{ "%.2f"|format(pos.market_value) }}</td>
                        <td class="{% if pos.unrealized_pnl >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
                            ${{ "%.2f"|format(pos.unrealized_pnl) }}
                        </td>
                        <td class="{% if pos.unrealized_pnl_pct >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
                            {{ "%.2f"|format(pos.unrealized_pnl_pct) }}%
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        {% if stats and stats.recent_trades %}
        <div class="card" style="margin-bottom: 30px;">
            <h2>Recent Trades (Last 10 from Alpaca)</h2>
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Qty</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Exit Time</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in stats.recent_trades %}
                    <tr>
                        <td><strong>{{ trade.symbol }}</strong></td>
                        <td>{{ trade.side|upper }}</td>
                        <td>{{ trade.qty|int }}</td>
                        <td>${{ "%.2f"|format(trade.entry_price) }}</td>
                        <td>${{ "%.2f"|format(trade.exit_price) }}</td>
                        <td class="{% if trade.pnl >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
                            ${{ "%.2f"|format(trade.pnl) }}
                        </td>
                        <td class="{% if trade.pnl_pct >= 0 %}pnl-positive{% else %}pnl-negative{% endif %}">
                            {{ "%.2f"|format(trade.pnl_pct) }}%
                        </td>
                        <td>{{ trade.exit_time.strftime('%Y-%m-%d %H:%M') }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        {% if recent_errors %}
        <div class="card" style="margin-bottom: 30px; border: 2px solid #ef4444;">
            <h2 style="color: #ef4444;">⚠️ Recent Errors & Warnings</h2>
            <div style="margin-top: 15px;">
                {% for error in recent_errors %}
                <div class="{% if error.level == 'ERROR' %}error-entry{% else %}warning-entry{% endif %}">
                    <div style="color: #888; font-size: 11px; margin-bottom: 4px;">{{ error.timestamp }}</div>
                    <div style="color: {% if error.level == 'ERROR' %}#ef4444{% else %}#fbbf24{% endif %}; font-weight: 600; margin-bottom: 4px;">{{ error.level }}</div>
                    <div>{{ error.message }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <div class="log-container">
            <h2 style="margin-bottom: 15px;">Recent Activity</h2>
            {% for log in logs %}
            <div class="log-entry">
                <span class="log-timestamp">{{ log.timestamp }}</span>
                <span class="log-{{ log.level|lower }}">{{ log.level }}</span>
                <span class="log-message">{{ log.message }}</span>
            </div>
            {% endfor %}
        </div>

        <div class="updated">Last updated: {{ now }} • All trade data from Alpaca API</div>
    </div>

    {% if equity_curve and equity_curve.dates %}
    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: {{ equity_curve.dates | tojson }},
                datasets: [{
                    label: 'Account Equity',
                    data: {{ equity_curve.equity | tojson }},
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: '#2a2a2a'
                        },
                        ticks: {
                            color: '#888',
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: '#2a2a2a'
                        },
                        ticks: {
                            color: '#888',
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
    </script>
    {% endif %}

    {% if daily_pnl and daily_pnl.dates %}
    <script>
        const dailyCtx = document.getElementById('dailyPnlChart').getContext('2d');
        const dailyPnlData = {{ daily_pnl.pnl | tojson }};
        new Chart(dailyCtx, {
            type: 'bar',
            data: {
                labels: {{ daily_pnl.dates | tojson }},
                datasets: [{
                    label: 'Daily P&L',
                    data: dailyPnlData,
                    backgroundColor: dailyPnlData.map(val => val >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)'),
                    borderColor: dailyPnlData.map(val => val >= 0 ? '#10b981' : '#ef4444'),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: '#2a2a2a'
                        },
                        ticks: {
                            color: '#888',
                            callback: function(value) {
                                return '$' + value.toFixed(0);
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: '#2a2a2a'
                        },
                        ticks: {
                            color: '#888',
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
    </script>
    {% endif %}

    <script>
        function emergencyStopBot() {
            if (!confirm('⚠️ STOP THE TRADING BOT?\n\nThis will send a stop signal to the bot.')) {
                return;
            }

            fetch('/api/emergency/stop-bot', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    const status = document.getElementById('emergency-status');
                    status.style.display = 'block';
                    status.style.background = data.success ? '#10b981' : '#ef4444';
                    status.textContent = data.message;
                    setTimeout(() => location.reload(), 2000);
                })
                .catch(err => alert('Error: ' + err));
        }

        function closeAllPositions() {
            if (!confirm('⚠️ CLOSE ALL POSITIONS?\n\nThis will market sell all open positions immediately.')) {
                return;
            }

            fetch('/api/emergency/close-all', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    const status = document.getElementById('emergency-status');
                    status.style.display = 'block';
                    status.style.background = data.success ? '#10b981' : '#ef4444';
                    status.textContent = data.message + (data.closed_count ? ` (${data.closed_count} positions)` : '');
                    setTimeout(() => location.reload(), 3000);
                })
                .catch(err => alert('Error: ' + err));
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    status = get_status()
    state = get_current_state()
    next_open = get_next_market_open()
    recent_logs = get_recent_logs(15)
    health = get_system_health()
    broker = get_broker_info()
    symbols = get_active_symbols()
    stats = get_trade_stats_from_log()  # Read from backfilled log file
    equity_curve = get_equity_curve()
    symbol_performance = get_symbol_performance()
    daily_pnl = get_daily_pnl()
    recent_errors = get_recent_errors(10)
    market_status = get_market_status()

    status_class = {
        'Running': 'running',
        'Stopped': 'stopped',
        'Unknown': 'unknown'
    }.get(status, 'unknown')

    logs = [parse_log_line(line) for line in recent_logs]

    return render_template_string(
        HTML_TEMPLATE,
        status=status,
        status_class=status_class,
        state=state,
        next_open=next_open,
        logs=logs,
        health=health,
        broker=broker,
        symbols=symbols,
        stats=stats,
        equity_curve=equity_curve,
        symbol_performance=symbol_performance,
        daily_pnl=daily_pnl,
        recent_errors=recent_errors,
        market_status=market_status,
        now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

@app.route('/api/status')
def api_status():
    """JSON API endpoint"""
    return jsonify({
        'status': get_status(),
        'state': get_current_state(),
        'next_market_open': get_next_market_open(),
        'health': get_system_health(),
        'broker': get_broker_info(),
        'symbols': get_active_symbols(),
        'stats': get_trade_stats_from_log(),
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# EMERGENCY CONTROL ENDPOINTS
# ============================================================================

@app.route('/api/emergency/stop-bot', methods=['POST'])
def emergency_stop_bot():
    """Emergency stop the trading bot"""
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 15)  # SIGTERM
            return jsonify({'success': True, 'message': 'Bot stop signal sent'})
        return jsonify({'success': False, 'message': 'Bot not running'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/emergency/close-all', methods=['POST'])
def emergency_close_all():
    """Close all open positions"""
    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            return jsonify({'success': False, 'message': 'API credentials not found'})

        from alpaca.trading.client import TradingClient

        client = TradingClient(api_key, secret_key, paper=True)
        result = client.close_all_positions(cancel_orders=True)

        return jsonify({
            'success': True,
            'message': 'Closed all positions',
            'closed_count': len(result) if result else 0
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
