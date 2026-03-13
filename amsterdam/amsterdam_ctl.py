#!/usr/bin/env python
"""
AutoTrader Control Script

Manage the autotrader daemon and symbol lists:
- Start/stop the daemon
- Check status
- View logs
- Manage trade and watch lists

Usage:
    python autotrader_ctl.py start [--symbols AAPL MSFT] [--dry-run]
    python autotrader_ctl.py stop
    python autotrader_ctl.py status
    python autotrader_ctl.py logs [-n 50]

Symbol List Management:
    python autotrader_ctl.py list                    # Show all symbols
    python autotrader_ctl.py list --trade            # Show trade list only
    python autotrader_ctl.py list --watch            # Show watch list only
    python autotrader_ctl.py add TSLA --trade        # Add to trade list
    python autotrader_ctl.py add NVDA --watch        # Add to watch list
    python autotrader_ctl.py move TSLA --to-watch    # Move to watch list
    python autotrader_ctl.py move NVDA --to-trade    # Move to trade list
    python autotrader_ctl.py remove AAPL             # Remove from all lists
"""

from __future__ import annotations

import os
import sys
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / "logs" / "autotrader.pid"
LOG_FILE = ROOT / "logs" / "autotrader.log"
STDOUT_LOG = ROOT / "logs" / "autotrader_stdout.log"

ET = ZoneInfo("America/New_York")


def get_pid() -> int | None:
    """Get the PID of running autotrader."""
    if not PID_FILE.exists():
        return None

    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process is actually running
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # PID file exists but process is not running
        PID_FILE.unlink(missing_ok=True)
        return None


def is_running() -> bool:
    """Check if autotrader is running."""
    return get_pid() is not None


def start(args) -> int:
    """Start the autotrader daemon."""
    if is_running():
        print(f"AutoTrader is already running (PID: {get_pid()})")
        return 1

    # Ensure logs directory exists
    (ROOT / "logs").mkdir(exist_ok=True)

    # Build command
    cmd = [sys.executable, str(ROOT / "autotrader.py")]

    if args.symbols:
        cmd.extend(["--symbols"] + args.symbols)

    if args.broker:
        cmd.extend(["--broker", args.broker])

    if args.dry_run:
        cmd.append("--dry-run")

    # Get symbols to display
    if args.symbols:
        display_symbols = args.symbols
    else:
        from core.symbol_list_manager import get_list_manager
        display_symbols = get_list_manager().get_trade_list() or ['(empty trade list)']

    print(f"Starting AutoTrader...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Symbols: {display_symbols}")
    print(f"  Broker: {args.broker or 'alpaca'}")
    print(f"  Dry run: {args.dry_run}")

    # Start as background process
    with open(STDOUT_LOG, 'a') as stdout_file:
        stdout_file.write(f"\n{'='*60}\n")
        stdout_file.write(f"AutoTrader started at {datetime.now(ET)}\n")
        stdout_file.write(f"Command: {' '.join(cmd)}\n")
        stdout_file.write(f"{'='*60}\n\n")
        stdout_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            start_new_session=True,  # Detach from terminal
        )

    # Save PID
    PID_FILE.write_text(str(process.pid))

    print(f"\nAutoTrader started (PID: {process.pid})")
    print(f"  Log file: {LOG_FILE}")
    print(f"  Stdout: {STDOUT_LOG}")
    print(f"\nMonitor with: python autotrader_ctl.py logs -f")

    return 0


def stop(args) -> int:
    """Stop the autotrader daemon."""
    pid = get_pid()

    if pid is None:
        print("AutoTrader is not running")
        return 1

    print(f"Stopping AutoTrader (PID: {pid})...")

    try:
        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)

        # Wait for process to stop
        import time
        for _ in range(30):  # Wait up to 30 seconds
            try:
                os.kill(pid, 0)
                time.sleep(1)
            except ProcessLookupError:
                break
        else:
            # Force kill if still running
            print("Process didn't stop gracefully, sending SIGKILL...")
            os.kill(pid, signal.SIGKILL)

        PID_FILE.unlink(missing_ok=True)
        print("AutoTrader stopped")
        return 0

    except ProcessLookupError:
        print("Process already stopped")
        PID_FILE.unlink(missing_ok=True)
        return 0
    except PermissionError:
        print(f"Permission denied stopping PID {pid}")
        return 1


def restart(args) -> int:
    """Restart the autotrader daemon."""
    print("Restarting AutoTrader...")

    # Stop if running
    if is_running():
        stop_result = stop(args)
        if stop_result != 0:
            print("Failed to stop AutoTrader")
            return stop_result
        import time
        time.sleep(1)  # Brief pause between stop and start

    # Start
    return start(args)


def status(args) -> int:
    """Show autotrader status."""
    pid = get_pid()

    print(f"\n{'='*50}")
    print("  AUTOTRADER STATUS")
    print(f"{'='*50}")

    if pid is None:
        print(f"  Status: STOPPED")
    else:
        print(f"  Status: RUNNING")
        print(f"  PID: {pid}")

    # Show market status
    from autotrader import MarketScheduler
    scheduler = MarketScheduler()
    now = scheduler.now_et()

    print(f"\n  Current time (ET): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Market open: {scheduler.is_market_open()}")
    print(f"  Trading day: {scheduler.is_trading_day()}")

    if not scheduler.is_market_open():
        next_open = scheduler.get_next_market_open()
        print(f"  Next open: {next_open.strftime('%Y-%m-%d %H:%M')} ET")

    # Show positions and P&L from Alpaca
    if args.positions or args.all:
        _show_positions()

    # Show trade/watch lists
    if args.lists or args.all:
        _show_lists_summary()

    # Show recent log entries
    if LOG_FILE.exists():
        print(f"\n  Recent log entries:")
        try:
            lines = LOG_FILE.read_text().splitlines()[-5:]
            for line in lines:
                print(f"    {line}")
        except Exception:
            pass

    print(f"{'='*50}\n")

    return 0


def logs(args) -> int:
    """View autotrader logs."""
    log_file = LOG_FILE if args.main else STDOUT_LOG

    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return 1

    if args.follow:
        # Tail -f mode
        print(f"Following {log_file} (Ctrl+C to stop)...\n")
        try:
            subprocess.run(["tail", "-f", str(log_file)])
        except KeyboardInterrupt:
            print("\n")
        return 0
    else:
        # Show last N lines
        lines = log_file.read_text().splitlines()
        for line in lines[-args.lines:]:
            print(line)
        return 0


# ========== Symbol List Commands ==========

def list_symbols(args) -> int:
    """List symbols in trade and/or watch lists."""
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()

    show_trade = args.trade or (not args.trade and not args.watch)
    show_watch = args.watch or (not args.trade and not args.watch)

    print(f"\n{'='*50}")
    print("  SYMBOL LISTS")
    print(f"{'='*50}")

    if show_trade:
        trade_list = manager.get_trade_list()
        print(f"\n  Trade List ({len(trade_list)} symbols):")
        if trade_list:
            for symbol in trade_list:
                entry = manager.get_symbol(symbol)
                notes = f" - {entry.notes}" if entry and entry.notes else ""
                print(f"    {symbol}{notes}")
        else:
            print("    (empty)")

    if show_watch:
        watch_list = manager.get_watch_list()
        print(f"\n  Watch List ({len(watch_list)} symbols):")
        if watch_list:
            for symbol in watch_list:
                entry = manager.get_symbol(symbol)
                notes = f" - {entry.notes}" if entry and entry.notes else ""
                print(f"    {symbol}{notes}")
        else:
            print("    (empty)")

    print(f"\n{'='*50}\n")
    return 0


def add_symbol(args) -> int:
    """Add a symbol to trade or watch list."""
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()
    symbol = args.symbol.upper()
    notes = args.notes or ""

    if args.watch:
        if manager.add_to_watch_list(symbol, notes):
            print(f"Added {symbol} to watch list")
        else:
            existing = manager.get_list_type(symbol)
            if existing == "watch":
                print(f"{symbol} is already in the watch list")
            else:
                print(f"Moved {symbol} from trade list to watch list")
    else:
        # Default to trade list
        if manager.add_to_trade_list(symbol, notes):
            print(f"Added {symbol} to trade list")
        else:
            existing = manager.get_list_type(symbol)
            if existing == "trade":
                print(f"{symbol} is already in the trade list")
            else:
                print(f"Moved {symbol} from watch list to trade list")

    return 0


def move_symbol(args) -> int:
    """Move a symbol between lists."""
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()
    symbol = args.symbol.upper()

    if not manager.symbol_exists(symbol):
        print(f"Symbol {symbol} not found in any list")
        print(f"Use 'add {symbol} --trade' or 'add {symbol} --watch' first")
        return 1

    if args.to_watch:
        if manager.move_to_watch_list(symbol):
            print(f"Moved {symbol} to watch list")
        else:
            print(f"{symbol} is already in the watch list")
    elif args.to_trade:
        if manager.move_to_trade_list(symbol):
            print(f"Moved {symbol} to trade list")
        else:
            print(f"{symbol} is already in the trade list")
    else:
        print("Specify --to-trade or --to-watch")
        return 1

    return 0


def remove_symbol(args) -> int:
    """Remove a symbol from all lists."""
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()
    symbol = args.symbol.upper()

    if manager.remove_symbol(symbol):
        print(f"Removed {symbol} from lists")
    else:
        print(f"Symbol {symbol} not found in any list")
        return 1

    return 0


def export_lists(args) -> int:
    """Export trade/watch lists to JSON file."""
    import json
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()

    data = {
        "trade_list": manager.get_trade_list(),
        "watch_list": manager.get_watch_list(),
        "exported_at": datetime.now(ET).isoformat(),
    }

    output_file = args.file or "symbol_lists.json"

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported lists to {output_file}")
    print(f"  Trade list: {len(data['trade_list'])} symbols")
    print(f"  Watch list: {len(data['watch_list'])} symbols")

    return 0


def import_lists(args) -> int:
    """Import trade/watch lists from JSON file."""
    import json
    from core.symbol_list_manager import get_list_manager

    input_file = args.file

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return 1

    with open(input_file) as f:
        data = json.load(f)

    manager = get_list_manager()

    # Import trade list
    trade_count = 0
    for symbol in data.get("trade_list", []):
        if manager.add_to_trade_list(symbol):
            trade_count += 1

    # Import watch list
    watch_count = 0
    for symbol in data.get("watch_list", []):
        if manager.add_to_watch_list(symbol):
            watch_count += 1

    print(f"Imported from {input_file}")
    print(f"  Trade list: {trade_count} symbols added")
    print(f"  Watch list: {watch_count} symbols added")

    return 0


def _show_positions():
    """Show current positions from Alpaca."""
    print(f"\n  Positions & P/L:")

    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            print("    (Alpaca credentials not configured)")
            return

        from alpaca.trading.client import TradingClient

        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        positions = client.get_all_positions()

        print(f"    Account Equity: ${float(account.equity):,.2f}")
        print(f"    Buying Power: ${float(account.buying_power):,.2f}")
        print(f"    Cash: ${float(account.cash):,.2f}")

        if positions:
            print(f"\n    Open Positions ({len(positions)}):")
            total_pnl = 0.0
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                total_pnl += pnl
                pnl_pct = float(pos.unrealized_plpc) * 100
                pnl_color = "+" if pnl >= 0 else ""
                print(f"      {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f} | P/L: {pnl_color}${pnl:.2f} ({pnl_color}{pnl_pct:.1f}%)")
            print(f"    Total Unrealized P/L: ${total_pnl:,.2f}")
        else:
            print("    No open positions")

    except Exception as e:
        print(f"    (Error fetching positions: {e})")


def _show_lists_summary():
    """Show trade/watch list summary."""
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()
    trade = manager.get_trade_list()
    watch = manager.get_watch_list()

    print(f"\n  Symbol Lists:")
    print(f"    Trade ({len(trade)}): {', '.join(trade) if trade else '(empty)'}")
    print(f"    Watch ({len(watch)}): {', '.join(watch) if watch else '(empty)'}")


def main():
    parser = argparse.ArgumentParser(
        description='AutoTrader control script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Start command
    start_parser = subparsers.add_parser('start', help='Start the autotrader')
    start_parser.add_argument('--symbols', '-s', nargs='+', help='Symbols to trade')
    start_parser.add_argument('--broker', '-b', choices=['alpaca', 'schwab'], help='Broker to use')
    start_parser.add_argument('--dry-run', action='store_true', help='No actual trading')
    start_parser.set_defaults(func=start)

    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop the autotrader')
    stop_parser.set_defaults(func=stop)

    # Restart command
    restart_parser = subparsers.add_parser('restart', help='Restart the autotrader')
    restart_parser.add_argument('--symbols', '-s', nargs='+', help='Symbols to trade')
    restart_parser.add_argument('--broker', '-b', choices=['alpaca', 'schwab'], help='Broker to use')
    restart_parser.add_argument('--dry-run', action='store_true', help='No actual trading')
    restart_parser.set_defaults(func=restart)

    # Status command
    status_parser = subparsers.add_parser('status', help='Show autotrader status')
    status_parser.add_argument('--positions', '-p', action='store_true', help='Show positions and P&L')
    status_parser.add_argument('--lists', '-l', action='store_true', help='Show trade/watch lists')
    status_parser.add_argument('--all', '-a', action='store_true', help='Show all info')
    status_parser.set_defaults(func=status)

    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View logs')
    logs_parser.add_argument('-n', '--lines', type=int, default=50, help='Number of lines')
    logs_parser.add_argument('-f', '--follow', action='store_true', help='Follow log output')
    logs_parser.add_argument('--main', action='store_true', help='Show main log instead of stdout')
    logs_parser.set_defaults(func=logs)

    # List command (symbol lists)
    list_parser = subparsers.add_parser('list', help='List symbols in trade/watch lists')
    list_parser.add_argument('--trade', '-t', action='store_true', help='Show trade list only')
    list_parser.add_argument('--watch', '-w', action='store_true', help='Show watch list only')
    list_parser.set_defaults(func=list_symbols)

    # Add command
    add_parser = subparsers.add_parser('add', help='Add symbol to a list')
    add_parser.add_argument('symbol', help='Stock symbol to add')
    add_parser.add_argument('--trade', '-t', action='store_true', help='Add to trade list (default)')
    add_parser.add_argument('--watch', '-w', action='store_true', help='Add to watch list')
    add_parser.add_argument('--notes', '-n', help='Notes about the symbol')
    add_parser.set_defaults(func=add_symbol)

    # Move command
    move_parser = subparsers.add_parser('move', help='Move symbol between lists')
    move_parser.add_argument('symbol', help='Stock symbol to move')
    move_parser.add_argument('--to-trade', action='store_true', help='Move to trade list')
    move_parser.add_argument('--to-watch', action='store_true', help='Move to watch list')
    move_parser.set_defaults(func=move_symbol)

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove symbol from all lists')
    remove_parser.add_argument('symbol', help='Stock symbol to remove')
    remove_parser.set_defaults(func=remove_symbol)

    # Export command
    export_parser = subparsers.add_parser('export', help='Export lists to JSON file')
    export_parser.add_argument('file', nargs='?', default='symbol_lists.json', help='Output file (default: symbol_lists.json)')
    export_parser.set_defaults(func=export_lists)

    # Import command
    import_parser = subparsers.add_parser('import', help='Import lists from JSON file')
    import_parser.add_argument('file', help='Input file to import')
    import_parser.set_defaults(func=import_lists)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
