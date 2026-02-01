#!/usr/bin/env python
"""
AutoTrader Control Script

Manage the autotrader daemon:
- Start/stop the daemon
- Check status
- View logs

Usage:
    python autotrader_ctl.py start [--symbols AAPL MSFT] [--dry-run]
    python autotrader_ctl.py stop
    python autotrader_ctl.py status
    python autotrader_ctl.py logs [-n 50]
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

    print(f"Starting AutoTrader...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Symbols: {args.symbols or ['AAPL', 'MSFT']}")
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

    # Status command
    status_parser = subparsers.add_parser('status', help='Show autotrader status')
    status_parser.set_defaults(func=status)

    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View logs')
    logs_parser.add_argument('-n', '--lines', type=int, default=50, help='Number of lines')
    logs_parser.add_argument('-f', '--follow', action='store_true', help='Follow log output')
    logs_parser.add_argument('--main', action='store_true', help='Show main log instead of stdout')
    logs_parser.set_defaults(func=logs)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
