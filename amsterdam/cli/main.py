#!/usr/bin/env python3
"""
Amsterdam CLI - Primary User Interface
======================================

PRIMARY ENTRYPOINT: This is the main user interface for the Amsterdam trading system.
Users interact with the system through this CLI.

CANONICAL PATH HIERARCHY:
    cli/main.py (this file) - User commands
         ↓
    app/bootstrap.py - Creates AppContext (canonical initialization)
         ↓
    app/container.py - Creates AppContainer (composition root)
         ↓
    RunnerFactory - Creates runners
         ↓
    Execution System - Trading loop

Commands delegate to specialized scripts, but all use the same canonical path:
    amsterdam start   -> app/daemon.py (daemon mode, uses bootstrap_app)
    amsterdam gui     -> monitoring/gui_app.py (GUI mode, uses bootstrap_app)
    amsterdam preflight -> preflight.py
    amsterdam token   -> token management

Usage:
    amsterdam start [OPTIONS]      Start the trading daemon
    amsterdam stop                 Stop the trading daemon
    amsterdam status               Check daemon status
    amsterdam gui [OPTIONS]        Launch GUI trading application
    amsterdam preflight [OPTIONS]  Run pre-flight checks
    amsterdam token COMMAND        Token management commands
    amsterdam test [OPTIONS]       Run test suite
    amsterdam logs [OPTIONS]       View logs
"""

import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure project root is in path (required before imports)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import click
from dotenv import load_dotenv

# Use canonical path helper for root path resolution
from core.config_loader import get_app_path

APP_ROOT = get_app_path()

# Load environment (canonical location)
load_dotenv(APP_ROOT / ".env")

# Canonical file locations for daemon process
PID_FILE = APP_ROOT / "logs" / "autotrader.pid"
LOG_FILE = APP_ROOT / "logs" / "autotrader.log"
STDOUT_LOG = APP_ROOT / "logs" / "autotrader_stdout.log"

ET = ZoneInfo("America/New_York")


# =============================================================================
# PROCESS MANAGEMENT UTILITIES
# =============================================================================


def _get_pid() -> int | None:
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


def _is_running() -> bool:
    """Check if autotrader is running."""
    return _get_pid() is not None


def _show_positions():
    """Show current positions from Alpaca."""
    click.echo("\n  Positions & P/L:")

    try:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            click.echo("    (Alpaca credentials not configured)")
            return

        from alpaca.trading.client import TradingClient

        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        positions = client.get_all_positions()

        click.echo(f"    Account Equity: ${float(account.equity):,.2f}")
        click.echo(f"    Buying Power: ${float(account.buying_power):,.2f}")
        click.echo(f"    Cash: ${float(account.cash):,.2f}")

        if positions:
            click.echo(f"\n    Open Positions ({len(positions)}):")
            total_pnl = 0.0
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                total_pnl += pnl
                pnl_pct = float(pos.unrealized_plpc) * 100
                pnl_sign = "+" if pnl >= 0 else ""
                click.echo(
                    f"      {pos.symbol}: {pos.qty} @ ${float(pos.avg_entry_price):.2f} | P/L: {pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_pct:.1f}%)"
                )
            click.echo(f"    Total Unrealized P/L: ${total_pnl:,.2f}")
        else:
            click.echo("    No open positions")

    except Exception as e:
        click.echo(f"    (Error fetching positions: {e})")


def _show_lists_summary():
    """Show trade/watch list summary."""
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()
    trade = manager.get_trade_list()
    watch = manager.get_watch_list()

    click.echo("\n  Symbol Lists:")
    click.echo(f"    Trade ({len(trade)}): {', '.join(trade) if trade else '(empty)'}")
    click.echo(f"    Watch ({len(watch)}): {', '.join(watch) if watch else '(empty)'}")


@click.group()
@click.version_option(version="1.0.0", prog_name="amsterdam")
def cli():
    """Amsterdam - Algorithmic Trading Platform

    A comprehensive trading system with support for Schwab and Alpaca brokers.

    \b
    Quick Start:
        amsterdam preflight        # Check system readiness
        amsterdam gui              # Launch simulation GUI
        amsterdam start            # Start autonomous trading

    \b
    Documentation:
        See docs/index.md for full documentation.
    """
    pass


# =============================================================================
# START COMMAND
# =============================================================================


@cli.command()
@click.option("--symbols", "-s", default=None, help="Comma-separated symbols (e.g., AAPL,MSFT)")
@click.option("--broker", "-b", type=click.Choice(["alpaca", "schwab", "hybrid"]), default="alpaca", help="Broker to use (hybrid = Schwab data + Alpaca execution)")
@click.option("--dry-run", is_flag=True, help="Run without executing real trades")
@click.option("--daemon", "-d", is_flag=True, default=True, help="Run as background daemon (default: True)")
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (not as daemon)")
def start(symbols, broker, dry_run, daemon, foreground):
    """Start the autonomous trading daemon.

    \b
    Examples:
        amsterdam start                           # Start as daemon (default)
        amsterdam start -s AAPL,MSFT              # Specific symbols
        amsterdam start --broker schwab           # Use Schwab
        amsterdam start --broker hybrid           # Schwab data + Alpaca execution
        amsterdam start --dry-run                 # No real trades
        amsterdam start --foreground              # Run in foreground
    """
    import subprocess

    # Check if already running
    if _is_running():
        click.secho(f"AutoTrader is already running (PID: {_get_pid()})", fg="yellow")
        click.echo("Use 'amsterdam restart' to restart, or 'amsterdam stop' first.")
        return

    # Ensure logs directory exists
    (APP_ROOT / "logs").mkdir(exist_ok=True)

    # Build command
    cmd = [sys.executable, str(APP_ROOT / "app" / "daemon.py")]

    if symbols:
        cmd.extend(["--symbols"] + symbols.split(","))
    if broker:
        cmd.extend(["--broker", broker])
    if dry_run:
        cmd.append("--dry-run")

    # Get symbols for display
    if symbols:
        display_symbols = symbols.split(",")
    else:
        from core.symbol_list_manager import get_list_manager

        display_symbols = get_list_manager().get_trade_list() or ["(from config)"]

    if foreground:
        # Run in foreground
        click.echo(f"Starting amsterdam (broker={broker})...")
        click.echo(f"  Symbols: {', '.join(display_symbols)}")
        os.execv(sys.executable, cmd)
    else:
        # Run as background daemon
        click.echo("Starting AutoTrader daemon...")
        click.echo(f"  Symbols: {', '.join(display_symbols)}")
        click.echo(f"  Broker: {broker}")
        click.echo(f"  Dry run: {dry_run}")

        with open(STDOUT_LOG, "a") as stdout_file:
            stdout_file.write(f"\n{'=' * 60}\n")
            stdout_file.write(f"AutoTrader started at {datetime.now(ET)}\n")
            stdout_file.write(f"Command: {' '.join(cmd)}\n")
            stdout_file.write(f"{'=' * 60}\n\n")
            stdout_file.flush()

            process = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                cwd=str(APP_ROOT),
                start_new_session=True,
            )

        # Save PID
        PID_FILE.write_text(str(process.pid))

        click.secho(f"\nAutoTrader started (PID: {process.pid})", fg="green")
        click.echo(f"  Log file: {LOG_FILE}")
        click.echo(f"  Stdout: {STDOUT_LOG}")
        click.echo("\nMonitor with: amsterdam logs -f")


# =============================================================================
# STOP COMMAND
# =============================================================================


@cli.command()
def stop():
    """Stop the trading daemon."""
    import time

    pid = _get_pid()

    if pid is None:
        click.echo("AutoTrader is not running")
        return

    click.echo(f"Stopping AutoTrader (PID: {pid})...")

    try:
        # Send SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)

        # Wait for process to stop
        for _ in range(30):  # Wait up to 30 seconds
            try:
                os.kill(pid, 0)
                time.sleep(1)
            except ProcessLookupError:
                break
        else:
            # Force kill if still running
            click.secho("Process didn't stop gracefully, sending SIGKILL...", fg="yellow")
            os.kill(pid, signal.SIGKILL)

        PID_FILE.unlink(missing_ok=True)
        click.secho("AutoTrader stopped", fg="green")

    except ProcessLookupError:
        click.echo("Process already stopped")
        PID_FILE.unlink(missing_ok=True)
    except PermissionError:
        click.secho(f"Permission denied stopping PID {pid}", fg="red")


# =============================================================================
# STATUS COMMAND
# =============================================================================


@cli.command()
@click.option("--positions", "-p", is_flag=True, help="Show positions and P&L")
@click.option("--lists", "-l", is_flag=True, help="Show trade/watch lists")
@click.option("--all", "-a", "show_all", is_flag=True, help="Show all info")
def status(positions, lists, show_all):
    """Check trading daemon status."""
    pid = _get_pid()

    click.echo(f"\n{'=' * 50}")
    click.echo("  AUTOTRADER STATUS")
    click.echo(f"{'=' * 50}")

    if pid is None:
        click.secho("  Status: STOPPED", fg="red")
    else:
        click.secho("  Status: RUNNING", fg="green")
        click.echo(f"  PID: {pid}")

    # Show market status
    try:
        from app.daemon import MarketScheduler

        scheduler = MarketScheduler()
        now = scheduler.now_et()

        click.echo(f"\n  Current time (ET): {now.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo(f"  Market open: {scheduler.is_market_open()}")
        click.echo(f"  Trading day: {scheduler.is_trading_day()}")

        if not scheduler.is_market_open():
            next_open = scheduler.get_next_market_open()
            click.echo(f"  Next open: {next_open.strftime('%Y-%m-%d %H:%M')} ET")
    except Exception as e:
        click.echo(f"  (Could not get market status: {e})")

    # Show positions and P&L from Alpaca
    if positions or show_all:
        _show_positions()

    # Show trade/watch lists
    if lists or show_all:
        _show_lists_summary()

    # Show recent log entries
    if LOG_FILE.exists():
        click.echo("\n  Recent log entries:")
        try:
            lines = LOG_FILE.read_text().splitlines()[-5:]
            for line in lines:
                click.echo(f"    {line}")
        except Exception:
            pass

    click.echo(f"{'=' * 50}\n")


# =============================================================================
# RESTART COMMAND
# =============================================================================


@cli.command()
@click.option("--symbols", "-s", default=None, help="Comma-separated symbols")
@click.option("--broker", "-b", type=click.Choice(["alpaca", "schwab", "hybrid"]), default="alpaca", help="Broker to use (hybrid = Schwab data + Alpaca execution)")
@click.option("--dry-run", is_flag=True, help="Run without executing real trades")
@click.pass_context
def restart(ctx, symbols, broker, dry_run):
    """Restart the trading daemon."""
    import time

    click.echo("Restarting AutoTrader...")

    # Stop if running
    if _is_running():
        ctx.invoke(stop)
        time.sleep(1)  # Brief pause between stop and start

    # Start with new options
    ctx.invoke(start, symbols=symbols, broker=broker, dry_run=dry_run, daemon=True, foreground=False)


# =============================================================================
# GUI COMMAND
# =============================================================================


@cli.command()
@click.option(
    "--mode", "-m", type=click.Choice(["simulation", "alpaca", "schwab"]), default="simulation", help="Trading mode"
)
@click.option("--symbols", "-s", default="AAPL,MSFT", help="Comma-separated symbols")
@click.option("--speed", type=float, default=0.1, help="Simulation speed (seconds per bar)")
@click.option("--steps", type=int, default=600, help="Number of bars to simulate (default: 600 = ~2 min)")
def gui(mode, symbols, speed, steps):
    """Launch the GUI trading application.

    \b
    Modes:
        simulation  - GBM price simulator (no real trades)
        alpaca      - Alpaca paper/live trading
        schwab      - Schwab live trading

    \b
    Examples:
        amsterdam gui                             # Simulation mode
        amsterdam gui --mode alpaca               # Alpaca trading
        amsterdam gui -s AAPL,GOOGL,TSLA          # Custom symbols
    """
    cmd = [
        sys.executable,
        str(APP_ROOT / "monitoring" / "gui_app.py"),
        "--mode",
        mode,
        "--symbols",
        symbols,
        "--speed",
        str(speed),
        "--steps",
        str(steps),
    ]

    click.echo(f"Launching GUI ({mode} mode, {steps} bars)...")
    os.execv(sys.executable, cmd)


# =============================================================================
# PREFLIGHT COMMAND
# =============================================================================


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--update-data", is_flag=True, help="Update stale historical data")
@click.option("--reauth-schwab", is_flag=True, help="Force Schwab re-authentication")
def preflight(verbose, update_data, reauth_schwab):
    """Run pre-flight system checks.

    Validates:
    - Broker credentials
    - Token expiry status
    - Historical data freshness
    - System configuration

    \b
    Examples:
        amsterdam preflight                       # Quick check
        amsterdam preflight -v                    # Verbose output
        amsterdam preflight --update-data         # Update stale data
    """
    cmd = [sys.executable, str(APP_ROOT / "preflight.py")]

    if verbose:
        cmd.append("-v")
    if update_data:
        cmd.append("--update-data")
    if reauth_schwab:
        cmd.append("--reauth-schwab")

    os.execv(sys.executable, cmd)


# =============================================================================
# TOKEN COMMAND GROUP
# =============================================================================


@cli.group()
def token():
    """Token management commands.

    \b
    Commands:
        amsterdam token status     Check token status
        amsterdam token refresh    Refresh tokens
        amsterdam token keeper     Run token keeper service
    """
    pass


@token.command("status")
def token_status():
    """Check Schwab token status."""
    import asyncio

    from core.credential_validator import CredentialValidator

    async def check():
        validator = CredentialValidator()
        result = await validator.validate_schwab()

        status_colors = {
            "VALID": "green",
            "EXPIRING_SOON": "yellow",
            "EXPIRED": "red",
            "MISSING": "red",
            "INVALID": "red",
        }

        color = status_colors.get(result.status.name, "white")
        click.echo("Schwab Token: ", nl=False)
        click.secho(result.status.name, fg=color, bold=True)
        click.echo(f"  {result.message}")

        # Also check Alpaca
        alpaca_result = await validator.validate_alpaca()
        color = status_colors.get(alpaca_result.status.name, "white")
        click.echo("Alpaca API:   ", nl=False)
        click.secho(alpaca_result.status.name, fg=color, bold=True)
        click.echo(f"  {alpaca_result.message}")

    asyncio.run(check())


@token.command("refresh")
@click.option("--force", "-f", is_flag=True, help="Force full re-authentication")
def token_refresh(force):
    """Refresh Schwab tokens.

    \b
    Examples:
        amsterdam token refresh           # Refresh if needed
        amsterdam token refresh --force   # Force browser login
    """
    cmd = [sys.executable, str(APP_ROOT / "refresh_schwab_token.py")]
    if force:
        cmd.append("--force")

    os.execv(sys.executable, cmd)


@token.command("keeper")
@click.option("--interval", "-i", type=int, default=60, help="Check interval in seconds")
@click.option("--daemon", "-d", is_flag=True, help="Run as background daemon")
def token_keeper(interval, daemon):
    """Run the token keeper service.

    Keeps Schwab tokens fresh by periodically checking and renewing them.

    \b
    Examples:
        amsterdam token keeper                    # Run in foreground
        amsterdam token keeper --daemon           # Background mode
        amsterdam token keeper -i 300             # Check every 5 minutes
    """
    cmd = [sys.executable, str(APP_ROOT / "token_keeper.py"), "--interval", str(interval)]

    if daemon:
        cmd.append("--daemon")

    os.execv(sys.executable, cmd)


# =============================================================================
# TEST COMMAND
# =============================================================================


@cli.command()
@click.option("--coverage", "-c", is_flag=True, help="Run with coverage report")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--unit", is_flag=True, help="Run only unit tests")
@click.option("--integration", is_flag=True, help="Run only integration tests")
@click.argument("path", required=False)
def test(coverage, verbose, unit, integration, path):
    """Run the test suite.

    \b
    Examples:
        amsterdam test                            # Run all tests
        amsterdam test -v                         # Verbose output
        amsterdam test --coverage                 # With coverage report
        amsterdam test tests/test_autotrader.py      # Specific file
    """
    cmd = [sys.executable, "-m", "pytest"]

    if path:
        cmd.append(path)
    else:
        cmd.append("tests/")

    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=core", "--cov=strategies", "--cov-report=html"])

    click.echo("Running tests...")
    os.execv(sys.executable, cmd)


# =============================================================================
# LOGS COMMAND
# =============================================================================


@cli.command()
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--lines", "-n", type=int, default=50, help="Number of lines to show")
@click.option(
    "--file", "-l", type=click.Choice(["app", "trades", "daemon", "preflight"]), default="app", help="Log file to view"
)
def logs(follow, lines, file):
    """View application logs.

    \b
    Log files:
        app         - Main application log
        trades      - Trade execution log
        daemon      - Daemon operations (autotrader.log)
        preflight   - Pre-flight checks

    \b
    Examples:
        amsterdam logs                            # Last 50 lines of app.log
        amsterdam logs -f                         # Follow in real-time
        amsterdam logs -l trades -n 100           # Last 100 trade entries
    """
    import subprocess

    log_files = {"app": "app.log", "trades": "trades.log", "daemon": "autotrader.log", "preflight": "preflight.log"}

    log_path = APP_ROOT / "logs" / log_files[file]

    if not log_path.exists():
        click.echo(f"Log file not found: {log_path}")
        return

    if follow:
        subprocess.run(["tail", "-f", str(log_path)])
    else:
        subprocess.run(["tail", f"-{lines}", str(log_path)])


# =============================================================================
# STATS COMMAND
# =============================================================================


@cli.command()
@click.option("--summary", is_flag=True, help="Show summary only")
@click.option("--by-day", is_flag=True, help="Show daily breakdown")
@click.option("--by-symbol", is_flag=True, help="Show per-symbol stats")
@click.option("--by-hour", is_flag=True, help="Show hourly performance")
@click.option("--by-strategy", is_flag=True, help="Show per-strategy stats")
@click.option("--worst", type=int, metavar="N", help="Show worst N trades")
def stats(summary, by_day, by_symbol, by_hour, by_strategy, worst):
    """Analyze trading performance and win rate.

    \b
    Shows:
        - Overall win rate and PnL
        - Performance by day, symbol, hour, strategy
        - Hold time analysis
        - Open positions

    \b
    Examples:
        amsterdam stats                   # Full analysis
        amsterdam stats --summary         # Quick summary only
        amsterdam stats --by-day          # Daily breakdown
        amsterdam stats --worst 10        # Show 10 worst trades
    """
    import subprocess

    cmd = [sys.executable, str(APP_ROOT / "tools" / "analyze_trades.py")]

    if summary:
        cmd.append("--summary")
    if by_day:
        cmd.append("--by-day")
    if by_symbol:
        cmd.append("--by-symbol")
    if by_hour:
        cmd.append("--by-hour")
    if by_strategy:
        cmd.append("--by-strategy")
    if worst:
        cmd.extend(["--worst", str(worst)])

    subprocess.run(cmd)


# =============================================================================
# SYMBOLS COMMAND GROUP
# =============================================================================


@cli.group()
def symbols():
    """Symbol list management.

    \b
    Commands:
        amsterdam symbols list         Show all symbols
        amsterdam symbols add          Add symbol to list
        amsterdam symbols remove       Remove symbol
        amsterdam symbols move         Move between lists
        amsterdam symbols export       Export lists to JSON
        amsterdam symbols import       Import lists from JSON
    """
    pass


@symbols.command("list")
@click.option("--trade", "-t", is_flag=True, help="Show trade list only")
@click.option("--watch", "-w", is_flag=True, help="Show watch list only")
def symbols_list(trade, watch):
    """List configured symbols."""
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()

    show_trade = trade or (not trade and not watch)
    show_watch = watch or (not trade and not watch)

    click.echo(f"\n{'=' * 50}")
    click.echo("  SYMBOL LISTS")
    click.echo(f"{'=' * 50}")

    if show_trade:
        trade_list = manager.get_trade_list()
        click.echo(f"\n  Trade List ({len(trade_list)} symbols):")
        if trade_list:
            for sym in trade_list:
                entry = manager.get_symbol(sym)
                notes = f" - {entry.notes}" if entry and entry.notes else ""
                click.echo(f"    {sym}{notes}")
        else:
            click.echo("    (empty)")

    if show_watch:
        watch_list = manager.get_watch_list()
        click.echo(f"\n  Watch List ({len(watch_list)} symbols):")
        if watch_list:
            for sym in watch_list:
                entry = manager.get_symbol(sym)
                notes = f" - {entry.notes}" if entry and entry.notes else ""
                click.echo(f"    {sym}{notes}")
        else:
            click.echo("    (empty)")

    click.echo(f"\n{'=' * 50}\n")


@symbols.command("add")
@click.argument("symbol")
@click.option("--trade", "-t", is_flag=True, help="Add to trade list (default)")
@click.option("--watch", "-w", is_flag=True, help="Add to watch list")
@click.option("--notes", "-n", default=None, help="Notes about the symbol")
def symbols_add(symbol, trade, watch, notes):
    """Add a symbol to trade or watch list."""
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()
    symbol = symbol.upper()
    notes = notes or ""

    if watch:
        if manager.add_to_watch_list(symbol, notes):
            click.secho(f"Added {symbol} to watch list", fg="green")
        else:
            existing = manager.get_list_type(symbol)
            if existing == "watch":
                click.echo(f"{symbol} is already in the watch list")
            else:
                click.echo(f"Moved {symbol} from trade list to watch list")
    else:
        # Default to trade list
        if manager.add_to_trade_list(symbol, notes):
            click.secho(f"Added {symbol} to trade list", fg="green")
        else:
            existing = manager.get_list_type(symbol)
            if existing == "trade":
                click.echo(f"{symbol} is already in the trade list")
            else:
                click.echo(f"Moved {symbol} from watch list to trade list")


@symbols.command("remove")
@click.argument("symbol")
def symbols_remove(symbol):
    """Remove a symbol from all lists."""
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()
    symbol = symbol.upper()

    if manager.remove_symbol(symbol):
        click.secho(f"Removed {symbol} from lists", fg="green")
    else:
        click.secho(f"Symbol {symbol} not found in any list", fg="yellow")


@symbols.command("move")
@click.argument("symbol")
@click.option("--to-trade", is_flag=True, help="Move to trade list")
@click.option("--to-watch", is_flag=True, help="Move to watch list")
def symbols_move(symbol, to_trade, to_watch):
    """Move a symbol between lists."""
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()
    symbol = symbol.upper()

    if not manager.symbol_exists(symbol):
        click.secho(f"Symbol {symbol} not found in any list", fg="red")
        click.echo(f"Use 'amsterdam symbols add {symbol} --trade' or '--watch' first")
        return

    if to_watch:
        if manager.move_to_watch_list(symbol):
            click.secho(f"Moved {symbol} to watch list", fg="green")
        else:
            click.echo(f"{symbol} is already in the watch list")
    elif to_trade:
        if manager.move_to_trade_list(symbol):
            click.secho(f"Moved {symbol} to trade list", fg="green")
        else:
            click.echo(f"{symbol} is already in the trade list")
    else:
        click.secho("Specify --to-trade or --to-watch", fg="red")


@symbols.command("export")
@click.argument("file", default="symbol_lists.json", required=False)
def symbols_export(file):
    """Export trade/watch lists to JSON file."""
    import json

    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()

    data = {
        "trade_list": manager.get_trade_list(),
        "watch_list": manager.get_watch_list(),
        "exported_at": datetime.now(ET).isoformat(),
    }

    with open(file, "w") as f:
        json.dump(data, f, indent=2)

    click.secho(f"Exported lists to {file}", fg="green")
    click.echo(f"  Trade list: {len(data['trade_list'])} symbols")
    click.echo(f"  Watch list: {len(data['watch_list'])} symbols")


@symbols.command("import")
@click.argument("file")
def symbols_import(file):
    """Import trade/watch lists from JSON file."""
    import json

    from core.symbol_list_manager import get_list_manager

    if not os.path.exists(file):
        click.secho(f"File not found: {file}", fg="red")
        return

    with open(file) as f:
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

    click.secho(f"Imported from {file}", fg="green")
    click.echo(f"  Trade list: {trade_count} symbols added")
    click.echo(f"  Watch list: {watch_count} symbols added")


# =============================================================================
# DATA COMMAND GROUP
# =============================================================================


@cli.group()
def data():
    """Historical data management.

    \b
    Commands:
        amsterdam data update      Update historical data
        amsterdam data status      Check data freshness
    """
    pass


@data.command("update")
@click.option("--symbols", "-s", default=None, help="Comma-separated symbols")
@click.option("--days", "-d", type=int, default=30, help="Days of history to fetch")
@click.option("--source", type=click.Choice(["alpaca", "schwab", "auto"]), default="auto", help="Data source")
@click.option("--timeframes", "-t", default=None, help="Comma-separated timeframes (e.g., 15min,30min,1hour)")
def data_update(symbols, days, source, timeframes):
    """Update historical data for symbols.

    \b
    Examples:
        amsterdam data update -s AAPL,TSLA                    # Default timeframe
        amsterdam data update -s AAPL -t 15min,30min,1hour    # Multiple timeframes
        amsterdam data update -d 750 -t 1hour --source alpaca # 2 years hourly
    """
    import asyncio

    from core.unified_data_pipeline import UnifiedDataPipeline

    async def update():
        pipeline = UnifiedDataPipeline()

        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
        else:
            # Load from config
            import json

            config_path = APP_ROOT / "config" / "symbols.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                symbol_list = cfg.get("trade_list", []) + cfg.get("watch_list", [])
            else:
                symbol_list = ["AAPL", "MSFT"]

        # Parse timeframes
        timeframe_list = None
        if timeframes:
            timeframe_list = [t.strip() for t in timeframes.split(",")]
            click.echo(f"Updating data for: {', '.join(symbol_list)} at {', '.join(timeframe_list)}")
        else:
            click.echo(f"Updating data for: {', '.join(symbol_list)}")

        src = None if source == "auto" else source
        results = await pipeline.update_symbols(symbol_list, days=days, source=src, timeframes=timeframe_list)

        for sym, count in results.items():
            if count > 0:
                click.secho(f"  {sym}: {count} bars", fg="green")
            else:
                click.secho(f"  {sym}: failed", fg="red")

    asyncio.run(update())


@data.command("status")
@click.option("--symbols", "-s", default=None, help="Comma-separated symbols")
def data_status(symbols):
    """Check data freshness for symbols."""
    from core.unified_data_pipeline import UnifiedDataPipeline

    pipeline = UnifiedDataPipeline()

    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbol_list = pipeline.list_available_symbols()[:10]

    click.echo("Data Status:")
    click.echo("-" * 50)

    for sym in symbol_list:
        info = pipeline.get_cache_info(sym)
        if info:
            age = info.get("age_minutes", 0)
            bars = info.get("bar_count", 0)
            if age < 60:
                color = "green"
            elif age < 1440:
                color = "yellow"
            else:
                color = "red"
            click.echo(f"  {sym}: ", nl=False)
            click.secho(f"{bars} bars, {age:.0f}min old", fg=color)
        else:
            click.echo(f"  {sym}: ", nl=False)
            click.secho("no data", fg="red")


# =============================================================================
# STRATEGY COMMAND GROUP
# =============================================================================


@cli.group()
def strategy():
    """Strategy management commands.

    \b
    Commands:
        amsterdam strategy select          Evaluate and select best strategies
        amsterdam strategy optimize-multitf Optimize across timeframes (multi-TF)
        amsterdam strategy list            List available strategies
        amsterdam strategy show            Show current routing configuration
    """
    pass


@strategy.command("select")
@click.argument("symbol")
@click.option("--days", "-d", type=int, default=365, help="Days of historical data")
@click.option("--top", "-n", type=int, default=3, help="Number of top strategies to select")
@click.option(
    "--metric",
    type=click.Choice(["composite", "sharpe_ratio", "sortino_ratio", "total_return"]),
    default="composite",
    help="Ranking metric",
)
@click.option("--no-walk-forward", is_flag=True, help="Disable walk-forward validation")
@click.option(
    "--regime",
    type=click.Choice(["low_volatility", "normal", "high_volatility"]),
    default="normal",
    help="Market regime to optimize for",
)
@click.option("--save", "-s", is_flag=True, help="Save results to config files")
@click.option("--capital", type=float, default=100000, help="Initial capital for backtesting")
def strategy_select(symbol, days, top, metric, no_walk_forward, regime, save, capital):
    """Evaluate all strategies and select the best performers for a symbol.

    Uses backtesting with walk-forward validation to find optimal strategies,
    then saves results to strategy_routing.json for use in live/simulated trading.

    \b
    Examples:
        amsterdam strategy select AAPL                    # Evaluate strategies for AAPL
        amsterdam strategy select AAPL --save             # Save best to config
        amsterdam strategy select MSFT -d 180 --save      # 180 days of data
        amsterdam strategy select TSLA --regime high_volatility --save
    """
    import asyncio

    from core.backtest.strategy_selector import StrategySelector
    from core.unified_data_pipeline import UnifiedDataPipeline

    symbol = symbol.upper()

    click.echo(f"\nLoading {days} days of data for {symbol}...")

    # Load data
    try:
        pipeline = UnifiedDataPipeline()
        data = pipeline.load_symbol_data(symbol)

        if data is None or data.empty:
            click.echo("No cached data, fetching from source...")

            async def update():
                return await pipeline.update_symbols([symbol], days=days)

            asyncio.run(update())
            data = pipeline.load_symbol_data(symbol)

        if data is None or data.empty:
            click.secho(f"Error: Could not load data for {symbol}", fg="red")
            return

        # Limit to requested days
        if len(data) > days:
            data = data.tail(days).reset_index(drop=True)

        click.echo(f"Loaded {len(data)} bars")

    except Exception as e:
        click.secho(f"Error loading data: {e}", fg="red")
        return

    # Run strategy selection
    click.echo(f"\nEvaluating strategies for {symbol}...")
    click.echo(f"  Metric: {metric}")
    click.echo(f"  Walk-forward: {'disabled' if no_walk_forward else 'enabled'}")
    click.echo(f"  Regime: {regime}")
    click.echo()

    try:
        selector = StrategySelector(data, initial_capital=capital)
        result = selector.select_best_strategies(
            symbol=symbol, top_n=top, metric=metric, use_walk_forward=not no_walk_forward, verbose=True
        )

        # Save if requested
        if save:
            routing_path, params_path = selector.save_to_config(result, regime=regime)
            click.echo()
            click.secho("Configuration saved:", fg="green")
            click.echo(f"  Routing: {routing_path}")
            click.echo(f"  Params:  {params_path}")
            click.echo()
            click.echo("Run 'amsterdam strategy show' to view current routing.")

    except Exception as e:
        click.secho(f"Error during strategy selection: {e}", fg="red")
        import traceback

        traceback.print_exc()


@strategy.command("optimize-multitf")
@click.option("--symbols", "-s", required=True, help="Comma-separated symbols (e.g., AAPL,TSLA,MSFT)")
@click.option("--timeframes", "-t", default="15min,30min,1hour", help="Comma-separated timeframes to test")
@click.option("--strategies", default="rsi,sma,meanreversion,bollinger", help="Comma-separated strategies to test")
@click.option("--days", "-d", type=int, default=750, help="Days of historical data")
@click.option(
    "--metric",
    type=click.Choice(["composite", "sharpe_ratio", "sortino_ratio", "total_return"]),
    default="sharpe_ratio",
    help="Optimization metric",
)
@click.option("--dry-run", is_flag=True, help="Preview without saving config")
def strategy_optimize_multitf(symbols, timeframes, strategies, days, metric, dry_run):
    """Optimize strategies across multiple timeframes.

    Tests all combinations of symbols × timeframes × strategies × regimes
    to find the optimal configuration. Results are saved to strategy_routing.json.

    \b
    Examples:
        amsterdam strategy optimize-multitf -s AAPL,TSLA,MSFT
        amsterdam strategy optimize-multitf -s AAPL -t 5min,15min,30min
        amsterdam strategy optimize-multitf -s AAPL,TSLA -d 365 --metric composite
        amsterdam strategy optimize-multitf -s AAPL --dry-run  # Preview without saving
    """
    import subprocess
    import sys

    # Build command
    cmd = [
        sys.executable,
        str(APP_ROOT / "tools" / "optimize_routing_multitf.py"),
        "--symbols",
        symbols,
        "--timeframes",
        timeframes,
        "--strategies",
        strategies,
        "--days",
        str(days),
        "--metric",
        metric,
    ]

    if dry_run:
        cmd.append("--dry-run")

    click.echo("Running multi-timeframe optimization...")
    click.echo(f"  Symbols: {symbols}")
    click.echo(f"  Timeframes: {timeframes}")
    click.echo(f"  Strategies: {strategies}")
    click.echo(f"  Days: {days}")
    click.echo(f"  Metric: {metric}")
    click.echo()

    # Run optimization
    result = subprocess.run(cmd)

    if result.returncode == 0 and not dry_run:
        click.echo()
        click.secho("Optimization complete!", fg="green")
        click.echo("Run 'amsterdam strategy show' to view routing configuration.")
    elif result.returncode != 0:
        click.secho(f"Optimization failed with exit code {result.returncode}", fg="red")


@strategy.command("list")
def strategy_list():
    """List all available trading strategies."""
    from strategies.strategy_registry import list_strategies

    strategies = sorted(list_strategies())

    click.echo("\nAvailable Strategies:")
    click.echo("-" * 40)

    for name in strategies:
        click.echo(f"  • {name}")

    click.echo(f"\nTotal: {len(strategies)} strategies")


@strategy.command("show")
@click.option("--symbol", "-s", default=None, help="Show routing for specific symbol")
def strategy_show(symbol):
    """Show current strategy routing configuration."""
    import json

    config_dir = ROOT / "config"
    routing_path = config_dir / "strategy_routing.json"

    if not routing_path.exists():
        click.echo("No strategy routing configured.")
        click.echo("Run 'amsterdam strategy select <SYMBOL> --save' to create one.")
        return

    with open(routing_path) as f:
        routing = json.load(f)

    click.echo("\nStrategy Routing Configuration:")
    click.echo("=" * 50)

    if symbol:
        symbol = symbol.upper()
        if symbol in routing:
            click.echo(f"\n{symbol}:")
            for regime, strat in routing[symbol].items():
                if not regime.endswith("_ranked"):
                    click.echo(f"  {regime}: {strat}")
        else:
            click.echo(f"No routing configured for {symbol}")
    else:
        for sym, regimes in routing.items():
            click.echo(f"\n{sym}:")
            if isinstance(regimes, dict):
                for regime, strat in regimes.items():
                    if not regime.endswith("_ranked"):
                        click.echo(f"  {regime}: {strat}")
            else:
                click.echo(f"  default: {regimes}")

    click.echo()


@strategy.command("refresh")
def strategy_refresh():
    """Hot-reload strategy routing configuration.

    Useful after editing config files manually or running 'strategy select --save'.
    """
    from core.logic.strategy_routing_manager import StrategyRoutingManager

    config_path = APP_ROOT / "config" / "strategy_routing.json"

    if not config_path.exists():
        click.secho("No routing config found.", fg="yellow")
        return

    try:
        router = StrategyRoutingManager(str(config_path))
        router.refresh()
        click.secho("Strategy routing reloaded successfully.", fg="green")

        # Show summary
        symbols = router.list_symbols()
        click.echo(f"Loaded routing for {len(symbols)} symbols: {', '.join(symbols[:5])}")
        if len(symbols) > 5:
            click.echo(f"  ... and {len(symbols) - 5} more")

    except Exception as e:
        click.secho(f"Error reloading config: {e}", fg="red")


# =============================================================================
# BACKTEST COMMAND GROUP
# =============================================================================


@cli.group()
def backtest():
    """Backtesting commands.

    \b
    Commands:
        amsterdam backtest run         Run single strategy backtest
        amsterdam backtest compare     Compare strategies
        amsterdam backtest hybrid      Compare hybrid vs standard sizing
    """
    pass


@backtest.command("run")
@click.argument("symbol")
@click.option("--strategy", "-s", default="sma", help="Strategy to backtest")
@click.option("--days", "-d", type=int, default=365, help="Days of historical data")
@click.option("--capital", type=float, default=10000, help="Initial capital")
@click.option("--hybrid/--no-hybrid", default=False, help="Use hybrid sizing")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def backtest_run(symbol, strategy, days, capital, hybrid, verbose):
    """Run a backtest for a single strategy.

    \b
    Examples:
        amsterdam backtest run AAPL -s sma
        amsterdam backtest run MSFT -s macd --hybrid
        amsterdam backtest run TSLA -s rsi -d 180 --capital 50000
    """
    import asyncio

    from core.backtest.unified_backtest_runner import BacktestConfig, UnifiedBacktestRunner
    from core.unified_data_pipeline import UnifiedDataPipeline

    symbol = symbol.upper()

    click.echo(f"\nLoading {days} days of data for {symbol}...")

    # Load data
    try:
        pipeline = UnifiedDataPipeline()
        data = pipeline.get_data(symbol)

        if data is None or data.empty:
            click.echo("No cached data, fetching...")

            async def fetch():
                return await pipeline.update_symbols([symbol], days=days)

            asyncio.run(fetch())
            data = pipeline.get_data(symbol)

        if data is None or data.empty:
            click.secho(f"Error: Could not load data for {symbol}", fg="red")
            return

        if len(data) > days:
            data = data.tail(days).reset_index(drop=True)

        click.echo(f"Loaded {len(data)} bars")

    except Exception as e:
        click.secho(f"Error loading data: {e}", fg="red")
        return

    # Run backtest
    click.echo(f"\nRunning backtest: {strategy} (hybrid={hybrid})")

    try:
        runner = UnifiedBacktestRunner(data)
        config = BacktestConfig(
            strategy_name=strategy,
            initial_capital=capital,
            use_hybrid_sizing=hybrid,
        )
        result = runner.run(config)

        # Display results
        m = result.metrics
        click.echo("\n" + "=" * 50)
        click.echo(f"  BACKTEST RESULTS - {symbol}")
        click.echo("=" * 50)
        click.echo(f"  Strategy: {strategy}")
        click.echo(f"  Hybrid Sizing: {'Yes' if hybrid else 'No'}")
        click.echo("-" * 50)
        click.echo(f"  Total Return:  {m.total_return:>+10.2%}")
        click.echo(f"  Sharpe Ratio:  {m.sharpe_ratio:>10.2f}")
        click.echo(f"  Sortino Ratio: {m.sortino_ratio:>10.2f}")
        click.echo(f"  Max Drawdown:  {m.max_drawdown:>10.2%}")
        click.echo(f"  Win Rate:      {m.win_rate:>10.2%}")
        click.echo(f"  Num Trades:    {m.num_trades:>10}")
        click.echo(f"  Final Value:   ${m.final_value:>10,.2f}")

        if hybrid:
            click.echo("-" * 50)
            click.echo(f"  Trades with trend:     {m.trades_with_trend}")
            click.echo(f"  Trades against trend:  {m.trades_against_trend}")
            click.echo(f"  Win rate (with):       {m.win_rate_with_trend:.1%}")
            click.echo(f"  Win rate (against):    {m.win_rate_against_trend:.1%}")

        click.echo("=" * 50)

    except Exception as e:
        click.secho(f"Error running backtest: {e}", fg="red")
        import traceback

        traceback.print_exc()


@backtest.command("compare")
@click.argument("symbol")
@click.option("--strategies", "-s", default="sma,ema,macd,rsi", help="Comma-separated strategies to compare")
@click.option("--days", "-d", type=int, default=365, help="Days of data")
@click.option(
    "--metric",
    "-m",
    default="sharpe_ratio",
    type=click.Choice(["sharpe_ratio", "total_return", "sortino_ratio"]),
    help="Metric for ranking",
)
@click.option("--capital", type=float, default=10000, help="Initial capital")
def backtest_compare(symbol, strategies, days, metric, capital):
    """Compare multiple strategies.

    \b
    Examples:
        amsterdam backtest compare AAPL -s sma,ema,macd,rsi
        amsterdam backtest compare MSFT -s momentum,breakout -m total_return
    """
    # Delegate to compare_strategies tool
    cmd = [
        sys.executable,
        str(APP_ROOT / "tools" / "compare_strategies.py"),
        symbol.upper(),
        "-s",
        strategies,
        "-d",
        str(days),
        "-m",
        metric,
        "--capital",
        str(capital),
    ]

    os.execv(sys.executable, cmd)


@backtest.command("hybrid")
@click.argument("symbol")
@click.option("--strategies", "-s", default="sma,macd,rsi", help="Comma-separated strategies")
@click.option("--days", "-d", type=int, default=365, help="Days of data")
@click.option("--capital", type=float, default=10000, help="Initial capital")
def backtest_hybrid(symbol, strategies, days, capital):
    """Compare hybrid vs standard sizing.

    \b
    Examples:
        amsterdam backtest hybrid AAPL -s sma,macd,rsi
        amsterdam backtest hybrid MSFT -s momentum,ema,bollinger
    """
    # Delegate to compare_strategies tool with hybrid flag
    cmd = [
        sys.executable,
        str(APP_ROOT / "tools" / "compare_strategies.py"),
        symbol.upper(),
        "-s",
        strategies,
        "-d",
        str(days),
        "--capital",
        str(capital),
        "--hybrid-comparison",
    ]

    os.execv(sys.executable, cmd)


@backtest.command("categories")
@click.argument("symbol")
@click.option("--categories", "-c", default="trend_following,mean_reversion", help="Comma-separated categories")
@click.option("--days", "-d", type=int, default=365, help="Days of data")
@click.option("--capital", type=float, default=10000, help="Initial capital")
def backtest_categories(symbol, categories, days, capital):
    """Compare strategy categories.

    \b
    Categories:
        trend_following - SMA, EMA, MACD, ADX, etc.
        mean_reversion  - RSI, Bollinger, Stochastic, etc.
        momentum        - Momentum-based strategies

    \b
    Examples:
        amsterdam backtest categories AAPL
        amsterdam backtest categories MSFT -c trend_following,momentum
    """
    # Delegate to compare_strategies tool
    cmd = [
        sys.executable,
        str(APP_ROOT / "tools" / "compare_strategies.py"),
        symbol.upper(),
        "--categories",
        categories,
        "-d",
        str(days),
        "--capital",
        str(capital),
    ]

    os.execv(sys.executable, cmd)


@backtest.command("full")
@click.argument("symbol")
@click.option("--strategies", "-s", default=None, help="Strategies to include")
@click.option("--days", "-d", type=int, default=365, help="Days of data")
@click.option("-o", "--output", default=None, help="Output file (.md or .json)")
@click.option("--capital", type=float, default=10000, help="Initial capital")
def backtest_full(symbol, strategies, days, output, capital):
    """Run full comparison analysis.

    \b
    Examples:
        amsterdam backtest full AAPL
        amsterdam backtest full MSFT -o reports/msft_analysis.md
        amsterdam backtest full TSLA -s sma,macd,rsi,momentum --days 180
    """
    cmd = [
        sys.executable,
        str(APP_ROOT / "tools" / "compare_strategies.py"),
        symbol.upper(),
        "-d",
        str(days),
        "--capital",
        str(capital),
        "--full",
    ]

    if strategies:
        cmd.extend(["-s", strategies])

    if output:
        cmd.extend(["-o", output])

    os.execv(sys.executable, cmd)


@backtest.command("optimize")
@click.argument("mode", required=False, default=None)
@click.option("--symbols", "-s", default=None, help="Comma-separated symbols (default: all)")
@click.option("--days", "-d", type=int, default=365, help="Days of data")
@click.option("--strategies", default=None, help="Comma-separated strategies to test")
@click.option(
    "--timeframes",
    "-t",
    default=None,
    help="Comma-separated timeframes (e.g., 15min,30min,1hour) - enables multi-TF optimization",
)
@click.option("--dry-run", is_flag=True, help="Don't save config")
def backtest_optimize(mode, symbols, days, strategies, timeframes, dry_run):
    """Optimize strategy routing for all symbols.

    Runs regime-aware backtests on all symbols and updates strategy_routing.json
    with the optimal strategy for each (symbol, regime) combination.

    With --timeframes, also optimizes the timeframe for each symbol/regime.

    Use 'all' mode for comprehensive optimization with all strategies and timeframes.

    \b
    Examples:
        amsterdam backtest optimize                          # All symbols, single TF
        amsterdam backtest optimize all                      # All strategies + timeframes
        amsterdam backtest optimize -s AAPL,MSFT            # Specific symbols
        amsterdam backtest optimize -d 180                  # Use 180 days
        amsterdam backtest optimize -t 15min,30min,1hour    # Multi-timeframe
        amsterdam backtest optimize --dry-run               # Don't save
    """
    # Handle 'all' mode - comprehensive optimization
    if mode == "all":
        # Default to all strategies
        if not strategies:
            strategies = "adx,bollinger,breakout,combined,donchian,ema,ichimoku,logisticregression,macd,meanreversion,momentum,psar,rsi,sma,stochastic,vwap"

        # Default to all timeframes
        if not timeframes:
            timeframes = "15min,30min,1hour,day"

        # Default to 750 days for comprehensive backtest
        if days == 365:  # If using default value
            days = 750

        click.echo("Running comprehensive optimization:")
        click.echo("  • All 16 strategies")
        click.echo("  • All 4 timeframes (15min, 30min, 1hour, day)")
        click.echo(f"  • {days} days of data")
        click.echo()

    # Determine which optimizer to use based on timeframes option
    if timeframes:
        # Use multi-timeframe optimizer
        cmd = [
            sys.executable,
            str(APP_ROOT / "tools" / "optimize_routing_multitf.py"),
            "-d",
            str(days),
            "--timeframes",
            timeframes,
        ]
    else:
        # Use single-timeframe optimizer (backward compatible)
        cmd = [
            sys.executable,
            str(APP_ROOT / "tools" / "optimize_routing.py"),
            "-d",
            str(days),
        ]

    if symbols:
        cmd.extend(["-s", symbols])

    if strategies:
        cmd.extend(["--strategies", strategies])

    if dry_run:
        cmd.append("--dry-run")

    os.execv(sys.executable, cmd)


# =============================================================================
# SCANNER
# =============================================================================


@cli.group()
def scan():
    """Stock scanner — screen and rank symbols for trading.

    \b
    Examples:
        amsterdam scan run                          # Scan full S&P 500
        amsterdam scan run -s AAPL,MSFT,GOOGL       # Scan specific symbols
        amsterdam scan run --apply                   # Scan and update trade/watch lists
        amsterdam scan run -n 10                     # Show top 10 results
    """
    pass


@scan.command("run")
@click.option("-s", "--symbols", default=None, help="Comma-separated symbols to scan (overrides universe)")
@click.option(
    "-u", "--universe", default=None, type=click.Choice(["sp500", "custom", "watchlist", "all"]), help="Universe source"
)
@click.option(
    "--apply", "apply_results", is_flag=True, default=False, help="Auto-update trade/watch lists with results"
)
@click.option("--optimize", is_flag=True, default=False, help="Run regime backtest optimization on trade candidates")
@click.option("-d", "--days", default=365, type=int, help="Days of data for optimization (default: 365)")
@click.option("-n", "--top", default=20, help="Number of results to display")
@click.option("--min-score", default=0.0, type=float, help="Minimum score filter")
@click.option("--json-output", is_flag=True, default=False, help="Output results as JSON")
def scan_run(symbols, universe, apply_results, optimize, days, top, min_score, json_output):
    """Run the stock scanner.

    Screens symbols using technical criteria, scores and ranks them.
    Use --optimize to also run regime backtest and assign best strategies.
    """
    import json as json_mod

    from core.config_loader import get_config
    from scanner.engine import get_scanner

    cfg = get_config()
    scanner_cfg = cfg.scanner.to_engine_config()

    # Override universe if specified
    if universe:
        scanner_cfg["universe_source"] = universe

    scanner = get_scanner(scanner_cfg)

    # Parse symbols
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

    click.echo(
        f"Scanning {'universe: ' + (universe or cfg.scanner.universe_source) if not symbol_list else str(len(symbol_list)) + ' symbols'}..."
    )

    report = scanner.scan(symbols=symbol_list, update_lists=apply_results)

    if json_output:
        click.echo(json_mod.dumps(report.to_dict(), indent=2))
        return

    # Display results table
    click.echo(f"\nScan completed in {report.duration_seconds}s — {report.universe_size} symbols screened")
    if report.errors:
        click.secho(f"  {len(report.errors)} errors encountered", fg="yellow")
    click.echo()

    results = [r for r in report.top_n(top) if r.total_score >= min_score]
    if not results:
        click.secho("No symbols matched the criteria.", fg="yellow")
        return

    # Header
    click.echo(f"{'Rank':<6}{'Symbol':<10}{'Score':<10}{'Rec':<10}{'RSI':<10}{'Vol Ratio':<12}{'Criteria'}")
    click.echo("-" * 85)

    for i, r in enumerate(results, 1):
        rec_color = {"trade": "green", "watch": "yellow", "skip": "white"}.get(r.recommendation, "white")

        rsi = r.metadata.get("RSI")
        rsi_str = f"{rsi:.1f}" if rsi is not None else "—"

        vol = r.metadata.get("volume")
        avg_vol = r.metadata.get("avg_volume_20")
        vol_ratio = f"{vol / avg_vol:.1f}x" if vol and avg_vol and avg_vol > 0 else "—"

        criteria_str = " | ".join(f"{k}={v:.2f}" for k, v in r.criteria_scores.items())

        click.echo(f"{i:<6}{r.symbol:<10}{r.total_score:<10.4f}", nl=False)
        click.secho(f"{r.recommendation:<10}", fg=rec_color, nl=False)
        click.echo(f"{rsi_str:<10}{vol_ratio:<12}{criteria_str}")

    # Summary
    trade_count = len(report.trade_candidates())
    watch_count = len(report.watch_candidates())
    click.echo()
    click.secho(f"Trade candidates: {trade_count}", fg="green")
    click.secho(f"Watch candidates: {watch_count}", fg="yellow")

    if apply_results:
        click.secho("\nTrade/watch lists updated.", fg="green")

    if optimize:
        trade_syms = report.trade_candidates()
        if not trade_syms:
            click.secho("\nNo trade candidates to optimize.", fg="yellow")
        else:
            click.echo(f"\nOptimizing strategies for {len(trade_syms)} trade candidates ({days} days)...")
            opt_results = scanner.optimize_strategies(symbols=trade_syms, days=days)
            for sym, strategies in opt_results.items():
                strat_str = ", ".join(f"{regime}={strat}" for regime, strat in strategies.items())
                click.echo(f"  {sym}: {strat_str}")
            click.secho(f"\nStrategy routing updated for {len(opt_results)} symbols.", fg="green")


@scan.command("results")
@click.option("--json-output", is_flag=True, default=False, help="Output as JSON")
def scan_results(json_output):
    """Show the last scan report."""
    import json as json_mod

    from core.config_loader import get_config
    from scanner.engine import get_scanner

    cfg = get_config()
    scanner = get_scanner(cfg.scanner.to_engine_config())

    report = scanner.last_report
    if not report:
        click.secho("No scan results available. Run 'amsterdam scan run' first.", fg="yellow")
        return

    if json_output:
        click.echo(json_mod.dumps(report.to_dict(), indent=2))
    else:
        click.echo(f"Last scan: {report.timestamp.isoformat()}")
        click.echo(f"Universe: {report.universe_size} symbols")
        click.echo(f"Trade candidates: {len(report.trade_candidates())}")
        click.echo(f"Watch candidates: {len(report.watch_candidates())}")
        click.echo(f"Duration: {report.duration_seconds}s")


@scan.command("apply")
@click.option("-t", "--trade-count", default=None, type=int, help="Max trade symbols to add")
@click.option("-w", "--watch-count", default=None, type=int, help="Max watch symbols to add")
@click.confirmation_option(prompt="Apply scan results to trade/watch lists?")
def scan_apply(trade_count, watch_count):
    """Apply last scan results to trade/watch lists."""
    from core.config_loader import get_config
    from core.symbol_list_manager import get_list_manager
    from scanner.engine import get_scanner

    cfg = get_config()
    scanner = get_scanner(cfg.scanner.to_engine_config())

    report = scanner.last_report
    if not report:
        click.secho("No scan results available. Run 'amsterdam scan run' first.", fg="yellow")
        return

    manager = get_list_manager()
    added = {"trade": 0, "watch": 0}

    for result in report.results:
        if result.recommendation == "skip":
            continue

        note = f"scanner score={result.total_score} @ {report.timestamp.isoformat()}"

        if result.recommendation == "trade":
            if trade_count is not None and added["trade"] >= trade_count:
                continue
            existing = manager.get_list_type(result.symbol)
            if existing != "trade":
                manager.add_to_trade_list(result.symbol, notes=note)
                added["trade"] += 1
                click.echo(f"  + {result.symbol} -> trade (score={result.total_score})")

        elif result.recommendation == "watch":
            if watch_count is not None and added["watch"] >= watch_count:
                continue
            existing = manager.get_list_type(result.symbol)
            if existing not in ("trade", "watch"):
                manager.add_to_watch_list(result.symbol, notes=note)
                added["watch"] += 1
                click.echo(f"  + {result.symbol} -> watch (score={result.total_score})")

    click.secho(f"\nAdded {added['trade']} to trade, {added['watch']} to watch.", fg="green")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
