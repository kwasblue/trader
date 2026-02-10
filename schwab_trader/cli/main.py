#!/usr/bin/env python3
"""
Schwab Trader CLI - Unified Command Line Interface

Usage:
    trader start [OPTIONS]      Start the trading daemon
    trader stop                 Stop the trading daemon
    trader status               Check daemon status
    trader gui [OPTIONS]        Launch GUI trading application
    trader preflight [OPTIONS]  Run pre-flight checks
    trader token COMMAND        Token management commands
    trader test [OPTIONS]       Run test suite
    trader logs [OPTIONS]       View logs
"""

import sys
import os
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import click
from dotenv import load_dotenv

# Load environment
load_dotenv(ROOT / ".venv" / ".env")


@click.group()
@click.version_option(version="1.0.0", prog_name="trader")
def cli():
    """Schwab Trader - Algorithmic Trading Platform

    A comprehensive trading system with support for Schwab and Alpaca brokers.

    \b
    Quick Start:
        trader preflight        # Check system readiness
        trader gui              # Launch simulation GUI
        trader start            # Start autonomous trading

    \b
    Documentation:
        See docs/index.md for full documentation.
    """
    pass


# =============================================================================
# START COMMAND
# =============================================================================

@cli.command()
@click.option('--symbols', '-s', default=None, help='Comma-separated symbols (e.g., AAPL,MSFT)')
@click.option('--broker', '-b', type=click.Choice(['alpaca', 'schwab']), default='alpaca', help='Broker to use')
@click.option('--dry-run', is_flag=True, help='Run without executing real trades')
@click.option('--daemon', '-d', is_flag=True, help='Run as background daemon')
def start(symbols, broker, dry_run, daemon):
    """Start the autonomous trading daemon.

    \b
    Examples:
        trader start                           # Start with defaults
        trader start -s AAPL,MSFT              # Specific symbols
        trader start --broker schwab           # Use Schwab
        trader start --dry-run                 # No real trades
        trader start --daemon                  # Background mode
    """
    import subprocess

    cmd = [sys.executable, str(ROOT / "autotrader.py")]

    if symbols:
        cmd.extend(['--symbols'] + symbols.split(','))
    if broker:
        cmd.extend(['--broker', broker])
    if dry_run:
        cmd.append('--dry-run')

    if daemon:
        # Run as background process
        click.echo(f"Starting trader daemon...")
        subprocess.Popen(
            cmd,
            stdout=open(ROOT / "logs" / "autotrader_stdout.log", 'a'),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        click.echo("Daemon started. Use 'trader status' to check.")
    else:
        # Run in foreground
        click.echo(f"Starting trader (broker={broker})...")
        os.execv(sys.executable, cmd)


# =============================================================================
# STOP COMMAND
# =============================================================================

@cli.command()
def stop():
    """Stop the trading daemon."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "autotrader_ctl.py"), "stop"],
        capture_output=True,
        text=True
    )
    click.echo(result.stdout)
    if result.stderr:
        click.echo(result.stderr, err=True)


# =============================================================================
# STATUS COMMAND
# =============================================================================

@cli.command()
def status():
    """Check trading daemon status."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "autotrader_ctl.py"), "status"],
        capture_output=True,
        text=True
    )
    click.echo(result.stdout)
    if result.stderr:
        click.echo(result.stderr, err=True)


# =============================================================================
# GUI COMMAND
# =============================================================================

@cli.command()
@click.option('--mode', '-m', type=click.Choice(['simulation', 'alpaca', 'schwab']),
              default='simulation', help='Trading mode')
@click.option('--symbols', '-s', default='AAPL,MSFT', help='Comma-separated symbols')
@click.option('--speed', type=float, default=0.1, help='Simulation speed (seconds per bar)')
@click.option('--steps', type=int, default=600, help='Number of bars to simulate (default: 600 = ~2 min)')
def gui(mode, symbols, speed, steps):
    """Launch the GUI trading application.

    \b
    Modes:
        simulation  - GBM price simulator (no real trades)
        alpaca      - Alpaca paper/live trading
        schwab      - Schwab live trading

    \b
    Examples:
        trader gui                             # Simulation mode
        trader gui --mode alpaca               # Alpaca trading
        trader gui -s AAPL,GOOGL,TSLA          # Custom symbols
    """
    cmd = [
        sys.executable, str(ROOT / "run_trading.py"),
        "--mode", mode,
        "--symbols", symbols,
        "--speed", str(speed),
        "--steps", str(steps)
    ]

    click.echo(f"Launching GUI ({mode} mode, {steps} bars)...")
    os.execv(sys.executable, cmd)


# =============================================================================
# PREFLIGHT COMMAND
# =============================================================================

@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--update-data', is_flag=True, help='Update stale historical data')
@click.option('--reauth-schwab', is_flag=True, help='Force Schwab re-authentication')
def preflight(verbose, update_data, reauth_schwab):
    """Run pre-flight system checks.

    Validates:
    - Broker credentials
    - Token expiry status
    - Historical data freshness
    - System configuration

    \b
    Examples:
        trader preflight                       # Quick check
        trader preflight -v                    # Verbose output
        trader preflight --update-data         # Update stale data
    """
    cmd = [sys.executable, str(ROOT / "preflight.py")]

    if verbose:
        cmd.append('-v')
    if update_data:
        cmd.append('--update-data')
    if reauth_schwab:
        cmd.append('--reauth-schwab')

    os.execv(sys.executable, cmd)


# =============================================================================
# TOKEN COMMAND GROUP
# =============================================================================

@cli.group()
def token():
    """Token management commands.

    \b
    Commands:
        trader token status     Check token status
        trader token refresh    Refresh tokens
        trader token keeper     Run token keeper service
    """
    pass


@token.command('status')
def token_status():
    """Check Schwab token status."""
    import asyncio
    from core.credential_validator import CredentialValidator

    async def check():
        validator = CredentialValidator()
        result = await validator.validate_schwab()

        status_colors = {
            'VALID': 'green',
            'EXPIRING_SOON': 'yellow',
            'EXPIRED': 'red',
            'MISSING': 'red',
            'INVALID': 'red'
        }

        color = status_colors.get(result.status.name, 'white')
        click.echo(f"Schwab Token: ", nl=False)
        click.secho(result.status.name, fg=color, bold=True)
        click.echo(f"  {result.message}")

        # Also check Alpaca
        alpaca_result = await validator.validate_alpaca()
        color = status_colors.get(alpaca_result.status.name, 'white')
        click.echo(f"Alpaca API:   ", nl=False)
        click.secho(alpaca_result.status.name, fg=color, bold=True)
        click.echo(f"  {alpaca_result.message}")

    asyncio.run(check())


@token.command('refresh')
@click.option('--force', '-f', is_flag=True, help='Force full re-authentication')
def token_refresh(force):
    """Refresh Schwab tokens.

    \b
    Examples:
        trader token refresh           # Refresh if needed
        trader token refresh --force   # Force browser login
    """
    cmd = [sys.executable, str(ROOT / "refresh_schwab_token.py")]
    if force:
        cmd.append('--force')

    os.execv(sys.executable, cmd)


@token.command('keeper')
@click.option('--interval', '-i', type=int, default=60, help='Check interval in seconds')
@click.option('--daemon', '-d', is_flag=True, help='Run as background daemon')
def token_keeper(interval, daemon):
    """Run the token keeper service.

    Keeps Schwab tokens fresh by periodically checking and renewing them.

    \b
    Examples:
        trader token keeper                    # Run in foreground
        trader token keeper --daemon           # Background mode
        trader token keeper -i 300             # Check every 5 minutes
    """
    cmd = [sys.executable, str(ROOT / "token_keeper.py"), '--interval', str(interval)]

    if daemon:
        cmd.append('--daemon')

    os.execv(sys.executable, cmd)


# =============================================================================
# TEST COMMAND
# =============================================================================

@cli.command()
@click.option('--coverage', '-c', is_flag=True, help='Run with coverage report')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--unit', is_flag=True, help='Run only unit tests')
@click.option('--integration', is_flag=True, help='Run only integration tests')
@click.argument('path', required=False)
def test(coverage, verbose, unit, integration, path):
    """Run the test suite.

    \b
    Examples:
        trader test                            # Run all tests
        trader test -v                         # Verbose output
        trader test --coverage                 # With coverage report
        trader test tests/test_autotrader.py   # Specific file
    """
    cmd = [sys.executable, '-m', 'pytest']

    if path:
        cmd.append(path)
    else:
        cmd.append('tests/')

    if verbose:
        cmd.append('-v')
    if coverage:
        cmd.extend(['--cov=core', '--cov=strategies', '--cov-report=html'])

    click.echo("Running tests...")
    os.execv(sys.executable, cmd)


# =============================================================================
# LOGS COMMAND
# =============================================================================

@cli.command()
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--lines', '-n', type=int, default=50, help='Number of lines to show')
@click.option('--file', '-l', type=click.Choice(['app', 'trades', 'autotrader', 'preflight']),
              default='app', help='Log file to view')
def logs(follow, lines, file):
    """View application logs.

    \b
    Log files:
        app         - Main application log
        trades      - Trade execution log
        autotrader  - Daemon operations
        preflight   - Pre-flight checks

    \b
    Examples:
        trader logs                            # Last 50 lines of app.log
        trader logs -f                         # Follow in real-time
        trader logs -l trades -n 100           # Last 100 trade entries
    """
    import subprocess

    log_files = {
        'app': 'app.log',
        'trades': 'trades.log',
        'autotrader': 'autotrader.log',
        'preflight': 'preflight.log'
    }

    log_path = ROOT / "logs" / log_files[file]

    if not log_path.exists():
        click.echo(f"Log file not found: {log_path}")
        return

    if follow:
        subprocess.run(['tail', '-f', str(log_path)])
    else:
        subprocess.run(['tail', f'-{lines}', str(log_path)])


# =============================================================================
# SYMBOLS COMMAND GROUP
# =============================================================================

@cli.group()
def symbols():
    """Symbol list management.

    \b
    Commands:
        trader symbols list         Show all symbols
        trader symbols add          Add symbol to list
        trader symbols remove       Remove symbol
        trader symbols move         Move between lists
    """
    pass


@symbols.command('list')
@click.option('--trade', is_flag=True, help='Show trade list only')
@click.option('--watch', is_flag=True, help='Show watch list only')
def symbols_list(trade, watch):
    """List configured symbols."""
    import subprocess

    cmd = [sys.executable, str(ROOT / "autotrader_ctl.py"), "list"]
    if trade:
        cmd.append('--trade')
    if watch:
        cmd.append('--watch')

    result = subprocess.run(cmd, capture_output=True, text=True)
    click.echo(result.stdout)


@symbols.command('add')
@click.argument('symbol')
@click.option('--trade', is_flag=True, help='Add to trade list')
@click.option('--watch', is_flag=True, help='Add to watch list')
def symbols_add(symbol, trade, watch):
    """Add a symbol to trade or watch list."""
    import subprocess

    cmd = [sys.executable, str(ROOT / "autotrader_ctl.py"), "add", symbol.upper()]
    if trade:
        cmd.append('--trade')
    elif watch:
        cmd.append('--watch')
    else:
        cmd.append('--trade')  # Default to trade list

    result = subprocess.run(cmd, capture_output=True, text=True)
    click.echo(result.stdout)


@symbols.command('remove')
@click.argument('symbol')
def symbols_remove(symbol):
    """Remove a symbol from all lists."""
    import subprocess

    cmd = [sys.executable, str(ROOT / "autotrader_ctl.py"), "remove", symbol.upper()]
    result = subprocess.run(cmd, capture_output=True, text=True)
    click.echo(result.stdout)


# =============================================================================
# DATA COMMAND GROUP
# =============================================================================

@cli.group()
def data():
    """Historical data management.

    \b
    Commands:
        trader data update      Update historical data
        trader data status      Check data freshness
    """
    pass


@data.command('update')
@click.option('--symbols', '-s', default=None, help='Comma-separated symbols')
@click.option('--days', '-d', type=int, default=30, help='Days of history to fetch')
@click.option('--source', type=click.Choice(['alpaca', 'schwab', 'auto']), default='auto',
              help='Data source')
def data_update(symbols, days, source):
    """Update historical data for symbols."""
    import asyncio
    from core.unified_data_pipeline import UnifiedDataPipeline

    async def update():
        pipeline = UnifiedDataPipeline()

        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
        else:
            # Load from config
            import json
            config_path = ROOT / "config" / "symbols.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                symbol_list = cfg.get('trade_list', []) + cfg.get('watch_list', [])
            else:
                symbol_list = ['AAPL', 'MSFT']

        click.echo(f"Updating data for: {', '.join(symbol_list)}")

        src = None if source == 'auto' else source
        results = await pipeline.update_symbols(symbol_list, days=days, source=src)

        for sym, count in results.items():
            if count > 0:
                click.secho(f"  {sym}: {count} bars", fg='green')
            else:
                click.secho(f"  {sym}: failed", fg='red')

    asyncio.run(update())


@data.command('status')
@click.option('--symbols', '-s', default=None, help='Comma-separated symbols')
def data_status(symbols):
    """Check data freshness for symbols."""
    from core.unified_data_pipeline import UnifiedDataPipeline

    pipeline = UnifiedDataPipeline()

    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
    else:
        symbol_list = pipeline.list_available_symbols()[:10]

    click.echo("Data Status:")
    click.echo("-" * 50)

    for sym in symbol_list:
        info = pipeline.get_cache_info(sym)
        if info:
            age = info.get('age_minutes', 0)
            bars = info.get('bar_count', 0)
            if age < 60:
                color = 'green'
            elif age < 1440:
                color = 'yellow'
            else:
                color = 'red'
            click.echo(f"  {sym}: ", nl=False)
            click.secho(f"{bars} bars, {age:.0f}min old", fg=color)
        else:
            click.echo(f"  {sym}: ", nl=False)
            click.secho("no data", fg='red')


# =============================================================================
# STRATEGY COMMAND GROUP
# =============================================================================

@cli.group()
def strategy():
    """Strategy management commands.

    \b
    Commands:
        trader strategy select      Evaluate and select best strategies
        trader strategy list        List available strategies
        trader strategy show        Show current routing configuration
    """
    pass


@strategy.command('select')
@click.argument('symbol')
@click.option('--days', '-d', type=int, default=365, help='Days of historical data')
@click.option('--top', '-n', type=int, default=3, help='Number of top strategies to select')
@click.option('--metric', type=click.Choice(['composite', 'sharpe_ratio', 'sortino_ratio', 'total_return']),
              default='composite', help='Ranking metric')
@click.option('--no-walk-forward', is_flag=True, help='Disable walk-forward validation')
@click.option('--regime', type=click.Choice(['low_volatility', 'normal', 'high_volatility']),
              default='normal', help='Market regime to optimize for')
@click.option('--save', '-s', is_flag=True, help='Save results to config files')
@click.option('--capital', type=float, default=100000, help='Initial capital for backtesting')
def strategy_select(symbol, days, top, metric, no_walk_forward, regime, save, capital):
    """Evaluate all strategies and select the best performers for a symbol.

    Uses backtesting with walk-forward validation to find optimal strategies,
    then saves results to strategy_routing.json for use in live/simulated trading.

    \b
    Examples:
        trader strategy select AAPL                    # Evaluate strategies for AAPL
        trader strategy select AAPL --save             # Save best to config
        trader strategy select MSFT -d 180 --save      # 180 days of data
        trader strategy select TSLA --regime high_volatility --save
    """
    import asyncio
    import logging
    import pandas as pd
    from pathlib import Path

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
            click.secho(f"Error: Could not load data for {symbol}", fg='red')
            return

        # Limit to requested days
        if len(data) > days:
            data = data.tail(days).reset_index(drop=True)

        click.echo(f"Loaded {len(data)} bars")

    except Exception as e:
        click.secho(f"Error loading data: {e}", fg='red')
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
            symbol=symbol,
            top_n=top,
            metric=metric,
            use_walk_forward=not no_walk_forward,
            verbose=True
        )

        # Save if requested
        if save:
            routing_path, params_path = selector.save_to_config(result, regime=regime)
            click.echo()
            click.secho("Configuration saved:", fg='green')
            click.echo(f"  Routing: {routing_path}")
            click.echo(f"  Params:  {params_path}")
            click.echo()
            click.echo("Run 'trader strategy show' to view current routing.")

    except Exception as e:
        click.secho(f"Error during strategy selection: {e}", fg='red')
        import traceback
        traceback.print_exc()


@strategy.command('list')
def strategy_list():
    """List all available trading strategies."""
    from strategies.strategy_registry import list_strategies

    strategies = sorted(list_strategies())

    click.echo("\nAvailable Strategies:")
    click.echo("-" * 40)

    for name in strategies:
        click.echo(f"  • {name}")

    click.echo(f"\nTotal: {len(strategies)} strategies")


@strategy.command('show')
@click.option('--symbol', '-s', default=None, help='Show routing for specific symbol')
def strategy_show(symbol):
    """Show current strategy routing configuration."""
    import json
    from pathlib import Path

    config_dir = ROOT / "config"
    routing_path = config_dir / "strategy_routing.json"

    if not routing_path.exists():
        click.echo("No strategy routing configured.")
        click.echo("Run 'trader strategy select <SYMBOL> --save' to create one.")
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
                if not regime.endswith('_ranked'):
                    click.echo(f"  {regime}: {strat}")
        else:
            click.echo(f"No routing configured for {symbol}")
    else:
        for sym, regimes in routing.items():
            click.echo(f"\n{sym}:")
            if isinstance(regimes, dict):
                for regime, strat in regimes.items():
                    if not regime.endswith('_ranked'):
                        click.echo(f"  {regime}: {strat}")
            else:
                click.echo(f"  default: {regimes}")

    click.echo()


@strategy.command('refresh')
def strategy_refresh():
    """Hot-reload strategy routing configuration.

    Useful after editing config files manually or running 'strategy select --save'.
    """
    from core.logic.strategy_routing_manager import StrategyRoutingManager

    config_path = ROOT / "config" / "strategy_routing.json"

    if not config_path.exists():
        click.secho("No routing config found.", fg='yellow')
        return

    try:
        router = StrategyRoutingManager(str(config_path))
        router.refresh()
        click.secho("Strategy routing reloaded successfully.", fg='green')

        # Show summary
        symbols = router.list_symbols()
        click.echo(f"Loaded routing for {len(symbols)} symbols: {', '.join(symbols[:5])}")
        if len(symbols) > 5:
            click.echo(f"  ... and {len(symbols) - 5} more")

    except Exception as e:
        click.secho(f"Error reloading config: {e}", fg='red')


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
