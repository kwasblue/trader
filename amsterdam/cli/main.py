#!/usr/bin/env python3
"""
Amsterdam CLI - Unified Command Line Interface

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
load_dotenv(ROOT / ".env")


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
@click.option('--symbols', '-s', default=None, help='Comma-separated symbols (e.g., AAPL,MSFT)')
@click.option('--broker', '-b', type=click.Choice(['alpaca', 'schwab']), default='alpaca', help='Broker to use')
@click.option('--dry-run', is_flag=True, help='Run without executing real trades')
@click.option('--daemon', '-d', is_flag=True, help='Run as background daemon')
def start(symbols, broker, dry_run, daemon):
    """Start the autonomous trading daemon.

    \b
    Examples:
        amsterdam start                           # Start with defaults
        amsterdam start -s AAPL,MSFT              # Specific symbols
        amsterdam start --broker schwab           # Use Schwab
        amsterdam start --dry-run                 # No real trades
        amsterdam start --daemon                  # Background mode
    """
    import subprocess

    cmd = [sys.executable, str(ROOT / "autoamsterdam.py")]

    if symbols:
        cmd.extend(['--symbols'] + symbols.split(','))
    if broker:
        cmd.extend(['--broker', broker])
    if dry_run:
        cmd.append('--dry-run')

    if daemon:
        # Run as background process
        click.echo(f"Starting amsterdam daemon...")
        subprocess.Popen(
            cmd,
            stdout=open(ROOT / "logs" / "autoamsterdam_stdout.log", 'a'),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        click.echo("Daemon started. Use 'amsterdam status' to check.")
    else:
        # Run in foreground
        click.echo(f"Starting amsterdam (broker={broker})...")
        os.execv(sys.executable, cmd)


# =============================================================================
# STOP COMMAND
# =============================================================================

@cli.command()
def stop():
    """Stop the trading daemon."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "autoatrader_ctl.py"), "stop"],
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
        [sys.executable, str(ROOT / "amsterdam_ctl.py"), "status"],
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
        amsterdam gui                             # Simulation mode
        amsterdam gui --mode alpaca               # Alpaca trading
        amsterdam gui -s AAPL,GOOGL,TSLA          # Custom symbols
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
        amsterdam preflight                       # Quick check
        amsterdam preflight -v                    # Verbose output
        amsterdam preflight --update-data         # Update stale data
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
        amsterdam token status     Check token status
        amsterdam token refresh    Refresh tokens
        amsterdam token keeper     Run token keeper service
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
        amsterdam token refresh           # Refresh if needed
        amsterdam token refresh --force   # Force browser login
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
        amsterdam token keeper                    # Run in foreground
        amsterdam token keeper --daemon           # Background mode
        amsterdam token keeper -i 300             # Check every 5 minutes
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
        amsterdam test                            # Run all tests
        amsterdam test -v                         # Verbose output
        amsterdam test --coverage                 # With coverage report
        amsterdam test tests/test_autoamsterdam.py   # Specific file
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
@click.option('--file', '-l', type=click.Choice(['app', 'trades', 'autoamsterdam', 'preflight']),
              default='app', help='Log file to view')
def logs(follow, lines, file):
    """View application logs.

    \b
    Log files:
        app         - Main application log
        trades      - Trade execution log
        autoamsterdam  - Daemon operations
        preflight   - Pre-flight checks

    \b
    Examples:
        amsterdam logs                            # Last 50 lines of app.log
        amsterdam logs -f                         # Follow in real-time
        amsterdam logs -l trades -n 100           # Last 100 trade entries
    """
    import subprocess

    log_files = {
        'app': 'app.log',
        'trades': 'trades.log',
        'autoamsterdam': 'autoamsterdam.log',
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
# STATS COMMAND
# =============================================================================

@cli.command()
@click.option('--summary', is_flag=True, help='Show summary only')
@click.option('--by-day', is_flag=True, help='Show daily breakdown')
@click.option('--by-symbol', is_flag=True, help='Show per-symbol stats')
@click.option('--by-hour', is_flag=True, help='Show hourly performance')
@click.option('--by-strategy', is_flag=True, help='Show per-strategy stats')
@click.option('--worst', type=int, metavar='N', help='Show worst N trades')
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

    cmd = [sys.executable, str(ROOT / "tools" / "analyze_trades.py")]

    if summary:
        cmd.append('--summary')
    if by_day:
        cmd.append('--by-day')
    if by_symbol:
        cmd.append('--by-symbol')
    if by_hour:
        cmd.append('--by-hour')
    if by_strategy:
        cmd.append('--by-strategy')
    if worst:
        cmd.extend(['--worst', str(worst)])

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
    """
    pass


@symbols.command('list')
@click.option('--trade', is_flag=True, help='Show trade list only')
@click.option('--watch', is_flag=True, help='Show watch list only')
def symbols_list(trade, watch):
    """List configured symbols."""
    import subprocess

    cmd = [sys.executable, str(ROOT / "autoamsterdam_ctl.py"), "list"]
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

    cmd = [sys.executable, str(ROOT / "autoamsterdam_ctl.py"), "add", symbol.upper()]
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

    cmd = [sys.executable, str(ROOT / "autoamsterdam_ctl.py"), "remove", symbol.upper()]
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
        amsterdam data update      Update historical data
        amsterdam data status      Check data freshness
    """
    pass


@data.command('update')
@click.option('--symbols', '-s', default=None, help='Comma-separated symbols')
@click.option('--days', '-d', type=int, default=30, help='Days of history to fetch')
@click.option('--source', type=click.Choice(['alpaca', 'schwab', 'auto']), default='auto',
              help='Data source')
@click.option('--timeframes', '-t', default=None,
              help='Comma-separated timeframes (e.g., 15min,30min,1hour)')
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

        # Parse timeframes
        timeframe_list = None
        if timeframes:
            timeframe_list = [t.strip() for t in timeframes.split(',')]
            click.echo(f"Updating data for: {', '.join(symbol_list)} at {', '.join(timeframe_list)}")
        else:
            click.echo(f"Updating data for: {', '.join(symbol_list)}")

        src = None if source == 'auto' else source
        results = await pipeline.update_symbols(
            symbol_list,
            days=days,
            source=src,
            timeframes=timeframe_list
        )

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
        amsterdam strategy select          Evaluate and select best strategies
        amsterdam strategy optimize-multitf Optimize across timeframes (multi-TF)
        amsterdam strategy list            List available strategies
        amsterdam strategy show            Show current routing configuration
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
        amsterdam strategy select AAPL                    # Evaluate strategies for AAPL
        amsterdam strategy select AAPL --save             # Save best to config
        amsterdam strategy select MSFT -d 180 --save      # 180 days of data
        amsterdam strategy select TSLA --regime high_volatility --save
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
            click.echo("Run 'amsterdam strategy show' to view current routing.")

    except Exception as e:
        click.secho(f"Error during strategy selection: {e}", fg='red')
        import traceback
        traceback.print_exc()


@strategy.command('optimize-multitf')
@click.option('--symbols', '-s', required=True, help='Comma-separated symbols (e.g., AAPL,TSLA,MSFT)')
@click.option('--timeframes', '-t', default='15min,30min,1hour',
              help='Comma-separated timeframes to test')
@click.option('--strategies', default='rsi,sma,meanreversion,bollinger',
              help='Comma-separated strategies to test')
@click.option('--days', '-d', type=int, default=750, help='Days of historical data')
@click.option('--metric', type=click.Choice(['composite', 'sharpe_ratio', 'sortino_ratio', 'total_return']),
              default='sharpe_ratio', help='Optimization metric')
@click.option('--dry-run', is_flag=True, help='Preview without saving config')
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
    import sys
    import subprocess

    # Build command
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "optimize_routing_multitf.py"),
        '--symbols', symbols,
        '--timeframes', timeframes,
        '--strategies', strategies,
        '--days', str(days),
        '--metric', metric
    ]

    if dry_run:
        cmd.append('--dry-run')

    click.echo(f"Running multi-timeframe optimization...")
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
        click.secho("Optimization complete!", fg='green')
        click.echo("Run 'amsterdam strategy show' to view routing configuration.")
    elif result.returncode != 0:
        click.secho(f"Optimization failed with exit code {result.returncode}", fg='red')


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


@backtest.command('run')
@click.argument('symbol')
@click.option('--strategy', '-s', default='sma', help='Strategy to backtest')
@click.option('--days', '-d', type=int, default=365, help='Days of historical data')
@click.option('--capital', type=float, default=10000, help='Initial capital')
@click.option('--hybrid/--no-hybrid', default=False, help='Use hybrid sizing')
@click.option('-v', '--verbose', is_flag=True, help='Verbose output')
def backtest_run(symbol, strategy, days, capital, hybrid, verbose):
    """Run a backtest for a single strategy.

    \b
    Examples:
        amsterdam backtest run AAPL -s sma
        amsterdam backtest run MSFT -s macd --hybrid
        amsterdam backtest run TSLA -s rsi -d 180 --capital 50000
    """
    import asyncio
    from core.unified_data_pipeline import UnifiedDataPipeline
    from core.backtest.unified_backtest_runner import UnifiedBacktestRunner, BacktestConfig

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
            click.secho(f"Error: Could not load data for {symbol}", fg='red')
            return

        if len(data) > days:
            data = data.tail(days).reset_index(drop=True)

        click.echo(f"Loaded {len(data)} bars")

    except Exception as e:
        click.secho(f"Error loading data: {e}", fg='red')
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
        click.secho(f"Error running backtest: {e}", fg='red')
        import traceback
        traceback.print_exc()


@backtest.command('compare')
@click.argument('symbol')
@click.option('--strategies', '-s', default='sma,ema,macd,rsi',
              help='Comma-separated strategies to compare')
@click.option('--days', '-d', type=int, default=365, help='Days of data')
@click.option('--metric', '-m', default='sharpe_ratio',
              type=click.Choice(['sharpe_ratio', 'total_return', 'sortino_ratio']),
              help='Metric for ranking')
@click.option('--capital', type=float, default=10000, help='Initial capital')
def backtest_compare(symbol, strategies, days, metric, capital):
    """Compare multiple strategies.

    \b
    Examples:
        amsterdam backtest compare AAPL -s sma,ema,macd,rsi
        amsterdam backtest compare MSFT -s momentum,breakout -m total_return
    """
    # Delegate to compare_strategies tool
    cmd = [
        sys.executable, str(ROOT / "tools" / "compare_strategies.py"),
        symbol.upper(),
        "-s", strategies,
        "-d", str(days),
        "-m", metric,
        "--capital", str(capital),
    ]

    os.execv(sys.executable, cmd)


@backtest.command('hybrid')
@click.argument('symbol')
@click.option('--strategies', '-s', default='sma,macd,rsi',
              help='Comma-separated strategies')
@click.option('--days', '-d', type=int, default=365, help='Days of data')
@click.option('--capital', type=float, default=10000, help='Initial capital')
def backtest_hybrid(symbol, strategies, days, capital):
    """Compare hybrid vs standard sizing.

    \b
    Examples:
        amsterdam backtest hybrid AAPL -s sma,macd,rsi
        amsterdam backtest hybrid MSFT -s momentum,ema,bollinger
    """
    # Delegate to compare_strategies tool with hybrid flag
    cmd = [
        sys.executable, str(ROOT / "tools" / "compare_strategies.py"),
        symbol.upper(),
        "-s", strategies,
        "-d", str(days),
        "--capital", str(capital),
        "--hybrid-comparison",
    ]

    os.execv(sys.executable, cmd)


@backtest.command('categories')
@click.argument('symbol')
@click.option('--categories', '-c', default='trend_following,mean_reversion',
              help='Comma-separated categories')
@click.option('--days', '-d', type=int, default=365, help='Days of data')
@click.option('--capital', type=float, default=10000, help='Initial capital')
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
        sys.executable, str(ROOT / "tools" / "compare_strategies.py"),
        symbol.upper(),
        "--categories", categories,
        "-d", str(days),
        "--capital", str(capital),
    ]

    os.execv(sys.executable, cmd)


@backtest.command('full')
@click.argument('symbol')
@click.option('--strategies', '-s', default=None, help='Strategies to include')
@click.option('--days', '-d', type=int, default=365, help='Days of data')
@click.option('-o', '--output', default=None, help='Output file (.md or .json)')
@click.option('--capital', type=float, default=10000, help='Initial capital')
def backtest_full(symbol, strategies, days, output, capital):
    """Run full comparison analysis.

    \b
    Examples:
        amsterdam backtest full AAPL
        amsterdam backtest full MSFT -o reports/msft_analysis.md
        amsterdam backtest full TSLA -s sma,macd,rsi,momentum --days 180
    """
    cmd = [
        sys.executable, str(ROOT / "tools" / "compare_strategies.py"),
        symbol.upper(),
        "-d", str(days),
        "--capital", str(capital),
        "--full",
    ]

    if strategies:
        cmd.extend(["-s", strategies])

    if output:
        cmd.extend(["-o", output])

    os.execv(sys.executable, cmd)


@backtest.command('optimize')
@click.argument('mode', required=False, default=None)
@click.option('--symbols', '-s', default=None, help='Comma-separated symbols (default: all)')
@click.option('--days', '-d', type=int, default=365, help='Days of data')
@click.option('--strategies', default=None, help='Comma-separated strategies to test')
@click.option('--timeframes', '-t', default=None,
              help='Comma-separated timeframes (e.g., 15min,30min,1hour) - enables multi-TF optimization')
@click.option('--dry-run', is_flag=True, help="Don't save config")
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
    if mode == 'all':
        # Default to all strategies
        if not strategies:
            strategies = 'adx,bollinger,breakout,combined,donchian,ema,ichimoku,logisticregression,macd,meanreversion,momentum,psar,rsi,sma,stochastic,vwap'

        # Default to all timeframes
        if not timeframes:
            timeframes = '15min,30min,1hour,day'

        # Default to 750 days for comprehensive backtest
        if days == 365:  # If using default value
            days = 750

        click.echo("Running comprehensive optimization:")
        click.echo(f"  • All 16 strategies")
        click.echo(f"  • All 4 timeframes (15min, 30min, 1hour, day)")
        click.echo(f"  • {days} days of data")
        click.echo()

    # Determine which optimizer to use based on timeframes option
    if timeframes:
        # Use multi-timeframe optimizer
        cmd = [
            sys.executable, str(ROOT / "tools" / "optimize_routing_multitf.py"),
            "-d", str(days),
            "--timeframes", timeframes,
        ]
    else:
        # Use single-timeframe optimizer (backward compatible)
        cmd = [
            sys.executable, str(ROOT / "tools" / "optimize_routing.py"),
            "-d", str(days),
        ]

    if symbols:
        cmd.extend(["-s", symbols])

    if strategies:
        cmd.extend(["--strategies", strategies])

    if dry_run:
        cmd.append("--dry-run")

    os.execv(sys.executable, cmd)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
