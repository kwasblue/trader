#!/usr/bin/env python
"""
AutoTrader - Autonomous Trading Daemon
=======================================

DAEMON MODULE: This is the autonomous trading daemon that runs continuously.
Invoked via: amsterdam start

This module provides:
- Market hour scheduling (MarketScheduler)
- Daily trading cycle (wait -> preflight -> trade -> update -> sleep)
- Daemon lifecycle management (start/stop/status)

It does NOT define application composition - that lives in app/bootstrap.py.

Canonical path:
    cli/main.py -> app/daemon.py -> bootstrap_app() -> AppContext
                                 -> RunnerFactory.create() -> runner.run()

The AutoTrader class receives an AppContext from bootstrap_app(), ensuring
all automation uses the same composition root as CLI and GUI modes.

Usage (via CLI - recommended):
    amsterdam start                        # Start with defaults
    amsterdam start -s AAPL,MSFT           # Specific symbols
    amsterdam start --broker schwab        # Use Schwab
    amsterdam start --dry-run              # No real trades
    amsterdam start --foreground           # Run in foreground

Usage (direct - for debugging):
    python app/daemon.py --symbols AAPL MSFT TSLA
    python app/daemon.py --broker alpaca --dry-run
"""

from __future__ import annotations

import os
import sys
import asyncio
import signal
import argparse
import atexit
import fcntl
import warnings
from pathlib import Path
from datetime import datetime, time, timedelta
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from enum import Enum
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from app.bootstrap import AppContext

# Suppress sklearn warnings for small sample sizes
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

# Add project root to path (daemon.py is in app/, so parent is project root)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# INSTANCE LOCK - Prevents multiple autotraders from running simultaneously
# ============================================================================

class InstanceLock:
    """
    Prevents multiple instances of autotrader from running.

    Uses a PID file with file locking to ensure only one instance runs at a time.
    This prevents websocket connection limit issues with Alpaca/Schwab.
    """

    def __init__(self, name: str = "autotrader"):
        self.lock_file = ROOT / "logs" / f"{name}.pid"
        self.lock_fd = None
        self._locked = False

    def acquire(self) -> bool:
        """
        Attempt to acquire the instance lock.

        Returns:
            True if lock acquired, False if another instance is running
        """
        # Ensure logs directory exists
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Open (or create) the lock file
            self.lock_fd = open(self.lock_file, 'w')

            # Try to get an exclusive lock (non-blocking)
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write our PID
            self.lock_fd.write(str(os.getpid()))
            self.lock_fd.flush()
            self._locked = True

            # Register cleanup on exit
            atexit.register(self.release)

            return True

        except (IOError, OSError):
            # Lock is held by another process
            if self.lock_fd:
                self.lock_fd.close()
                self.lock_fd = None

            # Try to read the other PID
            try:
                with open(self.lock_file, 'r') as f:
                    other_pid = f.read().strip()
                print(f"ERROR: Another autotrader instance is running (PID: {other_pid})")
                print(f"       Kill it with: kill {other_pid}")
                print(f"       Or force: kill -9 {other_pid}")
            except Exception:
                print("ERROR: Another autotrader instance is running")

            return False

    def release(self):
        """Release the instance lock."""
        if self._locked and self.lock_fd:
            try:
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
                self.lock_file.unlink(missing_ok=True)
            except Exception:
                pass
            self._locked = False
            self.lock_fd = None

    def __enter__(self):
        if not self.acquire():
            sys.exit(1)
        return self

    def __exit__(self, *args):
        self.release()

# Canonical path imports - all initialization goes through bootstrap
from app.bootstrap import bootstrap_app, AppContext
from core.config_loader import get_config


# US Eastern timezone for market hours
ET = ZoneInfo("America/New_York")

# Market hours (Eastern Time)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Default values (can be overridden by config)
DEFAULT_PRE_MARKET_BUFFER_MINUTES = 15
DEFAULT_POST_MARKET_DELAY_MINUTES = 5


class AutoTraderState(Enum):
    """Current state of the autotrader."""
    INITIALIZING = "initializing"
    WAITING_FOR_MARKET = "waiting_for_market"
    PRE_FLIGHT = "pre_flight"
    TRADING = "trading"
    POST_MARKET = "post_market"
    UPDATING_DATA = "updating_data"
    SLEEPING = "sleeping"
    STOPPED = "stopped"
    ERROR = "error"


class MarketScheduler:
    """
    Handles market schedule calculations.

    Knows when market opens/closes and handles weekends.
    For production, integrate with exchange_calendars for holiday support.
    """

    # US market holidays 2024-2025 (add more as needed)
    HOLIDAYS = {
        # 2024
        datetime(2024, 1, 1).date(),   # New Year's Day
        datetime(2024, 1, 15).date(),  # MLK Day
        datetime(2024, 2, 19).date(),  # Presidents Day
        datetime(2024, 3, 29).date(),  # Good Friday
        datetime(2024, 5, 27).date(),  # Memorial Day
        datetime(2024, 6, 19).date(),  # Juneteenth
        datetime(2024, 7, 4).date(),   # Independence Day
        datetime(2024, 9, 2).date(),   # Labor Day
        datetime(2024, 11, 28).date(), # Thanksgiving
        datetime(2024, 12, 25).date(), # Christmas
        # 2025
        datetime(2025, 1, 1).date(),   # New Year's Day
        datetime(2025, 1, 20).date(),  # MLK Day
        datetime(2025, 2, 17).date(),  # Presidents Day
        datetime(2025, 4, 18).date(),  # Good Friday
        datetime(2025, 5, 26).date(),  # Memorial Day
        datetime(2025, 6, 19).date(),  # Juneteenth
        datetime(2025, 7, 4).date(),   # Independence Day
        datetime(2025, 9, 1).date(),   # Labor Day
        datetime(2025, 11, 27).date(), # Thanksgiving
        datetime(2025, 12, 25).date(), # Christmas
        # 2026
        datetime(2026, 1, 1).date(),   # New Year's Day
        datetime(2026, 1, 19).date(),  # MLK Day
        datetime(2026, 2, 16).date(),  # Presidents Day
        datetime(2026, 4, 3).date(),   # Good Friday
        datetime(2026, 5, 25).date(),  # Memorial Day
        datetime(2026, 6, 19).date(),  # Juneteenth
        datetime(2026, 7, 3).date(),   # Independence Day (observed)
        datetime(2026, 9, 7).date(),   # Labor Day
        datetime(2026, 11, 26).date(), # Thanksgiving
        datetime(2026, 12, 25).date(), # Christmas
    }

    def __init__(self, logger=None):
        self.logger = logger

    def now_et(self) -> datetime:
        """Get current time in Eastern timezone."""
        return datetime.now(ET)

    def is_trading_day(self, date: datetime = None) -> bool:
        """Check if given date is a trading day."""
        if date is None:
            date = self.now_et()

        # Weekend check
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Holiday check
        if date.date() in self.HOLIDAYS:
            return False

        return True

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = self.now_et()

        if not self.is_trading_day(now):
            return False

        current_time = now.time()
        return MARKET_OPEN <= current_time < MARKET_CLOSE

    def get_next_market_open(self) -> datetime:
        """Get datetime of next market open."""
        now = self.now_et()

        # Start with today
        check_date = now.date()

        # If we're past today's open, start checking tomorrow
        if now.time() >= MARKET_OPEN:
            check_date += timedelta(days=1)

        # Find next trading day
        while True:
            candidate = datetime.combine(check_date, MARKET_OPEN, tzinfo=ET)
            if self.is_trading_day(candidate):
                return candidate
            check_date += timedelta(days=1)

            # Safety: don't loop forever
            if (check_date - now.date()).days > 10:
                raise RuntimeError("Could not find next trading day within 10 days")

    def get_market_close_today(self) -> datetime:
        """Get today's market close time."""
        now = self.now_et()
        return datetime.combine(now.date(), MARKET_CLOSE, tzinfo=ET)

    def seconds_until_market_open(self) -> float:
        """Get seconds until next market open."""
        now = self.now_et()
        next_open = self.get_next_market_open()
        return (next_open - now).total_seconds()

    def seconds_until_market_close(self) -> float:
        """Get seconds until today's market close."""
        now = self.now_et()
        close = self.get_market_close_today()
        return (close - now).total_seconds()


class AutoTrader:
    """
    Autonomous trading daemon.

    Manages the full daily trading cycle:
    - Pre-market preparation
    - Trading session
    - Post-market data updates
    - Sleep until next session

    Uses the canonical path: bootstrap_app() -> AppContext -> RunnerFactory

    Logs to: logs/autotrader.log
    """

    def __init__(
        self,
        ctx: AppContext,
        dry_run: bool = False,
        update_data_days: int = 5,
        day_trade: bool = False,
    ):
        # Store AppContext - single source of truth
        self.ctx = ctx
        self.symbols = ctx.symbols
        self.broker = ctx.metadata.get('broker', 'alpaca')
        self.dry_run = dry_run
        self.update_data_days = update_data_days
        self.day_trade = day_trade

        self.state = AutoTraderState.INITIALIZING
        self.scheduler = MarketScheduler()
        self.running = False
        self.trading_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "sessions_completed": 0,
            "total_trades": 0,
            "last_session_start": None,
            "last_session_end": None,
            "last_data_update": None,
            "errors": [],
        }

        # Use logger and config from canonical AppContext
        self.logger = ctx.logger
        self.config = ctx.config

        # Override swing mode if day trading is enabled
        if day_trade:
            from core.config_loader import enable_day_trade_mode
            self.config = enable_day_trade_mode()
            self.logger.info("DAY TRADE MODE: Swing mode disabled, same-day exits allowed")

        # Load timing settings from config
        self.pre_market_buffer = self.config.autotrader.pre_market_buffer_minutes
        self.post_market_delay = self.config.autotrader.post_market_delay_minutes

        self.logger.info("=" * 60)
        self.logger.info("AUTOTRADER INITIALIZED (via canonical path)")
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info(f"Broker: {self.broker}")
        self.logger.info(f"Dry run: {dry_run}")
        self.logger.info(f"Day trade: {day_trade}")
        self.logger.info("=" * 60)

    def _set_state(self, new_state: AutoTraderState) -> None:
        """Update state with logging."""
        old_state = self.state
        self.state = new_state
        self.logger.info(f"State change: {old_state.value} -> {new_state.value}")
        print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] State: {new_state.value}")

    async def run(self) -> None:
        """Main run loop - runs indefinitely until stopped."""
        self.running = True
        self._set_state(AutoTraderState.INITIALIZING)

        self.logger.info("Starting AutoTrader main loop")
        print(f"\n{'='*60}")
        print(f"  AUTOTRADER STARTED")
        print(f"  Symbols: {', '.join(self.symbols)}")
        print(f"  Broker: {self.broker}")
        print(f"  Dry run: {self.dry_run}")
        print(f"{'='*60}\n")

        try:
            while self.running:
                await self._run_daily_cycle()

        except asyncio.CancelledError:
            self.logger.info("AutoTrader cancelled")
        except Exception as e:
            self.logger.exception(f"AutoTrader error: {e}")
            self._set_state(AutoTraderState.ERROR)
            raise
        finally:
            self._set_state(AutoTraderState.STOPPED)
            self.logger.info("AutoTrader stopped")

    async def _run_daily_cycle(self) -> None:
        """Execute one full daily trading cycle."""
        self.logger.info("=" * 40)
        self.logger.info("STARTING DAILY CYCLE")
        self.logger.info("=" * 40)

        # 1. Wait for market to open
        await self._wait_for_market_open()

        if not self.running:
            return

        # 2. Run pre-flight checks
        preflight_ok = await self._run_preflight()

        if not preflight_ok:
            self.logger.error("Pre-flight checks failed, skipping trading session")
            self.stats["errors"].append(f"{self.scheduler.now_et()}: Pre-flight failed")
            # Still do post-market activities
        else:
            # 3. Wait for market to actually open (preflight runs in pre-market buffer)
            await self._wait_for_market_to_open()

            if not self.running:
                return

            # 4. Run trading session
            await self._run_trading_session()

        if not self.running:
            return

        # 5. Post-market: update historical data
        await self._run_post_market()

        if not self.running:
            return

        # 6. Post-market: scan for new symbols and optimize strategies
        await self._run_scan_and_optimize()

        if not self.running:
            return

        # 7. Sleep until next trading day
        await self._sleep_until_next_day()

    async def _wait_for_market_open(self) -> None:
        """Wait until market opens (with pre-market buffer)."""
        self._set_state(AutoTraderState.WAITING_FOR_MARKET)

        # Track last log time to avoid spamming logs
        last_log_time = None

        while self.running:
            now = self.scheduler.now_et()

            if self.scheduler.is_market_open():
                self.logger.info("Market is open!")
                return

            # Calculate wait time (recalculate each iteration for wall-clock accuracy)
            next_open = self.scheduler.get_next_market_open()

            # Account for pre-market buffer
            preflight_start = next_open - timedelta(minutes=self.pre_market_buffer)

            if now >= preflight_start:
                self.logger.info(f"Pre-market window reached, proceeding to preflight")
                return

            wait_seconds = (preflight_start - now).total_seconds()

            # Log status periodically (every 5 minutes, not every loop)
            if last_log_time is None or (now - last_log_time).total_seconds() >= 300:
                hours = int(wait_seconds // 3600)
                minutes = int((wait_seconds % 3600) // 60)

                self.logger.info(f"Market opens at {next_open.strftime('%Y-%m-%d %H:%M %Z')}")
                self.logger.info(f"Waiting {hours}h {minutes}m until pre-market window")
                print(f"[{now.strftime('%H:%M:%S')}] Waiting for market. Opens in {hours}h {minutes}m")
                last_log_time = now

            # Sleep in chunks to allow graceful shutdown
            # Use wall-clock time check after sleep to handle system suspend
            sleep_time = min(wait_seconds, 300)  # Max 5 minute sleep chunks
            await asyncio.sleep(sleep_time)

    async def _wait_for_market_to_open(self) -> None:
        """Wait until market is actually open (after preflight completes in pre-market buffer)."""
        if self.scheduler.is_market_open():
            return

        now = self.scheduler.now_et()
        next_open = self.scheduler.get_next_market_open()
        wait_seconds = (next_open - now).total_seconds()

        if wait_seconds <= 0:
            return

        minutes = int(wait_seconds // 60)
        seconds = int(wait_seconds % 60)

        self.logger.info(f"Waiting {minutes}m {seconds}s for market to open at {next_open.strftime('%H:%M')} ET")
        print(f"[{now.strftime('%H:%M:%S')}] Pre-flight complete. Market opens in {minutes}m {seconds}s")

        # Sleep in small chunks to allow graceful shutdown
        # Use wall-clock time check after each sleep to handle system suspend
        while self.running:
            # Check if market opened (handles edge cases and system suspend)
            if self.scheduler.is_market_open():
                return

            # Recalculate remaining time using wall clock
            now = self.scheduler.now_et()
            wait_seconds = (next_open - now).total_seconds()

            if wait_seconds <= 0:
                return

            chunk = min(wait_seconds, 30)  # 30 second chunks
            await asyncio.sleep(chunk)

    async def _check_network_connectivity(self) -> bool:
        """
        Check if network is available by attempting DNS resolution.

        Returns True if network is reachable, False otherwise.
        """
        import socket

        hosts_to_check = [
            ("paper-api.alpaca.markets", 443),
            ("api.alpaca.markets", 443),
            ("google.com", 443),
        ]

        for host, port in hosts_to_check:
            try:
                # Try DNS resolution
                socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
                return True
            except socket.gaierror:
                continue

        return False

    async def _wait_for_network(self) -> bool:
        """
        Wait for network connectivity until market opens.

        Returns True if network became available, False if market opened without network.
        """
        check_interval = 30  # seconds

        while self.running:
            if await asyncio.to_thread(self._check_network_connectivity_sync):
                self.logger.info("Network connectivity confirmed")
                return True

            # Check if we've run out of time (market is about to open)
            seconds_to_open = self.scheduler.seconds_until_market_open()
            if seconds_to_open <= 60:  # Less than 1 minute to market open
                self.logger.error("Market opening soon but no network connectivity")
                return False

            self.logger.warning(
                f"No network connectivity, retrying in {check_interval}s... "
                f"({int(seconds_to_open/60)} min until market open)"
            )
            print(
                f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] "
                f"Waiting for network... ({int(seconds_to_open/60)} min until market open)"
            )
            await asyncio.sleep(check_interval)

        return False

    def _check_network_connectivity_sync(self) -> bool:
        """Synchronous network check for use with asyncio.to_thread."""
        import socket

        hosts_to_check = [
            ("paper-api.alpaca.markets", 443),
            ("api.alpaca.markets", 443),
            ("google.com", 443),
        ]

        for host, port in hosts_to_check:
            try:
                socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
                return True
            except socket.gaierror:
                continue

        return False

    async def _run_preflight(self) -> bool:
        """
        Run pre-flight checks with network wait and retry logic.

        First waits for network connectivity (until market opens),
        then runs preflight with retries for other transient failures.
        """
        self._set_state(AutoTraderState.PRE_FLIGHT)

        # First, ensure we have network connectivity
        self.logger.info("Checking network connectivity...")
        print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Checking network connectivity...")

        if not await self._wait_for_network():
            self.logger.error("Failed to establish network connectivity")
            print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] No network connectivity")
            return False

        # Now run preflight with retries
        max_retries = getattr(self.config.autotrader, 'preflight_max_retries', 3)
        retry_delay = getattr(self.config.autotrader, 'preflight_retry_delay', 60)

        for attempt in range(1, max_retries + 1):
            self.logger.info(f"Running pre-flight checks (attempt {attempt}/{max_retries})...")
            print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Running pre-flight checks (attempt {attempt}/{max_retries})...")

            try:
                from preflight import PreFlightChecker

                checker = PreFlightChecker(verbose=False)
                success = await checker.run_all_checks(
                    symbols=self.symbols,
                    update_data=True,  # Update stale data
                    reauth_schwab=False,  # Don't attempt manual reauth in auto mode
                )

                if success:
                    self.logger.info("Pre-flight checks PASSED")
                    print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Pre-flight: PASSED")
                    return True

                # Check if we should retry
                if attempt < max_retries:
                    self.logger.warning(
                        f"Pre-flight checks failed, retrying in {retry_delay}s..."
                    )
                    print(
                        f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] "
                        f"Pre-flight failed, retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error("Pre-flight checks FAILED after all retries")
                    print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Pre-flight: FAILED")
                    return False

            except Exception as e:
                self.logger.exception(f"Pre-flight error (attempt {attempt}): {e}")

                if attempt < max_retries:
                    self.logger.warning(f"Retrying in {retry_delay}s...")
                    print(
                        f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] "
                        f"Pre-flight error, retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Pre-flight ERROR: {e}")
                    return False

        return False

    async def _run_trading_session(self) -> None:
        """Run the trading session until market close."""
        self._set_state(AutoTraderState.TRADING)

        self.stats["last_session_start"] = self.scheduler.now_et()

        self.logger.info("Starting trading session")
        print(f"\n[{self.scheduler.now_et().strftime('%H:%M:%S')}] TRADING SESSION STARTED")
        print(f"  Market closes at {MARKET_CLOSE.strftime('%H:%M')} ET\n")

        if self.dry_run:
            self.logger.info("DRY RUN - not executing actual trades")
            print("  ** DRY RUN MODE - No actual trades **\n")

            # In dry run, just wait until market close
            while self.running and self.scheduler.is_market_open():
                await asyncio.sleep(60)
                now = self.scheduler.now_et()
                remaining = self.scheduler.seconds_until_market_close()
                mins_remaining = int(remaining // 60)
                self.logger.debug(f"Dry run: {mins_remaining} minutes until close")

            self.stats["last_session_end"] = self.scheduler.now_et()
            self.stats["sessions_completed"] += 1
            return

        try:
            # Start the actual trading runner using factory
            await self._run_broker_session()

        except Exception as e:
            self.logger.exception(f"Trading session error: {e}")
            self.stats["errors"].append(f"{self.scheduler.now_et()}: {e}")
        finally:
            self.stats["last_session_end"] = self.scheduler.now_et()
            self.stats["sessions_completed"] += 1
            self.logger.info("Trading session ended")
            print(f"\n[{self.scheduler.now_et().strftime('%H:%M:%S')}] TRADING SESSION ENDED")

    async def _run_broker_session(self) -> None:
        """
        Run trading session for the configured broker.

        Uses RunnerFactory to create the appropriate runner based on
        self.broker. Adding new brokers only requires registering them
        with the factory - no changes to this method needed.
        """
        from core.runner_factory import RunnerFactory

        # Create runner via factory (supports alpaca, schwab, and any registered broker)
        runner = RunnerFactory.create(
            broker=self.broker,
            symbols=self.symbols,
            config=self.config
        )

        # Create task for the runner
        self.trading_task = asyncio.create_task(runner.run())

        # Config for market close position closing
        close_on_market_close = getattr(
            self.config.autotrader, 'close_positions_on_market_close', True
        )
        minutes_before = getattr(
            self.config.autotrader, 'market_close_minutes_before', 5
        )
        positions_closed = False

        try:
            # Wait until market close or stop signal
            while self.running and self.scheduler.is_market_open():
                await asyncio.sleep(10)

                # Check if we should close positions (X minutes before market close)
                if close_on_market_close and not positions_closed:
                    seconds_to_close = self.scheduler.seconds_until_market_close()
                    if seconds_to_close <= (minutes_before * 60):
                        self.logger.info(
                            f"Triggering position close {int(seconds_to_close/60)} minutes before market close"
                        )
                        await self._close_all_positions_market_close(runner)
                        positions_closed = True

            # Stop the runner gracefully
            self.logger.info(f"Market closed, stopping {self.broker} runner...")
            runner.stop()

            # Wait for runner to finish
            try:
                await asyncio.wait_for(self.trading_task, timeout=30)
            except asyncio.TimeoutError:
                self.logger.warning(f"{self.broker} runner did not stop gracefully, cancelling")
                self.trading_task.cancel()

        except asyncio.CancelledError:
            runner.stop()
            raise

    async def _close_all_positions_market_close(self, runner) -> None:
        """
        Close all positions before market close (safety net).

        This is a failsafe in case the EOD close logic in the runner didn't trigger
        due to missing bar data. Directly accesses the broker to close positions.
        Triggered X minutes before market close (configurable via market_close_minutes_before).
        """
        self.logger.info("=" * 60)
        self.logger.info("PRE-MARKET-CLOSE: Checking for open positions to close")
        self.logger.info("=" * 60)

        try:
            broker = runner.broker
            positions = await broker.get_positions()

            if not positions:
                self.logger.info("Market close: No positions to close")
                return

            # Filter to only positions with non-zero qty
            open_positions = [p for p in positions if p.qty != 0]

            if not open_positions:
                self.logger.info("Market close: No open positions to close")
                return

            self.logger.warning(
                f"Market close safety net: Found {len(open_positions)} open positions to close"
            )
            print(
                f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] "
                f"WARNING: Closing {len(open_positions)} positions at market close"
            )

            closed_count = 0
            for pos in open_positions:
                try:
                    qty = abs(int(pos.qty))
                    side = "sell" if pos.qty > 0 else "buy"

                    self.logger.info(f"[{pos.symbol}] Market close: {side} {qty} shares")

                    await broker.place_market_order(
                        symbol=pos.symbol,
                        qty=qty,
                        side=side
                    )
                    closed_count += 1

                except Exception as e:
                    self.logger.error(f"[{pos.symbol}] Market close failed: {e}")

            self.logger.info(f"Market close complete: {closed_count}/{len(open_positions)} positions closed")

        except Exception as e:
            self.logger.error(f"Market close position check failed: {e}")

    async def _run_post_market(self) -> None:
        """Run post-market activities (data updates)."""
        self._set_state(AutoTraderState.POST_MARKET)

        self.logger.info(f"Post-market: waiting {self.post_market_delay} minutes...")
        print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Post-market cooldown...")

        # Wait a bit for data to settle
        await asyncio.sleep(self.post_market_delay * 60)

        if not self.running:
            return

        # Update historical data
        self._set_state(AutoTraderState.UPDATING_DATA)

        self.logger.info("Updating historical data...")
        print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Updating historical data...")

        try:
            from core.unified_data_pipeline import UnifiedDataPipeline

            pipeline = UnifiedDataPipeline()
            results = await pipeline.update_symbols(
                symbols=self.symbols,
                days=self.update_data_days,
                process_data=True,
            )

            total_bars = sum(results.values())
            self.logger.info(f"Data update complete: {total_bars} total bars across {len(self.symbols)} symbols")
            print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Data updated: {total_bars} bars")

            self.stats["last_data_update"] = self.scheduler.now_et()

        except Exception as e:
            self.logger.exception(f"Data update error: {e}")
            self.stats["errors"].append(f"{self.scheduler.now_et()}: Data update failed - {e}")

    async def _run_scan_and_optimize(self) -> None:
        """Run stock scanner and optimize strategies for trade candidates.

        Post-market step: scans for new symbols, updates trade/watch lists,
        then runs regime backtest to assign optimal strategies.
        Non-fatal — errors are logged and trading continues with existing config.
        """
        from core.config_loader import get_config

        cfg = get_config()
        if not cfg.scanner.enabled or not cfg.scanner.schedule.get("pre_market_scan_enabled", True):
            return

        self._set_state(AutoTraderState.UPDATING_DATA)
        self.logger.info("Running post-market scan and strategy optimization...")
        print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Running scanner + optimizer...")

        try:
            from scanner.engine import get_scanner
            from core.symbol_list_manager import get_list_manager

            scanner_cfg = cfg.scanner.to_engine_config()
            scanner = get_scanner(scanner_cfg)
            schedule_cfg = cfg.scanner.schedule

            # 1. Run scan
            auto_update = schedule_cfg.get("auto_update_lists", False)
            report = scanner.scan(update_lists=auto_update)

            self.logger.info(
                f"Scan complete: {report.universe_size} screened, "
                f"{len(report.trade_candidates())} trade, "
                f"{len(report.watch_candidates())} watch"
            )

            # 2. Reload symbols if lists were updated
            if auto_update:
                updated_symbols = get_list_manager().get_trade_list()
                if updated_symbols and updated_symbols != self.symbols:
                    self.logger.info(f"Trade list updated: {len(self.symbols)} -> {len(updated_symbols)} symbols")
                    self.symbols = updated_symbols

            # 3. Optimize strategies
            if schedule_cfg.get("optimize_after_scan", True):
                opt_days = schedule_cfg.get("optimization_days", 365)
                opt_metric = schedule_cfg.get("optimization_metric", "sharpe_ratio")

                opt_results = scanner.optimize_strategies(
                    symbols=self.symbols,
                    days=opt_days,
                    metric=opt_metric,
                )

                self.logger.info(f"Strategy optimization complete: {len(opt_results)} symbols optimized")

                # 4. Reload strategy routing
                try:
                    from core.logic.strategy_routing_manager import StrategyRoutingManager
                    from pathlib import Path

                    routing_path = Path(__file__).resolve().parent.parent / "config" / "strategy_routing.json"
                    if routing_path.exists():
                        router = StrategyRoutingManager(str(routing_path))
                        router.refresh()
                        self.logger.info("Strategy routing reloaded")
                except Exception as e:
                    self.logger.warning(f"Could not reload strategy routing: {e}")

            print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Scan + optimize complete")

        except Exception as e:
            self.logger.exception(f"Scan/optimize error: {e}")
            self.stats["errors"].append(f"{self.scheduler.now_et()}: Scan/optimize failed - {e}")
            print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Scan/optimize failed (non-fatal): {e}")

    async def _sleep_until_next_day(self) -> None:
        """Sleep until next trading day."""
        self._set_state(AutoTraderState.SLEEPING)

        next_open = self.scheduler.get_next_market_open()
        now = self.scheduler.now_et()

        # Sleep until pre-market window of next day
        preflight_start = next_open - timedelta(minutes=self.pre_market_buffer)
        sleep_seconds = (preflight_start - now).total_seconds()

        if sleep_seconds <= 0:
            self.logger.info("Next trading day is imminent, not sleeping")
            return

        hours = int(sleep_seconds // 3600)
        minutes = int((sleep_seconds % 3600) // 60)

        self.logger.info(f"Sleeping until {preflight_start.strftime('%Y-%m-%d %H:%M %Z')}")
        self.logger.info(f"Sleep duration: {hours}h {minutes}m")

        print(f"\n[{now.strftime('%H:%M:%S')}] Daily cycle complete")
        print(f"  Sessions completed: {self.stats['sessions_completed']}")
        print(f"  Next session: {next_open.strftime('%Y-%m-%d %H:%M')} ET")
        print(f"  Sleeping for {hours}h {minutes}m...\n")

        # Sleep in chunks to allow graceful shutdown
        # Use wall-clock time after each sleep to handle system suspend correctly
        while self.running:
            now = self.scheduler.now_et()

            # Check if it's time to wake up (wall-clock check handles system suspend)
            if now >= preflight_start:
                self.logger.info("Wake time reached, starting new daily cycle")
                break

            # Calculate remaining time from wall clock
            remaining = (preflight_start - now).total_seconds()
            if remaining <= 0:
                break

            chunk = min(remaining, 300)  # 5 minute chunks
            await asyncio.sleep(chunk)

    def stop(self) -> None:
        """Signal the autotrader to stop."""
        self.logger.info("Stop signal received")
        print(f"\n[{self.scheduler.now_et().strftime('%H:%M:%S')}] Stopping AutoTrader...")
        self.running = False

        if self.trading_task and not self.trading_task.done():
            self.trading_task.cancel()

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "state": self.state.value,
            "running": self.running,
            "symbols": self.symbols,
            "broker": self.broker,
            "dry_run": self.dry_run,
            "market_open": self.scheduler.is_market_open(),
            "current_time_et": self.scheduler.now_et().isoformat(),
            "stats": self.stats,
        }


async def main():
    parser = argparse.ArgumentParser(
        description='Autonomous trading daemon (wrapper around canonical path)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (via CLI - recommended):
  amsterdam start                               # Start as daemon
  amsterdam start -s AAPL TSLA                  # Specific symbols
  amsterdam start --broker schwab               # Use Schwab broker
  amsterdam start --dry-run                     # No actual trading

Examples (direct - for debugging):
  python app/daemon.py                          # Run with defaults
  python app/daemon.py --symbols AAPL TSLA      # Specific symbols

Canonical Path:
  This script uses: bootstrap_app() -> AppContext -> RunnerFactory
  See app/bootstrap.py for the canonical initialization path.
        """
    )

    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=None,
        help='Symbols to trade (default: uses trade list from symbol manager)'
    )
    parser.add_argument(
        '--broker', '-b',
        choices=['alpaca', 'schwab', 'hybrid', 'alpaca-schwab'],
        default='alpaca',
        help='Broker to use (default: alpaca, hybrid: Alpaca execution + Schwab data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without executing actual trades'
    )
    parser.add_argument(
        '--day-trade',
        action='store_true',
        help='Enable day trading (disable swing mode, allow same-day exits)'
    )
    parser.add_argument(
        '--update-days',
        type=int,
        default=5,
        help='Days of historical data to update (default: 5)'
    )

    args = parser.parse_args()

    # Get symbols from trade list if not specified
    if args.symbols is None:
        from core.symbol_list_manager import get_list_manager
        symbols = get_list_manager().get_trade_list()
        if not symbols:
            print("Error: No symbols in trade list. Add symbols with:")
            print("  trader add AAPL --trade")
            sys.exit(1)
        print(f"Using trade list: {symbols}")
    else:
        symbols = args.symbols

    # =========================================================================
    # CANONICAL PATH: Use bootstrap_app() for all initialization
    # =========================================================================
    ctx = bootstrap_app(
        mode='daemon',
        symbols=symbols,
        broker=args.broker,
        trading_mode='live' if not args.dry_run else 'dry_run',
    )

    # Create autotrader with canonical AppContext
    trader = AutoTrader(
        ctx=ctx,
        dry_run=args.dry_run,
        update_data_days=args.update_days,
        day_trade=args.day_trade,
    )

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        trader.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Run
    try:
        await trader.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print(f"\nFinal status: {trader.get_status()}")


if __name__ == '__main__':
    # Acquire instance lock to prevent multiple autotraders
    lock = InstanceLock("autotrader")
    if not lock.acquire():
        print("\nTo run anyway, first stop the existing instance:")
        print("  python autotrader_ctl.py stop")
        sys.exit(1)

    try:
        asyncio.run(main())
    finally:
        lock.release()
