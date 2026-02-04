#!/usr/bin/env python
"""
AutoTrader - Autonomous Trading Daemon

Runs the trading system automatically:
1. Waits for market open
2. Runs pre-flight checks
3. Executes trading strategies during market hours
4. Stops at market close
5. Updates historical data
6. Sleeps until next trading day

Usage:
    # Run in foreground
    python autotrader.py

    # Run with specific symbols
    python autotrader.py --symbols AAPL MSFT TSLA

    # Run with specific broker
    python autotrader.py --broker alpaca

    # Dry run (no actual trading)
    python autotrader.py --dry-run

    # Run as background daemon
    nohup python autotrader.py > logs/autotrader_stdout.log 2>&1 &
"""

from __future__ import annotations

import os
import sys
import asyncio
import signal
import argparse
import atexit
import fcntl
from pathlib import Path
from datetime import datetime, time, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
from zoneinfo import ZoneInfo

# Add project root to path
ROOT = Path(__file__).resolve().parent
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

from dotenv import load_dotenv
load_dotenv(ROOT / ".venv" / ".env")
load_dotenv()

from loggers.logger import Logger
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

    Logs to: logs/autotrader.log
    """

    def __init__(
        self,
        symbols: List[str],
        broker: str = "alpaca",
        dry_run: bool = False,
        update_data_days: int = 5,
        day_trade: bool = False,
    ):
        self.symbols = symbols
        self.broker = broker
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

        # Setup logging
        self.logger = Logger(
            "autotrader.log",
            "AutoTrader",
            propagate=True
        ).get_logger()

        # Config
        self.config = get_config()

        # Override swing mode if day trading is enabled
        if day_trade:
            from core.config_loader import enable_day_trade_mode
            self.config = enable_day_trade_mode()
            self.logger.info("DAY TRADE MODE: Swing mode disabled, same-day exits allowed")

        # Load timing settings from config
        self.pre_market_buffer = self.config.autotrader.pre_market_buffer_minutes
        self.post_market_delay = self.config.autotrader.post_market_delay_minutes

        self.logger.info("=" * 60)
        self.logger.info("AUTOTRADER INITIALIZED")
        self.logger.info(f"Symbols: {symbols}")
        self.logger.info(f"Broker: {broker}")
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
            # 3. Run trading session
            await self._run_trading_session()

        if not self.running:
            return

        # 4. Post-market: update historical data
        await self._run_post_market()

        if not self.running:
            return

        # 5. Sleep until next trading day
        await self._sleep_until_next_day()

    async def _wait_for_market_open(self) -> None:
        """Wait until market opens (with pre-market buffer)."""
        self._set_state(AutoTraderState.WAITING_FOR_MARKET)

        while self.running:
            now = self.scheduler.now_et()

            if self.scheduler.is_market_open():
                self.logger.info("Market is open!")
                return

            # Calculate wait time
            next_open = self.scheduler.get_next_market_open()

            # Account for pre-market buffer
            preflight_start = next_open - timedelta(minutes=self.pre_market_buffer)

            if now >= preflight_start:
                self.logger.info(f"Pre-market window reached, proceeding to preflight")
                return

            wait_seconds = (preflight_start - now).total_seconds()

            # Log status periodically
            hours = int(wait_seconds // 3600)
            minutes = int((wait_seconds % 3600) // 60)

            self.logger.info(f"Market opens at {next_open.strftime('%Y-%m-%d %H:%M %Z')}")
            self.logger.info(f"Waiting {hours}h {minutes}m until pre-market window")
            print(f"[{now.strftime('%H:%M:%S')}] Waiting for market. Opens in {hours}h {minutes}m")

            # Sleep in chunks to allow graceful shutdown
            sleep_time = min(wait_seconds, 300)  # Max 5 minute sleep chunks
            await asyncio.sleep(sleep_time)

    async def _run_preflight(self) -> bool:
        """Run pre-flight checks."""
        self._set_state(AutoTraderState.PRE_FLIGHT)

        self.logger.info("Running pre-flight checks...")
        print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Running pre-flight checks...")

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
            else:
                self.logger.error("Pre-flight checks FAILED")
                print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Pre-flight: FAILED")

            return success

        except Exception as e:
            self.logger.exception(f"Pre-flight error: {e}")
            print(f"[{self.scheduler.now_et().strftime('%H:%M:%S')}] Pre-flight ERROR: {e}")
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
            # Start the actual trading runner
            if self.broker == "alpaca":
                await self._run_alpaca_session()
            elif self.broker == "schwab":
                await self._run_schwab_session()
            else:
                raise ValueError(f"Unknown broker: {self.broker}")

        except Exception as e:
            self.logger.exception(f"Trading session error: {e}")
            self.stats["errors"].append(f"{self.scheduler.now_et()}: {e}")
        finally:
            self.stats["last_session_end"] = self.scheduler.now_et()
            self.stats["sessions_completed"] += 1
            self.logger.info("Trading session ended")
            print(f"\n[{self.scheduler.now_et().strftime('%H:%M:%S')}] TRADING SESSION ENDED")

    async def _run_alpaca_session(self) -> None:
        """Run Alpaca trading session."""
        from utils.settings import Settings
        from core.alpaca_runner import AlpacaLiveRunner

        # Create settings from config directory
        settings = Settings(root=str(ROOT / "config"), include_root=True)

        # Override paper mode for safety in auto mode
        if hasattr(settings, '_cfg'):
            settings._cfg['ALPACA_PAPER'] = True

        runner = AlpacaLiveRunner(settings, self.symbols)

        # Create task for the runner
        self.trading_task = asyncio.create_task(runner.run())

        try:
            # Wait until market close or stop signal
            while self.running and self.scheduler.is_market_open():
                await asyncio.sleep(10)

            # Stop the runner gracefully
            self.logger.info("Market closed, stopping runner...")
            runner.stop()

            # Wait for runner to finish
            try:
                await asyncio.wait_for(self.trading_task, timeout=30)
            except asyncio.TimeoutError:
                self.logger.warning("Runner did not stop gracefully, cancelling")
                self.trading_task.cancel()

        except asyncio.CancelledError:
            runner.stop()
            raise

    async def _run_schwab_session(self) -> None:
        """Run Schwab trading session."""
        from utils.settings import Settings
        from core.schwab_runner import SchwabLiveRunner

        # Create settings from config directory
        settings = Settings(root=str(ROOT / "config"), include_root=True)

        runner = SchwabLiveRunner(settings, self.symbols)

        # Create task for the runner
        self.trading_task = asyncio.create_task(runner.run())

        try:
            # Wait until market close or stop signal
            while self.running and self.scheduler.is_market_open():
                await asyncio.sleep(10)

            # Stop the runner gracefully
            self.logger.info("Market closed, stopping Schwab runner...")
            runner.stop()

            # Wait for runner to finish
            try:
                await asyncio.wait_for(self.trading_task, timeout=30)
            except asyncio.TimeoutError:
                self.logger.warning("Schwab runner did not stop gracefully, cancelling")
                self.trading_task.cancel()

        except asyncio.CancelledError:
            runner.stop()
            raise

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
        while self.running and sleep_seconds > 0:
            chunk = min(sleep_seconds, 300)  # 5 minute chunks
            await asyncio.sleep(chunk)
            sleep_seconds -= chunk

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
        description='Autonomous trading daemon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autotrader.py                          # Run with defaults
  python autotrader.py --symbols AAPL TSLA      # Specific symbols
  python autotrader.py --broker schwab          # Use Schwab broker
  python autotrader.py --dry-run                # No actual trading

Background:
  nohup python autotrader.py > logs/autotrader_stdout.log 2>&1 &
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
        choices=['alpaca', 'schwab'],
        default='alpaca',
        help='Broker to use (default: alpaca)'
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
            print("  python autotrader_ctl.py add AAPL --trade")
            sys.exit(1)
        print(f"Using trade list: {symbols}")
    else:
        symbols = args.symbols

    # Create autotrader
    trader = AutoTrader(
        symbols=symbols,
        broker=args.broker,
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
