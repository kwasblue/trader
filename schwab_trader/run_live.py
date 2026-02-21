#!/usr/bin/env python3
"""
===============================================================================
DEPRECATED ENTRY POINT
===============================================================================

This entry point is deprecated. Use the CLI instead:

    # Preferred way to run the trading system:
    schwab-trader run --mode alpaca --symbols AAPL,MSFT
    python -m cli.main run --mode alpaca --symbols AAPL,MSFT

    # For Schwab:
    schwab-trader run --mode schwab --symbols AAPL,MSFT

See: python -m cli.main --help

===============================================================================
Consolidated Live Trading Entry Point (Legacy)
===============================================================================
Runs the AlpacaLiveRunner alongside the monitoring GUI.
Both share the same EventHandler for real-time data flow.

Usage (deprecated):
    python run_live.py
    python run_live.py --symbols AAPL,MSFT,GOOGL
    python run_live.py --no-gui  # headless mode
"""
import sys
import asyncio
import argparse
import logging
import warnings
from pathlib import Path

# Emit deprecation warning
warnings.warn(
    "\n"
    "=" * 70 + "\n"
    "DEPRECATION WARNING: run_live.py is deprecated!\n"
    "=" * 70 + "\n"
    "Use the CLI instead: schwab-trader run --mode alpaca --symbols AAPL\n"
    "Or: python -m cli.main run --mode alpaca --symbols AAPL\n"
    "=" * 70,
    DeprecationWarning,
    stacklevel=2
)

# Project root setup
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".venv" / ".env")

# Initialize root logger with proper file handler
# This ensures all component logs with propagate=True end up in app.log
from loggers.bootstrap import init_root_logger
init_root_logger(log_dir="logs", root_file="app.log", level=logging.DEBUG, console=True)

logger = logging.getLogger("LiveTrader")


def parse_args():
    parser = argparse.ArgumentParser(description="Run live trading with optional GUI")
    parser.add_argument(
        "--symbols", "-s",
        type=str,
        default="AAPL,MSFT",
        help="Comma-separated list of symbols to trade"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in headless mode without GUI"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Use paper trading (default: True)"
    )
    return parser.parse_args()


async def run_headless(symbols: list[str]):
    """Run the live runner without GUI."""
    from utils.settings import Settings
    from core.alpaca_runner import AlpacaLiveRunner

    settings = Settings(root="config", include_root=True)
    runner = AlpacaLiveRunner(settings, symbols)

    logger.info(f"Starting headless live runner for: {symbols}")
    await runner.run()


def run_with_gui(symbols: list[str]):
    """Run both the live runner and monitoring GUI."""
    from PySide6 import QtWidgets
    from qasync import QEventLoop

    from monitoring.theme import apply_dark_palette
    from monitoring.views.main_window import MainWindow
    from core.events.eventhandler import get_event_handler
    from utils.settings import Settings
    from core.alpaca_runner import AlpacaLiveRunner

    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)

    # Setup async event loop with Qt
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Shared event bus
    bus = get_event_handler()

    # Create GUI components
    # NOTE: MainWindow creates its own DataFeeder in __init__, so don't create one here
    win = MainWindow()
    win.bus = bus
    # Use MainWindow's feeder instead of creating a duplicate
    feeder = win.feeder
    win.show()

    async def startup():
        """Start both feeder and live runner."""
        try:
            await asyncio.sleep(0.5)  # Let Qt initialize

            # Start the data feeder (listens to events)
            await feeder.start()
            win._append_log("[INIT] Data feeder connected to event bus")

            # Start the live runner
            settings = Settings(root="config", include_root=True)
            runner = AlpacaLiveRunner(settings, symbols)

            win._append_log(f"[INIT] Starting live runner for: {symbols}")

            # Run the live runner in the background
            asyncio.create_task(runner.run())

            win._append_log("[INIT] Live trading system active")

        except Exception as e:
            logger.exception(f"Startup failed: {e}")
            win._append_log(f"[ERROR] Startup failed: {e}")

    async def shutdown():
        """Graceful shutdown."""
        try:
            logger.info("Shutting down...")
            await feeder.stop()
            logger.info("Shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    # Wire shutdown handler
    app.aboutToQuit.connect(lambda: asyncio.create_task(shutdown()))

    # Schedule startup
    loop.create_task(startup())

    # Run the event loop
    try:
        with loop:
            loop.run_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        QtWidgets.QApplication.processEvents()
        sys.exit(0)


def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    logger.info(f"Symbols: {symbols}")
    logger.info(f"Mode: {'Headless' if args.no_gui else 'GUI'}")
    logger.info(f"Paper: {args.paper}")

    if args.no_gui:
        asyncio.run(run_headless(symbols))
    else:
        run_with_gui(symbols)


if __name__ == "__main__":
    main()
