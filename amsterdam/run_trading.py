#!/usr/bin/env python3
"""
GUI Trading Application
=======================

SECONDARY WRAPPER: This is a GUI-specific wrapper around the canonical path.
For the canonical runtime path, see: app/bootstrap.py -> app/container.py

This file is NOT the main application shell. It provides:
- Qt GUI with qasync event loop integration
- Simulation mode (GBM price simulator) for testing
- Visual monitoring of live trading

The primary runtime path is:
    cli/main.py -> app/bootstrap.py -> app/container.py -> RunnerFactory

This GUI wrapper:
    run_trading.py -> bootstrap_app() -> TradingApplication (GUI-specific)

Usage:
    python run_trading.py                     # Default: Simulation mode
    python run_trading.py --mode simulation   # Explicit simulation
    python run_trading.py --mode alpaca       # Alpaca paper trading
    python run_trading.py --mode schwab       # Schwab live trading
    python run_trading.py --symbols AAPL,MSFT,GOOGL
"""
import sys
import asyncio
import argparse
import logging
import os
from pathlib import Path
from typing import Optional, List
from enum import Enum

# Project root setup
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bootstrap will be initialized in main() after parsing args
# This allows us to pass symbols and mode to bootstrap_app()
logger = None  # Will be set in main()


class TradingMode(Enum):
    SIMULATION = "simulation"
    ALPACA = "alpaca"
    SCHWAB = "schwab"


def parse_args():
    parser = argparse.ArgumentParser(description="Run trading system with GUI")
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["simulation", "alpaca", "schwab"],
        default="simulation",
        help="Trading mode (default: simulation)"
    )
    parser.add_argument(
        "--symbols", "-s",
        type=str,
        default="AAPL,MSFT",
        help="Comma-separated list of symbols"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Use paper trading for Alpaca (default: True)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.1,
        help="Simulation speed in seconds per bar (default: 0.1)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Number of bars to simulate (default: 600 = ~2 min)"
    )
    return parser.parse_args()


class TradingBackend:
    """
    Abstraction over different trading backends.
    All backends emit events to the shared EventHandler.
    """

    def __init__(self, mode: TradingMode, symbols: List[str], **kwargs):
        self.mode = mode
        self.symbols = symbols
        self.kwargs = kwargs
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._runner = None  # Store runner reference for clean shutdown

        # Import here to avoid circular deps
        from core.events.eventhandler import get_event_handler
        self.event_handler = get_event_handler()

    async def start(self):
        """Start the trading backend."""
        self._running = True

        if self.mode == TradingMode.SIMULATION:
            self._task = asyncio.create_task(self._run_simulation())
        elif self.mode == TradingMode.ALPACA:
            self._task = asyncio.create_task(self._run_alpaca())
        elif self.mode == TradingMode.SCHWAB:
            self._task = asyncio.create_task(self._run_schwab())

        logger.info(f"Backend started: {self.mode.value} for {self.symbols}")

    async def stop(self):
        """Stop the trading backend."""
        self._running = False

        # Stop the runner first (graceful shutdown)
        if self._runner is not None:
            if hasattr(self._runner, 'stop'):
                self._runner.stop()
                logger.info("Runner stop requested")

        # Then cancel the task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Backend stopped")

    async def _run_simulation(self):
        """Run GBM price simulation."""
        from core.simulator.gbm_simulator import GBMSimulator
        from core.simulator.simulation import SimConfig, SimulationRunner
        from core.events import events
        from datetime import datetime, timezone

        speed = self.kwargs.get('speed', 0.1)

        # Create simulation config
        # Use config file settings or default (600 steps = ~2 min at 0.1s/bar)
        steps = self.kwargs.get('steps', 600)
        config = SimConfig(
            symbols=self.symbols,
            steps=steps,
            bar_sleep=speed,
        )

        # Create simulation runner and store reference for clean shutdown
        self._runner = SimulationRunner(config)

        logger.info(f"[SIM] Starting simulation for {self.symbols}")

        try:
            # Use the runner's run method which handles all the simulation logic
            await self._runner.run()
        except asyncio.CancelledError:
            self._runner.stop()  # Ensure stop flag is set
            logger.info("[SIM] Simulation stopped")
        except Exception as e:
            logger.exception(f"[SIM] Error: {e}")
        finally:
            self._runner = None

    async def _run_alpaca(self):
        """Run Alpaca live/paper trading."""
        from dataclasses import replace
        from core.config_loader import get_config
        from core.alpaca_runner import AlpacaLiveRunner

        paper = self.kwargs.get('paper', True)

        # Get config and override paper mode if specified
        config = get_config()
        if paper != config.alpaca.paper:
            config = replace(config, alpaca=replace(config.alpaca, paper=paper))

        runner = AlpacaLiveRunner(symbols=self.symbols, config=config)

        try:
            await runner.run()
        except asyncio.CancelledError:
            logger.info("[ALPACA] Runner stopped")
        except Exception as e:
            logger.exception(f"[ALPACA] Error: {e}")

    async def _run_schwab(self):
        """Run Schwab live trading using SchwabLiveRunner."""
        from core.schwab_runner import SchwabLiveRunner
        import os

        logger.info("[SCHWAB] Initializing Schwab connection...")

        try:
            # Get credentials from environment variables
            api_key = os.getenv("SCHWAB_API_KEY")
            secret_key = os.getenv("SCHWAB_SECRET")

            if not api_key or not secret_key:
                logger.error("[SCHWAB] Missing credentials. Set SCHWAB_API_KEY and SCHWAB_SECRET in .env")
                return

            # Create and run the Schwab runner (uses get_config() internally)
            self._schwab_runner = SchwabLiveRunner(symbols=self.symbols)
            await self._schwab_runner.run()

        except asyncio.CancelledError:
            logger.info("[SCHWAB] Runner stopped")
            if hasattr(self, '_schwab_runner'):
                await self._schwab_runner.stop()
        except ValueError as e:
            logger.error(f"[SCHWAB] Configuration error: {e}")
        except Exception as e:
            logger.exception(f"[SCHWAB] Error: {e}")


class TradingApplication:
    """
    Main application that ties together:
    - Qt GUI with qasync event loop
    - DataFeeder for event->GUI signal bridging
    - TradingBackend for market data generation
    """

    def __init__(self, mode: TradingMode, symbols: List[str], **kwargs):
        self.mode = mode
        self.symbols = symbols
        self.kwargs = kwargs

        self.app = None
        self.loop = None
        self.window = None
        self.backend = None
        self.feeder = None

    def run(self):
        """Main entry point - sets up Qt + asyncio and runs the app."""
        from PySide6 import QtWidgets, QtCore
        from qasync import QEventLoop

        # Create Qt application first
        self.app = QtWidgets.QApplication(sys.argv)

        # Apply dark theme
        from monitoring.theme import apply_dark_palette
        apply_dark_palette(self.app)

        # Create qasync event loop - bridges Qt and asyncio
        self.loop = QEventLoop(self.app)
        asyncio.set_event_loop(self.loop)

        # Import after event loop is set
        from monitoring.views.main_window import MainWindow
        from core.events.eventhandler import get_event_handler

        # Create window and data feeder
        self.window = MainWindow()
        self.window.setWindowTitle(f"Trading Monitor — {self.mode.value.upper()}")

        # Backend reference for the window
        self.backend = TradingBackend(self.mode, self.symbols, **self.kwargs)

        # Store references in window for access
        self.window._trading_mode = self.mode
        self.window._trading_backend = self.backend

        self.window.show()

        # Wire shutdown
        self.app.aboutToQuit.connect(self._on_quit)

        # Schedule async startup after Qt is running
        self.loop.create_task(self._startup())

        # Run the event loop
        logger.info(f"Starting {self.mode.value} mode with symbols: {self.symbols}")

        try:
            with self.loop:
                self.loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            sys.exit(0)

    async def _startup(self):
        """Async startup sequence - initializes but does NOT auto-start trading."""
        try:
            await asyncio.sleep(0.5)  # Let Qt settle

            # Start the data feeder (already subscribed in __init__, this is for legacy compatibility)
            await self.window.feeder.start()
            self.window._append_log(f"[INIT] Data feeder connected to event bus")

            # Pre-populate the symbol input with command-line symbols
            if hasattr(self.window, 'symbol_input'):
                self.window.symbol_input.setText(','.join(self.symbols))

            # Set the mode dropdown to match command-line mode
            if hasattr(self.window, 'mode_combo'):
                mode_map = {'simulation': 0, 'alpaca': 1, 'schwab': 2}
                idx = mode_map.get(self.mode.value, 0)
                self.window.mode_combo.setCurrentIndex(idx)

            # Emit initial health status (system ready, not trading yet)
            from core.events import events
            from datetime import datetime, timezone
            await self.backend.event_handler.emit(events.EVENT_HEALTH_UPDATE, {
                "broker": self.mode.value,
                "status": "ready",
                "details": {"symbols": self.symbols},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            self.window._append_log(f"[INIT] Mode: {self.mode.value.upper()}")
            self.window._append_log(f"[INIT] Symbols: {', '.join(self.symbols)}")
            self.window._append_log("[INIT] System ready - click Start to begin trading")

        except Exception as e:
            logger.exception(f"Startup failed: {e}")
            self.window._append_log(f"[ERROR] Startup failed: {e}")

    def _on_quit(self):
        """Handle application quit."""
        logger.info("Application quitting...")
        if self.backend:
            asyncio.create_task(self.backend.stop())


def main():
    global logger

    args = parse_args()

    mode = TradingMode(args.mode)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Use unified bootstrap for initialization
    from app.bootstrap import bootstrap_app

    ctx = bootstrap_app(
        mode='gui',
        symbols=symbols,
        trading_mode=mode.value,
        log_level=logging.DEBUG,
        console_logging=True
    )
    logger = ctx.logger

    logger.info(f"Mode: {mode.value}")
    logger.info(f"Symbols: {symbols}")

    app = TradingApplication(
        mode=mode,
        symbols=symbols,
        paper=args.paper,
        speed=args.speed,
        steps=args.steps,
    )
    app.run()


if __name__ == "__main__":
    main()
