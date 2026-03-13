#!/usr/bin/env python3
"""
Trading Monitor Application
---------------------------
INTERNAL MODULE - Use 'trader gui' instead.

This module is called by the CLI. Do not run directly.

Usage:
    trader gui                    # Simulation mode
    trader gui --mode alpaca      # Alpaca trading
    trader gui --mode schwab      # Schwab trading
"""
import sys
import warnings
from pathlib import Path

# Suppress sklearn warnings for small sample sizes
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

# Project root setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import logging
from dotenv import load_dotenv

load_dotenv(ROOT / ".venv" / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TradingApp")


def main():
    """Launch the trading monitor application."""
    from PySide6 import QtWidgets
    from qasync import QEventLoop

    from monitoring.theme import apply_dark_palette
    from monitoring.views.main_window import MainWindow

    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)

    # Create qasync event loop - bridges Qt and asyncio
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Create and show main window
    window = MainWindow()
    window.show()

    logger.info("Trading Monitor started")
    logger.info("Select mode (Simulation/Alpaca/Schwab) and click Start")

    # Run the event loop
    try:
        with loop:
            loop.run_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    import warnings
    warnings.warn(
        "\n"
        "  Direct execution is deprecated. Use the CLI instead:\n"
        "    trader gui                    # Simulation mode\n"
        "    trader gui --mode alpaca      # Alpaca trading\n"
        "    trader gui --mode schwab      # Schwab trading\n",
        DeprecationWarning,
        stacklevel=1
    )
    main()
