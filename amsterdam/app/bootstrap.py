"""
Amsterdam Canonical Path - Application Bootstrap
=================================================

This module is the CANONICAL ENTRY POINT for all Amsterdam initialization.
All entry points should use bootstrap_app() to ensure consistent setup.

CANONICAL PATH HIERARCHY
------------------------

1. Primary Entry Point:
   - cli/main.py         -> All user commands (start, stop, gui, etc.)

2. Internal Modules (invoked by CLI):
   - app/daemon.py        -> Daemon automation (invoked by 'amsterdam start')
   - monitoring/gui_app.py -> GUI application (invoked by 'amsterdam gui')

3. Canonical Initialization (this module):
   bootstrap_app(mode, symbols, broker, ...) -> AppContext

4. Composition Root:
   AppContainer(ctx) -> Lazy-loads dependencies via factories

5. Runtime Creation:
   container.get_runner() -> RunnerFactory.create() -> BaseLiveRunner

6. Trading Loop:
   runner.run() -> bar normalization -> regime detection -> strategy routing
                -> trade approval -> sizing -> execution -> portfolio update

The answer to "how does the app start?" is:
    cli/main.py -> bootstrap_app() -> AppContext -> AppContainer -> runner.run()

All entry points should follow this path. If you're adding a new entry point,
import bootstrap_app() and use it as the first initialization step.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Canonical app root path - use get_app_path() from config_loader for consistency
from core.config_loader import TradingConfig, get_app_path, get_config
from loggers.bootstrap import init_root_logger

ROOT = get_app_path()


@dataclass
class AppContext:
    """Application context returned by bootstrap

    Attributes:
        config: Trading configuration
        logger: Configured logger instance
        mode: Application mode (daemon/gui/cli)
        symbols: List of symbols to trade
        root_path: Path to project root
        metadata: Additional context (broker, trading_mode, operation, etc.)
    """

    config: TradingConfig
    logger: logging.Logger
    mode: str
    symbols: list[str]
    root_path: Path
    metadata: dict


def bootstrap_app(
    mode: Literal["daemon", "gui", "cli"],
    symbols: list[str] | None = None,
    broker: str | None = None,
    trading_mode: str | None = None,
    operation: str | None = None,
    log_level: int = logging.INFO,
    console_logging: bool = True,
) -> AppContext:
    """Canonical initialization path for all Amsterdam entrypoints

    This function provides consistent setup for logging, configuration,
    and environment across all application modes.

    Args:
        mode: Application mode (daemon/gui/cli)
        symbols: List of symbols to trade (defaults to config.general.default_symbols)
        broker: Broker name (e.g., 'schwab')
        trading_mode: Trading mode (e.g., 'simulation', 'live')
        operation: Operation name for CLI mode (e.g., 'preflight', 'optimize')
        log_level: Logging level (default: INFO)
        console_logging: Enable console logging (default: True)

    Returns:
        AppContext with config, logger, symbols, and metadata

    Example:
        >>> from app.bootstrap import bootstrap_app
        >>> ctx = bootstrap_app(mode='daemon', symbols=['AAPL'], broker='schwab')
        >>> ctx.logger.info(f"Trading {ctx.symbols}")
    """
    # 1. Load environment variables
    load_dotenv(ROOT / ".venv" / ".env")
    load_dotenv()

    # 2. Initialize unified logger
    log_files = {"daemon": "autotrader.log", "gui": "app.log", "cli": "cli.log"}
    init_root_logger(
        log_dir=str(ROOT / "logs"),
        root_file=log_files[mode],
        level=log_level,
        console=console_logging,
    )
    logger = logging.getLogger(f"Amsterdam.{mode.capitalize()}")

    # 3. Load trading configuration
    config = get_config()

    # 4. Resolve symbols (use provided or fall back to config defaults)
    resolved_symbols = symbols or config.general.default_symbols or []

    # 5. Log initialization
    logger.info(f"AMSTERDAM {mode.upper()} MODE INITIALIZED")
    logger.info(f"Symbols: {resolved_symbols}")
    if broker:
        logger.info(f"Broker: {broker}")
    if trading_mode:
        logger.info(f"Trading mode: {trading_mode}")
    if operation:
        logger.info(f"Operation: {operation}")

    # 6. Build metadata dict
    metadata = {}
    if broker:
        metadata["broker"] = broker
    if trading_mode:
        metadata["trading_mode"] = trading_mode
    if operation:
        metadata["operation"] = operation

    # 7. Create and return context
    return AppContext(
        config=config,
        logger=logger,
        mode=mode,
        symbols=resolved_symbols,
        root_path=ROOT,
        metadata=metadata,
    )
