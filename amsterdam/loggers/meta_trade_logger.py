"""
Meta Trade Logger - ML-ready trade logging in JSONL format.

Logs trade entry conditions and outcomes for training a meta-model
that can improve trade selection.

Output format: JSONL (JSON Lines) for easy streaming and processing.

Usage:
    from loggers.meta_trade_logger import MetaTradeLogger
    from core.contracts.meta_types import TradeEntryContext, TradeExitContext

    logger = MetaTradeLogger()

    # On entry
    logger.log_entry(entry_context)

    # On exit
    logger.log_exit(exit_context)
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.contracts.meta_types import TradeEntryContext, TradeExitContext
from loggers.logger import Logger


class MetaTradeLogger:
    """
    Logger for ML training data in JSONL format.

    Writes structured trade events (entry/exit) to a JSONL file that can
    be converted to Parquet for model training.

    Example:
        logger = MetaTradeLogger(log_dir="logs")

        # Log entry
        entry = TradeEntryContext(
            trade_id="20240224_AAPL_001",
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            side="buy",
            qty=50,
            price=150.25,
            ...
        )
        logger.log_entry(entry)

        # Log exit
        exit_ctx = TradeExitContext(
            trade_id="20240224_AAPL_001",
            timestamp=datetime.now(timezone.utc),
            price=152.10,
            pnl_dollars=92.50,
            ...
        )
        logger.log_exit(exit_ctx)
    """

    def __init__(
        self,
        log_file: str = "meta_trades.jsonl",
        log_dir: str = "logs",
        logger_name: str = "MetaTradeLogger",
    ):
        """
        Initialize meta trade logger.

        Args:
            log_file: JSONL filename
            log_dir: Directory for log files
            logger_name: Logger name for app logs
        """
        # Ensure directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Full path to JSONL
        self.log_path = os.path.join(log_dir, log_file)

        # Thread lock for file writes
        self._lock = threading.Lock()

        # Setup app logger
        self._app_logger = Logger(log_file=f"{logger_name}.log", logger_name=logger_name, log_dir=log_dir).get_logger()

        self._app_logger.info(f"MetaTradeLogger initialized: {self.log_path}")

    def log_entry(self, context: TradeEntryContext) -> None:
        """
        Log a trade entry event.

        Args:
            context: TradeEntryContext with all entry features
        """
        event = {
            "event": "entry",
            "trade_id": context.trade_id,
            "timestamp": context.timestamp.isoformat(),
            "symbol": context.symbol,
            "side": context.side,
            "qty": context.qty,
            "price": context.price,
            "features": {
                "strategy": context.strategy,
                "regime": context.regime,
                "atr": context.atr,
                "atr_percentile": context.atr_percentile,
                "drawdown_portfolio_pct": context.drawdown_portfolio_pct,
                "drawdown_symbol_pct": context.drawdown_symbol_pct,
                "position_size_pct": context.position_size_pct,
                "hour_of_day": context.hour_of_day,
                "day_of_week": context.day_of_week,
                "minutes_since_open": context.minutes_since_open,
                "bars_in_regime": context.bars_in_regime,
                "hours_since_last_trade": context.hours_since_last_trade,
                "signal_strength": context.signal_strength,
            },
        }

        self._write_event(event)
        self._app_logger.info(
            f"[ENTRY] {context.trade_id} | {context.side.upper()} {context.qty} {context.symbol} @ ${context.price:.2f}"
        )

    def log_exit(self, context: TradeExitContext) -> None:
        """
        Log a trade exit event.

        Args:
            context: TradeExitContext with exit details and outcome
        """
        event = {
            "event": "exit",
            "trade_id": context.trade_id,
            "timestamp": context.timestamp.isoformat(),
            "price": context.price,
            "outcome": {
                "pnl_dollars": context.pnl_dollars,
                "pnl_percent": context.pnl_percent,
                "hold_bars": context.hold_bars,
                "mae_percent": context.mae_percent,
                "mfe_percent": context.mfe_percent,
                "exit_reason": context.exit_reason,
            },
        }

        self._write_event(event)
        pnl_sign = "+" if context.pnl_dollars >= 0 else ""
        self._app_logger.info(
            f"[EXIT] {context.trade_id} | ${pnl_sign}{context.pnl_dollars:.2f} "
            f"({pnl_sign}{context.pnl_percent:.2%}) | {context.exit_reason}"
        )

    def _write_event(self, event: dict[str, Any]) -> None:
        """
        Write an event to the JSONL file (thread-safe).

        Args:
            event: Dictionary to write as JSON line
        """
        with self._lock:
            try:
                with open(self.log_path, mode="a", encoding="utf-8") as f:
                    json.dump(event, f, separators=(",", ":"))
                    f.write("\n")
            except Exception as e:
                self._app_logger.error(f"Failed to write meta trade event: {e}")

    def flush(self) -> None:
        """Flush is a no-op since we write immediately."""
        pass

    def close(self) -> None:
        """Close the logger."""
        self._app_logger.info("MetaTradeLogger closed")


def generate_trade_id(symbol: str, timestamp: datetime | None = None) -> str:
    """
    Generate a unique trade ID.

    Format: YYYYMMDD_SYMBOL_HHMMSS_microseconds

    Args:
        symbol: Trading symbol
        timestamp: Trade timestamp (defaults to now)

    Returns:
        Unique trade ID string
    """
    ts = timestamp or datetime.now(timezone.utc)
    return f"{ts.strftime('%Y%m%d')}_{symbol}_{ts.strftime('%H%M%S')}_{ts.microsecond:06d}"


__all__ = [
    "MetaTradeLogger",
    "generate_trade_id",
]
