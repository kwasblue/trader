"""
File Trade Logger - CSV-based trade logging with flexible signatures

Extends TradeLoggerBase and provides:
- CSV file logging
- Flexible log_trade signatures (dict, positional, keyword)
- Automatic header creation
- Thread-safe file operations
- Async batched writes for high-frequency trading
- Integration with logger system
"""

from __future__ import annotations

import asyncio
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.base.trade_logger_base import TradeLoggerBase
from core.enums import OrderSide
from loggers.logger import Logger


class FileTradeLogger(TradeLoggerBase):
    """
    CSV-based trade logger with flexible interface.

    Supports multiple call signatures:
    1. Dict payload: log_trade({"symbol": "AAPL", "action": "buy", ...})
    2. Symbol-first: log_trade("AAPL", "buy", 150.0, 100, ...)
    3. Timestamp-first: log_trade(datetime.now(), "AAPL", "buy", ...)

    Example:
        logger = FileTradeLogger(
            log_file="trades.csv",
            log_dir="logs"
        )

        # Simple call
        logger.log_trade("AAPL", "buy", 150.25, 100)

        # With extras
        logger.log_trade(
            "AAPL", "buy", 150.25, 100,
            sl=148.0, tp=154.0,
            strategy="momentum"
        )

        # Dict payload
        logger.log_trade({
            "symbol": "AAPL",
            "action": "buy",
            "price": 150.25,
            "quantity": 100,
            "strategy": "momentum"
        })
    """

    def __init__(
        self,
        log_file: str = "trades.csv",
        log_dir: str = "logs",
        logger_name: str = "TradeLogger",
        batch_size: int = 10,
        flush_interval: float = 5.0,
        enable_async: bool = True,
    ):
        """
        Initialize file trade logger.

        Args:
            log_file: CSV filename
            log_dir: Directory for log files
            logger_name: Logger name for app logs
            batch_size: Number of trades to batch before writing (async mode)
            flush_interval: Seconds between auto-flushes (async mode)
            enable_async: Enable async batched writes (default True)
        """
        super().__init__(log_file=log_file, log_dir=log_dir)

        # Ensure directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Full path to CSV
        self.log_path = os.path.join(log_dir, log_file)

        # Thread lock for file writes
        self._lock = threading.Lock()

        # Async batching configuration
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._enable_async = enable_async
        self._write_buffer: list[list[Any]] = []
        self._buffer_lock = threading.Lock()
        self._flush_task: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trade_logger")
        self._running = False

        # Setup logger
        self.logger = Logger(log_file=f"{logger_name}.log", logger_name=logger_name, log_dir=log_dir).get_logger()

        # Create CSV with header if doesn't exist
        self._init_csv()

        self.logger.info(f"FileTradeLogger initialized: {self.log_path} (async={enable_async})")

    def _init_csv(self) -> None:
        """Create CSV file with header if it doesn't exist."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "symbol",
                        "action",
                        "price",
                        "quantity",
                        "order_id",
                        "sl",
                        "tp",
                        "trailing_stop",
                        "strategy",
                        "regime",
                        "commission",
                        "slippage",
                        "cash_before",
                        "cash_after",
                        "position_before",
                        "position_after",
                        "pnl",
                        "notes",
                    ]
                )

    # ========================================================================
    # ASYNC BATCHED WRITING
    # ========================================================================

    async def start_async_writer(self) -> None:
        """
        Start the background flush task for async batched writes.

        Call this at application startup if using async mode.
        """
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._background_flush_loop())
        self.logger.info("Async trade logger writer started")

    async def stop_async_writer(self) -> None:
        """
        Stop the background flush task and flush remaining buffer.

        Call this at application shutdown.
        """
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush any remaining buffered trades
        await self._flush_buffer_async()
        self._executor.shutdown(wait=True)
        self.logger.info("Async trade logger writer stopped")

    async def _background_flush_loop(self) -> None:
        """Background task that flushes buffer periodically."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_buffer_async()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in background flush: {e}")

    def _add_to_buffer(self, row: list[Any]) -> None:
        """Add a row to the write buffer (thread-safe)."""
        with self._buffer_lock:
            self._write_buffer.append(row)
            buffer_len = len(self._write_buffer)

        # Trigger immediate flush if buffer is full (sync fallback)
        if buffer_len >= self._batch_size and not self._running:
            self._flush_buffer_sync()

    async def _flush_buffer_async(self) -> None:
        """Flush the buffer to disk asynchronously."""
        with self._buffer_lock:
            if not self._write_buffer:
                return
            rows_to_write = self._write_buffer.copy()
            self._write_buffer.clear()

        # Write in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._do_batch_write, rows_to_write)

    def _flush_buffer_sync(self) -> None:
        """Flush the buffer to disk synchronously."""
        with self._buffer_lock:
            if not self._write_buffer:
                return
            rows_to_write = self._write_buffer.copy()
            self._write_buffer.clear()

        self._do_batch_write(rows_to_write)

    def _do_batch_write(self, rows: list[list[Any]]) -> None:
        """Perform the actual batch write to CSV (runs in executor)."""
        if not rows:
            return

        with self._lock:
            try:
                with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                self.logger.debug(f"Batch wrote {len(rows)} trades to CSV")
            except Exception as e:
                self.logger.error(f"Failed to batch write trades: {e}")

    # ========================================================================
    # MAIN LOGGING METHOD (Flexible signatures)
    # ========================================================================

    def log_trade(self, *args: Any, **kwargs: Any) -> None:
        """
        Log a trade with flexible signature.

        Signatures:

        1. Dict payload:
            log_trade({
                "symbol": "AAPL",
                "action": "buy",  # or OrderSide enum
                "price": 150.25,
                "quantity": 100,
                "timestamp": datetime.now(),  # optional
                "sl": 148.0,  # optional
                "tp": 154.0,  # optional
                ...
            })

        2. Symbol-first (recommended):
            log_trade(
                "AAPL", "buy", 150.25, 100,
                sl=148.0, tp=154.0, strategy="momentum"
            )

        3. Timestamp-first:
            log_trade(
                datetime.now(), "AAPL", "buy", 150.25, 100,
                sl=148.0, tp=154.0
            )
        """
        # Case 1: Dict payload
        if len(args) == 1 and isinstance(args[0], dict):
            self._log_from_dict(args[0])
            return

        # Case 2: All keyword arguments (symbol=, action=, price=, quantity=)
        if not args and kwargs:
            self._log_from_kwargs(**kwargs)
            return

        # Cases 3 & 4: Positional arguments
        if not args:
            raise TypeError("log_trade requires arguments")

        # Detect timestamp-first vs symbol-first
        if isinstance(args[0], datetime):
            self._log_timestamp_first(*args, **kwargs)
        else:
            self._log_symbol_first(*args, **kwargs)

    def _log_from_kwargs(self, **kwargs) -> None:
        """Log from keyword arguments only."""
        # Required fields
        symbol = kwargs.pop("symbol")
        action = kwargs.pop("action")
        price = float(kwargs.pop("price"))
        quantity = int(kwargs.pop("quantity"))

        # Optional timestamp
        timestamp = kwargs.pop("timestamp", datetime.now(timezone.utc))

        self._write_row(timestamp, symbol, action, price, quantity, **kwargs)

    def _log_from_dict(self, payload: dict[str, Any]) -> None:
        """Log from dictionary payload."""
        # Required fields
        symbol = payload["symbol"]
        action = payload["action"]
        price = float(payload["price"])
        quantity = int(payload["quantity"])

        # Optional fields
        timestamp = payload.get("timestamp", datetime.now(timezone.utc))

        # Extract all optional fields
        extras = {
            "order_id": payload.get("order_id"),
            "sl": payload.get("sl"),
            "tp": payload.get("tp"),
            "trailing_stop": payload.get("trailing_stop"),
            "strategy": payload.get("strategy"),
            "regime": payload.get("regime"),
            "commission": payload.get("commission"),
            "slippage": payload.get("slippage"),
            "cash_before": payload.get("cash_before"),
            "cash_after": payload.get("cash_after"),
            "position_before": payload.get("position_before"),
            "position_after": payload.get("position_after"),
            "pnl": payload.get("pnl"),
            "notes": payload.get("notes"),
        }

        self._write_row(timestamp, symbol, action, price, quantity, **extras)

    def _log_timestamp_first(self, *args, **kwargs) -> None:
        """Log with timestamp-first signature."""
        if len(args) < 5:
            raise TypeError("Timestamp-first requires: (timestamp, symbol, action, price, quantity)")

        timestamp = args[0]
        symbol = args[1]
        action = args[2]
        price = float(args[3])
        quantity = int(args[4])

        self._write_row(timestamp, symbol, action, price, quantity, **kwargs)

    def _log_symbol_first(self, *args, **kwargs) -> None:
        """Log with symbol-first signature."""
        if len(args) < 4:
            raise TypeError("Symbol-first requires: (symbol, action, price, quantity)")

        timestamp = kwargs.pop("timestamp", datetime.now(timezone.utc))
        symbol = args[0]
        action = args[1]
        price = float(args[2])
        quantity = int(args[3])

        self._write_row(timestamp, symbol, action, price, quantity, **kwargs)

    def _write_row(
        self,
        timestamp: datetime,
        symbol: str,
        action: Any,  # str or OrderSide
        price: float,
        quantity: int,
        order_id: str | None = None,
        sl: float | None = None,
        tp: float | None = None,
        trailing_stop: float | None = None,
        strategy: str | None = None,
        regime: str | None = None,
        commission: float | None = None,
        slippage: float | None = None,
        cash_before: float | None = None,
        cash_after: float | None = None,
        position_before: int | None = None,
        position_after: int | None = None,
        pnl: float | None = None,
        notes: str | None = None,
        **ignored,  # Catch any extra kwargs
    ) -> None:
        """Write trade to CSV file (thread-safe, optionally batched)."""
        # Normalize action (handle OrderSide enum)
        if isinstance(action, OrderSide):
            action_str = action.value.lower()
        else:
            action_str = str(action).lower()

        # Build row data
        row = [
            timestamp.isoformat(),
            symbol,
            action_str,
            f"{price:.2f}",
            quantity,
            order_id or "",
            f"{sl:.2f}" if sl is not None else "",
            f"{tp:.2f}" if tp is not None else "",
            f"{trailing_stop:.2f}" if trailing_stop is not None else "",
            strategy or "",
            regime or "",
            f"{commission:.2f}" if commission is not None else "",
            f"{slippage:.4f}" if slippage is not None else "",
            f"{cash_before:.2f}" if cash_before is not None else "",
            f"{cash_after:.2f}" if cash_after is not None else "",
            position_before if position_before is not None else "",
            position_after if position_after is not None else "",
            f"{pnl:.2f}" if pnl is not None else "",
            notes or "",
        ]

        # Use batched writing if async mode enabled
        if self._enable_async:
            self._add_to_buffer(row)
        else:
            # Direct write (legacy behavior)
            with self._lock:
                try:
                    with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                except Exception as e:
                    self.logger.error(f"Failed to write trade to CSV: {e}")

        # Mirror to app log
        self.logger.info(
            f"[TRADE] {timestamp.isoformat()} | {action_str.upper()} {quantity} {symbol} @ ${price:.2f} "
            f"| SL:{sl} TP:{tp} TS:{trailing_stop} | strategy={strategy} regime={regime}"
        )

    # ========================================================================
    # TRADELOGGERBASE IMPLEMENTATION
    # ========================================================================

    def log_order_status(
        self,
        order_id: str,
        symbol: str,
        status: str,
        side: OrderSide,
        quantity: float,
        filled_qty: float = 0.0,
        avg_price: float | None = None,
        timestamp: datetime | None = None,
        **kwargs,
    ) -> None:
        """Log order status update."""
        timestamp or datetime.now(timezone.utc)
        self.logger.info(
            f"[ORDER] {order_id} | {symbol} {side.value} {quantity} | "
            f"status={status} filled={filled_qty} avg_price={avg_price}"
        )

    def log_error(
        self,
        message: str,
        error: Exception | None = None,
        context: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Log error."""
        msg = f"[ERROR] {message}"
        if error:
            msg += f" | Error: {error}"
        if context:
            msg += f" | Context: {context}"
        self.logger.error(msg)

    def log_info(self, message: str, context: dict[str, Any] | None = None, timestamp: datetime | None = None) -> None:
        """Log info message."""
        msg = f"[INFO] {message}"
        if context:
            msg += f" | Context: {context}"
        self.logger.info(msg)

    def log_warning(
        self, message: str, context: dict[str, Any] | None = None, timestamp: datetime | None = None
    ) -> None:
        """Log warning."""
        msg = f"[WARNING] {message}"
        if context:
            msg += f" | Context: {context}"
        self.logger.warning(msg)

    def log_summary(
        self, summary_stats: dict[str, Any], period: str | None = None, timestamp: datetime | None = None
    ) -> None:
        """Log summary statistics."""
        self.logger.info("[SUMMARY]")
        if period:
            self.logger.info(f"  Period: {period}")
        for key, value in summary_stats.items():
            self.logger.info(f"  {key}: {value}")

    def log_position_update(
        self,
        symbol: str,
        quantity: int,
        avg_price: float,
        current_price: float,
        unrealized_pnl: float,
        timestamp: datetime | None = None,
        **kwargs,
    ) -> None:
        """Log position update."""
        self.logger.info(
            f"[POSITION] {symbol} | qty={quantity} avg=${avg_price:.2f} "
            f"current=${current_price:.2f} upnl=${unrealized_pnl:.2f}"
        )

    def flush(self) -> None:
        """Flush any buffered data to disk."""
        if self._enable_async:
            self._flush_buffer_sync()

    def close(self) -> None:
        """Close logger and flush remaining buffer."""
        # Flush any remaining buffered trades
        if self._enable_async:
            self._flush_buffer_sync()
        self._executor.shutdown(wait=True)
        self.logger.info("FileTradeLogger closed")

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_trades(
        self, symbol: str | None = None, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        """
        Read trades from CSV file.

        Args:
            symbol: Filter by symbol (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)

        Returns:
            List of trade dictionaries
        """
        trades = []

        try:
            with open(self.log_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Apply filters
                    if symbol and row["symbol"] != symbol:
                        continue

                    trade_time = datetime.fromisoformat(row["timestamp"])
                    if start_time and trade_time < start_time:
                        continue
                    if end_time and trade_time > end_time:
                        continue

                    trades.append(row)

        except FileNotFoundError:
            self.logger.warning(f"Trade log file not found: {self.log_path}")
        except Exception as e:
            self.logger.error(f"Failed to read trades: {e}")

        return trades

    def get_metrics(self) -> dict[str, Any]:
        """
        Calculate basic metrics from trade log.

        Returns:
            Dictionary of metrics
        """
        trades = self.get_trades()

        if not trades:
            return {"total_trades": 0, "winning_trades": 0, "losing_trades": 0, "total_pnl": 0.0, "win_rate": 0.0}

        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get("pnl") and float(t["pnl"]) > 0)
        losing_trades = sum(1 for t in trades if t.get("pnl") and float(t["pnl"]) < 0)
        total_pnl = sum(float(t.get("pnl", 0)) for t in trades if t.get("pnl"))

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_pnl": total_pnl,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0.0,
        }
