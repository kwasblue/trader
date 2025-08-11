# loggers/file_trade_logger.py
import csv
import os
from datetime import datetime, timezone
from typing import Optional, Any, Dict

from .logger import Logger

class FileTradeLogger:
    def __init__(self, log_file: str = "trades.csv", logger_name: str = "TradeLogger", log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)
        self.logger = Logger(log_file=f"{logger_name}.log", logger_name=logger_name, log_dir=log_dir).get_logger()

        # Create header if file doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "action", "price", "quantity",
                    "sl", "tp", "trailing_stop", "strategy", "regime",
                    "cash_before", "cash_after", "position_before", "position_after",
                    "notes"
                ])

    def log_trade(self, *args: Any, **kwargs: Any) -> None:
        """
        Flexible signatures supported:

        1) symbol-first (recommended):
            log_trade(symbol, action, price, quantity, **extras)

        2) timestamp-first:
            log_trade(timestamp, symbol, action, price, quantity, **extras)

        3) dict payload:
            log_trade({
              "symbol": "...", "action": "...", "price": 123.45, "quantity": 10,  # required
              "timestamp": dt, "sl":..., "tp":..., "trailing_stop":..., "strategy":..., "regime":...,
              "cash_before":..., "cash_after":..., "position_before":..., "position_after":..., "notes":...
            })
        """
        # --- Case 3: dict payload
        if len(args) == 1 and isinstance(args[0], dict):
            d = args[0]
            symbol = d["symbol"]
            action = d["action"]
            price = float(d["price"])
            quantity = int(d["quantity"])
            timestamp = d.get("timestamp", datetime.now(timezone.utc))
            extras = {k: d.get(k) for k in (
                "sl", "tp", "trailing_stop", "strategy", "regime",
                "cash_before", "cash_after", "position_before", "position_after", "notes"
            )}
            return self._write_row(timestamp, symbol, action, price, quantity, **extras)

        # --- Case 1/2: positional parsing
        if not args:
            raise TypeError("log_trade requires positional arguments or a dict payload.")

        # Detect timestamp-first vs symbol-first
        if isinstance(args[0], datetime):
            # timestamp-first
            if len(args) < 5:
                raise TypeError("timestamp-first usage requires: (timestamp, symbol, action, price, quantity)")
            timestamp = args[0]
            symbol = args[1]
            action = args[2]
            price = float(args[3])
            quantity = int(args[4])
            rest = args[5:]
        else:
            # symbol-first
            if len(args) < 4:
                raise TypeError("symbol-first usage requires: (symbol, action, price, quantity)")
            timestamp = kwargs.pop("timestamp", datetime.now(timezone.utc))
            symbol = args[0]
            action = args[1]
            price = float(args[2])
            quantity = int(args[3])
            rest = args[4:]

        # Merge extras from positional (ignored here) and kwargs
        extras = {
            "sl": kwargs.get("sl"),
            "tp": kwargs.get("tp"),
            "trailing_stop": kwargs.get("trailing_stop"),
            "strategy": kwargs.get("strategy"),
            "regime": kwargs.get("regime"),
            "cash_before": kwargs.get("cash_before"),
            "cash_after": kwargs.get("cash_after"),
            "position_before": kwargs.get("position_before"),
            "position_after": kwargs.get("position_after"),
            "notes": kwargs.get("notes"),
        }

        self._write_row(timestamp, symbol, action, price, quantity, **extras)

    def _write_row(
        self,
        timestamp: datetime,
        symbol: str,
        action: str,
        price: float,
        quantity: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        cash_before: Optional[float] = None,
        cash_after: Optional[float] = None,
        position_before: Optional[int] = None,
        position_after: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> None:
        # Write CSV
        with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.isoformat(), symbol, action, price, quantity,
                sl, tp, trailing_stop, strategy, regime,
                cash_before, cash_after, position_before, position_after, notes
            ])
        # Mirror to app log
        self.logger.info(
            f"[TRADE] {timestamp.isoformat()} | {action} {quantity} {symbol} @ {price:.2f} "
            f"| SL:{sl} TP:{tp} TS:{trailing_stop} | strat={strategy} regime={regime}"
        )

    # Optional: convenience wrappers
    def log_info(self, msg: str) -> None:
        self.logger.info(msg)

    def log_error(self, msg: str) -> None:
        self.logger.error(msg)

    def log_summary(self, msg: str) -> None:
        self.logger.info(f"[SUMMARY] {msg}")

    def flush(self) -> None:
        # nothing to flush for CSV, but method retained for interface compatibility
        pass

    def log_error(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        if context:
            message += f" | Context: {context}"
        self.logger.error(f"[ERROR] {message}")

    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        if context:
            message += f" | Context: {context}"
        self.logger.info(f"[INFO] {message}")

    def log_summary(self, summary_stats: Dict[str, Any]) -> None:
        self.logger.info("[SUMMARY]")
        for key, value in summary_stats.items():
            self.logger.info(f"  {key}: {value}")

    def flush(self) -> None:
        # No buffering, but placeholder if needed in future
        pass
