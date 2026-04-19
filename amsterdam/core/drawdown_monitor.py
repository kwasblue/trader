from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.events.eventhandler import get_event_handler
from loggers.logger import Logger


def _try_create_task(coro) -> None:
    """
    Try to create an async task for event emission.

    If no event loop is running (e.g., when called from asyncio.to_thread),
    close the coroutine properly and skip the event emission gracefully.
    """
    try:
        asyncio.create_task(coro)
    except RuntimeError:
        # No running event loop - this happens when called via asyncio.to_thread
        # Close the coroutine to prevent "coroutine was never awaited" warning
        coro.close()


class DrawdownMonitor:
    """
    Unified risk guard for per-symbol and portfolio-level drawdown control.

    Features
    --------
    • Per-symbol controls:
        - Intraday peak tracking, drawdown limit -> lock
        - Daily start value tracking, daily drawdown limit -> lock
        - Cooldown window after unlock
    • Portfolio-level controls:
        - Portfolio peak tracking, intraday drawdown limit -> global lock
        - Portfolio daily start tracking, daily drawdown limit -> global lock
        - Portfolio cooldown window after unlock
    • Day reset helper to set fresh daily baselines.
    • Query helpers to decide if trading is allowed.

    Typical usage
    -------------
        ddm = DrawdownMonitor(...)
        # each bar/tick:
        ddm.update_portfolio(total_equity)               # must be called first per tick
        ddm.update_symbol(symbol, symbol_equity)         # then per symbol
        if ddm.can_trade(symbol):
            ... place orders ...
    """

    def __init__(
        self,
        # --- Per-symbol limits ---
        max_symbol_drawdown: float = 0.30,  # 30% symbol intraday
        max_symbol_daily_drawdown: float = 0.15,  # 15% symbol daily
        symbol_cooldown_seconds: int = 5,
        # --- Portfolio limits ---
        max_portfolio_drawdown: float = 0.25,  # 25% portfolio intraday
        max_portfolio_daily_drawdown: float = 0.10,  # 10% portfolio daily
        portfolio_cooldown_seconds: int = 5,
    ):
        # Per-symbol state
        self.max_symbol_drawdown = max_symbol_drawdown
        self.max_symbol_daily_drawdown = max_symbol_daily_drawdown
        self.symbol_cooldown_seconds = symbol_cooldown_seconds

        self.symbol_peak: dict[str, float] = {}
        self.symbol_daily_start: dict[str, float] = {}
        self.symbol_locked = defaultdict(lambda: False)
        self.symbol_last_unlock_time: dict[str, datetime] = {}

        # Track current drawdown values
        self.current_portfolio_dd: float = 0.0
        self.current_portfolio_daily_dd: float = 0.0
        self.current_symbol_dd: dict[str, float] = defaultdict(float)
        self.current_symbol_daily_dd: dict[str, float] = defaultdict(float)

        # Portfolio state
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.max_portfolio_daily_drawdown = max_portfolio_daily_drawdown
        self.portfolio_cooldown_seconds = portfolio_cooldown_seconds

        self.portfolio_peak: float | None = None
        self.portfolio_daily_start: float | None = None
        self.portfolio_locked: bool = False
        self.portfolio_last_unlock_time: datetime | None = None

        # Thread safety - RLock for reentrant locking
        self._lock = threading.RLock()

        # Persistence path
        self._persistence_path: Path | None = None

        # Logger - own file with propagation to app.log
        self.logger = Logger(
            log_file="drawdown_monitor.log", logger_name="DrawdownMonitor", propagate=True
        ).get_logger()
        self.event_handler = get_event_handler()

        self.logger.info(
            f"DrawdownMonitor initialized: symbol_dd={max_symbol_drawdown:.1%}, "
            f"portfolio_dd={max_portfolio_drawdown:.1%}, cooldown={symbol_cooldown_seconds}s"
        )

    # ----------------------------- Public API -----------------------------

    def start_new_day(
        self,
        portfolio_equity: float | None = None,
        per_symbol_equity: dict[str, float] | None = None,
    ) -> None:
        """
        Reset daily baselines. Call once at session start.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            if portfolio_equity is not None:
                self.portfolio_daily_start = portfolio_equity
                self.logger.info(f"[DAILY RESET] Portfolio start set to {portfolio_equity:,.2f}")

            if per_symbol_equity:
                for sym, eq in per_symbol_equity.items():
                    self.symbol_daily_start[sym] = eq
                self.logger.info(
                    f"[DAILY RESET] Per-symbol daily starts set for {len(per_symbol_equity)} symbols at {now.isoformat()}"
                )

            self._persist_state()

    def update_portfolio(self, portfolio_equity: float) -> bool:
        """
        Update portfolio-level drawdown state.
        Returns True if portfolio trading is allowed, False if locked / cooling.
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # init peaks and daily start
            if self.portfolio_peak is None:
                self.portfolio_peak = portfolio_equity
            else:
                self.portfolio_peak = max(self.portfolio_peak, portfolio_equity)

            if self.portfolio_daily_start is None:
                self.portfolio_daily_start = portfolio_equity

            # daily drawdown
            if self.portfolio_daily_start:
                self.current_portfolio_daily_dd = (
                    portfolio_equity - self.portfolio_daily_start
                ) / self.portfolio_daily_start
            # intraday drawdown
            if self.portfolio_peak:
                self.current_portfolio_dd = (portfolio_equity - self.portfolio_peak) / self.portfolio_peak

            # daily drawdown
            daily_dd = (portfolio_equity - self.portfolio_daily_start) / self.portfolio_daily_start
            if daily_dd < -self.max_portfolio_daily_drawdown:
                if not self.portfolio_locked:
                    self.portfolio_locked = True
                    self._persist_state()
                    self.logger.warning(f"[PORTFOLIO DAILY LOCK] Daily DD {daily_dd:.2%} breached.")
                    _try_create_task(
                        self.event_handler.emit_guardrail(
                            "portfolio.daily",  # guard_name (namespaced style is nice)
                            True,  # triggered
                            f"Portfolio daily drawdown breached {daily_dd:.2%}",
                            daily_dd,  # <- value (float)
                        )
                    )
                return False

            # intraday drawdown vs peak
            intraday_dd = (portfolio_equity - self.portfolio_peak) / self.portfolio_peak
            if intraday_dd < -self.max_portfolio_drawdown:
                if not self.portfolio_locked:
                    self.portfolio_locked = True
                    self._persist_state()
                    self.logger.warning(f"[PORTFOLIO LOCK] Intraday DD {intraday_dd:.2%} breached.")
                    _try_create_task(
                        self.event_handler.emit_guardrail(
                            "portfolio.intraday",
                            True,
                            f"Portfolio intraday drawdown breached {intraday_dd:.2%}",
                            intraday_dd,
                        )
                    )
                return False

            # cooldown if previously unlocked
            if not self.portfolio_locked and self.portfolio_last_unlock_time:
                elapsed = (now - self.portfolio_last_unlock_time).total_seconds()
                if elapsed < self.portfolio_cooldown_seconds:
                    self.logger.warning(f"[PORTFOLIO COOLDOWN] {elapsed:.1f}s elapsed — trading disabled.")
                    _try_create_task(
                        self.event_handler.emit_guardrail(
                            "portfolio.cooldown", True, f"Portfolio in cooldown ({elapsed:.1f}s elapsed)", elapsed
                        )
                    )
                    return False

            return True

    def update_symbol(self, symbol: str, symbol_equity: float) -> bool:
        """
        Update per-symbol drawdown state.
        Returns True if symbol trading is allowed, False if locked / cooling / portfolio locked.
        """
        with self._lock:
            if self.portfolio_locked:
                return False

            now = datetime.now(timezone.utc)

            # init peaks and daily start
            if symbol not in self.symbol_peak:
                self.symbol_peak[symbol] = symbol_equity
            else:
                self.symbol_peak[symbol] = max(self.symbol_peak[symbol], symbol_equity)

            if symbol not in self.symbol_daily_start:
                self.symbol_daily_start[symbol] = symbol_equity

            daily_start = self.symbol_daily_start[symbol]
            if daily_start:
                self.current_symbol_daily_dd[symbol] = (symbol_equity - daily_start) / daily_start
            # intraday drawdown
            peak = self.symbol_peak.get(symbol)
            if peak:
                self.current_symbol_dd[symbol] = (symbol_equity - peak) / peak

            # daily drawdown
            if daily_start > 0:
                daily_dd = (symbol_equity - daily_start) / daily_start
                if daily_dd < -self.max_symbol_daily_drawdown:
                    if not self.symbol_locked[symbol]:
                        self.symbol_locked[symbol] = True
                        self._persist_state()
                        self.logger.warning(f"[{symbol}] DAILY LOCK | DD {daily_dd:.2%}")
                        _try_create_task(
                            self._emit_guardrail(
                                f"{symbol}_daily", True, f"{symbol} daily drawdown breached {daily_dd:.2%}"
                            )
                        )
                    return False

            # intraday drawdown vs peak
            peak = self.symbol_peak[symbol]
            if peak > 0:
                intraday_dd = (symbol_equity - peak) / peak
                if intraday_dd < -self.max_symbol_drawdown:
                    if not self.symbol_locked[symbol]:
                        self.symbol_locked[symbol] = True
                        self._persist_state()
                        self.logger.warning(f"[{symbol}] LOCK | Intraday DD {intraday_dd:.2%}")
                    return False

            # cooldown if previously unlocked
            if not self.symbol_locked[symbol] and symbol in self.symbol_last_unlock_time:
                elapsed = (now - self.symbol_last_unlock_time[symbol]).total_seconds()
                if elapsed < self.symbol_cooldown_seconds:
                    self.logger.warning(f"[{symbol}] COOLDOWN {elapsed:.1f}s — trading disabled.")
                    return False

            return True

    def can_trade(self, symbol: str) -> bool:
        """
        Combined check: portfolio not locked/cooling AND symbol not locked/cooling.
        Use after calling update_portfolio(...) and update_symbol(...).
        """
        with self._lock:
            return not self.is_portfolio_blocked() and not self.is_symbol_blocked(symbol)

    # ----------------------------- Query helpers -----------------------------

    def is_symbol_blocked(self, symbol: str) -> bool:
        with self._lock:
            return self.symbol_locked[symbol] or self.is_symbol_in_cooldown(symbol)

    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        with self._lock:
            if symbol not in self.symbol_last_unlock_time:
                return False
            elapsed = (datetime.now(timezone.utc) - self.symbol_last_unlock_time[symbol]).total_seconds()
            return elapsed < self.symbol_cooldown_seconds

    def is_portfolio_blocked(self) -> bool:
        with self._lock:
            return self.portfolio_locked or self.is_portfolio_in_cooldown()

    def is_portfolio_in_cooldown(self) -> bool:
        with self._lock:
            if not self.portfolio_last_unlock_time:
                return False
            elapsed = (datetime.now(timezone.utc) - self.portfolio_last_unlock_time).total_seconds()
            return elapsed < self.portfolio_cooldown_seconds

    def get_portfolio_drawdown(self) -> float:
        """Current portfolio drawdown vs peak (negative = DD)."""
        with self._lock:
            return self.current_portfolio_dd

    def get_portfolio_daily_drawdown(self) -> float:
        """Current portfolio drawdown vs daily start."""
        with self._lock:
            return self.current_portfolio_daily_dd

    def get_symbol_drawdown(self, symbol: str) -> float:
        with self._lock:
            return self.current_symbol_dd.get(symbol, 0.0)

    def get_symbol_daily_drawdown(self, symbol: str) -> float:
        with self._lock:
            return self.current_symbol_daily_dd.get(symbol, 0.0)

    # ----------------------------- Admin / manual controls -----------------------------

    def unlock_symbol(self, symbol: str) -> None:
        with self._lock:
            if self.symbol_locked[symbol]:
                self.symbol_locked[symbol] = False
                self.symbol_last_unlock_time[symbol] = datetime.now(timezone.utc)
                self._persist_state()
                self.logger.info(f"[{symbol}] UNLOCKED (cooldown started)")
                _try_create_task(
                    self._emit_guardrail(f"{symbol}", False, f"{symbol} unlocked from guardrail (cooldown active)")
                )

    def reset_symbol(self, symbol: str) -> None:
        with self._lock:
            self.symbol_locked[symbol] = False
            self.symbol_peak[symbol] = 0.0
            self.symbol_daily_start[symbol] = 0.0
            if symbol in self.symbol_last_unlock_time:
                del self.symbol_last_unlock_time[symbol]
            self._persist_state()
            self.logger.info(f"[{symbol}] RESET")

    def unlock_portfolio(self) -> None:
        with self._lock:
            if self.portfolio_locked:
                self.portfolio_locked = False
                self.portfolio_last_unlock_time = datetime.now(timezone.utc)
                self._persist_state()
                self.logger.info("[PORTFOLIO UNLOCKED] (cooldown started)")

    def reset_portfolio(self) -> None:
        with self._lock:
            self.portfolio_locked = False
            self.portfolio_peak = None
            self.portfolio_daily_start = None
            self.portfolio_last_unlock_time = None
            self._persist_state()
            self.logger.info("[PORTFOLIO RESET]")

    async def _emit_guardrail(self, guard_name: str, triggered: bool, message: str, value: float = None) -> None:
        """Helper to emit guardrail events."""
        if self.event_handler:
            await self.event_handler.emit_guardrail(guard_name, triggered, message, value)

    # ----------------------------- Persistence -----------------------------

    def set_persistence_path(self, path: Path) -> None:
        """Set the path for state persistence file."""
        self._persistence_path = path
        self.logger.info(f"Persistence path set to: {path}")

    def _persist_state(self) -> None:
        """Persist current state to disk (internal helper)."""
        if self._persistence_path is None:
            # Use default path if not set
            self._persistence_path = Path(__file__).parent.parent / "data" / "drawdown_state.json"

        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            state = self._serialize_state()
            with open(self._persistence_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to persist drawdown state: {e}")

    def _serialize_state(self) -> dict[str, Any]:
        """Serialize current state to dictionary."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio": {
                "peak": self.portfolio_peak,
                "daily_start": self.portfolio_daily_start,
                "locked": self.portfolio_locked,
                "last_unlock_time": self.portfolio_last_unlock_time.isoformat()
                if self.portfolio_last_unlock_time
                else None,
                "current_dd": self.current_portfolio_dd,
                "current_daily_dd": self.current_portfolio_daily_dd,
            },
            "symbols": {
                sym: {
                    "peak": self.symbol_peak.get(sym),
                    "daily_start": self.symbol_daily_start.get(sym),
                    "locked": self.symbol_locked[sym],
                    "last_unlock_time": self.symbol_last_unlock_time.get(sym).isoformat()
                    if self.symbol_last_unlock_time.get(sym)
                    else None,
                    "current_dd": self.current_symbol_dd.get(sym, 0.0),
                    "current_daily_dd": self.current_symbol_daily_dd.get(sym, 0.0),
                }
                for sym in set(list(self.symbol_peak.keys()) + list(self.symbol_locked.keys()))
            },
        }

    def save_state(self) -> None:
        """Explicitly save current state to disk."""
        with self._lock:
            self._persist_state()
            self.logger.info("Drawdown state saved")

    def load_state(self, path: Path | None = None) -> bool:
        """
        Load state from disk.

        Args:
            path: Path to state file. Uses default if not provided.

        Returns:
            True if state was loaded successfully, False otherwise.
        """
        load_path = path or self._persistence_path
        if load_path is None:
            load_path = Path(__file__).parent.parent / "data" / "drawdown_state.json"

        if not load_path.exists():
            self.logger.info(f"No persisted state found at {load_path}")
            return False

        try:
            with open(load_path) as f:
                state = json.load(f)

            with self._lock:
                self._deserialize_state(state)

            self.logger.info(f"Drawdown state loaded from {load_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load drawdown state: {e}")
            return False

    def _deserialize_state(self, state: dict[str, Any]) -> None:
        """Restore state from dictionary."""
        # Portfolio state
        portfolio = state.get("portfolio", {})
        self.portfolio_peak = portfolio.get("peak")
        self.portfolio_daily_start = portfolio.get("daily_start")
        self.portfolio_locked = portfolio.get("locked", False)
        self.current_portfolio_dd = portfolio.get("current_dd", 0.0)
        self.current_portfolio_daily_dd = portfolio.get("current_daily_dd", 0.0)

        if portfolio.get("last_unlock_time"):
            self.portfolio_last_unlock_time = datetime.fromisoformat(portfolio["last_unlock_time"])

        # Symbol states
        symbols = state.get("symbols", {})
        for sym, sym_state in symbols.items():
            if sym_state.get("peak") is not None:
                self.symbol_peak[sym] = sym_state["peak"]
            if sym_state.get("daily_start") is not None:
                self.symbol_daily_start[sym] = sym_state["daily_start"]
            self.symbol_locked[sym] = sym_state.get("locked", False)
            self.current_symbol_dd[sym] = sym_state.get("current_dd", 0.0)
            self.current_symbol_daily_dd[sym] = sym_state.get("current_daily_dd", 0.0)

            if sym_state.get("last_unlock_time"):
                self.symbol_last_unlock_time[sym] = datetime.fromisoformat(sym_state["last_unlock_time"])

    # ----------------------------- Async Wrappers -----------------------------
    # These resolve the threading.RLock vs asyncio.Lock mismatch for use in async contexts

    async def can_trade_async(self, symbol: str) -> bool:
        """
        Async wrapper for can_trade() to avoid blocking the event loop.

        Use this in async contexts (e.g., LiveExecutionEngine.handle_signal_context).

        Args:
            symbol: Trading symbol to check

        Returns:
            True if trading is allowed, False if blocked by drawdown/cooldown
        """
        return await asyncio.to_thread(self.can_trade, symbol)

    async def update_portfolio_async(self, portfolio_equity: float) -> bool:
        """
        Async wrapper for update_portfolio() to avoid blocking the event loop.

        Args:
            portfolio_equity: Current portfolio equity value

        Returns:
            True if portfolio trading is allowed, False if locked/cooling
        """
        return await asyncio.to_thread(self.update_portfolio, portfolio_equity)

    async def update_symbol_async(self, symbol: str, symbol_equity: float) -> bool:
        """
        Async wrapper for update_symbol() to avoid blocking the event loop.

        Args:
            symbol: Trading symbol
            symbol_equity: Current equity value for this symbol

        Returns:
            True if symbol trading is allowed, False if locked/cooling
        """
        return await asyncio.to_thread(self.update_symbol, symbol, symbol_equity)
