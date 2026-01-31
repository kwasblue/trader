from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Optional
from core.events.eventhandler import EventHandler, get_event_handler
from core.events.events import EVENT_GUARDRAIL_TRIGGERED, GuardrailPayload

from loggers.logger import Logger

import asyncio


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
        max_symbol_drawdown: float = 0.05,         # 30% symbol intraday
        max_symbol_daily_drawdown: float = 0.02,   # 10% symbol daily
        symbol_cooldown_seconds: int = 5,

        # --- Portfolio limits ---
        max_portfolio_drawdown: float = 0.05,      # 25% portfolio intraday
        max_portfolio_daily_drawdown: float = 0.02,# 10% portfolio daily
        portfolio_cooldown_seconds: int = 5,
    ):
        # Per-symbol state
        self.max_symbol_drawdown = max_symbol_drawdown
        self.max_symbol_daily_drawdown = max_symbol_daily_drawdown
        self.symbol_cooldown_seconds = symbol_cooldown_seconds

        self.symbol_peak: Dict[str, float] = {}
        self.symbol_daily_start: Dict[str, float] = {}
        self.symbol_locked = defaultdict(lambda: False)
        self.symbol_last_unlock_time: Dict[str, datetime] = {}

        # Track current drawdown values
        self.current_portfolio_dd: float = 0.0
        self.current_portfolio_daily_dd: float = 0.0
        self.current_symbol_dd: Dict[str, float] = defaultdict(float)
        self.current_symbol_daily_dd: Dict[str, float] = defaultdict(float)

        # Portfolio state
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.max_portfolio_daily_drawdown = max_portfolio_daily_drawdown
        self.portfolio_cooldown_seconds = portfolio_cooldown_seconds

        self.portfolio_peak: Optional[float] = None
        self.portfolio_daily_start: Optional[float] = None
        self.portfolio_locked: bool = False
        self.portfolio_last_unlock_time: Optional[datetime] = None

        # Logger - own file with propagation to app.log
        self.logger = Logger(
            log_file='drawdown_monitor.log',
            logger_name='DrawdownMonitor',
            propagate=True
        ).get_logger()
        self.event_handler = get_event_handler()

        self.logger.info(
            f"DrawdownMonitor initialized: symbol_dd={max_symbol_drawdown:.1%}, "
            f"portfolio_dd={max_portfolio_drawdown:.1%}, cooldown={symbol_cooldown_seconds}s"
        )

    # ----------------------------- Public API -----------------------------

    def start_new_day(
        self,
        portfolio_equity: Optional[float] = None,
        per_symbol_equity: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Reset daily baselines. Call once at session start.
        """
        now = datetime.now(timezone.utc)
        if portfolio_equity is not None:
            self.portfolio_daily_start = portfolio_equity
            self.logger.info(f"[DAILY RESET] Portfolio start set to {portfolio_equity:,.2f}")

        if per_symbol_equity:
            for sym, eq in per_symbol_equity.items():
                self.symbol_daily_start[sym] = eq
            self.logger.info(f"[DAILY RESET] Per-symbol daily starts set for {len(per_symbol_equity)} symbols at {now.isoformat()}")

    def update_portfolio(self, portfolio_equity: float) -> bool:
        """
        Update portfolio-level drawdown state.
        Returns True if portfolio trading is allowed, False if locked / cooling.
        """
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
            self.current_portfolio_daily_dd = (portfolio_equity - self.portfolio_daily_start) / self.portfolio_daily_start
        # intraday drawdown
        if self.portfolio_peak:
            self.current_portfolio_dd = (portfolio_equity - self.portfolio_peak) / self.portfolio_peak

        # daily drawdown
        daily_dd = (portfolio_equity - self.portfolio_daily_start) / self.portfolio_daily_start
        if daily_dd < -self.max_portfolio_daily_drawdown:
            if not self.portfolio_locked:
                self.portfolio_locked = True
                self.logger.warning(f"[PORTFOLIO DAILY LOCK] Daily DD {daily_dd:.2%} breached.")
                asyncio.create_task(
                    self.event_handler.emit_guardrail(
                        "portfolio.daily",         # guard_name (namespaced style is nice)
                        True,                      # triggered
                        f"Portfolio daily drawdown breached {daily_dd:.2%}", 
                        daily_dd                   # <- value (float)
                    )
                )
            return False

        # intraday drawdown vs peak
        intraday_dd = (portfolio_equity - self.portfolio_peak) / self.portfolio_peak
        if intraday_dd < -self.max_portfolio_drawdown:
            if not self.portfolio_locked:
                self.portfolio_locked = True
                self.logger.warning(f"[PORTFOLIO LOCK] Intraday DD {intraday_dd:.2%} breached.")
                asyncio.create_task(
                    self.event_handler.emit_guardrail(
                        "portfolio.intraday",
                        True,
                        f"Portfolio intraday drawdown breached {intraday_dd:.2%}",
                        intraday_dd
                    )
                )
            return False


        # cooldown if previously unlocked
        if not self.portfolio_locked and self.portfolio_last_unlock_time:
            elapsed = (now - self.portfolio_last_unlock_time).total_seconds()
            if elapsed < self.portfolio_cooldown_seconds:
                self.logger.warning(f"[PORTFOLIO COOLDOWN] {elapsed:.1f}s elapsed — trading disabled.")
                asyncio.create_task(
                    self.event_handler.emit_guardrail(
                        "portfolio.cooldown",
                        True,
                        f"Portfolio in cooldown ({elapsed:.1f}s elapsed)",
                        elapsed
                    )
                )
                return False

        return True

    def update_symbol(self, symbol: str, symbol_equity: float) -> bool:
        """
        Update per-symbol drawdown state.
        Returns True if symbol trading is allowed, False if locked / cooling / portfolio locked.
        """
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
                    self.logger.warning(f"[{symbol}] DAILY LOCK | DD {daily_dd:.2%}")
                    asyncio.create_task(self._emit_guardrail(f"{symbol}_daily", True, f"{symbol} daily drawdown breached {daily_dd:.2%}"))
                return False

        # intraday drawdown vs peak
        peak = self.symbol_peak[symbol]
        if peak > 0:
            intraday_dd = (symbol_equity - peak) / peak
            if intraday_dd < -self.max_symbol_drawdown:
                if not self.symbol_locked[symbol]:
                    self.symbol_locked[symbol] = True
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
        return not self.is_portfolio_blocked() and not self.is_symbol_blocked(symbol)

    # ----------------------------- Query helpers -----------------------------

    def is_symbol_blocked(self, symbol: str) -> bool:
        return self.symbol_locked[symbol] or self.is_symbol_in_cooldown(symbol)

    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        if symbol not in self.symbol_last_unlock_time:
            return False
        elapsed = (datetime.now(timezone.utc) - self.symbol_last_unlock_time[symbol]).total_seconds()
        return elapsed < self.symbol_cooldown_seconds

    def is_portfolio_blocked(self) -> bool:
        return self.portfolio_locked or self.is_portfolio_in_cooldown()

    def is_portfolio_in_cooldown(self) -> bool:
        if not self.portfolio_last_unlock_time:
            return False
        elapsed = (datetime.now(timezone.utc) - self.portfolio_last_unlock_time).total_seconds()
        return elapsed < self.portfolio_cooldown_seconds
    
    def get_portfolio_drawdown(self) -> float:
        """Current portfolio drawdown vs peak (negative = DD)."""
        return self.current_portfolio_dd

    def get_portfolio_daily_drawdown(self) -> float:
        """Current portfolio drawdown vs daily start."""
        return self.current_portfolio_daily_dd

    def get_symbol_drawdown(self, symbol: str) -> float:
        return self.current_symbol_dd.get(symbol, 0.0)

    def get_symbol_daily_drawdown(self, symbol: str) -> float:
        return self.current_symbol_daily_dd.get(symbol, 0.0)

    # ----------------------------- Admin / manual controls -----------------------------

    def unlock_symbol(self, symbol: str) -> None:
        if self.symbol_locked[symbol]:
            self.symbol_locked[symbol] = False
            self.symbol_last_unlock_time[symbol] = datetime.now(timezone.utc)
            self.logger.info(f"[{symbol}] UNLOCKED (cooldown started)")
            asyncio.create_task(self._emit_guardrail(f"{symbol}", False,
            f"{symbol} unlocked from guardrail (cooldown active)"))

    def reset_symbol(self, symbol: str) -> None:
        self.symbol_locked[symbol] = False
        self.symbol_peak[symbol] = 0.0
        self.symbol_daily_start[symbol] = 0.0
        if symbol in self.symbol_last_unlock_time:
            del self.symbol_last_unlock_time[symbol]
        self.logger.info(f"[{symbol}] RESET")

    def unlock_portfolio(self) -> None:
        if self.portfolio_locked:
            self.portfolio_locked = False
            self.portfolio_last_unlock_time = datetime.now(timezone.utc)
            self.logger.info("[PORTFOLIO UNLOCKED] (cooldown started)")

    def reset_portfolio(self) -> None:
        self.portfolio_locked = False
        self.portfolio_peak = None
        self.portfolio_daily_start = None
        self.portfolio_last_unlock_time = None
        self.logger.info("[PORTFOLIO RESET]")
