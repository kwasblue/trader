from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Optional

from loggers.logger import Logger


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
        max_symbol_drawdown: float = 0.02,         # 30% symbol intraday
        max_symbol_daily_drawdown: float = 0.01,   # 10% symbol daily
        symbol_cooldown_seconds: int = 10,

        # --- Portfolio limits ---
        max_portfolio_drawdown: float = 0.02,      # 25% portfolio intraday
        max_portfolio_daily_drawdown: float = 0.01,# 10% portfolio daily
        portfolio_cooldown_seconds: int = 10,
    ):
        # Per-symbol state
        self.max_symbol_drawdown = max_symbol_drawdown
        self.max_symbol_daily_drawdown = max_symbol_daily_drawdown
        self.symbol_cooldown_seconds = symbol_cooldown_seconds

        self.symbol_peak: Dict[str, float] = {}
        self.symbol_daily_start: Dict[str, float] = {}
        self.symbol_locked = defaultdict(lambda: False)
        self.symbol_last_unlock_time: Dict[str, datetime] = {}

        # Portfolio state
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.max_portfolio_daily_drawdown = max_portfolio_daily_drawdown
        self.portfolio_cooldown_seconds = portfolio_cooldown_seconds

        self.portfolio_peak: Optional[float] = None
        self.portfolio_daily_start: Optional[float] = None
        self.portfolio_locked: bool = False
        self.portfolio_last_unlock_time: Optional[datetime] = None

        self.logger = Logger(log_file='TradeLogger.log', logger_name='Drawdown Monitor')

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
        daily_dd = (portfolio_equity - self.portfolio_daily_start) / self.portfolio_daily_start
        if daily_dd < -self.max_portfolio_daily_drawdown:
            if not self.portfolio_locked:
                self.portfolio_locked = True
                self.logger.warning(f"[PORTFOLIO DAILY LOCK] Daily DD {daily_dd:.2%} breached.")
            return False

        # intraday drawdown vs peak
        intraday_dd = (portfolio_equity - self.portfolio_peak) / self.portfolio_peak
        if intraday_dd < -self.max_portfolio_drawdown:
            if not self.portfolio_locked:
                self.portfolio_locked = True
                self.logger.warning(f"[PORTFOLIO LOCK] Intraday DD {intraday_dd:.2%} breached.")
            return False

        # cooldown if previously unlocked
        if not self.portfolio_locked and self.portfolio_last_unlock_time:
            elapsed = (now - self.portfolio_last_unlock_time).total_seconds()
            if elapsed < self.portfolio_cooldown_seconds:
                self.logger.warning(f"[PORTFOLIO COOLDOWN] {elapsed:.1f}s elapsed — trading disabled.")
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

        # daily drawdown
        daily_start = self.symbol_daily_start[symbol]
        if daily_start > 0:
            daily_dd = (symbol_equity - daily_start) / daily_start
            if daily_dd < -self.max_symbol_daily_drawdown:
                if not self.symbol_locked[symbol]:
                    self.symbol_locked[symbol] = True
                    self.logger.warning(f"[{symbol}] DAILY LOCK | DD {daily_dd:.2%}")
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

    # ----------------------------- Admin / manual controls -----------------------------

    def unlock_symbol(self, symbol: str) -> None:
        if self.symbol_locked[symbol]:
            self.symbol_locked[symbol] = False
            self.symbol_last_unlock_time[symbol] = datetime.now(timezone.utc)
            self.logger.info(f"[{symbol}] UNLOCKED (cooldown started)")

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
