"""
Trade Logic Manager - Business rules for trade approval and execution

Manages the decision logic of WHEN and IF to execute trades based on:
- Market conditions
- Position constraints
- Timing rules (cooldowns, hours)
- Risk limits
- Custom business rules
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
import logging

from core.enums import OrderSide

logger = logging.getLogger(__name__)


class TradeLogicManagerBase(ABC):
    """
    Abstract base class for trade approval logic.
    
    Trade Logic Manager decides WHETHER trades should be executed based on
    business rules and constraints. It acts as a gatekeeper before execution.
    
    Key Distinction:
    - TradeLogicManager: Approves/rejects trades (business rules)
    - Executor: Executes approved trades (mechanics)
    - RiskManager: Enforces risk limits (safety)
    
    Responsibilities:
    - Check cooldown periods
    - Verify market hours
    - Enforce position limits per symbol
    - Implement entry/exit rules
    - Apply custom business logic
    
    Does NOT:
    - Calculate position size (that's PositionSizer)
    - Execute trades (that's Executor)
    - Enforce global risk (that's RiskManager)
    
    Example:
        logic = TradeLogicManager(
            cooldown_seconds=300,
            max_positions=5,
            allow_after_hours=False
        )
        
        # Check if trade should execute
        should_trade, reason = logic.should_trade(
            symbol="AAPL",
            state=symbol_state,
            signal=1,
            regime="trending"
        )
        
        if should_trade:
            # Execute trade
            executor.execute(...)
        else:
            logger.info(f"Trade blocked: {reason}")
    """
    
    def __init__(self, **kwargs):
        """
        Initialize trade logic manager.

        Args:
            **kwargs: Configuration parameters like:
                - cooldown_seconds: Minimum time between trades per symbol (time-based)
                - cooldown_bars: Minimum bars between trades per symbol (bar-based)
                - cooldown_mode: 'time', 'bars', or 'both' (default: 'time')
                - max_positions: Maximum concurrent positions
                - allow_after_hours: Whether to trade outside regular hours
                - min_signal_strength: Minimum signal confidence
        """
        self.params = dict(kwargs)
        self.cooldown_seconds = kwargs.get('cooldown_seconds', 300)  # 5 min default
        self.cooldown_bars = kwargs.get('cooldown_bars', 5)  # 5 bars default
        self.cooldown_mode = kwargs.get('cooldown_mode', 'time')  # 'time', 'bars', or 'both'
        self.max_positions = kwargs.get('max_positions', 10)
        self.allow_after_hours = kwargs.get('allow_after_hours', False)

        # Track bars since last trade per symbol (for bar-based cooldown)
        self._bars_since_trade: Dict[str, int] = {}
    
    # ========================================================================
    # ABSTRACT METHODS - CORE LOGIC
    # ========================================================================
    
    @abstractmethod
    def should_trade(
        self,
        symbol: str,
        state: Any,
        signal: int,
        regime: str,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if trade should be executed.
        
        This is the main decision method. It should check all relevant
        conditions and return whether the trade is approved.
        
        Args:
            symbol: Trading symbol
            state: Symbol-specific state object containing:
                - last_trade_time: When last trade occurred
                - current_position: Current position size
                - entry_price: Average entry price (if in position)
                - pnl: Current P&L
                - trade_count: Number of trades today
                - etc.
            signal: Trading signal (1, -1, or 0)
            regime: Market regime ("trending", "ranging", "volatile", etc.)
            **kwargs: Additional context:
                - price: Current price
                - market_open: Whether market is open
                - account_positions: Total number of positions
                - etc.
        
        Returns:
            Tuple of (should_trade, reason):
            - should_trade: True if trade approved, False if blocked
            - reason: Explanation if blocked (None if approved)
            
        Example:
            should_trade, reason = logic.should_trade(
                symbol="AAPL",
                state=state,
                signal=1,
                regime="trending",
                price=150.25,
                market_open=True,
                account_positions=3
            )
            
            if not should_trade:
                logger.info(f"Trade blocked: {reason}")
        """
        pass
    
    @abstractmethod
    def can_enter_position(
        self,
        symbol: str,
        state: Any,
        side: OrderSide,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if new position can be opened.
        
        Validates entry conditions like:
        - Not already in position
        - Position limit not exceeded
        - Cooldown period elapsed
        - Market conditions suitable
        
        Args:
            symbol: Trading symbol
            state: Symbol state
            side: Order side (BUY or SELL)
            **kwargs: Additional context
            
        Returns:
            (can_enter, reason) tuple
        """
        pass
    
    @abstractmethod
    def can_exit_position(
        self,
        symbol: str,
        state: Any,
        side: OrderSide,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position can be closed.
        
        Validates exit conditions like:
        - Actually in position
        - Exit side matches position (sell long, buy short)
        - Minimum hold time elapsed
        - Exit conditions met
        
        Args:
            symbol: Trading symbol
            state: Symbol state
            side: Order side (BUY to close short, SELL to close long)
            **kwargs: Additional context
            
        Returns:
            (can_exit, reason) tuple
        """
        pass
    
    # ========================================================================
    # OPTIONAL METHODS - SPECIFIC CHECKS
    # ========================================================================
    
    def check_cooldown(
        self,
        symbol: str,
        state: Any,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if cooldown period has elapsed (supports time, bars, or both modes).

        Args:
            symbol: Trading symbol
            state: Symbol state with last_trade_time and optionally bars_since_trade
            current_time: Current time (defaults to now)

        Returns:
            (is_allowed, reason) tuple
        """
        # Bar-based cooldown check
        if self.cooldown_mode in ('bars', 'both'):
            bars_elapsed = self._bars_since_trade.get(symbol, self.cooldown_bars + 1)
            if bars_elapsed < self.cooldown_bars:
                remaining = self.cooldown_bars - bars_elapsed
                return False, f"Cooldown active ({remaining} bars remaining)"

        # Time-based cooldown check
        if self.cooldown_mode in ('time', 'both'):
            if hasattr(state, 'last_trade_time') and state.last_trade_time is not None:
                current_time = current_time or datetime.now(timezone.utc)
                time_since_last = (current_time - state.last_trade_time).total_seconds()

                if time_since_last < self.cooldown_seconds:
                    remaining = self.cooldown_seconds - time_since_last
                    return False, f"Cooldown active ({remaining:.0f}s remaining)"

        return True, None

    def on_bar(self, symbol: str) -> None:
        """
        Call this on each new bar to track bar-based cooldowns.

        Args:
            symbol: Trading symbol
        """
        if symbol in self._bars_since_trade:
            self._bars_since_trade[symbol] += 1

    def on_trade(self, symbol: str) -> None:
        """
        Call this when a trade executes to reset bar cooldown.

        Args:
            symbol: Trading symbol
        """
        self._bars_since_trade[symbol] = 0

    def reset_bar_cooldown(self, symbol: Optional[str] = None) -> None:
        """
        Reset bar cooldown counter.

        Args:
            symbol: Specific symbol to reset, or None for all
        """
        if symbol is None:
            self._bars_since_trade.clear()
        elif symbol in self._bars_since_trade:
            del self._bars_since_trade[symbol]
    
    def check_position_limit(
        self,
        current_positions: int,
        max_positions: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if position limit allows new position.
        
        Args:
            current_positions: Number of current positions
            max_positions: Maximum allowed (uses default if None)
            
        Returns:
            (is_allowed, reason) tuple
        """
        max_pos = max_positions or self.max_positions
        
        if current_positions >= max_pos:
            return False, f"Position limit reached ({current_positions}/{max_pos})"
        
        return True, None
    
    def check_market_hours(
        self,
        market_open: bool,
        allow_after_hours: Optional[bool] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if trading allowed based on market hours.
        
        Args:
            market_open: Whether market is currently open
            allow_after_hours: Override default setting
            
        Returns:
            (is_allowed, reason) tuple
        """
        allow = allow_after_hours if allow_after_hours is not None else self.allow_after_hours
        
        if not market_open and not allow:
            return False, "Market closed (after-hours trading disabled)"
        
        return True, None
    
    def check_signal_strength(
        self,
        signal: int,
        signal_strength: Optional[float] = None,
        min_strength: float = 0.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if signal is strong enough.
        
        Args:
            signal: Raw signal value
            signal_strength: Signal confidence (0-1)
            min_strength: Minimum required strength
            
        Returns:
            (is_allowed, reason) tuple
        """
        if signal == 0:
            return False, "Hold signal (no action)"
        
        if signal_strength is not None and signal_strength < min_strength:
            return False, f"Signal too weak ({signal_strength:.2f} < {min_strength:.2f})"
        
        return True, None
    
    def check_regime_compatibility(
        self,
        regime: str,
        allowed_regimes: Optional[list] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if current regime is suitable for trading.
        
        Args:
            regime: Current market regime
            allowed_regimes: List of regimes to trade in (None = all)
            
        Returns:
            (is_allowed, reason) tuple
        """
        if allowed_regimes is None:
            return True, None
        
        if regime not in allowed_regimes:
            return False, f"Regime '{regime}' not in allowed list"
        
        return True, None
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def is_in_position(self, state: Any) -> bool:
        """Check if currently in position."""
        if not hasattr(state, 'current_position'):
            return False
        return state.current_position != 0
    
    def is_long(self, state: Any) -> bool:
        """Check if currently long."""
        if not hasattr(state, 'current_position'):
            return False
        return state.current_position > 0
    
    def is_short(self, state: Any) -> bool:
        """Check if currently short."""
        if not hasattr(state, 'current_position'):
            return False
        return state.current_position < 0
    
    def is_flat(self, state: Any) -> bool:
        """Check if currently flat (no position)."""
        return not self.is_in_position(state)
    
    def get_position_side(self, state: Any) -> Optional[str]:
        """Get current position side."""
        if self.is_long(state):
            return "long"
        elif self.is_short(state):
            return "short"
        return None
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get configuration parameter."""
        return self.params.get(key, default)
    
    def set_param(self, key: str, value: Any) -> None:
        """Set configuration parameter."""
        self.params[key] = value
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"cooldown={self.cooldown_seconds}s, "
            f"max_positions={self.max_positions})"
        )