"""
Symbol State - Per-symbol trading state tracking

Tracks all state for a single symbol including:
- Position details (side, quantity, prices)
- Risk management (stops, targets)
- Performance tracking (excursions, bars held)
- Trade logic state (pyramiding, partials)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, timezone
from loggers.logger import Logger

from core.enums import PositionState

# Module-level logger instance
_logger_instance = Logger(
    log_file="symbol_state.log",
    logger_name="SymbolState",
    propagate=True
)
logger = _logger_instance.get_logger()


@dataclass
class SymbolState:
    """
    Per-symbol trading state.
    
    Tracks all information needed for trade logic decisions and
    position management for a single symbol.
    
    Position State:
    - side: "long", "short", or None
    - current_position: Quantity (from portfolio)
    - entry_price: Average entry price
    
    Risk Management:
    - stop_loss: Stop loss price level
    - take_profit: Take profit price level
    - partial_exit_targets: List of partial exit prices
    
    Performance Tracking:
    - bars_held: Bars since entry
    - max_favorable_excursion: Best price move
    - max_adverse_excursion: Worst price move
    
    Trade Logic:
    - pyramid_layer: Number of adds to position
    - last_trade_time: When last trade occurred
    - strategy_name: Active strategy
    
    Example:
        state = SymbolState(symbol="AAPL")
        
        # Position opened
        state.side = "long"
        state.current_position = 100
        state.entry_price = 150.25
        state.stop_loss = 148.00
        state.take_profit = 154.50
        
        # Update on each bar
        state.bars_held += 1
        state.update_excursions(151.50, "long", 150.25)
        
        # Position closed
        state.reset()
    """
    
    # Core identification
    symbol: str
    
    # Position state (synced from portfolio)
    side: Optional[str] = None  # "long", "short", or None
    current_position: int = 0    # Actual position from portfolio
    entry_price: Optional[float] = None
    
    # Risk management levels
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    partial_exit_targets: List[float] = field(default_factory=list)
    
    # Position tracking
    pyramid_layer: int = 0
    bars_held: int = 0
    last_trade_time: Optional[datetime] = None
    entry_date: Optional[datetime] = None  # Date position was opened (for swing mode)
    
    # Performance metrics
    max_favorable_excursion: Optional[float] = None  # MFE
    max_adverse_excursion: Optional[float] = None    # MAE
    
    # Strategy tracking
    strategy_name: Optional[str] = None
    
    # Legacy/optional fields
    portfolio_value: float = 0.0  # For reference if needed

    # Concurrency control
    position_state: PositionState = PositionState.NONE
    pending_order_id: Optional[str] = None

    def __post_init__(self):
        """Initialize asyncio lock for thread-safe operations."""
        self._lock = asyncio.Lock()
    
    # ========================================================================
    # PROPERTIES
    # ========================================================================
    
    @property
    def is_long(self) -> bool:
        """Check if currently long."""
        return self.side == "long"
    
    @property
    def is_short(self) -> bool:
        """Check if currently short."""
        return self.side == "short"
    
    @property
    def is_flat(self) -> bool:
        """Check if currently flat (no position)."""
        return self.side is None or self.current_position == 0
    
    @property
    def is_in_position(self) -> bool:
        """Check if in any position."""
        return not self.is_flat
    
    @property
    def unrealized_pnl(self) -> Optional[float]:
        """
        Calculate unrealized P&L if we have position.
        
        Note: This requires current price to be meaningful.
        Better to get from portfolio which tracks last_price.
        """
        if self.is_flat or self.entry_price is None:
            return None
        
        # Need current price - this is why portfolio tracking is better
        logger.warning(
            f"[{self.symbol}] unrealized_pnl called but state doesn't track current price"
        )
        return None
    
    @property
    def has_stop_loss(self) -> bool:
        """Check if stop loss is set."""
        return self.stop_loss is not None
    
    @property
    def has_take_profit(self) -> bool:
        """Check if take profit is set."""
        return self.take_profit is not None
    
    @property
    def has_partial_targets(self) -> bool:
        """Check if partial exit targets exist."""
        return len(self.partial_exit_targets) > 0
    
    # ========================================================================
    # EXCURSION TRACKING
    # ========================================================================
    
    def update_excursions(
        self,
        current_price: float,
        side: Optional[str] = None,
        avg_price: Optional[float] = None
    ) -> None:
        """
        Update max favorable and adverse excursions.
        
        Tracks the best and worst price moves since entry.
        
        Args:
            current_price: Current market price
            side: Position side (uses self.side if None)
            avg_price: Entry price (uses self.entry_price if None)
        """
        # Use instance values if not provided
        side = side or self.side
        avg_price = avg_price or self.entry_price
        
        # Can't calculate without position and entry
        if side is None or avg_price is None:
            return
        
        # Calculate signed excursion
        if side == "long":
            excursion = current_price - avg_price
        else:  # short
            excursion = avg_price - current_price
        
        # Update MFE (best move)
        if self.max_favorable_excursion is None:
            self.max_favorable_excursion = excursion
        else:
            self.max_favorable_excursion = max(
                self.max_favorable_excursion,
                excursion
            )
        
        # Update MAE (worst move)
        if self.max_adverse_excursion is None:
            self.max_adverse_excursion = excursion
        else:
            self.max_adverse_excursion = min(
                self.max_adverse_excursion,
                excursion
            )
    
    # ========================================================================
    # RISK LEVEL CHECKS
    # ========================================================================
    
    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop loss hit.
        
        Args:
            current_price: Current market price
            
        Returns:
            True if stop loss hit
        """
        if not self.has_stop_loss:
            return False
        
        if self.is_long:
            return current_price <= self.stop_loss
        elif self.is_short:
            return current_price >= self.stop_loss
        
        return False
    
    def check_take_profit(self, current_price: float) -> bool:
        """
        Check if take profit hit.
        
        Args:
            current_price: Current market price
            
        Returns:
            True if take profit hit
        """
        if not self.has_take_profit:
            return False
        
        if self.is_long:
            return current_price >= self.take_profit
        elif self.is_short:
            return current_price <= self.take_profit
        
        return False
    
    def check_partial_target(self, current_price: float) -> Optional[float]:
        """
        Check if partial exit target hit.
        
        Args:
            current_price: Current market price
            
        Returns:
            Target price if hit, None otherwise
        """
        if not self.has_partial_targets:
            return None
        
        # Check first target in list
        target = self.partial_exit_targets[0]
        
        if self.is_long and current_price >= target:
            return target
        elif self.is_short and current_price <= target:
            return target
        
        return None
    
    def pop_partial_target(self) -> Optional[float]:
        """
        Remove and return first partial target.
        
        Call this after executing partial exit.
        
        Returns:
            Target price, or None if no targets
        """
        if not self.has_partial_targets:
            return None
        
        return self.partial_exit_targets.pop(0)
    
    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================
    
    def reset(self) -> None:
        """
        Reset state when position closed.

        Clears all trade-specific fields but preserves symbol.
        """
        self.side = None
        self.current_position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.partial_exit_targets.clear()
        self.pyramid_layer = 0
        self.bars_held = 0
        self.max_favorable_excursion = None
        self.max_adverse_excursion = None
        # Reset position lifecycle state
        self.position_state = PositionState.NONE
        self.pending_order_id = None
        # Note: strategy_name preserved (may use same strategy next time)

        logger.debug(f"[{self.symbol}] State reset")
    
    def update_from_portfolio(
        self,
        qty: int,
        avg_price: Optional[float] = None
    ) -> None:
        """
        Sync state from portfolio position.
        
        Args:
            qty: Current position quantity
            avg_price: Average entry price (if available)
        """
        self.current_position = qty
        
        # Determine side from quantity
        if qty > 0:
            self.side = "long"
        elif qty < 0:
            self.side = "short"
        else:
            self.side = None
        
        # Update entry price if provided and we're in position
        if avg_price is not None and qty != 0:
            self.entry_price = avg_price
        elif qty == 0:
            self.entry_price = None
    
    # ========================================================================
    # TRAILING STOP MANAGEMENT
    # ========================================================================
    
    def update_trailing_stop(
        self,
        current_price: float,
        atr: float,
        multiplier: float = 1.5
    ) -> None:
        """
        Update trailing stop loss.
        
        For longs: trails up as price rises
        For shorts: trails down as price falls
        
        Args:
            current_price: Current market price
            atr: Average True Range
            multiplier: ATR multiplier for stop distance
        """
        if self.is_flat:
            return
        
        stop_distance = atr * multiplier
        
        if self.is_long:
            # Trail stop up
            new_stop = current_price - stop_distance
            if self.stop_loss is None:
                self.stop_loss = new_stop
            else:
                self.stop_loss = max(self.stop_loss, new_stop)
        
        elif self.is_short:
            # Trail stop down
            new_stop = current_price + stop_distance
            if self.stop_loss is None:
                self.stop_loss = new_stop
            else:
                self.stop_loss = min(self.stop_loss, new_stop)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_risk_distance(self) -> Optional[float]:
        """
        Get distance from entry to stop loss.
        
        Returns:
            Distance in dollars, or None if no stop
        """
        if self.entry_price is None or self.stop_loss is None:
            return None
        
        return abs(self.entry_price - self.stop_loss)
    
    def get_reward_distance(self) -> Optional[float]:
        """
        Get distance from entry to take profit.
        
        Returns:
            Distance in dollars, or None if no target
        """
        if self.entry_price is None or self.take_profit is None:
            return None
        
        return abs(self.take_profit - self.entry_price)
    
    def get_risk_reward_ratio(self) -> Optional[float]:
        """
        Calculate risk/reward ratio.
        
        Returns:
            Reward/risk ratio, or None if can't calculate
        """
        risk = self.get_risk_distance()
        reward = self.get_reward_distance()
        
        if risk is None or reward is None or risk == 0:
            return None
        
        return reward / risk
    
    def to_dict(self) -> dict:
        """
        Convert state to dictionary.
        
        Useful for logging or serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            'symbol': self.symbol,
            'side': self.side,
            'position': self.current_position,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'bars_held': self.bars_held,
            'mfe': self.max_favorable_excursion,
            'mae': self.max_adverse_excursion,
            'strategy': self.strategy_name,
        }
    
    def __repr__(self) -> str:
        position_str = f"{self.side.upper()} {abs(self.current_position)}" if self.side else "FLAT"
        return (
            f"SymbolState({self.symbol}: {position_str}, "
            f"entry=${self.entry_price:.2f if self.entry_price else 0:.2f}, "
            f"bars={self.bars_held})"
        )