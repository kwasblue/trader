"""
Symbol State - Per-symbol trading state (dumb data container)

Tracks state fields for a single symbol:
- Position details (side, quantity, prices)
- Risk levels (stops, targets) - set by PositionManager
- Performance tracking (excursions, bars held) - updated by PositionManager
- Trade logic state (pyramiding, partials)

NOTE: SymbolState is a DUMB data container. All "smart" logic
(checking stops, updating trailing, exit decisions) belongs in PositionManager.
SymbolState only stores fields and provides simple property accessors.
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

        # Position opened (fields set by PositionManager)
        state.side = "long"
        state.current_position = 100
        state.entry_price = 150.25
        state.stop_loss = 148.00
        state.take_profit = 154.50

        # Check simple properties
        if state.is_long:
            print(f"Long {state.current_position} shares")

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
    entry_regime: Optional[str] = None  # Regime when position was opened
    
    # Legacy/optional fields
    portfolio_value: float = 0.0  # For reference if needed

    # Concurrency control
    position_state: PositionState = PositionState.NONE
    pending_order_id: Optional[str] = None

    # Meta trade logging (for ML training)
    trade_id: Optional[str] = None  # Links entry/exit for meta-model training

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
    def unrealized_pnl(self) -> float:
        """
        Calculate unrealized P&L if we have position.

        Note: This requires current price to be meaningful.
        For accurate P&L tracking, use PortfolioState which tracks last_price.

        Returns:
            0.0 if flat or no entry price (cannot calculate)
        """
        if self.is_flat or self.entry_price is None:
            return 0.0

        # Note: SymbolState doesn't track current price directly.
        # For accurate P&L, portfolio should be used.
        # Return 0.0 rather than None to avoid None propagation.
        return 0.0
    
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
    # PARTIAL TARGET HELPERS
    # ========================================================================

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
        self.trade_id = None  # Clear trade_id for meta logging
        # Clear strategy ownership - allows other strategies to manage next position
        # The cooldown mechanism (in TradeApprover) prevents rapid re-entry
        self.strategy_name = None
        self.entry_regime = None

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