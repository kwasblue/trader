"""
Position Sizer - Calculate position sizes based on risk management rules

Provides framework for different position sizing strategies:
- Fixed dollar amount
- Percentage of equity
- Risk-based (volatility adjusted)
- Kelly criterion
- Custom algorithms
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class PositionSizerBase(ABC):
    """
    Abstract base class for position sizing logic.

    Position sizing determines HOW MUCH to trade based on:
    - Account size and available capital
    - Risk tolerance per trade
    - Market volatility (ATR, volatility regime)
    - Signal strength/confidence
    - Position limits and constraints

    Design Philosophy:
    - Position sizer only calculates quantity
    - Does NOT enforce limits (that's risk manager's job)
    - Does NOT execute trades (that's executor's job)
    - Does NOT generate signals (that's strategy's job)

    Example:
        sizer = ATRPositionSizer(risk_per_trade=0.01, atr_multiplier=2.0)

        qty = sizer.calculate_position_size(
            symbol="AAPL",
            price=150.00,
            account_value=100000,
            atr=2.5,
            signal_strength=0.85
        )
        # Returns: 266 shares (risk $1000, stop at $3.75 distance)
    """

    def __init__(self, **kwargs):
        """
        Initialize position sizer with configuration.

        Args:
            **kwargs: Sizer-specific configuration parameters
        """
        self.params = dict(kwargs)

    # ========================================================================
    # ABSTRACT METHODS
    # ========================================================================

    @abstractmethod
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        account_value: float,
        signal_strength: float = 1.0,
        atr: float | None = None,
        stop_loss_price: float | None = None,
        **kwargs,
    ) -> int:
        """
        Calculate appropriate position size for a trade.

        This is the main method that determines how many shares to trade.
        Implementations should consider:
        - Account size and available capital
        - Risk tolerance
        - Volatility (ATR or other measures)
        - Signal strength/confidence
        - Any position limits

        Args:
            symbol: Trading symbol
            price: Current market price
            account_value: Total account value (or available capital)
            signal_strength: Signal confidence (0.0 to 1.0)
            atr: Average True Range (for volatility-based sizing)
            stop_loss_price: Stop loss price (if known)
            **kwargs: Additional context:
                - market_conditions: "high_volatility", "low_volatility", etc.
                - signal: Raw signal value (1, -1, 0)
                - regime: Market regime classification
                - existing_position: Current position size

        Returns:
            Number of shares to trade (integer)
            Returns 0 if position should not be taken

        Example:
            # Risk-based sizing
            qty = sizer.calculate_position_size(
                symbol="AAPL",
                price=150.00,
                account_value=100000,
                signal_strength=0.8,
                atr=2.5,
                market_conditions="trending"
            )
        """
        pass

    # ========================================================================
    # OPTIONAL METHODS (Can be overridden)
    # ========================================================================

    def adjust_for_volatility(self, base_size: float, atr: float, price: float) -> float:
        """
        Adjust position size based on volatility.

        Higher volatility → smaller position
        Lower volatility → larger position

        Args:
            base_size: Base position size before adjustment
            atr: Average True Range
            price: Current price

        Returns:
            Adjusted position size
        """
        if atr <= 0:
            return base_size

        # ATR as percentage of price
        volatility_pct = atr / price

        # Reduce size as volatility increases
        # Example: 2% volatility → 1.0x, 4% → 0.5x, 1% → 2.0x
        target_volatility = 0.02  # 2% baseline
        adjustment = target_volatility / volatility_pct

        # Cap adjustment to reasonable range
        adjustment = max(0.25, min(adjustment, 4.0))

        return base_size * adjustment

    def adjust_for_signal_strength(self, base_size: float, signal_strength: float) -> float:
        """
        Adjust position size based on signal confidence.

        Higher confidence → larger position
        Lower confidence → smaller position

        Args:
            base_size: Base position size before adjustment
            signal_strength: Signal confidence (0.0 to 1.0)

        Returns:
            Adjusted position size
        """
        # Linear scaling: 0.5 strength → 0.5x size, 1.0 strength → 1.0x size
        return base_size * signal_strength

    def apply_minimum_size(self, size: float, min_size: int = 1) -> int:
        """
        Apply minimum position size constraint.

        Args:
            size: Calculated position size
            min_size: Minimum viable trade size

        Returns:
            Size as integer, or 0 if below minimum
        """
        size_int = int(size)
        return size_int if size_int >= min_size else 0

    def apply_maximum_size(self, size: int, max_size: int) -> int:
        """
        Apply maximum position size constraint.

        Args:
            size: Calculated position size
            max_size: Maximum allowed position size

        Returns:
            Size capped at maximum
        """
        return min(size, max_size)

    def calculate_stop_distance(
        self, price: float, stop_loss_price: float | None = None, atr: float | None = None, atr_multiplier: float = 2.0
    ) -> float:
        """
        Calculate stop loss distance.

        Uses explicit stop price if provided, otherwise calculates
        from ATR.

        Args:
            price: Current price
            stop_loss_price: Explicit stop price (if known)
            atr: Average True Range
            atr_multiplier: Multiplier for ATR-based stop

        Returns:
            Stop distance in dollars
        """
        if stop_loss_price is not None:
            return abs(price - stop_loss_price)
        elif atr is not None:
            return atr * atr_multiplier
        else:
            # Default to 2% of price
            return price * 0.02

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
        return f"{self.__class__.__name__}(params={self.params})"
