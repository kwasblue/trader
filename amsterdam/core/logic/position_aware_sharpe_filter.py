"""
Position-Aware Sharpe Filter - Advanced version

Blocks ENTRY signals in poor-performing regimes but allows EXIT signals
for existing positions. This gives better control than the basic filter.

Usage:
    # In strategy_routing_manager.py, after getting the strategy:
    if self.sharpe_filter and not state.is_in_position:
        # Block entries in bad regimes
        if not self.sharpe_filter.should_trade(symbol, regime):
            return self._create_noop_strategy()

    # If already in position, use the real strategy (can exit)
    return strategy
"""

from typing import Optional
from core.logic.sharpe_filter import SharpeFilter


class PositionAwareSharpeFilter:
    """
    Enhanced Sharpe filter that's position-aware.

    Behavior:
    - NOT in position + bad regime → Block (NoOp strategy)
    - IN position + bad regime → Allow (real strategy can exit)

    This prevents entering losing strategies but allows proper exits.
    """

    def __init__(self, min_sharpe: float = 0.5):
        """
        Initialize position-aware filter.

        Args:
            min_sharpe: Minimum Sharpe to allow new positions
        """
        self.sharpe_filter = SharpeFilter(min_sharpe=min_sharpe)

    def should_trade(
        self,
        symbol: str,
        regime: str,
        is_in_position: bool = False
    ) -> bool:
        """
        Check if trading should be allowed.

        Args:
            symbol: Trading symbol
            regime: Market regime
            is_in_position: True if already holding a position

        Returns:
            True if should trade

        Logic:
        - If in position: Always allow (need to exit properly)
        - If not in position: Check Sharpe threshold
        """
        # Always allow if in position (for exits)
        if is_in_position:
            return True

        # Block new entries if Sharpe too low
        return self.sharpe_filter.should_trade(symbol, regime)

    def get_sharpe(self, symbol: str, regime: str) -> Optional[float]:
        """Get Sharpe ratio."""
        return self.sharpe_filter.get_sharpe(symbol, regime)


# Example integration in base_live_runner.py:
"""
# In _process_bar method, around line 800:

# Get strategy - but need position awareness
if state.is_in_position:
    # Already in position - use real strategy (for proper exits)
    strategy = self.router.get_strategy(symbol, regime, allow_blocked=True)
else:
    # Not in position - filter applies
    strategy = self.router.get_strategy(symbol, regime)

signal = strategy.generate_signal(df)
"""
