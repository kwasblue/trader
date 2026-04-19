"""
Trade Approver Router

Routes to appropriate trade approver based on context (symbol, strategy, regime).

Architecture:
    DynamicTradeLogicManager (config-driven factory)
              |
              v
    TradeApproverRouter (runtime resolver - single authority)
              |
              v
    TradeApprover (approval logic - StandardTradeApprover etc.)

The router is the ONLY authority for approver selection.
DynamicTradeLogicManager can populate the router on startup.
Engines depend on the router, not the manager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.base.trade_logic_manager_base import TradeApprover
from core.tracing import trace
from loggers.logger import Logger

if TYPE_CHECKING:
    from core.logic.trade_logic_manager import DynamicTradeLogicManager

# Dedicated log file for trade approver routing
logger = Logger(log_file="trade_logic_router.log", logger_name="TradeApproverRouter", propagate=True).get_logger()


class TradeApproverRouter:
    """
    Routes to appropriate trade approver based on context.

    Supports multiple routing strategies:
    - By symbol: Different approver per symbol
    - By strategy: Different approver per strategy
    - By regime: Different approver per market condition
    - Fallback to default

    This is the SINGLE AUTHORITY for approver selection.
    Engines should depend on this router, not on DynamicTradeLogicManager directly.

    Example:
        # Create router with default approver
        default_approver = StandardTradeApprover()
        router = TradeApproverRouter(default_approver)

        # Register overrides
        router.register_symbol_approver("BTC-USD", crypto_approver)
        router.register_regime_approver("high_volatility", conservative_approver)

        # Get approver for context
        approver = router.get_approver(symbol="AAPL", strategy="momentum", regime="normal")
    """

    def __init__(self, default_approver: TradeApprover | DynamicTradeLogicManager):
        """
        Initialize router with default approver.

        Args:
            default_approver: Fallback approver if no specific match.
                              Can be a TradeApprover or DynamicTradeLogicManager.
                              If DynamicTradeLogicManager, its .get() method is used
                              as the fallback resolver.
        """
        self.default_approver = default_approver

        # Check if default_approver is a DynamicTradeLogicManager (has .get() method)
        self._is_dynamic_router = hasattr(default_approver, "get") and callable(getattr(default_approver, "get", None))

        # Routing tables
        self.approver_by_symbol: dict[str, TradeApprover] = {}
        self.approver_by_strategy: dict[str, TradeApprover] = {}
        self.approver_by_regime: dict[str, TradeApprover] = {}

    def register_symbol_approver(self, symbol: str, approver: TradeApprover) -> None:
        """Register approver for specific symbol."""
        self.approver_by_symbol[symbol] = approver
        logger.debug(f"Registered symbol approver for {symbol}: {approver.__class__.__name__}")

    def register_strategy_approver(self, strategy: str, approver: TradeApprover) -> None:
        """Register approver for specific strategy."""
        self.approver_by_strategy[strategy] = approver
        logger.debug(f"Registered strategy approver for {strategy}: {approver.__class__.__name__}")

    def register_regime_approver(self, regime: str, approver: TradeApprover) -> None:
        """Register approver for specific regime."""
        self.approver_by_regime[regime] = approver
        logger.debug(f"Registered regime approver for {regime}: {approver.__class__.__name__}")

    @trace
    def get_approver(self, symbol: str, strategy: str | None = None, regime: str | None = None) -> TradeApprover:
        """
        Get appropriate approver for context.

        Priority:
        1. Symbol-specific approver
        2. Strategy-specific approver
        3. Regime-specific approver
        4. Default approver (or DynamicTradeLogicManager.get() if dynamic)

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            regime: Market regime

        Returns:
            TradeApprover instance with should_trade() method
        """
        # Check symbol-specific
        if symbol in self.approver_by_symbol:
            approver = self.approver_by_symbol[symbol]
            logger.debug(f"[{symbol}] Routed to symbol-specific approver: {approver.__class__.__name__}")
            return approver

        # Check strategy-specific
        if strategy and strategy in self.approver_by_strategy:
            approver = self.approver_by_strategy[strategy]
            logger.debug(f"[{symbol}] Routed to strategy-specific approver ({strategy}): {approver.__class__.__name__}")
            return approver

        # Check regime-specific
        if regime and regime in self.approver_by_regime:
            approver = self.approver_by_regime[regime]
            logger.debug(f"[{symbol}] Routed to regime-specific approver ({regime}): {approver.__class__.__name__}")
            return approver

        # Fallback to default
        # If default_approver is a DynamicTradeLogicManager, call its .get() method
        if self._is_dynamic_router:
            approver = self.default_approver.get(symbol, regime or "normal")
            logger.debug(f"[{symbol}] Routed via DynamicTradeLogicManager ({regime}): {approver.__class__.__name__}")
            return approver

        logger.debug(f"[{symbol}] Using default approver: {self.default_approver.__class__.__name__}")
        return self.default_approver

    def clear_overrides(self) -> None:
        """Clear all registered overrides (keep default)."""
        self.approver_by_symbol.clear()
        self.approver_by_strategy.clear()
        self.approver_by_regime.clear()
        logger.debug("Cleared all approver overrides")

    def __repr__(self) -> str:
        return (
            f"TradeApproverRouter("
            f"default={self.default_approver.__class__.__name__}, "
            f"symbols={len(self.approver_by_symbol)}, "
            f"strategies={len(self.approver_by_strategy)}, "
            f"regimes={len(self.approver_by_regime)})"
        )


__all__ = ["TradeApproverRouter"]
