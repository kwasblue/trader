
from __future__ import annotations

from typing import Optional, Dict, Union, TYPE_CHECKING
from core.base.trade_logic_manager_base import TradeLogicManagerBase
from loggers.logger import Logger

if TYPE_CHECKING:
    from core.logic.trade_logic_manager import DynamicTradeLogicManager

# Dedicated log file for trade logic routing
logger = Logger(
    log_file="trade_logic_router.log",
    logger_name="TradeLogicRouter",
    propagate=True
).get_logger()


class TradeLogicRouter:
    """
    Routes to appropriate trade logic based on context.
    
    Supports multiple routing strategies:
    - By symbol: Different logic per symbol
    - By strategy: Different logic per strategy
    - By regime: Different logic per market condition
    - Fallback to default
    """
    
    def __init__(self, default_logic: Union[TradeLogicManagerBase, "DynamicTradeLogicManager"]):
        """
        Initialize router with default logic.

        Args:
            default_logic: Fallback logic if no specific match.
                          Can be a TradeLogicManagerBase or DynamicTradeLogicManager.
        """
        self.default_logic = default_logic

        # Check if default_logic is a DynamicTradeLogicManager (has .get() method)
        self._is_dynamic_router = hasattr(default_logic, 'get') and callable(getattr(default_logic, 'get', None))

        # Routing tables
        self.logic_by_symbol: Dict[str, TradeLogicManagerBase] = {}
        self.logic_by_strategy: Dict[str, TradeLogicManagerBase] = {}
        self.logic_by_regime: Dict[str, TradeLogicManagerBase] = {}
    
    def register_symbol_logic(self, symbol: str, logic: TradeLogicManagerBase) -> None:
        """Register logic for specific symbol."""
        self.logic_by_symbol[symbol] = logic
    
    def register_strategy_logic(self, strategy: str, logic: TradeLogicManagerBase) -> None:
        """Register logic for specific strategy."""
        self.logic_by_strategy[strategy] = logic
    
    def register_regime_logic(self, regime: str, logic: TradeLogicManagerBase) -> None:
        """Register logic for specific regime."""
        self.logic_by_regime[regime] = logic
    
    def get_logic(
        self,
        symbol: str,
        strategy: Optional[str] = None,
        regime: Optional[str] = None
    ) -> TradeLogicManagerBase:
        """
        Get appropriate logic for context.

        Priority:
        1. Symbol-specific logic
        2. Strategy-specific logic
        3. Regime-specific logic
        4. Default logic (or DynamicTradeLogicManager.get() if dynamic)

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            regime: Market regime

        Returns:
            Trade logic manager instance with should_trade() method
        """
        # Check symbol-specific
        if symbol in self.logic_by_symbol:
            logic = self.logic_by_symbol[symbol]
            logger.debug(f"[{symbol}] Routed to symbol-specific logic: {logic.__class__.__name__}")
            return logic

        # Check strategy-specific
        if strategy and strategy in self.logic_by_strategy:
            logic = self.logic_by_strategy[strategy]
            logger.debug(f"[{symbol}] Routed to strategy-specific logic ({strategy}): {logic.__class__.__name__}")
            return logic

        # Check regime-specific
        if regime and regime in self.logic_by_regime:
            logic = self.logic_by_regime[regime]
            logger.debug(f"[{symbol}] Routed to regime-specific logic ({regime}): {logic.__class__.__name__}")
            return logic

        # Fallback to default
        # If default_logic is a DynamicTradeLogicManager, call its .get() method
        if self._is_dynamic_router:
            logic = self.default_logic.get(symbol, regime or "normal")
            logger.debug(f"[{symbol}] Routed via DynamicTradeLogicManager ({regime}): {logic.__class__.__name__}")
            return logic

        logger.debug(f"[{symbol}] Using default logic: {self.default_logic.__class__.__name__}")
        return self.default_logic

