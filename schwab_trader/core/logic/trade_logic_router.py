
from __future__ import annotations

from typing import Optional, Dict
import logging
from core.base.trade_logic_manager_base import TradeLogicManagerBase
from loggers.logger import Logger



class TradeLogicRouter:
    """
    Routes to appropriate trade logic based on context.
    
    Supports multiple routing strategies:
    - By symbol: Different logic per symbol
    - By strategy: Different logic per strategy
    - By regime: Different logic per market condition
    - Fallback to default
    """
    
    def __init__(self, default_logic: TradeLogicManagerBase):
        """
        Initialize router with default logic.
        
        Args:
            default_logic: Fallback logic if no specific match
        """
        self.default_logic = default_logic
        
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
        4. Default logic
        
        Args:
            symbol: Trading symbol
            strategy: Strategy name
            regime: Market regime
            
        Returns:
            Trade logic manager
        """
        # Check symbol-specific
        if symbol in self.logic_by_symbol:
            return self.logic_by_symbol[symbol]
        
        # Check strategy-specific
        if strategy and strategy in self.logic_by_strategy:
            return self.logic_by_strategy[strategy]
        
        # Check regime-specific
        if regime and regime in self.logic_by_regime:
            return self.logic_by_regime[regime]
        
        # Fallback to default
        return self.default_logic

