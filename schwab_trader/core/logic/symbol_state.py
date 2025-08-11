# core/logic/symbol_state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SymbolState:
    """
    Per-symbol trading state tracked by trade logic / execution engines.

    Attributes
    ----------
    symbol : str
        Ticker this state refers to.
    side : Optional[str]
        "long", "short", or None when flat.
    qty : int
        Current position size.
    entry_price : Optional[float]
        Price at which current position was opened (or last averaged).
    stop_loss : Optional[float]
        Current stop-loss level.
    take_profit : Optional[float]
        Current take-profit level.
    partial_exit_targets : List[float]
        Price targets for partial exits (front of list is next).
    pyramid_layer : int
        How many times we've added to the position.
    bars_held : int
        Bars since entry (or since last reset).
    max_favorable_excursion : Optional[float]
        Best move in our favor since entry (signed by side).
    max_adverse_excursion : Optional[float]
        Worst move against us since entry (signed by side).
    strategy_name : Optional[str]
        Strategy currently driving decisions for this symbol.
    portfolio_value : float
        Optional snapshot for logic that sizes vs. equity.
    """
    symbol: str
    side: Optional[str] = None
    qty: int = 0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    partial_exit_targets: List[float] = field(default_factory=list)
    pyramid_layer: int = 0
    bars_held: int = 0
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    strategy_name: Optional[str] = None
    portfolio_value: float = 0.0
    

    def update_excursions(self, price: float, side: Optional[str], avg_price: Optional[float]) -> None:
        if avg_price is None or side is None:
            return
        signed = (price - avg_price) if side == "long" else (avg_price - price)
        self.max_favorable_excursion = signed if self.max_favorable_excursion is None else max(self.max_favorable_excursion, signed)
        self.max_adverse_excursion   = signed if self.max_adverse_excursion   is None else min(self.max_adverse_excursion,   signed)
    
    def reset(self) -> None:
        """Clear trade-specific fields when we flatten."""
        self.side = None
        self.qty = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.partial_exit_targets.clear()
        self.pyramid_layer = 0
        self.bars_held = 0
        self.max_favorable_excursion = None
        self.max_adverse_excursion = None
        

 