# core/logic/portfolio_state.py
from dataclasses import dataclass, field
from typing import Dict
from core.types import BrokerSnapshot  # <- the dataclass we defined earlier

@dataclass
class SymbolPosition:
    qty: int = 0
    avg_price: float = 0.0
    last_price: float = 0.0

@dataclass
class PortfolioState:
    cash: float = 100_000.0
    positions: Dict[str, SymbolPosition] = field(default_factory=dict)

    def update_price(self, symbol: str, price: float) -> None:
        pos = self.positions.setdefault(symbol, SymbolPosition())
        pos.last_price = float(price)

    # keep apply_fill ONLY for legacy backtests; don't call it in live/sim now
    def apply_fill(self, symbol: str, side: str, qty: int, price: float) -> None:
        ...

    # unrealized pnl
    def total_unrealized(self) -> float:
        pnl = 0.0
        for p in self.positions.values():
            if p.qty != 0:
                pnl += (p.last_price - p.avg_price) * p.qty
        return pnl

    def total_equity(self) -> float:
        return self.cash + self.total_unrealized()

    # NEW: mirror from broker snapshot
    def sync_from_snapshot(self, snap: BrokerSnapshot) -> None:
        self.cash = snap.cash
        self.positions.clear()
        for sym, pv in snap.positions.items():
            self.positions[sym] = SymbolPosition(
                qty=pv.qty, avg_price=pv.avg_price, last_price=pv.last_price
            )
