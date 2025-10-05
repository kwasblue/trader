from dataclasses import dataclass, field
from typing import Dict, List
from core.app_types import BrokerSnapshot

@dataclass
class SymbolPosition:
    qty: int = 0
    avg_price: float = 0.0
    last_price: float = 0.0

@dataclass
class PortfolioState:
    cash: float = 100_000.0
    positions: Dict[str, SymbolPosition] = field(default_factory=dict)
    realized_pnl: float = 0.0
    equity_history: List[float] = field(default_factory=list)
    unrealized_pnl: float = 0.0
    drawdown: float = 0.0



    @property
    def equity(self) -> float:
        """Return latest total equity (cash + unrealized)."""
        if self.equity_history:
            return self.equity_history[-1]
        return self.total_equity()
    
    def update_price(self, symbol: str, price: float) -> None:
        pos = self.positions.setdefault(symbol, SymbolPosition())
        pos.last_price = float(price)
        # Track equity movement due to MTM
        self.equity_history.append(self.total_equity())

    def apply_fill(self, symbol: str, side: str, qty: int, price: float) -> None:
        qty = int(qty)
        if qty <= 0:
            return

        side = side.lower().strip()
        if side in ("long", "buy", "cover"):
            trade_qty = +qty
        elif side in ("short", "sell"):
            trade_qty = -qty
        else:
            raise ValueError(f"Unknown side: {side}")

        p = self.positions.setdefault(symbol, SymbolPosition())
        px = float(price)

        old_qty = p.qty
        new_qty = old_qty + trade_qty

        # --- Realized PnL ---
        if old_qty > 0 and trade_qty < 0:  # closing/reducing long
            closed = min(old_qty, -trade_qty)
            self.realized_pnl += (px - p.avg_price) * closed
        elif old_qty < 0 and trade_qty > 0:  # closing/reducing short
            closed = min(-old_qty, trade_qty)
            self.realized_pnl += (p.avg_price - px) * closed

        # --- Cash ---
        self.cash -= trade_qty * px

        # --- Position update ---
        if old_qty == 0 or (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
            total_same_side = abs(old_qty) + abs(trade_qty)
            p.avg_price = (p.avg_price * abs(old_qty) + px * abs(trade_qty)) / total_same_side if abs(old_qty) else px
            p.qty = new_qty
        elif (old_qty > 0 and new_qty >= 0) or (old_qty < 0 and new_qty <= 0):
            p.qty = new_qty
            if p.qty == 0:
                p.avg_price = 0.0
        else:  # flip
            p.qty = new_qty
            p.avg_price = px

        p.last_price = px
        self.equity_history.append(self.total_equity())

    # --- Metrics ---
    def total_unrealized(self) -> float:
        return sum((p.last_price - p.avg_price) * p.qty for p in self.positions.values() if p.qty)

    def total_equity(self) -> float:
        return self.cash + self.total_unrealized()

    def current_drawdown(self) -> float:
        if not self.equity_history:
            return 0.0
        peak = max(self.equity_history)
        current = self.equity_history[-1]
        return (current - peak) / peak

    # --- Sync from broker snapshot ---
    def sync_from_snapshot(self, snap: BrokerSnapshot) -> None:
        self.cash = snap.cash
        self.positions.clear()
        for sym, pv in snap.positions.items():
            self.positions[sym] = SymbolPosition(
                qty=pv.qty, avg_price=pv.avg_price, last_price=pv.last_price
            )
        self.equity_history.append(snap.portfolio_value or self.total_equity())
