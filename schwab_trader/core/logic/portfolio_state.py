# core/logic/portfolio_state.py
from dataclasses import dataclass, field
from typing import Dict
from core.app_types import BrokerSnapshot  # <- the dataclass we defined earlier

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

    def apply_fill(self, symbol: str, side: str, qty: int, price: float) -> None:
        """
        Apply a trade fill to the portfolio (SIM/exec path).
        - Signed qty convention: position.qty > 0 long, < 0 short
        - side: "long"/"buy" increases qty, "short"/"sell" decreases qty
        - price: fill price
        - fee_rate: e.g. 0.001 for 10 bps (cash is reduced by fees)

        Cash update uses trade cash-flows:
          buy/long adds +qty shares  → cash -= qty*price + fees
          sell/short adds -qty shares → cash += qty*price (where qty passed in is positive) minus fees
        """
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
        #fees = abs(qty * px) * float(fee_rate)

        # 1) Cash flow (proceeds positive, purchases negative)
        #    cash -= trade_qty * price yields:
        #      + buy/long (trade_qty>0)  → cash decreases
        #      + sell/short (trade_qty<0) → cash increases
        self.cash -= trade_qty * px
        #self.cash -= fees

        # 2) Position/avg_price update
        old_qty = p.qty
        new_qty = old_qty + trade_qty

        if old_qty == 0 or (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
            # (a) Opening or adding to same-side position → VWAP the entry price
            total_shares_same_side = abs(old_qty) + abs(trade_qty)
            if abs(old_qty) == 0:
                p.avg_price = px
            else:
                # Weighted average on absolute shares (avg_price is always positive)
                p.avg_price = (p.avg_price * abs(old_qty) + px * abs(trade_qty)) / total_shares_same_side
            p.qty = new_qty

        elif (old_qty > 0 and new_qty >= 0) or (old_qty < 0 and new_qty <= 0):
            # (b) Reducing position without flipping side (partial or full close)
            #     Keep avg_price for the remaining shares if not flat.
            p.qty = new_qty
            if p.qty == 0:
                p.avg_price = 0.0  # flat resets avg

        else:
            # (c) Crossing through flat (flip): close the old side fully and open the new side
            #     The portion that crosses past zero becomes a fresh position at current price.
            #     Example: old +50, trade -70 → close 50, open -20 at px
            p.qty = new_qty
            p.avg_price = px  # new side starts at current fill price

        # 3) Keep last price fresh for MTM
        p.last_price = px

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
