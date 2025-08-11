from dataclasses import dataclass

@dataclass
class OrderGuardConfig:
    max_notional_per_symbol: float = 50_000
    max_qty_per_order: int = 5_000
    blocklist: set[str] = None
    allow_short: bool = True
    enforce_market_hours: bool = True

class OrderGuard:
    def __init__(self, cfg: OrderGuardConfig):
        self.cfg = cfg

    def validate(self, symbol: str, side: str, qty: int, price: float, market_open: bool) -> bool:
        if self.cfg.blocklist and symbol in self.cfg.blocklist: return False
        if side.lower() == "sell" and not self.cfg.allow_short and qty > 0: return False
        if qty <= 0 or qty > self.cfg.max_qty_per_order: return False
        if qty * price > self.cfg.max_notional_per_symbol: return False
        if self.cfg.enforce_market_hours and not market_open: return False
        return True
