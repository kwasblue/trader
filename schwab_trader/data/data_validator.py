from datetime import timedelta

class BarValidator:
    def __init__(self, max_age_sec: int = 120, max_move_pct: float = 0.2):
        self.max_age = timedelta(seconds=max_age_sec)
        self.max_move_pct = max_move_pct

    def is_valid(self, prev_close: float | None, bar: dict) -> bool:
        if any(k not in bar for k in ("timestamp","open","high","low","close")): return False
        if prev_close is not None:
            move = abs(bar["close"] - prev_close) / max(prev_close, 1e-9)
            if move > self.max_move_pct: return False
        return True
