# strategies/strategy_registry/sma_strategy.py
from core.base.base_strategy import BaseStrategy
import pandas as pd

class SMAStrategy(BaseStrategy):
    def __init__(self, fast: int = 10, slow: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.fast = fast
        self.slow = slow

    def generate_signal(self, data: pd.DataFrame) -> int:
        df = data.copy()
        close = df["Close"] if "Close" in df.columns else df["close"]
        s_fast = close.rolling(self.fast).mean()
        s_slow = close.rolling(self.slow).mean()
        if len(df) < max(self.fast, self.slow):
            return 0
        return 1 if s_fast.iloc[-1] > s_slow.iloc[-1] else -1 if s_fast.iloc[-1] < s_slow.iloc[-1] else 0
