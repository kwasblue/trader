# strategies/strategy_registry/momentum_strategy.py
from core.base.base_strategy import BaseStrategy
import pandas as pd

class MomentumStrategy(BaseStrategy):
    def __init__(self, lookback: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback

    def generate_signal(self, data: pd.DataFrame) -> int:
        df = data.copy()
        close = df["Close"] if "Close" in df.columns else df["close"]
        if len(df) < self.lookback + 1:
            return 0
        return 1 if close.iloc[-1] > close.iloc[-self.lookback - 1] else -1
