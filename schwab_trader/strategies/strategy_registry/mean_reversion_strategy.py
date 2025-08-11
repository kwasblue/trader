# strategies/strategy_registry/mean_reversion_strategy.py
from core.base.base_strategy import BaseStrategy
import pandas as pd

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, window: int = 14, threshold: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.threshold = threshold

    def generate_signal(self, data: pd.DataFrame) -> int:
        df = data.copy()
        close = df["Close"] if "Close" in df.columns else df["close"]
        if len(df) < self.window:
            return 0
        z = (close - close.rolling(self.window).mean()) / close.rolling(self.window).std(ddof=0)
        if z.iloc[-1] > self.threshold:
            return -1
        if z.iloc[-1] < -self.threshold:
            return 1
        return 0
