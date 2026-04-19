# strategies/strategy_registry/momentum_strategy.py

import numpy as np
import pandas as pd

from core.base.base_strategy import BaseStrategy


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

    def generate_signals_vectorized(self, data: pd.DataFrame) -> list[int] | None:
        """Vectorized momentum signal generation for fast backtesting."""
        close = data["Close"].values if "Close" in data.columns else data["close"].values
        n = len(close)

        # Compare current price with price `lookback` periods ago
        signals = np.zeros(n, dtype=int)

        for i in range(self.lookback + 1, n):
            if close[i] > close[i - self.lookback - 1]:
                signals[i] = 1
            else:
                signals[i] = -1

        return signals.tolist()
