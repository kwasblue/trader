# strategies/strategy_registry/sma_strategy.py

import numpy as np
import pandas as pd

from core.base.base_strategy import BaseStrategy


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

    def generate_signals_vectorized(self, data: pd.DataFrame) -> list[int] | None:
        """Vectorized SMA crossover signal generation for fast backtesting."""
        close = data["Close"] if "Close" in data.columns else data["close"]

        # Calculate SMAs
        sma_fast = close.rolling(self.fast).mean()
        sma_slow = close.rolling(self.slow).mean()

        # Generate signals: 1 when fast > slow, -1 when fast < slow, 0 otherwise
        signals = np.where(sma_fast > sma_slow, 1, np.where(sma_fast < sma_slow, -1, 0))

        # No signal during warmup
        warmup = max(self.fast, self.slow)
        signals[:warmup] = 0

        return signals.tolist()
