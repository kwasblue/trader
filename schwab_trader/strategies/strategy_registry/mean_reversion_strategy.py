# strategies/strategy_registry/mean_reversion_strategy.py
import numpy as np
from typing import Optional, List
from core.base.base_strategy import BaseStrategy
import pandas as pd


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Z-score.

    Generates buy signal when price is significantly below mean (oversold),
    sell signal when price is significantly above mean (overbought).

    Optimizations:
    - Single-pass rolling statistics computation
    - Vectorized signal generation for backtesting
    """

    def __init__(self, window: int = 14, threshold: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.threshold = threshold

    def generate_signal(self, data: pd.DataFrame) -> int:
        """Generate signal for the latest bar."""
        close = data.get("Close", data.get("close"))
        if close is None or len(data) < self.window:
            return 0

        # Optimization: Single rolling object for both mean and std
        rolling = close.rolling(self.window)
        mean = rolling.mean()
        std = rolling.std(ddof=0)

        # Compute z-score for last value only
        last_z = (close.iat[-1] - mean.iat[-1]) / (std.iat[-1] + 1e-10)

        if last_z > self.threshold:
            return -1  # Overbought -> Sell
        if last_z < -self.threshold:
            return 1   # Oversold -> Buy
        return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        """
        Vectorized signal generation for fast backtesting.
        """
        close = data.get("Close", data.get("close"))
        if close is None or len(data) < self.window:
            return None

        # Single-pass rolling stats
        rolling = close.rolling(self.window)
        mean = rolling.mean()
        std = rolling.std(ddof=0)

        # Vectorized z-score
        z = (close - mean) / (std + 1e-10)

        # Vectorized signal generation
        signals = np.where(z > self.threshold, -1,
                  np.where(z < -self.threshold, 1, 0))

        # Set early signals to 0 (not enough data)
        signals[:self.window] = 0

        return signals.tolist()
