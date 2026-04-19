import numpy as np
import pandas as pd

from core.base.base_strategy import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 20)
        data["Rolling_High"] = data["High"].rolling(window=window).max()
        data["Rolling_Low"] = data["Low"].rolling(window=window).min()
        data["Signal"] = np.where(
            data["Close"] > data["Rolling_High"].shift(1),
            1,
            np.where(data["Close"] < data["Rolling_Low"].shift(1), -1, 0),
        )
        if data.empty or "Signal" not in data.columns:
            return 0
        return int(data["Signal"].iloc[-1])

    def generate_signals_vectorized(self, data: pd.DataFrame) -> list[int] | None:
        """Vectorized breakout signal generation for fast backtesting."""
        window = self.params.get("window", 20)

        high = data["High"] if "High" in data.columns else data["high"]
        low = data["Low"] if "Low" in data.columns else data["low"]
        close = data["Close"] if "Close" in data.columns else data["close"]

        # Calculate rolling high/low (pandas rolling is already efficient)
        rolling_high = high.rolling(window=window).max().shift(1)
        rolling_low = low.rolling(window=window).min().shift(1)

        # Generate signals: buy on breakout above high, sell on breakdown below low
        signals = np.where(close > rolling_high, 1, np.where(close < rolling_low, -1, 0))

        # No signal during warmup
        signals[: window + 1] = 0

        return signals.tolist()
