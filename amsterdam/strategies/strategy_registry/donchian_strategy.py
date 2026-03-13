import numpy as np
import pandas as pd
from typing import Optional, List
from core.base.base_strategy import BaseStrategy


class DonchianStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 20)
        data["Donchian_High"] = data["High"].rolling(window=window).max()
        data["Donchian_Low"] = data["Low"].rolling(window=window).min()
        data["Signal"] = np.where(data["Close"] > data["Donchian_High"].shift(1), 1,
                                  np.where(data["Close"] < data["Donchian_Low"].shift(1), -1, 0))
        if data.empty or "Signal" not in data.columns:
            return 0
        return int(data["Signal"].iloc[-1])

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        """Vectorized Donchian channel signal generation for fast backtesting."""
        window = self.params.get("window", 20)

        high = data['High'] if 'High' in data.columns else data['high']
        low = data['Low'] if 'Low' in data.columns else data['low']
        close = data['Close'] if 'Close' in data.columns else data['close']

        # Calculate Donchian channels (pandas rolling is already efficient)
        donchian_high = high.rolling(window=window).max().shift(1)
        donchian_low = low.rolling(window=window).min().shift(1)

        # Generate signals: buy on breakout above channel high, sell on breakdown below low
        signals = np.where(close > donchian_high, 1,
                          np.where(close < donchian_low, -1, 0))

        # No signal during warmup
        signals[:window + 1] = 0

        return signals.tolist()