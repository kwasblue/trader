import numpy as np
import pandas as pd
from typing import Optional, List
from ta.trend import ADXIndicator
from core.base.base_strategy import BaseStrategy


class ADXStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 14)
        threshold = self.params.get("threshold", 25)
        adx = ADXIndicator(data["High"], data["Low"], data["Close"], window=window)
        data["ADX"] = adx.adx()
        data["+DI"] = adx.adx_pos()
        data["-DI"] = adx.adx_neg()
        data["Signal"] = np.where((data["ADX"] > threshold) & (data["+DI"] > data["-DI"]), 1,
                                  np.where((data["ADX"] > threshold) & (data["-DI"] > data["+DI"]), -1, 0))
        if data.empty or "Signal" not in data.columns:
            return 0
        return int(data["Signal"].iloc[-1])

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        """Vectorized ADX signal generation for fast backtesting."""
        window = self.params.get("window", 14)
        threshold = self.params.get("threshold", 25)

        high = data['High'] if 'High' in data.columns else data['high']
        low = data['Low'] if 'Low' in data.columns else data['low']
        close = data['Close'] if 'Close' in data.columns else data['close']

        # Use ta library for ADX calculation (already optimized)
        adx_indicator = ADXIndicator(high, low, close, window=window)
        adx = adx_indicator.adx()
        plus_di = adx_indicator.adx_pos()
        minus_di = adx_indicator.adx_neg()

        # Generate signals: buy when ADX > threshold and +DI > -DI
        signals = np.where(
            (adx > threshold) & (plus_di > minus_di), 1,
            np.where((adx > threshold) & (minus_di > plus_di), -1, 0)
        )

        # No signal during warmup (ADX needs 2*window periods)
        warmup = 2 * window
        signals[:warmup] = 0

        return signals.tolist()