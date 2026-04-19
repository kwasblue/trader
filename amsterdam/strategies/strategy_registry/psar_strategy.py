import numpy as np
import pandas as pd
from ta.trend import PSARIndicator

from core.base.base_strategy import BaseStrategy


class PSARStrategy(BaseStrategy):
    def generate_signal(self, data):
        psar = PSARIndicator(data["High"], data["Low"], data["Close"])
        data["PSAR"] = psar.psar()
        data["Signal"] = np.where(data["Close"] > data["PSAR"], 1, -1)
        if data.empty or "Signal" not in data.columns:
            return 0
        return int(data["Signal"].iloc[-1])

    def generate_signals_vectorized(self, data: pd.DataFrame) -> list[int] | None:
        """Vectorized Parabolic SAR signal generation for fast backtesting."""
        high = data["High"] if "High" in data.columns else data["high"]
        low = data["Low"] if "Low" in data.columns else data["low"]
        close = data["Close"] if "Close" in data.columns else data["close"]

        # Use ta library for PSAR calculation (already optimized)
        psar_indicator = PSARIndicator(high, low, close)
        psar = psar_indicator.psar()

        # Ensure PSAR values match input length (ta library may return different length)
        n = len(close)
        if len(psar) != n:
            psar = psar.iloc[:n] if len(psar) > n else psar.reindex(close.index, fill_value=close.iloc[0])

        # Generate signals: buy when price above PSAR, sell when below
        # Use .values to avoid index mismatch issues
        signals = np.where(close.values > psar.values, 1, -1)

        # No signal during initial periods (PSAR needs some warmup)
        signals[:5] = 0

        return signals.tolist()
