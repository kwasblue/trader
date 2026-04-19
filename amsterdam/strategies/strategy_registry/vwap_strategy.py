import numpy as np
import pandas as pd

from core.base.base_strategy import BaseStrategy


class VWAPStrategy(BaseStrategy):
    def generate_signal(self, data):
        data["VWAP"] = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()
        data["Signal"] = np.where(data["Close"] < data["VWAP"], 1, -1)
        if data.empty or "Signal" not in data.columns:
            return 0
        return int(data["Signal"].iloc[-1])

    def generate_signals_vectorized(self, data: pd.DataFrame) -> list[int] | None:
        """Vectorized VWAP signal generation for fast backtesting."""
        close = data["Close"] if "Close" in data.columns else data["close"]
        volume = data["Volume"] if "Volume" in data.columns else data["volume"]

        # Calculate VWAP using numpy for speed
        close_vals = close.values
        volume_vals = volume.values

        # Cumulative sums for VWAP calculation
        cum_pv = np.cumsum(close_vals * volume_vals)
        cum_vol = np.cumsum(volume_vals)

        # Avoid division by zero
        vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv, dtype=float), where=cum_vol != 0)

        # Generate signals: buy when price below VWAP (undervalued), sell when above
        signals = np.where(close_vals < vwap, 1, -1)

        # No signal for first few periods
        signals[:5] = 0

        return signals.tolist()
