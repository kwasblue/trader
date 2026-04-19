import numpy as np
import pandas as pd
from typing import Optional, List
from core.base.base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    def generate_signal(self, data):
        fast = self.params.get("fast_window", 12)
        slow = self.params.get("slow_window", 26)
        signal = self.params.get("signal_window", 9)
        data["EMA_Fast"] = data["Close"].ewm(span=fast, adjust=False).mean()
        data["EMA_Slow"] = data["Close"].ewm(span=slow, adjust=False).mean()
        data["MACD"] = data["EMA_Fast"] - data["EMA_Slow"]
        data["MACD_Signal"] = data["MACD"].ewm(span=signal, adjust=False).mean()
        data["Signal"] = np.where(data["MACD"] > data["MACD_Signal"], 1, -1)
        if data.empty or "Signal" not in data.columns:
            return 0
        return int(data["Signal"].iloc[-1])

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        """Vectorized MACD signal generation for fast backtesting."""
        fast = self.params.get("fast_window", 12)
        slow = self.params.get("slow_window", 26)
        signal_window = self.params.get("signal_window", 9)

        close = data['Close'] if 'Close' in data.columns else data['close']

        # Calculate EMAs using pandas (already optimized)
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        # MACD line and signal line
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal_window, adjust=False).mean()

        # Generate signals: 1 when MACD > Signal, -1 otherwise
        signals = np.where(macd > macd_signal, 1, -1)

        # No signal during warmup period
        warmup = slow + signal_window
        signals[:warmup] = 0

        return signals.tolist()