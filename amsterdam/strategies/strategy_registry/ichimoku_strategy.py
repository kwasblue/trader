import numpy as np
import pandas as pd

from core.base.base_strategy import BaseStrategy


class IchimokuStrategy(BaseStrategy):
    def generate_signal(self, data):
        data["Tenkan_Sen"] = (data["High"].rolling(window=9).max() + data["Low"].rolling(window=9).min()) / 2
        data["Kijun_Sen"] = (data["High"].rolling(window=26).max() + data["Low"].rolling(window=26).min()) / 2
        data["Senkou_Span_A"] = ((data["Tenkan_Sen"] + data["Kijun_Sen"]) / 2).shift(26)
        data["Senkou_Span_B"] = (
            (data["High"].rolling(window=52).max() + data["Low"].rolling(window=52).min()) / 2
        ).shift(26)
        data["Signal"] = np.where(
            data["Close"] > data["Senkou_Span_A"], 1, np.where(data["Close"] < data["Senkou_Span_B"], -1, 0)
        )
        if data.empty or "Signal" not in data.columns:
            return 0
        return int(data["Signal"].iloc[-1])

    def generate_signals_vectorized(self, data: pd.DataFrame) -> list[int] | None:
        """Vectorized Ichimoku Cloud signal generation for fast backtesting."""
        high = data["High"] if "High" in data.columns else data["high"]
        low = data["Low"] if "Low" in data.columns else data["low"]
        close = data["Close"] if "Close" in data.columns else data["close"]

        # Calculate Ichimoku components
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)

        # Generate signals: buy when price above Span A, sell when below Span B
        signals = np.where(close > senkou_span_a, 1, np.where(close < senkou_span_b, -1, 0))

        # No signal during warmup (need 52 + 26 = 78 periods)
        warmup = 78
        signals[:warmup] = 0

        return signals.tolist()
