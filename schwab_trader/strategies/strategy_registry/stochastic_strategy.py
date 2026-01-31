import numpy as np
import pandas as pd
from typing import Optional, List
from core.base.base_strategy import BaseStrategy


class StochasticStrategy(BaseStrategy):
    def generate_signal(self, data):
        k_window = self.params.get("k_window", 14)
        d_window = self.params.get("d_window", 3)
        oversold = self.params.get("oversold", 20)
        overbought = self.params.get("overbought", 80)

        data["Lowest_Low"] = data["Low"].rolling(window=k_window).min()
        data["Highest_High"] = data["High"].rolling(window=k_window).max()
        data["%K"] = 100 * (data["Close"] - data["Lowest_Low"]) / (data["Highest_High"] - data["Lowest_Low"])
        data["%D"] = data["%K"].rolling(window=d_window).mean()

        data["Signal"] = np.where(
            (data["%K"].shift(1) < data["%D"].shift(1)) & (data["%K"] > data["%D"]) & (data["%K"] < oversold), 1,
            np.where(
                (data["%K"].shift(1) > data["%D"].shift(1)) & (data["%K"] < data["%D"]) & (data["%K"] > overbought), -1,
                0
            )
        )

        if data.empty or "Signal" not in data.columns:
            return 0
        return int(data["Signal"].iloc[-1])

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        """Vectorized Stochastic Oscillator signal generation for fast backtesting."""
        k_window = self.params.get("k_window", 14)
        d_window = self.params.get("d_window", 3)
        oversold = self.params.get("oversold", 20)
        overbought = self.params.get("overbought", 80)

        high = data['High'] if 'High' in data.columns else data['high']
        low = data['Low'] if 'Low' in data.columns else data['low']
        close = data['Close'] if 'Close' in data.columns else data['close']

        # Calculate %K and %D
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(window=d_window).mean()

        # Get shifted values for crossover detection
        k_prev = k.shift(1)
        d_prev = d.shift(1)

        # Generate signals: buy on bullish crossover in oversold, sell on bearish crossover in overbought
        signals = np.where(
            (k_prev < d_prev) & (k > d) & (k < oversold), 1,
            np.where(
                (k_prev > d_prev) & (k < d) & (k > overbought), -1,
                0
            )
        )

        # No signal during warmup
        warmup = k_window + d_window
        signals[:warmup] = 0

        return signals.tolist()
