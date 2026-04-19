import numpy as np
import pandas as pd

from core.base.base_strategy import BaseStrategy


class BollingerStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 20)
        num_std = self.params.get("num_std", 2)
        data["Rolling_Mean"] = data["Close"].rolling(window=window).mean()
        data["Rolling_Std"] = data["Close"].rolling(window=window).std()
        data["Upper_Band"] = data["Rolling_Mean"] + (data["Rolling_Std"] * num_std)
        data["Lower_Band"] = data["Rolling_Mean"] - (data["Rolling_Std"] * num_std)
        data["Signal"] = np.where(
            data["Close"] < data["Lower_Band"], 1, np.where(data["Close"] > data["Upper_Band"], -1, 0)
        )
        if data.empty or "Signal" not in data.columns:
            return 0
        return int(data["Signal"].iloc[-1])

    def generate_signals_vectorized(self, data: pd.DataFrame) -> list[int] | None:
        """Vectorized Bollinger Bands signal generation for fast backtesting."""
        window = self.params.get("window", 20)
        num_std = self.params.get("num_std", 2)

        close = data["Close"] if "Close" in data.columns else data["close"]

        # Calculate rolling mean and std (pandas rolling is already efficient)
        rolling_mean = close.rolling(window=window).mean()
        rolling_std = close.rolling(window=window).std()

        # Calculate bands
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)

        # Generate signals: buy when below lower band, sell when above upper band
        signals = np.where(close < lower_band, 1, np.where(close > upper_band, -1, 0))

        # No signal during warmup
        signals[:window] = 0

        return signals.tolist()
