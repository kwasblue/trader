import numpy as np
from strategies.base_strategy import BaseStrategy

class SMAStrategy(BaseStrategy):
    def generate_signal(self, data):
        short_window = self.params.get("short_window", 20)
        long_window = self.params.get("long_window", 50)
        data["SMA_Short"] = data["Close"].rolling(window=short_window).mean()
        data["SMA_Long"] = data["Close"].rolling(window=long_window).mean()
        data["Signal"] = np.where(data["SMA_Short"] > data["SMA_Long"], 1, -1)
        return data