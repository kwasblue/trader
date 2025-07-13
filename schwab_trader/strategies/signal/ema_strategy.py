import numpy as np
from strategies.base_strategy import BaseStrategy

class EMAStrategy(BaseStrategy):
    def generate_signal(self, data):
        short_window = self.params.get("short_window", 20)
        long_window = self.params.get("long_window", 50)
        data["EMA_Short"] = data["Close"].ewm(span=short_window, adjust=False).mean()
        data["EMA_Long"] = data["Close"].ewm(span=long_window, adjust=False).mean()
        data["Signal"] = np.where(data["EMA_Short"] > data["EMA_Long"], 1, -1)
        return data