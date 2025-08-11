import numpy as np
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
        return data