import numpy as np
from core.base.base_strategy import BaseStrategy

class VWAPStrategy(BaseStrategy):
    def generate_signal(self, data):
        data["VWAP"] = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()
        data["Signal"] = np.where(data["Close"] < data["VWAP"], 1, -1)
        return data