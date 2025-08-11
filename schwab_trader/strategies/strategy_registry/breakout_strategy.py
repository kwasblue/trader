import numpy as np
from core.base.base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 20)
        data["Rolling_High"] = data["High"].rolling(window=window).max()
        data["Rolling_Low"] = data["Low"].rolling(window=window).min()
        data["Signal"] = np.where(data["Close"] > data["Rolling_High"].shift(1), 1,
                                  np.where(data["Close"] < data["Rolling_Low"].shift(1), -1, 0))
        return data