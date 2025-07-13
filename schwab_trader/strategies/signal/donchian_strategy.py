import numpy as np
from strategies.base_strategy import BaseStrategy

class DonchianStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 20)
        data["Donchian_High"] = data["High"].rolling(window=window).max()
        data["Donchian_Low"] = data["Low"].rolling(window=window).min()
        data["Signal"] = np.where(data["Close"] > data["Donchian_High"].shift(1), 1,
                                  np.where(data["Close"] < data["Donchian_Low"].shift(1), -1, 0))
        return data