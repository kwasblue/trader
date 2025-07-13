import numpy as np
from strategies.base_strategy import BaseStrategy

class StochasticStrategy(BaseStrategy):
    def generate_signal(self, data):
        k_window = self.params.get("k_window", 14)
        d_window = self.params.get("d_window", 3)
        data["Lowest_Low"] = data["Low"].rolling(window=k_window).min()
        data["Highest_High"] = data["High"].rolling(window=k_window).max()
        data["%K"] = 100 * (data["Close"] - data["Lowest_Low"]) / (data["Highest_High"] - data["Lowest_Low"])
        data["%D"] = data["%K"].rolling(window=d_window).mean()
        data["Signal"] = np.where((data["%K"] > data["%D"]) & (data["%K"] < 20), 1,
                                  np.where((data["%K"] < data["%D"]) & (data["%K"] > 80), -1, 0))
        return data