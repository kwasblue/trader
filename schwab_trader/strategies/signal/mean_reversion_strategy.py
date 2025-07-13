import numpy as np
from strategies.base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 10)
        data["Rolling_Mean"] = data["Close"].rolling(window=window).mean()
        data["Signal"] = np.where(data["Close"] > data["Rolling_Mean"], -1, 1)
        return data