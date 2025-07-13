import numpy as np
from strategies.base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 10)
        data["Momentum"] = data["Close"].diff(window)
        data["Signal"] = np.where(data["Momentum"] > 0, 1, -1)
        return data