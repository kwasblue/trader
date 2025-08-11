import numpy as np
from ta.trend import PSARIndicator
from core.base.base_strategy import BaseStrategy

class PSARStrategy(BaseStrategy):
    def generate_signal(self, data):
        psar = PSARIndicator(data["High"], data["Low"], data["Close"])
        data["PSAR"] = psar.psar()
        data["Signal"] = np.where(data["Close"] > data["PSAR"], 1, -1)
        return data