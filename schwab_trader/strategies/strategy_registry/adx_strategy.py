import numpy as np
from ta.trend import ADXIndicator
from core.base.base_strategy import BaseStrategy

class ADXStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 14)
        threshold = self.params.get("threshold", 25)
        adx = ADXIndicator(data["High"], data["Low"], data["Close"], window=window)
        data["ADX"] = adx.adx()
        data["+DI"] = adx.adx_pos()
        data["-DI"] = adx.adx_neg()
        data["Signal"] = np.where((data["ADX"] > threshold) & (data["+DI"] > data["-DI"]), 1,
                                  np.where((data["ADX"] > threshold) & (data["-DI"] > data["+DI"]), -1, 0))
        return data