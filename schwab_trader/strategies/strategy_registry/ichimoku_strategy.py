import numpy as np
from core.base.base_strategy import BaseStrategy

class IchimokuStrategy(BaseStrategy):
    def generate_signal(self, data):
        data["Tenkan_Sen"] = (data["High"].rolling(window=9).max() + data["Low"].rolling(window=9).min()) / 2
        data["Kijun_Sen"] = (data["High"].rolling(window=26).max() + data["Low"].rolling(window=26).min()) / 2
        data["Senkou_Span_A"] = ((data["Tenkan_Sen"] + data["Kijun_Sen"]) / 2).shift(26)
        data["Senkou_Span_B"] = ((data["High"].rolling(window=52).max() + data["Low"].rolling(window=52).min()) / 2).shift(26)
        data["Signal"] = np.where(data["Close"] > data["Senkou_Span_A"], 1,
                                  np.where(data["Close"] < data["Senkou_Span_B"], -1, 0))
        return data