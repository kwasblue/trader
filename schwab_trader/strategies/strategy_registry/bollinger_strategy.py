import numpy as np
from strategies.base_strategy import BaseStrategy

class BollingerStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 20)
        num_std = self.params.get("num_std", 2)
        data["Rolling_Mean"] = data["Close"].rolling(window=window).mean()
        data["Rolling_Std"] = data["Close"].rolling(window=window).std()
        data["Upper_Band"] = data["Rolling_Mean"] + (data["Rolling_Std"] * num_std)
        data["Lower_Band"] = data["Rolling_Mean"] - (data["Rolling_Std"] * num_std)
        data["Signal"] = np.where(data["Close"] < data["Lower_Band"], 1,
                                  np.where(data["Close"] > data["Upper_Band"], -1, 0))
        return data