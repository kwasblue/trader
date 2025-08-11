import numpy as np
from core.base.base_strategy import BaseStrategy

class RSIStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 14)
        oversold = self.params.get("oversold", 30)
        overbought = self.params.get("overbought", 70)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['Signal'] = np.where(data['RSI'] < oversold, 1, np.where(data['RSI'] > overbought, -1, 0))
        return data