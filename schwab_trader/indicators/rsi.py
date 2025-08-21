from core.base.base_indicator import BaseIndicator

class RSIIndicator(BaseIndicator):
    def __init__(self, df, periods=14):
        super().__init__(df)
        self.periods = periods

    def compute(self):
        delta = self.df['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(com=self.periods - 1, adjust=False).mean()
        ma_down = down.ewm(com=self.periods - 1, adjust=False).mean()
        rs = ma_up / ma_down
        self.df['RSI'] = 100 - (100 / (1 + rs))
        return self.df