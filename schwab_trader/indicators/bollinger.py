from indicators.base_indicator import BaseIndicator

class BollingerBandsIndicator(BaseIndicator):
    def __init__(self, df, window=20):
        super().__init__(df)
        self.window = window

    def compute(self):
        tp = (self.df['Close'] + self.df['High'] + self.df['Low']) / 3
        sigma = tp.rolling(self.window).std(ddof=0)
        ma = tp.rolling(self.window).mean()
        self.df['Bollinger_Upper'] = ma + 2 * sigma
        self.df['Bollinger_Lower'] = ma - 2 * sigma
        return self.df