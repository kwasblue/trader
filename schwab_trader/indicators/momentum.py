from indicators.base_indicator import BaseIndicator

class MomentumIndicator(BaseIndicator):
    def __init__(self, df, window=10):
        super().__init__(df)
        self.window = window

    def compute(self):
        self.df['Momentum'] = self.df['Close'] - self.df['Close'].shift(self.window)
        return self.df