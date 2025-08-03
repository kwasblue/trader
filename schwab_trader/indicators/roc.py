from indicators.base_indicator import BaseIndicator

class ROCIndicator(BaseIndicator):
    def __init__(self, df, window=10):
        super().__init__(df)
        self.window = window

    def compute(self):
        self.df['ROC'] = self.df['Close'].pct_change(periods=self.window) * 100
        return self.df