from indicators.base_indicator import BaseIndicator

class PSARIndicator(BaseIndicator):
    def compute(self):
        self.df['SAR'] = self.df['High'].rolling(5).max()  # Approximation
        return self.df