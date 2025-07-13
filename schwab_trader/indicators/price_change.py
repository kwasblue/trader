from indicators.base_indicator import BaseIndicator

class PriceChangeIndicator(BaseIndicator):
    def compute(self):
        self.df['Price_Change'] = self.df['Close'] - self.df['Open']
        return self.df