from schwab_trader.core.base.base_indicator import BaseIndicator

class PercentChangeIndicator(BaseIndicator):
    def compute(self):
        self.df['Percent_Change'] = ((self.df['Close'] - self.df['Open']) / self.df['Open']) * 100
        return self.df