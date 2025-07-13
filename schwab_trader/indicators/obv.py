import numpy as np
from indicators.base_indicator import BaseIndicator

class OBVIndicator(BaseIndicator):
    def compute(self):
        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()
        return self.df