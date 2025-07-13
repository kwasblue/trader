import pandas as pd
from indicators.ema import EMAIndicator
from indicators.sma import SMAIndicator
from indicators.macd import MACDIndicator
from indicators.rsi import RSIIndicator
from indicators.atr import ATRIndicator
from indicators.vwap import VWAPIndicator
from indicators.obv import OBVIndicator
from indicators.momentum import MomentumIndicator
from indicators.roc import ROCIndicator
from indicators.bollinger import BollingerBandsIndicator
from indicators.psar import PSARIndicator
from indicators.price_change import PriceChangeIndicator
from indicators.percent_change import PercentChangeIndicator

class TechnicalIndicators:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def apply_all(self, sma_window=20, ema_window=20):
        self.df = SMAIndicator().calculate(self.df, sma_window)
        self.df = EMAIndicator().calculate(self.df, ema_window)
        self.df = MACDIndicator().calculate(self.df)
        self.df = RSIIndicator().calculate(self.df)
        self.df = ATRIndicator().calculate(self.df)
        self.df = VWAPIndicator().calculate(self.df)
        self.df = OBVIndicator().calculate(self.df)
        self.df = MomentumIndicator().calculate(self.df)
        self.df = ROCIndicator().calculate(self.df)
        self.df = BollingerBandsIndicator().calculate(self.df)
        self.df = PSARIndicator().calculate(self.df)
        self.df = PriceChangeIndicator().calculate(self.df)
        self.df = PercentChangeIndicator().calculate(self.df)
        return self.df
