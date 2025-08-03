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
        df = self.df.copy()
        df = SMAIndicator(df, window=sma_window).compute()
        df = EMAIndicator(df, window=ema_window).compute()
        df = MACDIndicator(df).compute()
        df = RSIIndicator(df).compute()
        df = ATRIndicator(df).compute()
        df = VWAPIndicator(df).compute()
        df = OBVIndicator(df).compute()
        df = MomentumIndicator(df).compute()
        df = ROCIndicator(df).compute()
        df = BollingerBandsIndicator(df).compute()
        df = PSARIndicator(df).compute()
        df = PriceChangeIndicator(df).compute()
        df = PercentChangeIndicator(df).compute()
        return df
