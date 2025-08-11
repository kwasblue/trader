import time
from datetime import datetime, UTC
from collections import defaultdict
import pandas as pd

from schwab_trader.loggers.logger import Logger
from core.position_sizer import DynamicPositionSizer
from core.base.base_broker_interface import BaseBrokerInterface


class Executor:
    """
    Broker-agnostic executor for live trading.
    Handles trading decisions, order submission, sizing, and logging.
    """

    def __init__(
        self,
        broker: BaseBrokerInterface,
        sizer: DynamicPositionSizer,
        dry_run: bool = False,
        risk_percentage: float = 0.7,
    ):
        self.broker = broker
        self.sizer = sizer
        self.dry_run = dry_run
        self.risk_percentage = risk_percentage
        self.logger = Logger("app.log", "LiveExecutor").get_logger()
        self.position = defaultdict(int)

    def execute(self, symbol: str, df: pd.DataFrame, signal: int, price: float, atr_value: float):
        """
        Execute a buy/sell/hold decision using broker interface.

        Args:
            symbol (str): Ticker
            df (pd.DataFrame): Price and indicator data
            signal (int): +1=Buy, -1=Sell, 0=Hold
            price (float): Current price
            atr_value (float): ATR value used for stop-loss sizing
        """
        if pd.isna(atr_value) or atr_value <= 0:
            return

        atr_25 = df['ATR'].quantile(0.25)
        atr_75 = df['ATR'].quantile(0.75)

        market_conditions = (
            "low_volatility" if atr_value < atr_25 else
            "high_volatility" if atr_value > atr_75 else
            "normal"
        )

        stop_loss_price = price - (atr_value * 2)

        cash = self.broker.get_available_funds(symbol)
        qty = int(self.sizer.calculate_position_size(
            price=price,
            stop_loss_price=stop_loss_price,
            current_cash=cash,
            market_conditions=market_conditions,
            signal=signal
        ))

        while qty > 0 and not self.broker.has_sufficient_funds(symbol, qty):
            qty -= 1

        if qty <= 0:
            self.logger.warning(f"No affordable position size for {symbol} at ${price:.2f}")
            return

        now = datetime.now(UTC)

        if signal == 1 and self.position[symbol] == 0:
            self._place_order("BUY", symbol, qty, price)

        elif signal == -1 and self.position[symbol] > 0:
            self._place_order("SELL", symbol, self.position[symbol], price)

        elif signal == 0:
            self.logger.info(f"[{symbol}] HOLD - No trade action taken")

    def _place_order(self, side: str, symbol: str, qty: int, price: float):
        if self.dry_run:
            self.logger.info(f"[DRY RUN] {side} {qty} {symbol} @ {price:.2f}")
            return

        try:
            response = self.broker.place_market_order(symbol, qty, side)
            self.logger.info(f"[{side}] {symbol}: {qty} @ {price:.2f} → Response: {response}")
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return

        if side == "BUY":
            self.position[symbol] += qty
        elif side == "SELL":
            self.position[symbol] = 0
