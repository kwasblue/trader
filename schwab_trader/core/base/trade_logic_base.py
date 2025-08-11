from abc import ABC, abstractmethod

class TradeLogic(ABC):
    """
    Abstract base class for defining custom trade logic.
    Subclasses should implement the `execute` method.
    """

    @abstractmethod
    def execute(self, symbol, state, signal, price, atr, regime, broker, sizer, performance_tracker):
        """
        Execute trade logic for a given signal.

        Args:
            symbol (str): Ticker symbol
            state (object): Symbol-specific trading state
            signal (int): +1 for Buy, -1 for Sell, 0 for Hold
            price (float): Current price
            atr (float): Average True Range for volatility sizing
            regime (str): Market regime
            broker (BrokerInterface): Broker for executing trades
            sizer (PositionSizerBase): Determines position size
            performance_tracker (Any): Tracks PnL, trades, etc.
        """
        pass
