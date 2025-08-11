from abc import ABC, abstractmethod


class PositionSizerBase(ABC):
    """
    Abstract base class for position sizing logic.

    Subclasses should implement a method to calculate position size
    based on price, risk, cash available, market conditions, and signal.
    """

    @abstractmethod
    def calculate_position_size(
        self,
        price: float,
        stop_loss_price: float,
        current_cash: float,
        market_conditions: str,
        signal: int
    ) -> float:
        """
        Calculate the appropriate position size for a trade.

        Args:
            price (float): Current market price of the asset.
            stop_loss_price (float): Price level at which trade will be exited if it moves against you.
            current_cash (float): Cash available for trading.
            market_conditions (str): Label describing current market volatility (e.g., 'high_volatility').
            signal (int): Trading signal (+1 for buy, -1 for sell, 0 for hold).

        Returns:
            float: The number of shares/units to trade.
        """
        pass
