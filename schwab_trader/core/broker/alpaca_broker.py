from alpaca.trading.client import TradingClient
from alpaca.data.live.stock import StockDataStream
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from core.base.base_broker_interface import BaseBrokerInterface
from loggers.logger import Logger


def _map_side(side: str) -> OrderSide:
    return OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL


class AlpacaBroker(BaseBrokerInterface):
    """
    Broker implementation for Alpaca using the Alpaca SDK (TradingClient + Data Stream).
    """

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.trading_client = None
        self.stream = None
        self.bar_callback = None
        self.logger = Logger("alpaca.log", "AlpacaBroker").get_logger()

    def connect(self):
        """
        Establish trading and streaming connections.
        """
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
        self.stream = StockDataStream(self.api_key, self.api_secret)
        self.logger.info("✅ Connected to Alpaca.")

    async def start_stream(self):
        """
        Run the websocket stream (blocks).
        """
        if self.stream:
            self.logger.info("🔄 Starting Alpaca stream...")
            await self.stream.run()

    def subscribe_bars(self, callback, symbol: str):
        """
        Subscribe to real-time bars for a symbol.

        Args:
            callback: Async function with (bar) -> None
            symbol (str): Symbol to subscribe to
        """
        if self.stream:
            self.logger.info(f"📡 Subscribed to {symbol} bars")
            self.stream.subscribe_bars(callback, symbol)
        else:
            self.logger.warning("⚠️ Stream not initialized. Call `connect()` first.")

    def execute_trade(self, trade: dict) -> bool:
        """
        Execute a trade using market order.

        Args:
            trade (dict): Dict with keys: symbol, qty, side

        Returns:
            bool: Whether the order succeeded
        """
        return self.submit_market_order(
            symbol=trade["symbol"],
            qty=trade["qty"],
            side=trade["side"]
        )

    def submit_market_order(self, symbol: str, qty: int, side: str) -> bool:
        """
        Submit a market order through Alpaca.

        Args:
            symbol (str): Symbol to trade
            qty (int): Quantity
            side (str): "buy" or "sell"

        Returns:
            bool: True if order submission was successful
        """
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=_map_side(side),
                time_in_force=TimeInForce.DAY
            )
            self.trading_client.submit_order(order)
            self.logger.info(f"🟢 Submitted {side.upper()} order for {qty} {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"🔴 Order failed for {symbol}: {e}")
            return False

    def get_account_info(self):
        try:
            return self.trading_client.get_account()
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            return None

    def get_open_position(self, symbol: str):
        try:
            return self.trading_client.get_open_position(symbol)
        except Exception as e:
            self.logger.warning(f"No open position for {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> bool:
        try:
            self.trading_client.close_position(symbol)
            self.logger.info(f"🔻 Closed position for {symbol}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to close position for {symbol}: {e}")
            return False
