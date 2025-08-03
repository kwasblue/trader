import time
import pandas as pd
from datetime import datetime, UTC
from collections import defaultdict
from utils.logger import Logger
from core.position_sizer import DynamicPositionSizer
from data.streaming.schwab_client import SchwabClient


class Executor:
    """
    Live trading Executor for placing and managing orders via Schwab's API.

    - Supports MARKET, LIMIT, STOP, and OCO orders.
    - Handles live position tracking and funds checks.
    - Can operate in dry_run mode for simulation or testing without real orders.
    """

    def __init__(self, client: SchwabClient, sizer:DynamicPositionSizer, dry_run: bool = False, risk_percentage: int = 0.7):
        self.client = client
        self.account_number = self._get_default_account()
        self.logger = Logger('app.log', 'LiveExecutor', log_dir=self.client.config['folders']['logs']).get_logger()
        self.sizer = sizer
        self.risk_percentage = risk_percentage

        # Track open positions and portfolio state locally
        self.position = defaultdict(int)

    def _get_default_account(self) -> str:
        """
        Get the primary account number from the Schwab client.
        """
        accounts = self.client.account_number()
        return accounts.get("accountNumbers", [{}])[0].get("accountNumber")

    def execute(self, symbol: str, df: pd.DataFrame, signal: int, price: float, atr_value: float):
        """
        Main execution method called by the strategy callback.
        Handles buy/sell decisions, position sizing, funds check, and logging.

        Parameters:
        - symbol: Ticker symbol to trade
        - df: Historical + live DataFrame (with ATR and signals)
        - signal: +1 = Buy, -1 = Sell, 0 = Hold
        - price: Current market price
        - atr_value: Latest ATR for volatility-based SL sizing
        """
        if pd.isna(atr_value) or atr_value <= 0:
            return

        # Volatility classification
        atr_25 = df['ATR'].quantile(0.25)
        atr_75 = df['ATR'].quantile(0.75)
        market_conditions = (
            "low_volatility" if atr_value < atr_25 else
            "high_volatility" if atr_value > atr_75 else
            "normal"
        )

        stop_loss_price = price - (atr_value * 2)

        # Determine ideal quantity via position sizer
        raw_qty = self.sizer.calculate_position_size(
            price=price,
            stop_loss_price=stop_loss_price,
            current_cash=self.cash[symbol],
            market_conditions=market_conditions,
            signal=signal
        )

        qty = int(raw_qty)
        # Auto-scale down if not enough funds
        while qty > 0 and not self._has_sufficient_funds(symbol, qty):
            qty -= 1

        if qty <= 0:
            self.logger.warning(f"No affordable position size for {symbol} at ${price:.2f}")
            return

        now = datetime.now(UTC)

        # Execute Buy
        if signal == 1 and self.position[symbol] == 0:
            response = self.buy(symbol, qty)
            self.log_order_response(response)
            self.logger.info(f"[LIVE BUY] {symbol}: {qty} @ {price:.2f} | SL: {stop_loss_price:.2f}")
            self.position[symbol] += qty

        # Execute Sell
        elif signal == -1 and self.position[symbol] > 0:
            response = self.sell(symbol, self.position[symbol])
            self.log_order_response(response)
            self.logger.info(f"[LIVE SELL] {symbol}: {self.position[symbol]} @ {price:.2f}")
            self.position[symbol] = 0
        
        # hold signal
        elif signal == 0: 
            self.logger.info(f'[HOLD]. Will not buy or sell until the next signal ')

    # --- ORDER WRAPPERS ---

    def buy(self, symbol: str, qty: int, order_type: str = "MARKET", **kwargs):
        return self._place_order("BUY", symbol, qty, order_type, **kwargs)

    def sell(self, symbol: str, qty: int, order_type: str = "MARKET", **kwargs):
        return self._place_order("SELL", symbol, qty, order_type, **kwargs)

    def _place_order(self, instruction: str, symbol: str, qty: int, order_type: str, **kwargs):
        """
        Core method to place orders via Schwab's API or simulate if dry_run is enabled.
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] {instruction} {qty} {symbol} as {order_type}")
            return {"status": "dry_run"}

        order_data = self.client.generate_order(
            orderType=order_type,
            session=kwargs.get("session", "NORMAL"),
            duration=kwargs.get("duration", "DAY"),
            orderStrategyType=kwargs.get("orderStrategyType", "SINGLE"),
            instruction=instruction,
            quantity=qty,
            symbol=symbol,
            assetType=kwargs.get("assetType", "EQUITY")
        )

        return self.client.place_orders(self.account_number, order_data)

    # --- OCO SUPPORT ---

    def place_oco_order(self, symbol: str, qty: int, stop_price: float, limit_price: float):
        if self.dry_run:
            self.logger.info(f"[DRY RUN] OCO for {symbol} - {qty} shares")
            return {"status": "dry_run"}

        order_data = {
            "orderStrategyType": "OCO",
            "childOrderStrategies": [
                self.client.generate_order(
                    orderType="LIMIT",
                    session="NORMAL",
                    duration="DAY",
                    orderStrategyType="SINGLE",
                    instruction="SELL",
                    quantity=qty,
                    symbol=symbol,
                    assetType="EQUITY",
                ),
                self.client.generate_order(
                    orderType="STOP",
                    session="NORMAL",
                    duration="DAY",
                    orderStrategyType="SINGLE",
                    instruction="SELL",
                    quantity=qty,
                    symbol=symbol,
                    assetType="EQUITY",
                )
            ]
        }

        order_data["childOrderStrategies"][0]["price"] = str(limit_price)
        order_data["childOrderStrategies"][1]["stopPrice"] = str(stop_price)

        return self.client.place_orders(self.account_number, order_data)

    # --- ACCOUNT UTILITIES ---

    def _has_sufficient_funds(self, symbol: str, qty: int) -> bool:
        """
        Check if you have enough buying power to execute a trade of `qty` shares.
        """
        quote = self.client.quote(symbol)
        price = quote.get(symbol, {}).get("lastPrice")
        if price is None:
            self.logger.warning(f"No quote available for {symbol}")
            return False

        required_cash = price * qty
        account_info = self.client.accounts_number(self.account_number)
        cash_available = account_info.get("securitiesAccount", {}).get("currentBalances", {}).get("availableFunds")

        return cash_available is not None and cash_available >= required_cash

    def get_open_orders(self):
        return self.client.all_orders(self.account_number)

    def cancel_order(self, order_id: str):
        endpoint = f"{self.client.config['api']['base_url']}/accounts/{self.account_number}/orders/{order_id}"
        headers = {"Authorization": f"Bearer {self.client.authenticator.access_token()}"}
        return self.client._request("DELETE", endpoint, headers)

    def get_order_status(self, order_id: str):
        endpoint = f"{self.client.config['api']['base_url']}/accounts/{self.account_number}/orders/{order_id}"
        headers = {"Authorization": f"Bearer {self.client.authenticator.access_token()}"}
        return self.client._request("GET", endpoint, headers)

    def log_order_response(self, response: dict):
        if "error" in response:
            self.logger.error(f"Order error: {response['error']}")
        else:
            self.logger.info(f"Order response: {response}")

    def retry_failed_order(self, *args, max_retries=3, delay=2, **kwargs):
        for attempt in range(max_retries):
            result = self._place_order(*args, **kwargs)
            if "error" not in result:
                return result
            self.logger.warning(f"Retrying order (attempt {attempt + 1})")
            time.sleep(delay)
        self.logger.error("Order failed after retries")
        return {"error": "Retries exceeded"}
