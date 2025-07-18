import time
import logging
from data.streaming.schwab_client import SchwabClient

class Executor:
    """
    A comprehensive executor for sending, managing, and validating live trades through the Schwab API.
    """

    def __init__(self, client: SchwabClient, dry_run: bool = False):
        self.client = client
        self.account_number = self._get_default_account()
        self.dry_run = dry_run
        self.logger = logging.getLogger("Executor")

    def _get_default_account(self):
        accounts = self.client.account_number()
        return accounts.get("accountNumbers", [{}])[0].get("accountNumber")

    def buy(self, symbol: str, qty: int, order_type: str = "MARKET", **kwargs):
        return self._place_order("BUY", symbol, qty, order_type, **kwargs)

    def sell(self, symbol: str, qty: int, order_type: str = "MARKET", **kwargs):
        return self._place_order("SELL", symbol, qty, order_type, **kwargs)

    def _place_order(self, instruction, symbol, qty, order_type, **kwargs):
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

    def cancel_order(self, order_id: str):
        endpoint = f"{self.client.config['api']['base_url']}/accounts/{self.account_number}/orders/{order_id}"
        headers = {"Authorization": f"Bearer {self.client.authenticator.access_token()}"}
        return self.client._request("DELETE", endpoint, headers)

    def get_order_status(self, order_id: str):
        endpoint = f"{self.client.config['api']['base_url']}/accounts/{self.account_number}/orders/{order_id}"
        headers = {"Authorization": f"Bearer {self.client.authenticator.access_token()}"}
        return self.client._request("GET", endpoint, headers)

    def get_open_orders(self):
        return self.client.all_orders(self.account_number)

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

    def has_sufficient_funds(self, symbol: str, qty: int) -> bool:
        quote = self.client.quote(symbol)
        price = quote.get(symbol, {}).get("lastPrice")
        if price is None:
            self.logger.warning(f"No quote available for {symbol}")
            return False

        required_cash = price * qty
        account_info = self.client.accounts_number(self.account_number)
        cash_available = account_info.get("securitiesAccount", {}).get("currentBalances", {}).get("availableFunds")

        return cash_available is not None and cash_available >= required_cash

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
