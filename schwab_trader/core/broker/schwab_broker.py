from core.base.base_broker_interface import BaseBrokerInterface
from data.streaming.schwab_client import SchwabClient

class SchwabBroker(BaseBrokerInterface):
    def __init__(self, client: SchwabClient):
        self.client = client
        self.account_number = self.get_default_account()

    def get_default_account(self) -> str:
        accounts = self.client.account_number()
        return accounts.get("accountNumbers", [{}])[0].get("accountNumber")

    def place_market_order(self, symbol: str, qty: int, side: str) -> dict:
        order_data = self.client.generate_order(
            orderType="MARKET",
            session="NORMAL",
            duration="DAY",
            orderStrategyType="SINGLE",
            instruction=side,
            quantity=qty,
            symbol=symbol,
            assetType="EQUITY"
        )
        return self.client.place_orders(self.account_number, order_data)

    def place_oco_order(self, symbol: str, qty: int, stop_price: float, limit_price: float) -> dict:
        order_data = {
            "orderStrategyType": "OCO",
            "childOrderStrategies": [
                self.client.generate_order(
                    orderType="LIMIT",
                    instruction="SELL",
                    quantity=qty,
                    symbol=symbol,
                    price=str(limit_price),
                ),
                self.client.generate_order(
                    orderType="STOP",
                    instruction="SELL",
                    quantity=qty,
                    symbol=symbol,
                    stopPrice=str(stop_price),
                )
            ]
        }
        return self.client.place_orders(self.account_number, order_data)

    def get_quote(self, symbol: str) -> float:
        quote = self.client.quote(symbol)
        return quote.get(symbol, {}).get("lastPrice", None)

    def get_available_funds(self) -> float:
        account_info = self.client.accounts_number(self.account_number)
        return account_info.get("securitiesAccount", {}).get("currentBalances", {}).get("availableFunds", 0.0)

    def get_open_orders(self) -> list:
        return self.client.all_orders(self.account_number)

    def cancel_order(self, order_id: str) -> dict:
        return self.client.cancel_order(self.account_number, order_id)

    def get_order_status(self, order_id: str) -> dict:
        return self.client.order_status(self.account_number, order_id)
