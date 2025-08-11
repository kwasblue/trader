import requests
from data.streaming.authenticator import Authenticator
from schwab_trader.logger.logger import Logger
from utils.configloader import ConfigLoader


class SchwabClient:
    """
    Client for interacting with the Schwab API.
    Handles authentication, request throttling, and logging.
    """

    def __init__(self, apikey: str, secretkey: str):
        """
        Initializes the SchwabClient with API credentials.

        Args:
            apikey (str): API key for authentication.
            secretkey (str): Secret key for authentication.
        """
        self.authenticator = Authenticator()
        self.config = ConfigLoader().load_config()
        self.apikey = apikey
        self.secretkey = secretkey
        self.rate_lim = 0.5  # Hard rate limit in seconds
        self.logger = Logger(
            log_file='app.log',
            logger_name='SchwabClient',
            log_dir=self.config['folders']['logs']
        ).get_logger()

    def _throttle_requests(self):
        """
        Implements a rate limit to avoid overwhelming the API.
        """
        import time
        time.sleep(self.rate_lim)

    def _request(self, method: str, endpoint: str, headers: dict, params: dict = None, data: dict = None) -> dict:
        """
        Sends a request to the Schwab API.

        Args:
            method (str): HTTP method (GET, POST, etc.).
            endpoint (str): API endpoint URL.
            headers (dict): HTTP headers.
            params (dict, optional): Query parameters. Defaults to None.
            data (dict, optional): Request body. Defaults to None.

        Returns:
            dict: API response as a dictionary.
        """
        try:
            self.logger.info(f'Sending {method.upper()} request to {endpoint}')
            response = requests.request(
                method=method,
                url=endpoint,
                headers=headers,
                params=params,
                json=data
            )
            self._throttle_requests()
            if response.status_code == 200:
                self.logger.info(f'Successful {method.upper()} request to {endpoint}')
                return response.json()
            else:
                self.logger.error(f'{method.upper()} request to {endpoint} failed with status {response.status_code}')
                return {"error": f"HTTP {response.status_code}"}
        except requests.RequestException as e:
            self.logger.error(f'Error in {method.upper()} request to {endpoint}: {str(e)}')
            return {"error": str(e)}

    def _get(self, endpoint: str, headers: dict, params: dict) -> dict:
        """Helper method for making GET requests."""
        return self._request("GET", endpoint, headers, params)

    def _post(self, endpoint: str, headers: dict, data: dict) -> dict:
        """Helper method for making POST requests."""
        return self._request("POST", endpoint, headers, data=data)

    def _set_headers_params(self, endpoint: str, **kwargs) -> tuple:
        """
        Sets headers and parameters for API requests.
        
        Args:
            endpoint (str): API endpoint.
            **kwargs: Additional parameters for the request.
        
        Returns:
            tuple: (endpoint, headers, params)
        """
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.authenticator.access_token()}'
        }

        params = {}
        params.update(kwargs)
        self.logger.info(f'Setting headers and params for {endpoint} with {params}')
        return endpoint, headers, params

    def account_number(self) -> dict:
        """
        Retrieves the account number associated with the API user.

        Returns:
            dict: API response containing the account number.
        """
        endpoint = f"{self.config['api']['base_url']}/accounts/accountNumbers"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        return self._get(endpoint, headers, {})

    def accounts(self) -> dict:
        """
        Retrieves all accounts linked to the user.
        
        Returns:
            dict: API response containing account details.
        """
        endpoint = f"{self.config['api']['base_url']}/accounts"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        self.logger.info('Retrieving all accounts information')
        return self._get(endpoint, headers, {})

    def accounts_number(self, account_number: str) -> dict:
        """
        Retrieves details of a specific account.
        
        Args:
            account_number (str): The account number to retrieve details for.
        
        Returns:
            dict: API response containing account details.
        """
        endpoint = f"{self.config['api']['base_url']}/accounts/{account_number}"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        self.logger.info(f'Retrieving information for account number {account_number}')
        return self._get(endpoint, headers, {})

    def all_orders(self, account_number: str) -> dict:
        """
        Retrieves all orders associated with a specific account.
        
        Args:
            account_number (str): The account number to retrieve orders for.
        
        Returns:
            dict: API response containing order details.
        """
        endpoint = f"{self.config['api']['base_url']}/accounts/{account_number}/orders"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        self.logger.info(f'Retrieving all orders for account number {account_number}')
        return self._get(endpoint, headers, {})

    def generate_order(self, orderType: str, session: str, duration: str, orderStrategyType: str, instruction: str, quantity: int, symbol: str, assetType: str) -> dict:
        """
        Generates an order with user-defined parameters.
        """
        return {
            "orderType": orderType,
            "session": session,
            "duration": duration,
            "orderStrategyType": orderStrategyType,
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": assetType
                    }
                }
            ]
        }

    def place_orders(self, account_number: str, order_data: dict) -> dict:
        """Places an order for a specified account."""
        endpoint = f"{self.config['api']['base_url']}/accounts/{account_number}/orders"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        self.logger.info(f'Placing order for account number {account_number} with data {order_data}')
        return self._post(endpoint, headers, order_data)

    def quote(self, symbol: str) -> dict:
        endpoint = f"{self.config['api']['market_url']}/quotes"
        endpoint, headers, params = self._set_headers_params(endpoint=endpoint, symbol=symbol)
        self.logger.info(f'Retrieving quote for {symbol}')
        return self._get(endpoint, headers, params)

    def daily_price_history(self, symbol: str, start: int = '', end: int = '') -> dict:
        endpoint = f"{self.config['api']['market_url']}/pricehistory"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        endpoint, header, params = self._set_headers_params(
            endpoint=endpoint,
            periodType='year',
            frequencyType='daily',
            frequency=1,
            period=20,
            start=start,
            end=end,
            symbol=symbol
        )
        self.logger.info(f'Retrieving daily price history for {symbol}')
        return self._get(endpoint, headers, params)
    
    def custom_price_history(self, symbol: str, start: int = '', end: int = '',
                             periodType: int = '', frequencyType: str = '',
                             period: int = '', frequency: int = '') -> dict:
        endpoint = f"{self.config['api']['market_url']}/pricehistory"
        endpoint, headers, params = self._set_headers_params(
            endpoint=endpoint,
            periodType=periodType,
            frequencyType=frequencyType,
            frequency=frequency,
            period=period,
            start=start,
            end=end,
            symbol=symbol
        )
        self.logger.info(f'Retrieving custom price history for {symbol}')
        return self._get(endpoint, headers, params)

    def get_order_by_id(self, account_number: str, order_id: str) -> dict:
        endpoint = f"{self.config['api']['base_url']}/accounts/{account_number}/orders/{order_id}"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        return self._get(endpoint, headers, {})

    def cancel_order(self, account_number: str, order_id: str) -> dict:
        endpoint = f"{self.config['api']['base_url']}/accounts/{account_number}/orders/{order_id}"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        return self._request("DELETE", endpoint, headers)

    def replace_order(self, account_number: str, order_id: str, new_order_data: dict) -> dict:
        endpoint = f"{self.config['api']['base_url']}/accounts/{account_number}/orders/{order_id}"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        return self._request("PUT", endpoint, headers, data=new_order_data)

    def all_orders_all_accounts(self) -> dict:
        endpoint = f"{self.config['api']['base_url']}/orders"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        return self._get(endpoint, headers, {})

    def transactions(self, account_number: str) -> dict:
        endpoint = f"{self.config['api']['base_url']}/accounts/{account_number}/transactions"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        return self._get(endpoint, headers, {})

    def transaction_by_id(self, account_number: str, transaction_id: str) -> dict:
        endpoint = f"{self.config['api']['base_url']}/accounts/{account_number}/transactions/{transaction_id}"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        return self._get(endpoint, headers, {})

    def user_preferences(self) -> dict:
        endpoint = f"{self.config['api']['base_url']}/userPreference"
        headers = {'Authorization': f'Bearer {self.authenticator.access_token()}'}
        return self._get(endpoint, headers, {})
    
    def close_all_positions(self, account_number: str) -> dict:
        """
        Closes all open positions in the specified account.

        Args:
            account_number (str): The Schwab account number.

        Returns:
            dict: A dictionary summarizing results of close operations.
        """
        positions = self.accounts_number(account_number).get("securitiesAccount", {}).get("positions", [])
        if not positions:
            self.logger.info(f"No open positions found for account {account_number}.")
            return {"status": "No positions to close"}

        results = {}
        for pos in positions:
            symbol = pos.get("instrument", {}).get("symbol")
            quantity = int(float(pos.get("longQuantity", 0) or pos.get("shortQuantity", 0)))
            asset_type = pos.get("instrument", {}).get("assetType", "EQUITY")

            if quantity == 0:
                continue

            if pos.get("longQuantity", 0) > 0:
                instruction = "SELL"
            elif pos.get("shortQuantity", 0) > 0:
                instruction = "BUY_TO_COVER"
            else:
                self.logger.warning(f"Unable to determine position type for {symbol}")
                continue

            order_data = self.generate_order(
                orderType="MARKET",
                session="NORMAL",
                duration="DAY",
                orderStrategyType="SINGLE",
                instruction=instruction,
                quantity=quantity,
                symbol=symbol,
                assetType=asset_type
            )

            self.logger.info(f"Closing position for {symbol}: {instruction} {quantity}")
            response = self.place_orders(account_number, order_data)
            results[symbol] = response

        return results