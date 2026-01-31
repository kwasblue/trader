import time
import threading
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from data.streaming.authenticator import Authenticator
from loggers.logger import Logger
from utils.configloader import ConfigLoader

# Optimization constants
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.3
RATE_LIMIT_REQUESTS = 2  # requests per second
POOL_CONNECTIONS = 10
POOL_MAXSIZE = 20


class TokenBucketRateLimiter:
    """Non-blocking token bucket rate limiter."""

    def __init__(self, rate: float = 2.0, capacity: float = 5.0):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 5.0) -> bool:
        """Acquire a token, waiting if necessary. Returns False if timeout."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                # Refill tokens based on elapsed time
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True

            # Wait a bit before retrying
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)


class SchwabClient:
    """
    Optimized client for interacting with the Schwab API.

    Optimizations:
    - Connection pooling via requests.Session
    - Non-blocking token bucket rate limiter
    - Automatic retry with exponential backoff
    - Request timeouts to prevent hanging
    """

    def __init__(self, apikey: str, secretkey: str):
        self.authenticator = Authenticator()
        self.config = ConfigLoader().load_config()
        self.apikey = apikey
        self.secretkey = secretkey
        self.logger = Logger(
            log_file='app.log',
            logger_name='SchwabClient',
            log_dir=self.config['folders']['logs']
        ).get_logger()

        # Optimization: Connection pooling with session
        self._session = requests.Session()
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(
            pool_connections=POOL_CONNECTIONS,
            pool_maxsize=POOL_MAXSIZE,
            max_retries=retry_strategy
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # Optimization: Non-blocking rate limiter
        self._rate_limiter = TokenBucketRateLimiter(
            rate=RATE_LIMIT_REQUESTS,
            capacity=5.0
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        headers: dict,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        timeout: float = REQUEST_TIMEOUT
    ) -> dict:
        """
        Sends a request with connection pooling, rate limiting, and timeout.
        """
        # Rate limit (non-blocking with timeout)
        if not self._rate_limiter.acquire(timeout=5.0):
            self.logger.warning(f"Rate limit timeout for {endpoint}")
            return {"error": "Rate limit timeout"}

        try:
            self.logger.debug(f'Sending {method.upper()} request to {endpoint}')
            response = self._session.request(
                method=method,
                url=endpoint,
                headers=headers,
                params=params,
                json=data,
                timeout=timeout
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 201:
                return {"status": "created", "headers": dict(response.headers)}
            else:
                self.logger.error(
                    f'{method.upper()} {endpoint} failed: HTTP {response.status_code}'
                )
                return {"error": f"HTTP {response.status_code}", "body": response.text[:200]}

        except requests.Timeout:
            self.logger.error(f'Timeout on {method.upper()} {endpoint}')
            return {"error": "Request timeout"}
        except requests.RequestException as e:
            self.logger.error(f'Request error on {endpoint}: {e}')
            return {"error": str(e)}

    def close(self):
        """Close the session and release connections."""
        self._session.close()

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