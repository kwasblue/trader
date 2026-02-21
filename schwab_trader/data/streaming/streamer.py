import json
import requests
import websockets
from data.streaming.authenticator import Authenticator
from loggers.logger import Logger
from utils.configloader import ConfigLoader
from data.streaming.schwab_client import SchwabClient
from core.events.eventhandler import EventHandler, get_event_handler
from core.contracts.events import EVENT_PRICE_UPDATE
import asyncio
from typing import Callable, Dict, Optional, Any
from datetime import datetime, timezone


class SchwabStreamingClient():
    """
    WebSocket client for Schwab real-time market data streaming.

    Features:
    - Level 1 equity quotes
    - Automatic reconnection on disconnect
    - Quote callback system for subscribers
    - Event bus integration for price updates
    """

    def __init__(self, apikey, secretkey):
        self.authenticator = Authenticator()
        self.config = ConfigLoader().load_config()
        self.apikey = apikey
        self.secretkey = secretkey
        self.streamer_info = None
        self.connection = None
        self.streaming_logger = Logger('app.log', 'SchwabStreamingClient', log_dir=f'{self.config["folders"]["logs"]}').get_logger()
        self.price_dict = {}
        self.client = SchwabClient(apikey=self.apikey, secretkey=self.secretkey)

        # Event handler for publishing quotes
        self._event_handler = get_event_handler()

        # Quote callback - set by SchwabBroker
        self._quote_callback: Optional[Callable] = None

        # Reconnection settings
        self._reconnect_enabled = True
        self._reconnect_delay = 5  # seconds
        self._max_reconnect_attempts = 10
        self._reconnect_attempts = 0

        # Running state
        self._running = False
        self._symbols: list = []

    def set_quote_callback(self, callback: Callable):
        """
        Set the callback function for quote updates.

        The callback will be invoked with (symbol, quote_dict) when quotes arrive.

        Args:
            callback: Async function to call with quote data
        """
        self._quote_callback = callback

    async def _dispatch_quote(self, symbol: str, quote: Dict):
        """
        Dispatch a quote to the registered callback and event bus.

        Args:
            symbol: The symbol for this quote
            quote: Quote data dictionary
        """
        # Call registered callback
        if self._quote_callback:
            try:
                if asyncio.iscoroutinefunction(self._quote_callback):
                    await self._quote_callback(symbol, quote)
                else:
                    self._quote_callback(symbol, quote)
            except Exception as e:
                self.streaming_logger.exception(f"Error in quote callback for {symbol}: {e}")

        # Emit to event bus
        try:
            await self._event_handler.emit(EVENT_PRICE_UPDATE, {
                "symbol": symbol,
                "price": quote.get("last_price", 0),
                "bid": quote.get("bid_price", 0),
                "ask": quote.get("ask_price", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            self.streaming_logger.warning(f"Failed to emit price update: {e}")

    async def websocket_client(self, symbols):
        """
        Main WebSocket client loop for receiving streaming quotes.

        Args:
            symbols: List of symbols to subscribe to
        """
        self._symbols = symbols
        self._running = True

        try:
            self.streamer_info = self.client.user_preferences()['streamerInfo'][0]
            self.streaming_logger.info("Retrieved user preferences successfully")
        except Exception as e:
            self.streaming_logger.error(f"Failed to retrieve user preferences: {e}")
            return

        login_request = {
            'service': 'ADMIN',
            'requestid': 0,
            'command': 'LOGIN',
            'SchwabClientCustomerId': self.streamer_info['schwabClientCustomerId'],
            'SchwabClientCorrelId': self.streamer_info['schwabClientCorrelId'],
            'parameters': {
                'Authorization': self.authenticator.access_token(),
                'SchwabClientChannel': self.streamer_info['schwabClientChannel'],
                'SchwabClientFunctionId': self.streamer_info['schwabClientFunctionId'],
            }
        }

        symbol_request = {
            'service': 'LEVELONE_EQUITIES',
            'requestid': 1,
            'command': 'SUBS',
            'SchwabClientCustomerId': self.streamer_info['schwabClientCustomerId'],
            'SchwabClientCorrelId': self.streamer_info['schwabClientCorrelId'],
            'parameters': {
                'keys': ','.join(symbols),
                'fields': ','.join(str(field) for field in range(0, 42))
            }
        }

        try:
            async with websockets.connect(
                self.streamer_info['streamerSocketUrl'],
                ping_interval=30,
                ping_timeout=10,
            ) as ws:
                self.connection = ws

                # Send login request
                await ws.send(json.dumps(login_request))

                # Wait for login response
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=30)
                    self.streaming_logger.info("Login response received")
                    self.streaming_logger.debug(message)

                    # Check login success
                    login_resp = json.loads(message)
                    if login_resp.get('response', [{}])[0].get('content', {}).get('code') != 0:
                        self.streaming_logger.error(f"Login failed: {login_resp}")
                        return
                except asyncio.TimeoutError:
                    self.streaming_logger.error("Login timeout")
                    return

                # Send symbol subscription request
                await ws.send(json.dumps(symbol_request))
                self.streaming_logger.info(f"Subscribed to symbols: {symbols}")

                # Reset reconnect counter on successful connection
                self._reconnect_attempts = 0

                # Main message loop
                while self._running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=60)
                        self.streaming_logger.debug(f"Received message: {message[:200]}...")

                        # Parse message
                        try:
                            parsed_message = json.loads(message)
                        except json.JSONDecodeError:
                            self.streaming_logger.error("Failed to decode message as JSON")
                            continue

                        # Process data messages
                        data_list = parsed_message.get('data', [])
                        for data in data_list:
                            content = data.get('content', [])
                            for item in content:
                                symbol = item.get('key')
                                if not symbol:
                                    continue

                                # Extract quote fields
                                # Field mapping: 1=last, 2=bid, 3=ask, 4=bid_size, 5=ask_size,
                                # 8=volume, 29=close, 30=open, 31=high, 32=low
                                quote = {
                                    'last_price': item.get('1'),
                                    'bid_price': item.get('2'),
                                    'ask_price': item.get('3'),
                                    'bid_size': item.get('4'),
                                    'ask_size': item.get('5'),
                                    'volume': item.get('8'),
                                    'close_price': item.get('29'),
                                    'open_price': item.get('30'),
                                    'high_price': item.get('31'),
                                    'low_price': item.get('32'),
                                }

                                # Update local price dict
                                self.price_dict[symbol] = quote

                                # Dispatch to callback
                                await self._dispatch_quote(symbol, quote)

                    except asyncio.TimeoutError:
                        # Send heartbeat/ping to keep connection alive
                        self.streaming_logger.debug("Heartbeat timeout, connection still active")
                        continue

                    except websockets.ConnectionClosed as e:
                        self.streaming_logger.warning(f"Connection closed: {e}")
                        break

        except Exception as e:
            self.streaming_logger.exception(f"WebSocket error: {e}")

        finally:
            self.connection = None

    async def _reconnect_loop(self, symbols):
        """
        Reconnection loop with exponential backoff.

        Args:
            symbols: Symbols to resubscribe to after reconnection
        """
        while self._running and self._reconnect_enabled:
            if self._reconnect_attempts >= self._max_reconnect_attempts:
                self.streaming_logger.error(
                    f"Max reconnection attempts ({self._max_reconnect_attempts}) reached"
                )
                break

            self._reconnect_attempts += 1
            delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))
            delay = min(delay, 300)  # Cap at 5 minutes

            self.streaming_logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} "
                f"in {delay} seconds"
            )

            await asyncio.sleep(delay)

            try:
                await self.websocket_client(symbols)
                # If we get here without error, connection was successful
                if not self._running:
                    break
            except Exception as e:
                self.streaming_logger.exception(f"Reconnection failed: {e}")
                continue

    async def run(self, symbols):
        """
        Run the streaming client with automatic reconnection.

        Args:
            symbols: List of symbols to stream
        """
        self._running = True
        self._symbols = symbols

        while self._running:
            try:
                await self.websocket_client(symbols)
            except Exception as e:
                self.streaming_logger.exception(f"Stream error: {e}")

            # If we exited and still running, attempt reconnection
            if self._running and self._reconnect_enabled:
                await self._reconnect_loop(symbols)
            else:
                break

        self.streaming_logger.info("Streaming client stopped")

    async def stop(self):
        """Stop the streaming client."""
        self._running = False
        if self.connection:
            try:
                await self.connection.close()
            except Exception:
                pass
        self.streaming_logger.info("Stop requested")

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get the latest cached quote for a symbol.

        Args:
            symbol: The symbol to get quote for

        Returns:
            Quote dictionary or None if not available
        """
        return self.price_dict.get(symbol)

