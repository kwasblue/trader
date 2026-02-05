from __future__ import annotations
import asyncio
import inspect
from typing import Optional, List, Dict, Any

import alpaca
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed
from datetime import datetime, timezone

from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderClass

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

from core.base.base_broker_interface import BaseBrokerInterface
from core.app_types import OrderResult, PositionView, BrokerSnapshot
from loggers.logger import Logger
from core.events.eventhandler import EventHandler, get_event_handler
from core.events.events import (
    EVENT_ORDER_STATUS, OrderStatusPayload,
    EVENT_NEW_TRADE, TradePayload,
    EVENT_POSITION_UPDATE, PositionPayload,
    EVENT_PNL_UPDATE, PnLPayload,
    EVENT_HEALTH_UPDATE, HealthPayload,
    EVENT_ALERT, AlertPayload
)
from core.utils.retry import retry, async_retry, get_circuit_breaker, CircuitOpenError
from core.utils.health import get_health_checker


def _map_side(side: str) -> OrderSide:
    return OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL


def _map_tif(tif: str) -> TimeInForce:
    m = {
        "day": TimeInForce.DAY,
        "gtc": TimeInForce.GTC,
        "opg": TimeInForce.OPG,
        "cls": TimeInForce.CLS,
        "ioc": TimeInForce.IOC,
        "fok": TimeInForce.FOK,
    }
    return m.get(tif.lower(), TimeInForce.GTC)


class AlpacaBroker(BaseBrokerInterface):
    """
    Broker implementation for Alpaca using the Alpaca SDK (TradingClient + Data Stream).

    Minimal surface to match your BaseBrokerInterface plus the small helpers
    you originally used (connect/start_stream/subscribe_bars/execute_trade).
    """

    def __init__(self, api_key: str, api_secret: str, event_handler: EventHandler | None = None, paper: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.trading_client: Optional[TradingClient] = None
        self.stream: Optional[StockDataStream] = None
        self.data_rest: Optional[StockHistoricalDataClient] = None
        self._last_price: Dict[str, float] = {}
        self._connected = False
        self.logger = Logger("alpaca.log", "AlpacaBroker").get_logger()
        self.event_handler = get_event_handler()

    # ---------------------------------------------------------
    # Connections / Streaming
    # ---------------------------------------------------------
    @async_retry(max_attempts=3, base_delay=2.0, circuit_breaker="alpaca_connect")
    async def connect(self) -> None:
        """
        Establish trading and streaming connections with retry logic.

        Returns:
            None (connection status stored in self._connected)
        """
        # Close any existing connections first to avoid connection limit issues
        if self.stream is not None:
            self.logger.warning("Closing existing stream before reconnecting...")
            self.disconnect()

        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
        self.stream = StockDataStream(self.api_key, self.api_secret, feed=DataFeed.IEX)
        self.data_rest = StockHistoricalDataClient(self.api_key, self.api_secret)
        self._connected = True
        self.logger.info("Connected to Alpaca.")

    @retry(max_attempts=3, base_delay=2.0, circuit_breaker="alpaca_connect")
    def connect_sync(self) -> None:
        """
        Sync wrapper for connect(). For backward compatibility.

        Use `await connect()` in async code.
        """
        # Close any existing connections first to avoid connection limit issues
        if self.stream is not None:
            self.logger.warning("Closing existing stream before reconnecting...")
            self.disconnect()

        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
        self.stream = StockDataStream(self.api_key, self.api_secret, feed=DataFeed.IEX)
        self.data_rest = StockHistoricalDataClient(self.api_key, self.api_secret)
        self._connected = True
        self.logger.info("Connected to Alpaca.")

    def disconnect(self):
        """Close all connections and clean up resources."""
        self._connected = False
        if self.stream is not None:
            try:
                # Stop the websocket stream
                if hasattr(self.stream, 'stop'):
                    self.stream.stop()
                elif hasattr(self.stream, 'close'):
                    self.stream.close()
                self.logger.info("Stream disconnected.")
            except Exception as e:
                self.logger.warning(f"Error closing stream: {e}")
            finally:
                self.stream = None

        self.trading_client = None
        self.data_rest = None
        self.logger.info("Disconnected from Alpaca.")

        # Register health check
        health_checker = get_health_checker()
        health_checker.register(
            "alpaca_broker",
            self._health_check,
            metadata={"paper": self.paper}
        )

        if self.event_handler:
            payload: HealthPayload = {
                "broker": "alpaca",
                "status": "connected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            asyncio.create_task(self.event_handler.emit(EVENT_HEALTH_UPDATE, payload))

    async def _health_check(self) -> bool:
        """Health check for Alpaca connection."""
        try:
            if not self.trading_client:
                return False
            # Try to get account info as health check
            account = await asyncio.to_thread(self.trading_client.get_account)
            return account is not None
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False

    async def _heartbeat_loop(self, interval: int = 10):
        """Emit periodic broker heartbeat."""
        while True:
            if self.event_handler:
                payload: HealthPayload = {
                    "broker": "alpaca",
                    "status": "alive",  # ✅ must be one of the allowed literals
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_HEALTH_UPDATE, payload)
            await asyncio.sleep(interval)

    async def start_stream(self, retry_seconds: int = 300):
        if not self.stream:
            self.logger.warning("Stream not initialized. Call connect() first.")
            if self.event_handler:
                await self.event_handler.emit(EVENT_HEALTH_UPDATE, {
                    "broker": "alpaca",
                    "status": "not_initialized",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            return

        self.logger.info(
            f"[stream] starting... alpaca-py={getattr(alpaca,'__version__','unknown')} feed={getattr(self, 'feed', '?')}"
        )
        if self.event_handler:
            payload: HealthPayload = {
                "broker": "alpaca",
                "status": "stream_starting",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.event_handler.emit(EVENT_HEALTH_UPDATE, payload)
        # launch heartbeat task
        asyncio.create_task(self._heartbeat_loop(interval=15))

        while True:
            try:
                run_fn = self.stream.run
                if inspect.iscoroutinefunction(run_fn):
                    await run_fn()
                else:
                    await asyncio.to_thread(run_fn)

                self.logger.warning("[stream] exited cleanly; restarting in 5s...")
                if self.event_handler:
                    payload: HealthPayload = {
                        "broker": "alpaca",
                        "status": "stream_exited",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.event_handler.emit(EVENT_HEALTH_UPDATE, payload)

                await asyncio.sleep(5)

            except Exception as e:
                self.logger.error(f"[stream] error: {e}; retrying in {retry_seconds}s")
                if self.event_handler:
                    # alert
                    alert_payload: AlertPayload = {
                        "level": "error",
                        "message": f"Stream error: {e}",
                        "symbol": None,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.event_handler.emit(EVENT_ALERT, alert_payload)

                    # health
                    health_payload: HealthPayload = {
                        "broker": "alpaca",
                        "status": "reconnecting",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.event_handler.emit(EVENT_HEALTH_UPDATE, health_payload)

            await asyncio.sleep(retry_seconds)

    def subscribe_bars(self, callback, symbol: str):
        if not self.stream:
            self.logger.warning("Stream not initialized. Call connect() first.")
            return

        if not asyncio.iscoroutinefunction(callback):
            async def _async_wrap(bar):
                return callback(bar)
            cb = _async_wrap
        else:
            cb = callback

        self.stream.subscribe_bars(cb, symbol)
        self.logger.info(f"[stream] subscribed bars: {symbol}")

        if self.event_handler:
            asyncio.create_task(self.event_handler.emit(EVENT_HEALTH_UPDATE, {
                "broker": "alpaca",
                "status": "subscribed",
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }))

    # ---------------------------------------------------------
    # Convenience (from your original snippet)
    # ---------------------------------------------------------
    def execute_trade(self, trade: dict) -> bool:
        """Execute a trade using market order.

        trade = {"symbol": str, "qty": int, "side": "buy"|"sell"}
        """
        return self.submit_market_order(
            symbol=trade["symbol"],
            qty=trade["qty"],
            side=trade["side"],
        )

    @retry(max_attempts=3, base_delay=1.0, circuit_breaker="alpaca_orders")
    def submit_market_order(self, symbol: str, qty: int, side: str) -> bool:
        """
        Submit a market order through Alpaca with retry logic.

        Features:
        - Automatic retry on transient failures (3 attempts)
        - Circuit breaker to prevent cascading failures
        - Event emission for order status tracking
        """
        if not self.trading_client:
            self.logger.error("Trading client not initialized. Call `connect()` first.")
            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": "N/A",
                    "symbol": symbol,
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                asyncio.create_task(self.event_handler.emit(EVENT_ORDER_STATUS, payload))
            return False

        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=_map_side(side),
                time_in_force=TimeInForce.DAY,
            )
            res = self.trading_client.submit_order(order)

            self.logger.info(f"Submitted {side.upper()} order for {qty} {symbol}")

            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": getattr(res, "id", "N/A"),
                    "symbol": symbol,
                    "status": "submitted",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                asyncio.create_task(self.event_handler.emit(EVENT_ORDER_STATUS, payload))

            return True

        except Exception as e:
            self.logger.error(f"Order failed for {symbol}: {e}")
            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": "N/A",
                    "symbol": symbol,
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                asyncio.create_task(self.event_handler.emit(EVENT_ORDER_STATUS, payload))
            return False

    # ---------------------------------------------------------
    # BaseBrokerInterface implementation (async + sync)
    # ---------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "gtc",
        **kwargs,
    ) -> OrderResult:
        """
        Place an order asynchronously.

        Returns:
            OrderResult with order details and status
        """
        if not self.trading_client:
            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": "N/A",
                    "symbol": symbol,
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_ORDER_STATUS, payload)
            raise RuntimeError("Not connected: call connect() first")

        tif = _map_tif(time_in_force)
        s = _map_side(side)

        def _submit():
            ot = (order_type or "market").lower()
            if ot == "market":
                req = MarketOrderRequest(symbol=symbol, qty=qty, side=s, time_in_force=tif)
            elif ot == "limit":
                if limit_price is None:
                    raise ValueError("limit_price is required for limit orders")
                req = LimitOrderRequest(symbol=symbol, qty=qty, side=s, time_in_force=tif, limit_price=limit_price)
            else:
                raise ValueError(f"Unsupported order_type: {order_type}")
            o = self.trading_client.submit_order(req)
            return self._mk_order_result(o)

        try:
            res = await asyncio.to_thread(_submit)

            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": res.order_id or "N/A",
                    "symbol": res.symbol or symbol,
                    "status": res.status or "submitted",
                    "filled_qty": res.filled_qty or 0.0,
                    "avg_price": res.avg_fill_price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_ORDER_STATUS, payload)

            return res

        except Exception as e:
            self.logger.error(f"Order failed for {symbol}: {e}")
            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": "N/A",
                    "symbol": symbol,
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_ORDER_STATUS, payload)
            raise  # Re-raise so caller knows order failed


    async def cancel_order(self, order_id: str) -> None:  # no return; emits instead
        if not self.trading_client:
            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": order_id,
                    "symbol": "N/A",
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_ORDER_STATUS, payload)
            raise RuntimeError("Not connected: call connect() first")

        def _cancel():
            self.trading_client.cancel_order_by_id(order_id)
            try:
                o = self.trading_client.get_order_by_id(order_id)
                return self._mk_order_result(o)
            except Exception:
                return OrderResult(order_id=order_id, status="canceled")

        try:
            res = await asyncio.to_thread(_cancel)

            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": res.order_id or order_id,
                    "symbol": res.symbol or "N/A",
                    "status": res.status or "canceled",
                    "filled_qty": res.filled_qty or 0.0,
                    "avg_price": res.avg_fill_price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_ORDER_STATUS, payload)

        except Exception as e:
            self.logger.error(f"Cancel order failed for {order_id}: {e}")
            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": order_id,
                    "symbol": "N/A",
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_ORDER_STATUS, payload)
            return res

    async def place_market_order(self, symbol: str, qty: int, side, price: float | None = None) -> OrderResult:
        """
        Place a market order asynchronously.

        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            side: OrderSide enum or string ("buy"/"sell")
            price: Optional price hint (ignored for market orders)

        Returns:
            OrderResult with execution details
        """
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        # Handle both OrderSide enum and string
        if isinstance(side, str):
            alpaca_side = _map_side(side)
        else:
            # Assume it's an OrderSide enum from core.enums
            alpaca_side = _map_side(side.value if hasattr(side, 'value') else str(side))

        def _submit():
            return self.trading_client.submit_order(
                MarketOrderRequest(symbol=symbol, qty=qty, side=alpaca_side, time_in_force=TimeInForce.DAY)
            )

        o = await asyncio.to_thread(_submit)
        res = self._mk_order_result(o)

        # Emit event
        if self.event_handler:
            payload = {
                "order_id": res.order_id,
                "symbol": res.symbol,
                "status": res.status,
                "side": res.side,
                "qty": res.qty,
                "filled_qty": res.filled_qty,
                "avg_price": res.avg_fill_price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.event_handler.emit(EVENT_ORDER_STATUS, payload)

        return res


    async def place_oco_order(self, symbol: str, qty: int, stop_price: float, limit_price: float) -> OrderResult:
        """
        Place OCO (One-Cancels-Other) bracket order asynchronously.

        Uses Alpaca's bracket order_class with take_profit/stop_loss.

        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            stop_price: Stop-loss trigger price
            limit_price: Take-profit limit price

        Returns:
            OrderResult for the OCO order group
        """
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _submit():
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,  # typical for exiting a long
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,  # type: ignore[arg-type]
                take_profit={"limit_price": limit_price},
                stop_loss={"stop_price": stop_price},
            )
            return self.trading_client.submit_order(req)

        o = await asyncio.to_thread(_submit)
        res = self._mk_order_result(o)

        # Emit order status
        if self.event_handler:
            payload = {
                "order_id": res.order_id,
                "symbol": res.symbol,
                "status": res.status,
                "side": res.side,
                "qty": res.qty,
                "filled_qty": res.filled_qty,
                "avg_price": res.avg_fill_price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.event_handler.emit(EVENT_ORDER_STATUS, payload)

        return res


    async def get_position(self, symbol: str) -> Optional[PositionView]:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _get():
            try:
                p = self.trading_client.get_open_position(symbol)
                return self._mk_position_view(p)
            except Exception:
                return None

        return await asyncio.to_thread(_get)

    async def get_account_info(self) -> BrokerSnapshot:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _acct():
            a = self.trading_client.get_account()
            # Also fetch all positions
            positions_list = self.trading_client.get_all_positions()
            positions = {}
            for p in positions_list:
                symbol = getattr(p, "symbol", None)
                if symbol:
                    positions[symbol] = self._mk_position_view(p)
            return self._mk_broker_snapshot(a, positions)

        return await asyncio.to_thread(_acct)

    async def is_market_open(self) -> bool:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _clock():
            c = self.trading_client.get_clock()
            return bool(getattr(c, "is_open", False))

        return await asyncio.to_thread(_clock)

    def get_default_account(self) -> str:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")
        a = self.trading_client.get_account()
        return getattr(a, "account_number", getattr(a, "id", ""))

    def get_quote(self, symbol: str) -> float:
        if not self.data_rest:
            raise RuntimeError("Not connected: call connect() first")
        q = self.data_rest.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        quote = q[symbol]
        price = quote.ask_price or quote.bid_price or quote.midpoint
        if price is None:
            raise RuntimeError(f"No quote available for {symbol}")
        return float(price)

    def get_available_funds(self) -> float:
        """Get available funds (cash)."""
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")
        a = self.trading_client.get_account()
        return float(getattr(a, "cash", 0.0))

    def get_buying_power(self) -> float:
        """
        Get buying power from broker API.

        Returns:
            Available buying power in dollars (as reported by Alpaca)
        """
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")
        a = self.trading_client.get_account()
        bp = float(getattr(a, "buying_power", 0.0) or 0.0)
        self.logger.debug(f"Buying power from API: ${bp:,.2f}")
        return bp

    async def get_open_orders(self) -> List[OrderResult]:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _orders():
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, limit=500)
            orders = self.trading_client.get_orders(filter=req)
            return [self._mk_order_result(o) for o in orders]

        return await asyncio.to_thread(_orders)

    async def get_order_status(self, order_id: str) -> OrderResult:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _status():
            o = self.trading_client.get_order_by_id(order_id)
            return self._mk_order_result(o)

        return await asyncio.to_thread(_status)

    def mark_price(self, symbol: str, price: float) -> None:
        self._last_price[symbol] = float(price)

    # ---------------------------------------------------------
    # Minimal mappers to your domain types
    # ---------------------------------------------------------
    def _mk_order_result(self, o) -> OrderResult:
        try:
            return OrderResult(
                order_id=getattr(o, "id", None) or getattr(o, "client_order_id", None),
                client_order_id=getattr(o, "client_order_id", None),
                symbol=getattr(o, "symbol", None),
                side=str(getattr(o, "side", "")).lower(),
                qty=float(getattr(o, "qty", 0) or getattr(o, "quantity", 0) or 0),
                filled_qty=float(getattr(o, "filled_qty", 0) or 0),
                type=str(getattr(o, "type", "")).lower(),
                time_in_force=str(getattr(o, "time_in_force", "")).lower(),
                status=str(getattr(o, "status", "")).lower(),
                limit_price=_to_float(getattr(o, "limit_price", None)),
                stop_price=_to_float(getattr(o, "stop_price", None)),
                avg_fill_price=_to_float(getattr(o, "filled_avg_price", None)),
            )
        except Exception:
            # Fallback to a minimal dict-like if your dataclass signature differs
            return OrderResult(order_id=getattr(o, "id", None))

    def _mk_position_view(self, p) -> PositionView:
        current_price = _to_float(getattr(p, "current_price", None))
        return PositionView(
            symbol=getattr(p, "symbol", None),
            qty=float(getattr(p, "qty", 0) or 0),
            avg_entry_price=_to_float(getattr(p, "avg_entry_price", None)),
            market_price=current_price,
            last_price=current_price,  # Set last_price for compatibility
            unrealized_pl=_to_float(getattr(p, "unrealized_pl", None)),
            unrealized_plpc=_to_float(getattr(p, "unrealized_plpc", None)),
            side=str(getattr(p, "side", "")).lower(),
        )

    def _mk_broker_snapshot(self, a, positions: dict = None) -> BrokerSnapshot:
        return BrokerSnapshot(
            account_number=getattr(a, "account_number", getattr(a, "id", "")),
            status=getattr(a, "status", ""),
            cash=_to_float(getattr(a, "cash", 0.0)) or 0.0,
            buying_power=_to_float(getattr(a, "buying_power", 0.0)) or 0.0,
            equity=_to_float(getattr(a, "equity", 0.0)) or 0.0,
            portfolio_value=_to_float(getattr(a, "portfolio_value", 0.0)) or 0.0,
            positions=positions,
        )


def _to_float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except Exception:
        return None
