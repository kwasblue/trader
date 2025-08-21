from __future__ import annotations
import asyncio
import inspect
from typing import Optional, List, Dict, Any

import alpaca
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

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

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.trading_client: Optional[TradingClient] = None
        self.stream: Optional[StockDataStream] = None
        self.data_rest: Optional[StockHistoricalDataClient] = None
        self._last_price: Dict[str, float] = {}
        self.logger = Logger("alpaca.log", "AlpacaBroker").get_logger()

    # ---------------------------------------------------------
    # Connections / Streaming
    # ---------------------------------------------------------
    def connect(self):
        """Establish trading and streaming connections."""
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
        self.stream = StockDataStream(self.api_key, self.api_secret, feed=DataFeed.IEX)
        self.data_rest = StockHistoricalDataClient(self.api_key, self.api_secret)
        self.logger.info("Connected to Alpaca.")

    async def start_stream(self, retry_seconds: int = 300):
        if not self.stream:
            self.logger.warning("Stream not initialized. Call connect() first.")
            return

        self.logger.info(
            f"[stream] starting... alpaca-py={getattr(alpaca,'__version__','unknown')} feed={getattr(self, 'feed', '?')}"
        )

        while True:
            try:
                run_fn = self.stream.run  # don't CALL it yet
                if inspect.iscoroutinefunction(run_fn):
                    await run_fn()                   # async variant (some versions)
                else:
                    await asyncio.to_thread(run_fn)  # sync variant (0.42.0 etc.)

                self.logger.warning("[stream] exited cleanly; restarting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"[stream] error: {e}; retrying in {retry_seconds}s")
                await asyncio.sleep(retry_seconds)

    def subscribe_bars(self, callback, symbol: str):
        if not self.stream:
            self.logger.warning("Stream not initialized. Call connect() first.")
            return

        # Wrap sync callbacks so the SDK gets an async function
        if not asyncio.iscoroutinefunction(callback):
            async def _async_wrap(bar):
                return callback(bar)
            cb = _async_wrap
        else:
            cb = callback

        self.stream.subscribe_bars(cb, symbol)
        self.logger.info(f"[stream] subscribed bars: {symbol}")

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

    def submit_market_order(self, symbol: str, qty: int, side: str) -> bool:
        """Submit a market order through Alpaca (bool result for convenience)."""
        if not self.trading_client:
            self.logger.error("Trading client not initialized. Call `connect()` first.")
            return False
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=_map_side(side),
                time_in_force=TimeInForce.DAY,
            )
            self.trading_client.submit_order(order)
            self.logger.info(f"Submitted {side.upper()} order for {qty} {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Order failed for {symbol}: {e}")
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
        if not self.trading_client:
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

        return await asyncio.to_thread(_submit)

    async def cancel_order(self, order_id: str) -> OrderResult:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _cancel():
            self.trading_client.cancel_order_by_id(order_id)
            try:
                o = self.trading_client.get_order_by_id(order_id)
                return self._mk_order_result(o)
            except Exception:
                return OrderResult(order_id=order_id, status="canceled")  # type: ignore[arg-type]

        return await asyncio.to_thread(_cancel)

    def place_market_order(self, symbol: str, qty: int, side: str, price: float | None = None) -> OrderResult:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")
        o = self.trading_client.submit_order(
            MarketOrderRequest(symbol=symbol, qty=qty, side=_map_side(side), time_in_force=TimeInForce.DAY)
        )
        return self._mk_order_result(o)

    def place_oco_order(self, symbol: str, qty: int, stop_price: float, limit_price: float) -> OrderResult:
        """Simple OCO via bracket: TP limit + SL stop.
        Note: Alpaca uses bracket order_class with take_profit/stop_loss.
        """
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,  # typical for exiting a long
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,  # type: ignore[arg-type]
            take_profit={"limit_price": limit_price},
            stop_loss={"stop_price": stop_price},
        )
        o = self.trading_client.submit_order(req)
        return self._mk_order_result(o)

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
            return self._mk_broker_snapshot(a)

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
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")
        a = self.trading_client.get_account()
        return float(getattr(a, "cash", 0.0))

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
        return PositionView(
            symbol=getattr(p, "symbol", None),
            qty=float(getattr(p, "qty", 0) or 0),
            avg_entry_price=_to_float(getattr(p, "avg_entry_price", None)),
            market_price=_to_float(getattr(p, "current_price", None)),
            unrealized_pl=_to_float(getattr(p, "unrealized_pl", None)),
            unrealized_plpc=_to_float(getattr(p, "unrealized_plpc", None)),
            side=str(getattr(p, "side", "")).lower(),
        )

    def _mk_broker_snapshot(self, a) -> BrokerSnapshot:
        return BrokerSnapshot(
            account_number=getattr(a, "account_number", getattr(a, "id", "")),
            status=getattr(a, "status", ""),
            cash=_to_float(getattr(a, "cash", 0.0)) or 0.0,
            buying_power=_to_float(getattr(a, "buying_power", 0.0)) or 0.0,
            equity=_to_float(getattr(a, "equity", 0.0)) or 0.0,
            portfolio_value=_to_float(getattr(a, "portfolio_value", 0.0)) or 0.0,
            multiplier=_to_float(getattr(a, "multiplier", 1.0)) or 1.0,
        )


def _to_float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except Exception:
        return None
