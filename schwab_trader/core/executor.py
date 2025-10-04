import time
from datetime import datetime, UTC
from collections import defaultdict
import pandas as pd
import asyncio
import uuid

from loggers.logger import Logger
from core.position_sizer import DynamicPositionSizer
from core.base.base_broker_interface import BaseBrokerInterface
from core.events.eventhandler import EventHandler, get_event_handler
from core.events.events import (
    EVENT_NEW_TRADE,
    EVENT_ORDER_STATUS,
    EVENT_PNL_UPDATE,
    EVENT_ALERT,
    EVENT_MANUAL_ORDER,
    EVENT_FLATTEN_ALL,
    EVENT_FLATTEN_SYMBOL,
    EVENT_CANCEL_ALL,
    EVENT_POSITION_UPDATE,
    TradePayload,
    OrderStatusPayload,
    PnLPayload,
    AlertPayload,
    PositionPayload
)


class Executor:
    """
    Broker-agnostic executor for live trading.
    Handles trading decisions, order submission, sizing, logging,
    and responds to GUI-issued events (manual order, flatten, cancel).
    """

    def __init__(
        self,
        broker: BaseBrokerInterface,
        sizer: DynamicPositionSizer,
        dry_run: bool = False,
        risk_percentage: float = 0.7,
    ):
        self.broker = broker
        self.sizer = sizer
        self.dry_run = dry_run
        self.risk_percentage = risk_percentage
        self.logger = Logger("app.log", "LiveExecutor").get_logger()
        self.bus = get_event_handler()
        self.position = defaultdict(int)
        self.stops = {}   # symbol → stop loss
        self.targets = {} # symbol → take profit

        # Wire GUI -> Executor event handlers
        asyncio.create_task(self.bus.subscribe(EVENT_MANUAL_ORDER, self.handle_manual_order))
        asyncio.create_task(self.bus.subscribe(EVENT_FLATTEN_ALL, self.handle_flatten_all))
        asyncio.create_task(self.bus.subscribe(EVENT_FLATTEN_SYMBOL, self.handle_flatten_symbol))
        asyncio.create_task(self.bus.subscribe(EVENT_CANCEL_ALL, self.handle_cancel_all))


    # ------------------- Core Execution -------------------
    def execute(self, symbol: str, df: pd.DataFrame, signal: int, price: float, atr_value: float):
        """
        Execute a buy/sell/hold decision using broker interface.
        """
        if pd.isna(atr_value) or atr_value <= 0:
            return

        atr_25 = df['ATR'].quantile(0.25)
        atr_75 = df['ATR'].quantile(0.75)

        market_conditions = (
            "low_volatility" if atr_value < atr_25 else
            "high_volatility" if atr_value > atr_75 else
            "normal"
        )

        stop_loss_price = price - (atr_value * 2)

        cash = self.broker.get_available_funds(symbol)
        qty = int(self.sizer.calculate_position_size(
            price=price,
            stop_loss_price=stop_loss_price,
            current_cash=cash,
            market_conditions=market_conditions,
            signal=signal
        ))

        while qty > 0 and not self.broker.has_sufficient_funds(symbol, qty):
            qty -= 1

        if qty <= 0:
            self.logger.warning(f"No affordable position size for {symbol} at ${price:.2f}")
            return

        if signal == 1 and self.position[symbol] == 0:
            self._place_order("BUY", symbol, qty, price)

        elif signal == -1 and self.position[symbol] > 0:
            self._place_order("SELL", symbol, self.position[symbol], price)

        elif signal == 0:
            self.logger.info(f"[{symbol}] HOLD - No trade action taken")


    def _place_order(self, side: str, symbol: str, qty: int, price: float):
        """
        Internal helper to place an order and emit schema-compliant events.
        """
        now = datetime.now(UTC)

        if self.dry_run:
            self.logger.info(f"[DRY RUN] {side} {qty} {symbol} @ {price:.2f}")
            alert: AlertPayload = {
                "level": "info",
                "message": f"[DRY RUN] {side} {qty} {symbol} @ {price:.2f}",
                "symbol": symbol,
                "timestamp": now.isoformat(),
            }
            asyncio.create_task(self.bus.emit(EVENT_ALERT, alert))
            return

        try:
            response = self.broker.place_market_order(symbol, qty, side)
            self.logger.info(f"[{side}] {symbol}: {qty} @ {price:.2f} → Response: {response}")

            # Normalize response (supports dict or dataclass OrderResult)
            order_id = getattr(response, "order_id", None) or (response.get("id") if isinstance(response, dict) else None)
            status   = getattr(response, "status", None) or (response.get("status") if isinstance(response, dict) else "unknown")

            # ✅ emit order status
            order_status: OrderStatusPayload = {
                "order_id": str(order_id) if order_id else f"mock_{uuid.uuid4()}",
                "symbol": symbol,
                "status": status,
                "filled_qty": float(qty),
                "avg_price": float(price),
                "timestamp": now.isoformat(),
            }
            asyncio.create_task(self.bus.emit(EVENT_ORDER_STATUS, order_status))

            # ✅ emit trade event
            trade: TradePayload = {
                "symbol": symbol,
                "side": side.lower(),
                "qty": float(qty),
                "price": float(price),
                "timestamp": now.isoformat(),
                "pnl": None,  # updated elsewhere
            }
            asyncio.create_task(self.bus.emit(EVENT_NEW_TRADE, trade))

        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            alert: AlertPayload = {
                "level": "error",
                "message": f"Order failed for {symbol}: {e}",
                "symbol": symbol,
                "timestamp": now.isoformat(),
            }
            asyncio.create_task(self.bus.emit(EVENT_ALERT, alert))
            return

        # ✅ Update local position
        if side == "BUY":
            self.position[symbol] += qty
        elif side == "SELL":
            self.position[symbol] = 0

        # ✅ emit PnL update (must match schema)
        cash = self.broker.get_available_funds()
        portfolio_value = cash + sum(pos * price for pos in self.position.values())
        pnl_payload: PnLPayload = {
            "portfolio_value": portfolio_value,
            "equity_curve": [],  # optional to fill in
            "unrealized": 0.0,   # TODO: calc unrealized per-position
            "realized": 0.0,     # TODO: track realized PnL
            "drawdown": 0.0,     # TODO: hook into drawdown monitor
            "timestamp": now.isoformat(),
        }
        asyncio.create_task(self.bus.emit(EVENT_PNL_UPDATE, pnl_payload))

        pos_payload: PositionPayload = {
            "symbol": symbol,
            "qty": self.position[symbol],
            "avg_price": self.entry_price[symbol],
            "cash": self.cash[symbol],
            "last_price": price,
        }
        asyncio.create_task(self.bus.emit(EVENT_POSITION_UPDATE, pos_payload))




    # ------------------- GUI Event Handlers -------------------
    async def handle_manual_order(self, event):
        payload = event.payload
        symbol  = payload["symbol"]
        qty     = int(payload["qty"])
        side    = payload["side"].upper()
        price   = float(payload.get("price", 0.0))
        order_type = payload.get("type", "market").lower()
        tif     = payload.get("tif", "DAY")

        self.logger.info(f"[MANUAL ORDER] {side} {qty} {symbol} ({order_type.upper()})")

        if self.dry_run:
            alert: AlertPayload = {
                "level": "info",
                "message": f"[MANUAL-DRYRUN] {side} {qty} {symbol}",
                "symbol": symbol,
                "timestamp": datetime.now(UTC).isoformat()
            }
            await self.bus.emit(EVENT_ALERT, alert)
            return

        try:
            # Call broker directly depending on order type
            if order_type == "limit":
                resp = self.broker.place_order(
                    symbol, qty, side, order_type="limit",
                    limit_price=price, time_in_force=tif
                )
            else:
                resp = self.broker.place_market_order(symbol, qty, side)

            # Delegate to unified emitter logic
            self._place_order(side, symbol, qty, price)

        except Exception as e:
            self.logger.error(f"[MANUAL ORDER FAILED] {symbol} {side} {qty} @ {price}: {e}")
            alert: AlertPayload = {
                "level": "error",
                "message": f"Manual order failed for {symbol}: {e}",
                "symbol": symbol,
                "timestamp": datetime.now(UTC).isoformat()
            }
            await self.bus.emit(EVENT_ALERT, alert)



    async def handle_flatten_all(self, event):
        """
        Close ALL open positions across symbols.
        """
        self.logger.info("[EXECUTOR] Flatten ALL positions")
        for sym, qty in list(self.position.items()):
            if qty > 0:
                try:
                    # TODO: fetch live price instead of 0.0
                    live_price = 0.0
                    self._place_order("SELL", sym, qty, live_price)
                except Exception as e:
                    self.logger.error(f"[EXECUTOR] Failed to flatten {sym}: {e}")
                    alert: AlertPayload = {
                        "level": "error",
                        "message": f"Flatten all failed for {sym}: {e}",
                        "symbol": sym,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    await self.bus.emit(EVENT_ALERT, alert)


    async def handle_flatten_symbol(self, event):
        """
        Close the position for a specific symbol.
        """
        sym = event.payload["symbol"]
        qty = self.position.get(sym, 0)
        self.logger.info(f"[EXECUTOR] Flatten {sym} ({qty})")

        if qty > 0:
            try:
                # TODO: fetch live price instead of 0.0
                live_price = 0.0
                self._place_order("SELL", sym, qty, live_price)
            except Exception as e:
                self.logger.error(f"[EXECUTOR] Failed to flatten {sym}: {e}")
                alert: AlertPayload = {
                    "level": "error",
                    "message": f"Flatten symbol failed for {sym}: {e}",
                    "symbol": sym,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                await self.bus.emit(EVENT_ALERT, alert)


    async def handle_cancel_all(self, event):
        self.logger.info("[EXECUTOR] Cancel all open orders")
        try:
            open_orders = await self.broker.get_open_orders()
            for o in open_orders:
                order_id = getattr(o, "order_id", None) or (o.get("id") if isinstance(o, dict) else None)
                if order_id:
                    await self.broker.cancel_order(order_id)
        except Exception as e:
            alert: AlertPayload = {
                "level": "error",
                "message": f"Cancel all failed: {e}",
                "symbol": None,
                "timestamp": datetime.now(UTC).isoformat()
            }
            await self.bus.emit(EVENT_ALERT, alert)

