# core/mock_executor.py
from loggers.logger import Logger
import pandas as pd
from pathlib import Path
from datetime import datetime, UTC
from collections import defaultdict
from core.position_sizer import DynamicPositionSizer
import asyncio

from core.events.eventhandler import EventHandler, get_event_handler
from core.events.events import (
    EVENT_NEW_TRADE,
    EVENT_PNL_UPDATE,
    EVENT_ALERT,
    EVENT_POSITION_UPDATE,
    EVENT_MANUAL_ORDER,
    TradePayload,
    PnLPayload,
    AlertPayload,
    PositionPayload
)
import asyncio


class MockExecutor:
    """
    Simulated execution environment for strategies + manual orders.
    Emits events so GUI sees trades, PnL, and positions like live mode.
    """

    def __init__(self, risk_percentage=0.07, starting_cash: float = 100_000):
        self.logger = Logger('app.log', 'MockExecutor', log_dir='logs').get_logger()
        self.peak_portfolio_value = defaultdict(lambda: 0.0)
        self.portfolio_history = defaultdict(list)

        self.cash = defaultdict(lambda: starting_cash)
        self.position = defaultdict(int)
        self.entry_price = defaultdict(lambda: 0.0)
        self.realized_pnl = defaultdict(float)
        self.total_fees = defaultdict(float)

        self.risk_percentage = risk_percentage
        self.sizer = DynamicPositionSizer(risk_percentage=self.risk_percentage)
        self.bus = get_event_handler()
        asyncio.create_task(self.bus.subscribe(EVENT_MANUAL_ORDER, self._on_manual_order))

    # ---------------- Algorithmic execution ----------------
    def execute(self, symbol, df, signal, price, atr_value):
        if signal == 0 or pd.isna(atr_value) or atr_value <= 0:
            self.logger.debug(f"[HOLD] {symbol}: No action taken.")
            return

        atr_25 = df['ATR'].quantile(0.25) if df is not None else atr_value * 0.8
        atr_75 = df['ATR'].quantile(0.75) if df is not None else atr_value * 1.2
        if atr_value < atr_25:
            market_conditions = "low_volatility"
        elif atr_value > atr_75:
            market_conditions = "high_volatility"
        else:
            market_conditions = "normal"

        stop_loss_price = price - (atr_value * 2) if signal == 1 else price + (atr_value * 2)

        qty = self.sizer.calculate_position_size(
            price=price,
            stop_loss_price=stop_loss_price,
            current_cash=self.cash[symbol],
            market_conditions=market_conditions,
            signal=signal,
        )

        fee = 0.001 * price * qty
        max_qty = int(self.cash[symbol] // (price + fee))
        qty = min(qty, max_qty)

        now = datetime.now(UTC)

        # --- BUY ---
        if signal == 1 and qty > 0 and self.position[symbol] == 0:
            self.cash[symbol] -= (price * qty + fee)
            self.position[symbol] = qty
            self.entry_price[symbol] = price
            self.total_fees[symbol] += fee
            self.logger.info(f"[BUY] {symbol} {qty} @ {price:.2f}")

            trade: TradePayload = {
                "symbol": symbol, "side": "buy", "qty": qty, "price": price,
                "timestamp": now.isoformat(), "pnl": None,
            }
            asyncio.create_task(self.bus.emit(EVENT_NEW_TRADE, trade))

            pos_payload: PositionPayload = {
                "symbol": symbol,
                "qty": float(self.position[symbol]),
                "avg_price": float(self.entry_price[symbol]),
                "unrealized": 0.0,
                "realized": float(self.realized_pnl[symbol]),
                "timestamp": now.isoformat(),
            }
            asyncio.create_task(self.bus.emit(EVENT_POSITION_UPDATE, pos_payload))


        # --- SELL ---
        elif signal == -1 and self.position[symbol] > 0:
            qty = self.position[symbol]
            self.cash[symbol] += price * qty - fee
            self.realized_pnl[symbol] += (price - self.entry_price[symbol]) * qty
            self.position[symbol] = 0
            self.total_fees[symbol] += fee
            self.logger.info(f"[SELL] {symbol} {qty} @ {price:.2f}")

            trade: TradePayload = {
                "symbol": symbol, "side": "sell", "qty": qty, "price": price,
                "timestamp": now.isoformat(), "pnl": self.realized_pnl[symbol],
            }
            asyncio.create_task(self.bus.emit(EVENT_NEW_TRADE, trade))

            pos_payload: PositionPayload = {
                "symbol": symbol,
                "qty": float(self.position[symbol]),
                "avg_price": float(self.entry_price[symbol]),
                "unrealized": 0.0,
                "realized": float(self.realized_pnl[symbol]),
                "timestamp": now.isoformat(),
            }
            asyncio.create_task(self.bus.emit(EVENT_POSITION_UPDATE, pos_payload))


        # Track + emit
        self._update_and_emit(symbol, price, now)

    # ---------------- Manual order handler ----------------
    async def _on_manual_order(self, event):
        payload = event.payload
        symbol = payload["symbol"]
        side = payload["side"].lower()
        qty = int(payload["qty"])
        price = payload.get("price") or 100.0
        now = datetime.now(UTC)

        if side in ("buy", "long"):
            if price * qty > self.cash[symbol]:
                alert: AlertPayload = {
                    "level": "error",
                    "message": f"Insufficient funds for {symbol}",
                    "timestamp": now.isoformat(),
                }
                return asyncio.create_task(self.bus.emit(EVENT_ALERT, alert))

            self.cash[symbol] -= price * qty
            self.position[symbol] += qty
            if self.entry_price[symbol] == 0.0:  # first entry
                self.entry_price[symbol] = price
            self.logger.info(f"[MANUAL BUY] {symbol} {qty} @ {price:.2f}")

            trade: TradePayload = {
                "symbol": symbol, "side": "buy", "qty": qty, "price": price,
                "timestamp": now.isoformat(), "pnl": None,
            }
            asyncio.create_task(self.bus.emit(EVENT_NEW_TRADE, trade))

            pos_payload: PositionPayload = {
                "symbol": symbol,
                "qty": self.position[symbol],
                "avg_price": self.entry_price[symbol],
                "cash": self.cash[symbol],
                "last_price": price,
            }
            asyncio.create_task(self.bus.emit(EVENT_POSITION_UPDATE, pos_payload))

        elif side in ("sell", "cover"):
            if qty > self.position[symbol]:
                alert: AlertPayload = {
                    "level": "error",
                    "message": f"Not enough {symbol} to sell",
                    "timestamp": now.isoformat(),
                }
                return asyncio.create_task(self.bus.emit(EVENT_ALERT, alert))

            self.cash[symbol] += price * qty
            self.position[symbol] -= qty
            self.realized_pnl[symbol] += (price - self.entry_price[symbol]) * qty
            self.logger.info(f"[MANUAL SELL] {symbol} {qty} @ {price:.2f}")

            trade: TradePayload = {
                "symbol": symbol, "side": "sell", "qty": qty, "price": price,
                "timestamp": now.isoformat(), "pnl": self.realized_pnl[symbol],
            }
            asyncio.create_task(self.bus.emit(EVENT_NEW_TRADE, trade))

            pos_payload: PositionPayload = {
                "symbol": symbol,
                "qty": self.position[symbol],
                "avg_price": self.entry_price[symbol],
                "cash": self.cash[symbol],
                "last_price": price,
            }
            asyncio.create_task(self.bus.emit(EVENT_POSITION_UPDATE, pos_payload))


        # Track + emit
        self._update_and_emit(symbol, price, now)

    # ---------------- Shared helpers ----------------
    def _update_and_emit(self, symbol, price, now):
        portfolio_value = self.cash[symbol] + self.position[symbol] * price
        self.peak_portfolio_value[symbol] = max(self.peak_portfolio_value[symbol], portfolio_value)
        drawdown = (portfolio_value - self.peak_portfolio_value[symbol]) / max(self.peak_portfolio_value[symbol], 1.0)

        self.portfolio_history[symbol].append({
            "Date": now, "Portfolio_Value": portfolio_value,
            "Cash": self.cash[symbol], "Position": self.position[symbol],
            "Price": price, "Drawdown": drawdown,
            "Fees": self.total_fees[symbol], "RealizedPnL": self.realized_pnl[symbol],
        })

        pnl_payload: PnLPayload = {
            "portfolio_value": portfolio_value,
            "equity_curve": [p["Portfolio_Value"] for p in self.portfolio_history[symbol]],
            "unrealized": (price - self.entry_price[symbol]) * self.position[symbol] if self.position[symbol] else 0.0,
            "realized": self.realized_pnl[symbol],
            "drawdown": drawdown,
            "timestamp": now.isoformat(),
        }
        asyncio.create_task(self.bus.emit(EVENT_PNL_UPDATE, pnl_payload))

        pos_payload: PositionPayload = {
            "symbol": symbol,
            "qty": float(self.position[symbol]),
            "avg_price": float(self.entry_price[symbol]),
            "unrealized": float((price - self.entry_price[symbol]) * self.position[symbol]) if self.position[symbol] else 0.0,
            "realized": float(self.realized_pnl[symbol]),
            "timestamp": now.isoformat(),
        }
        asyncio.create_task(self.bus.emit(EVENT_POSITION_UPDATE, pos_payload))

    def save_results(self, path: str):
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        for symbol, history in self.portfolio_history.items():
            df = pd.DataFrame(history)
            df.to_csv(output_path / f"mock_results_{symbol}.csv", index=False)
            self.logger.info(f"Saved mock results for {symbol} to {output_path}")
