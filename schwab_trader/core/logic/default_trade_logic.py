# core/logic/default_trade_logic.py
from __future__ import annotations
from typing import Optional
from core.base.trade_logic_base import TradeLogic
from core.base.base_broker_interface import BaseBrokerInterface
from core.base.position_sizer_base import PositionSizerBase
from loggers.file_trade_logger import FileTradeLogger
from core.logic.symbol_state import SymbolState
from core.logic.portfolio_state import PortfolioState
from datetime import datetime, UTC

class DefaultTradeLogic(TradeLogic):
    def __init__(
        self,
        tp_mult_low: float = 1.5,
        tp_mult_normal: float = 2.0,
        tp_mult_high: float = 3.0,
        sl_mult_low: float = 1.0,
        sl_mult_normal: float = 1.5,
        sl_mult_high: float = 2.0,
        exit_fraction: float = 0.25,
        trailing_stop: bool = True,
    ):
        self.tp = {"low_volatility": tp_mult_low, "normal": tp_mult_normal, "high_volatility": tp_mult_high}
        self.sl = {"low_volatility": sl_mult_low, "normal": sl_mult_normal, "high_volatility": sl_mult_high}
        self.exit_fraction = exit_fraction
        self.trailing_stop = trailing_stop
    
    def _get_cash(self, broker: BaseBrokerInterface, state: SymbolState) -> float:
        """
        Try broker first, then fall back to portfolio_value on the state.
        Expected broker methods (use whatever your interface provides):
          - get_available_funds() -> float  (preferred)
          - get_account_balance() -> mapping with 'cash' or similar
        """
        cash = None
        if hasattr(broker, "get_available_funds"):
            try:
                cash = float(broker.get_available_funds())
            except Exception:
                cash = None

        if cash is None and hasattr(broker, "get_account_balance"):
            try:
                bal = broker.get_account_balance()
                # support dict or object with .cash attribute
                if isinstance(bal, dict):
                    # common keys people use
                    for k in ("available_funds", "buying_power", "cash"):
                        if k in bal:
                            cash = float(bal[k])
                            break
                elif hasattr(bal, "cash"):
                    cash = float(bal.cash)
            except Exception:
                cash = None

        # final fallback (crude): use total equity cached on the symbol state
        if cash is None:
            cash = float(getattr(state, "portfolio_value", 0.0))

        return max(0.0, cash)

    def _can_afford(self, broker: BaseBrokerInterface, symbol: str, side: str, qty: int, price: float, cash: float) -> bool:
        """
        Conservative affordability check. For BUY we require notional <= cash.
        For SELL (closing long or opening short), we let it through (no cash needed).
        If your system supports real margin, enhance this to use a preview endpoint.
        """
        if qty <= 0:
            return False
        notional = qty * price

        # if broker has a preview endpoint, prefer that
        if hasattr(broker, "preview_order"):
            try:
                ok = broker.preview_order(symbol=symbol, qty=qty, side=side, price=price)
                if isinstance(ok, bool):
                    return ok
            except Exception:
                pass  # fall back to simple rule

        if side.lower() == "buy":
            return notional <= cash
        # for sells we assume either closing longs or allowing shorting without upfront cash here
        return True

    def execute(
        self,
        symbol: str,
        state: SymbolState,
        signal: int,
        price: float,
        atr: Optional[float],
        regime: Optional[str],
        broker: BaseBrokerInterface,
        sizer: PositionSizerBase,
        performance_tracker: FileTradeLogger,
        portfolio: PortfolioState,                     
    ) -> None:
        # snapshot position from portfolio
        pos = portfolio.positions.get(symbol)
        qty_now = 0 if not pos else pos.qty
        avg_now = None if not pos else pos.avg_price
        side_now = "long" if qty_now > 0 else ("short" if qty_now < 0 else None)

        # keep tactical bookkeeping
        state.side = side_now
        if state.side:
            state.bars_held += 1
            state.update_excursions(price, state.side, avg_now)

        cond = self._get_market_condition(atr, regime)
        if atr is None or atr <= 0:
            return

        # size using portfolio cash
        stop_loss_price = price - atr * self.sl[cond] if signal == 1 else price + atr * self.sl[cond]
        qty = sizer.calculate_position_size(
            price=price,
            stop_loss_price=stop_loss_price,
            current_cash=portfolio.cash,
            market_conditions=cond,
            signal=signal,
        )
        if qty <= 0 and state.side is None:
            return

        # --- entries (flat -> position) ---
        if state.side is None:
            if signal == 1 and broker.place_market_order(symbol, qty, "buy", price):
                self._entered_long(state, price, atr, cond)
                performance_tracker.log_trade(symbol, "enter_long", price, qty,
                                              sl=state.stop_loss, tp=state.take_profit,
                                              strategy=state.strategy_name or type(self).__name__, regime=cond,
                                              timestamp = datetime.now(UTC))
            elif signal == -1 and broker.place_market_order(symbol, qty, "sell", price):
                self._entered_short(state, price, atr, cond)
                performance_tracker.log_trade(symbol, "enter_short", price, qty,
                                              sl=state.stop_loss, tp=state.take_profit,
                                              strategy=state.strategy_name or type(self).__name__, regime=cond,
                                              timestamp = datetime.now(UTC))
            return

        # --- manage / exits (already in a position) ---
        if state.side == "long":
            self._manage_long(symbol, state, signal, price, atr, cond, broker, performance_tracker, qty_now)
        else:
            self._manage_short(symbol, state, signal, price, atr, cond, broker, performance_tracker, qty_now)

    def _get_market_condition(self, atr: Optional[float], regime: Optional[str]) -> str:
        return regime if regime in ("low_volatility", "normal", "high_volatility") else "normal"

    # On entry, only set tactical targets/stops; do NOT touch qty/avg here.
    def _entered_long(self, state, price, atr, cond):
        state.stop_loss  = price - atr * self.sl[cond]
        state.take_profit = price + atr * self.tp[cond]
        state.partial_exit_targets = [price + atr * lvl for lvl in (1.0, 2.0)]
        state.pyramid_layer = 1
        state.bars_held = 0
        state.max_favorable_excursion = None
        state.max_adverse_excursion = None

    def _entered_short(self, state, price, atr, cond):
        state.stop_loss  = price + atr * self.sl[cond]
        state.take_profit = price - atr * self.tp[cond]
        state.partial_exit_targets = [price - atr * lvl for lvl in (1.0, 2.0)]
        state.pyramid_layer = 1
        state.bars_held = 0
        state.max_favorable_excursion = None
        state.max_adverse_excursion = None

    def _manage_long(self, symbol, state, signal, price, atr, cond, broker, tracker, qty_now: int):
        if signal == -1:
            self._close(symbol, state, price, broker, tracker, abs(qty_now))
            return

        # partial exits sized from current position
        if qty_now > 0 and state.partial_exit_targets and price >= state.partial_exit_targets[0]:
            qty_exit = max(int(qty_now * self.exit_fraction), 1)
            if broker.place_market_order(symbol, qty_exit, "sell", price):
                tracker.log_trade(symbol, "partial_exit", price, qty_exit,
                                  strategy=state.strategy_name, regime=cond, notes="long",
                                  timestamp = datetime.now(UTC))
                state.partial_exit_targets.pop(0)

        if self.trailing_stop and atr is not None:
            state.stop_loss = max(state.stop_loss, price - atr * self.sl[cond])

        if price >= state.take_profit or price <= state.stop_loss:
            self._close(symbol, state, price, broker, tracker, abs(qty_now))

    def _manage_short(self, symbol, state, signal, price, atr, cond, broker, tracker, qty_now: int):
        if signal == 1:
            self._close(symbol, state, price, broker, tracker, abs(qty_now))
            return

        if qty_now < 0 and state.partial_exit_targets and price <= state.partial_exit_targets[0]:
            qty_exit = max(int(abs(qty_now) * self.exit_fraction), 1)
            if broker.place_market_order(symbol, qty_exit, "buy", price):
                tracker.log_trade(symbol, "partial_exit", price, qty_exit,
                                  strategy=state.strategy_name, regime=cond, notes="short",
                                  timestamp = datetime.now(UTC))
                state.partial_exit_targets.pop(0)

        if self.trailing_stop and atr is not None:
            state.stop_loss = min(state.stop_loss, price + atr * self.sl[cond])

        if price <= state.take_profit or price >= state.stop_loss:
            self._close(symbol, state, price, broker, tracker, abs(qty_now))

    def _close(self, symbol, state, price, broker, tracker, qty_to_flatten: int):
        if qty_to_flatten <= 0:
            return
        side = "sell" if state.side == "long" else "buy"
        if broker.place_market_order(symbol, qty_to_flatten, side, price):
            tracker.log_trade(symbol, f"exit_{state.side}", price, qty_to_flatten,
                              sl=state.stop_loss, tp=state.take_profit,
                              strategy=state.strategy_name,
                              notes=f"bars={state.bars_held}, MFE={state.max_favorable_excursion}, MAE={state.max_adverse_excursion}",
                              timestamp = datetime.now(UTC))
            state.reset()