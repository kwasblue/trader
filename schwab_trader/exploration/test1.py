# trading_system/config.yaml

# strategy_routing:
#   default:
#     class: MomentumStrategy
#     params: {}

#   overrides:
#     symbols:
#       TSLA:
#         class: MomentumStrategy
#         params:
#           lookback: 14
#     regimes:
#       high_volatility:
#         class: MomentumStrategy
#         params:
#           lookback: 10

# trade_logic:
#   default:
#     class: DefaultTradeLogic
#     params:
#       max_pyramid_layers: 1
#       exit_fraction: 0.25
#       trailing_stop: true
#       partial_exit_levels: [1.0, 2.0]
#       regime_tp_sl_multipliers:
#         low_volatility: {tp_mult: 1.5, sl_mult: 1.0}
#         normal: {tp_mult: 2.0, sl_mult: 1.5}
#         high_volatility: {tp_mult: 3.0, sl_mult: 2.0}
#       max_consecutive_losses: 3
#       max_symbol_daily_loss_pct: 0.02

# trading_system/main.py
import sys
from pathlib import Path
project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import asyncio 
from collections import defaultdict
import asyncio
import logging
from strategies.strategy_registry.momentum_strategy import MomentumStrategy
from abc import ABC, abstractmethod
import yaml 
from datetime import datetime
import importlib
import yaml
import logging
import os
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from enum import Enum
from alpaca.trading.enums import OrderSide as AlpacaOrderSide

async def main():
    config = ConfigLoader.load("config.yaml")
    broker = ''
    engine = ExecutionEngine(broker=broker, config=config)
    await engine.run()

if __name__ == "__main__":
    asyncio.run(main())

class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.drawdowns = {}
        self.daily_pnl = {}

    def log_trade(self, trade, state):
        symbol = trade['symbol']
        price = trade['price']
        qty = trade['qty']
        side = trade['side']
        pnl = trade.get('pnl', 0)

        self.trades.append({
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'price': price,
            'qty': qty,
            'side': side,
            'pnl': pnl
        })

        logger.info(f"Trade executed: {side.upper()} {qty} {symbol} @ {price:.2f} | PnL: {pnl:.2f}")

        # Optionally update rolling metrics here

class PositionState:
    def __init__(self):
        self.cash = 100000
        self.positions = defaultdict(int)
        self.avg_price = defaultdict(float)
        self.last_price = {}

    def update_price(self, symbol, price):
        self.last_price[symbol] = price

    def apply_trade(self, trade):
        symbol = trade['symbol']
        qty = trade['qty']
        price = trade['price']
        side = trade['side']
        cost = qty * price

        if side == 'buy':
            prev_qty = self.positions[symbol]
            new_qty = prev_qty + qty
            self.avg_price[symbol] = (
                (self.avg_price[symbol] * prev_qty + cost) / new_qty
                if new_qty != 0 else 0
            )
            self.positions[symbol] += qty
            self.cash -= cost
        elif side == 'sell':
            self.positions[symbol] -= qty
            self.cash += cost

    def get_position(self, symbol):
        return self.positions[symbol]

    def get_avg_price(self, symbol):
        return self.avg_price[symbol]

    def get_price(self, symbol):
        return self.last_price.get(symbol, None)

class ExecutionEngine:
    def __init__(self, broker, config):
        self.broker = broker
        self.config = config
        self.state = PositionState()
        self.performance_tracker = PerformanceTracker()
        self.strategy_instances = {}
        self.trade_logic_instances = {}

    async def run(self):
        while True:
            bar = self.broker.get_next_bar()
            if bar is None:
                await asyncio.sleep(1)
                continue

            symbol = bar['symbol']
            self.state.update_price(symbol, bar['close'])

            strategy = self.get_strategy(symbol)
            signal = strategy.generate_signal(bar)

            trade_logic = self.get_trade_logic(symbol)
            trade = trade_logic.decide_trade(symbol, signal, bar, self.state)

            if trade:
                executed = self.broker.execute_trade(trade)
                if executed:
                    self.state.apply_trade(trade)
                    self.performance_tracker.log_trade(trade, self.state)

    def get_strategy(self, symbol):
        if symbol not in self.strategy_instances:
            self.strategy_instances[symbol] = MomentumStrategy(**self.config["strategy_routing"]["default"]["params"])
        return self.strategy_instances[symbol]

    def get_trade_logic(self, symbol):
        if symbol not in self.trade_logic_instances:
            self.trade_logic_instances[symbol] = DefaultTradeLogic(**self.config["trade_logic"]["default"]["params"])
        return self.trade_logic_instances[symbol]

class DefaultTradeLogic:
    def __init__(self,
                 max_pyramid_layers=1,
                 exit_fraction=0.25,
                 trailing_stop=True,
                 partial_exit_levels=None,
                 regime_tp_sl_multipliers=None,
                 max_consecutive_losses=3,
                 max_symbol_daily_loss_pct=0.02):

        self.max_pyramid_layers = max_pyramid_layers
        self.exit_fraction = exit_fraction
        self.trailing_stop = trailing_stop
        self.partial_exit_levels = partial_exit_levels or [1.0, 2.0]
        self.regime_tp_sl_multipliers = regime_tp_sl_multipliers or {
            "low_volatility": {"tp_mult": 1.5, "sl_mult": 1.0},
            "normal": {"tp_mult": 2.0, "sl_mult": 1.5},
            "high_volatility": {"tp_mult": 3.0, "sl_mult": 2.0},
        }
        self.max_consecutive_losses = max_consecutive_losses
        self.max_symbol_daily_loss_pct = max_symbol_daily_loss_pct
        self.pyramiding_count = {}

    def decide_trade(self, symbol, signal, bar, state):
        price = bar['close']
        position = state.get_position(symbol)
        avg_price = state.get_avg_price(symbol)

        # === ENTRY ===
        if signal == 1 and position <= 0:
            return {
                "symbol": symbol,
                "qty": 10,
                "price": price,
                "side": "buy"
            }

        if signal == -1 and position >= 0:
            return {
                "symbol": symbol,
                "qty": 10,
                "price": price,
                "side": "sell"
            }

        # === TP/SL Placeholder ===
        # Add your trailing stop or take-profit exit logic here

        return None

class BrokerInterface(ABC):
    @abstractmethod
    def get_next_bar(self):
        pass

    @abstractmethod
    def execute_trade(self, trade):
        pass

    @abstractmethod
    def get_account_info(self):
        pass

    @abstractmethod
    def get_open_position(self, symbol):
        pass

    @abstractmethod
    def close_position(self, symbol):
        pass

class ConfigLoader:
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

class TradeLogicBase(ABC):
    """
    Abstract base class for all trade logic implementations.
    Each trade logic module is responsible for evaluating incoming signals
    and making decisions about entries, exits, sizing, and risk management.
    """

    @abstractmethod
    def execute(
        self,
        symbol: str,
        state,
        signal: int,
        price: float,
        atr: float,
        regime: str,
        broker,
        sizer,
        performance_tracker
    ):
        """
        Evaluate and act on the current market state and trading signal.

        Args:
            symbol (str): The ticker symbol.
            state: PositionState object holding current trade context.
            signal (int): Output from strategy (1=buy, -1=sell, 0=hold).
            price (float): Current market price.
            atr (float): ATR value for volatility-based SL/TP sizing.
            regime (str): Current regime classification (e.g., "low_volatility").
            broker: An implementation of BrokerInterface to place orders.
            sizer: Position sizing engine (e.g., DynamicPositionSizer).
            performance_tracker: Logs and evaluates trade performance.
        """
        pass

class DefaultTradeLogic(TradeLogicBase):
    def __init__(self,
                 max_pyramid_layers=1,
                 exit_fraction=0.25,
                 trailing_stop=True,
                 partial_exit_levels=(1.0, 2.0),  # ATR multiples
                 regime_tp_sl_multipliers=None,
                 max_daily_drawdown_pct=0.03,   # 3% portfolio
                 max_consecutive_losses=3,
                 max_symbol_daily_loss_pct=0.02):  # 2% per symbol
        self.max_pyramid_layers = max_pyramid_layers
        self.exit_fraction = exit_fraction
        self.trailing_stop = trailing_stop
        self.partial_exit_levels = partial_exit_levels
        self.regime_tp_sl_multipliers = regime_tp_sl_multipliers or {
            "low_volatility": {"tp_mult": 1.5, "sl_mult": 1.0},
            "normal": {"tp_mult": 2.0, "sl_mult": 1.5},
            "high_volatility": {"tp_mult": 3.0, "sl_mult": 2.0}
        }
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_symbol_daily_loss_pct = max_symbol_daily_loss_pct

        # Tracking
        self.daily_start_value = None
        self.consecutive_losses = 0
        self.symbol_daily_loss = {}
        self.last_trade_day = None

    def execute(self, symbol, state, signal, price, atr, regime, broker, sizer, performance_tracker):
        today = datetime.utcnow().date()

        # Reset daily counters if new day
        if self.last_trade_day != today:
            self.daily_start_value = broker.get_account().cash
            self.consecutive_losses = 0
            self.symbol_daily_loss.clear()
            self.last_trade_day = today

        # --- Portfolio protection checks ---
        current_value = broker.get_account().cash
        if self.daily_start_value and (current_value < self.daily_start_value * (1 - self.max_daily_drawdown_pct)):
            return  # stop trading for day

        if self.consecutive_losses >= self.max_consecutive_losses:
            return  # pause until next day

        if self.symbol_daily_loss.get(symbol, 0) <= -self.max_symbol_daily_loss_pct:
            return  # stop trading this symbol today

        # --- Normal execution flow ---
        if state.side:
            state.bars_held += 1
            state.update_excursions(price)

        qty = sizer.calculate_position_size(
            price,
            price - atr * self.regime_tp_sl_multipliers[regime]["sl_mult"] if signal == 1
            else price + atr * self.regime_tp_sl_multipliers[regime]["sl_mult"],
            state.portfolio_value,
            regime,
            signal
        )

        # Entry
        if state.side is None:
            if signal == 1 and broker.submit_market_order(symbol, qty, "buy"):
                self._enter_long(state, qty, price, atr, regime)
            elif signal == -1 and broker.submit_market_order(symbol, qty, "sell"):
                self._enter_short(state, qty, price, atr, regime)

        # Manage open trades
        elif state.side == "long":
            self._manage_long(symbol, state, signal, price, atr, regime, broker, performance_tracker)
        elif state.side == "short":
            self._manage_short(symbol, state, signal, price, atr, regime, broker, performance_tracker)

    def _enter_long(self, state, qty, price, atr, regime):
        mults = self.regime_tp_sl_multipliers[regime]
        state.side = "long"
        state.qty = qty
        state.entry_price = price
        state.stop_loss = price - atr * mults["sl_mult"]
        state.take_profit = price + atr * mults["tp_mult"]
        state.partial_exit_targets = [price + atr * lvl for lvl in self.partial_exit_levels]
        state.pyramid_layer = 1
        state.bars_held = 0
        state.max_favorable_excursion = None
        state.max_adverse_excursion = None

    def _enter_short(self, state, qty, price, atr, regime):
        mults = self.regime_tp_sl_multipliers[regime]
        state.side = "short"
        state.qty = qty
        state.entry_price = price
        state.stop_loss = price + atr * mults["sl_mult"]
        state.take_profit = price - atr * mults["tp_mult"]
        state.partial_exit_targets = [price - atr * lvl for lvl in self.partial_exit_levels]
        state.pyramid_layer = 1
        state.bars_held = 0
        state.max_favorable_excursion = None
        state.max_adverse_excursion = None

    def _manage_long(self, symbol, state, signal, price, atr, regime, broker, performance_tracker):
        if signal == -1:  # flip
            self._close_trade(symbol, state, price, regime, broker, performance_tracker)
            return

        if state.partial_exit_targets and price >= state.partial_exit_targets[0]:
            qty_exit = max(int(state.qty * self.exit_fraction), 1)
            if broker.submit_market_order(symbol, qty_exit, "sell"):
                state.qty -= qty_exit
                state.partial_exit_targets.pop(0)

        if self.trailing_stop:
            state.stop_loss = max(state.stop_loss, price - atr * self.regime_tp_sl_multipliers[regime]["sl_mult"])

        if price >= state.take_profit or price <= state.stop_loss:
            self._close_trade(symbol, state, price, regime, broker, performance_tracker)

    def _manage_short(self, symbol, state, signal, price, atr, regime, broker, performance_tracker):
        if signal == 1:  # flip
            self._close_trade(symbol, state, price, regime, broker, performance_tracker)
            return

        if state.partial_exit_targets and price <= state.partial_exit_targets[0]:
            qty_exit = max(int(state.qty * self.exit_fraction), 1)
            if broker.submit_market_order(symbol, qty_exit, "buy"):
                state.qty -= qty_exit
                state.partial_exit_targets.pop(0)

        if self.trailing_stop:
            state.stop_loss = min(state.stop_loss, price + atr * self.regime_tp_sl_multipliers[regime]["sl_mult"])

        if price <= state.take_profit or price >= state.stop_loss:
            self._close_trade(symbol, state, price, regime, broker, performance_tracker)

    def _close_trade(self, symbol, state, price, regime, broker, performance_tracker):
        if broker.submit_market_order(symbol, state.qty, "sell" if state.side == "long" else "buy"):
            pnl = performance_tracker.record_trade(
                symbol, state.strategy_name, regime, state.side,
                state.entry_price, price, state.qty, state.bars_held,
                state.max_favorable_excursion, state.max_adverse_excursion
            )

            # Loss tracking for risk limits
            if pnl < 0:
                self.consecutive_losses += 1
                self.symbol_daily_loss[symbol] = self.symbol_daily_loss.get(symbol, 0) + (pnl / state.portfolio_value)
            else:
                self.consecutive_losses = 0

            # Reset state
            state.side = None
            state.qty = 0
            state.pyramid_layer = 0
            state.bars_held = 0
            state.max_favorable_excursion = None
            state.max_adverse_excursion = None

class StrategyRoutingManager:
    def __init__(self, config_path="config.yaml", hot_reload=True):
        self.config_path = config_path
        self.hot_reload = hot_reload
        self.strategy_cache = {}
        self.config = None
        self.default_strategy = None
        self.last_loaded = 0
        self._load_config(force=True)

    def _load_config(self, force=False):
        try:
            last_modified = os.path.getmtime(self.config_path)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return

        if not force and last_modified == self.last_loaded:
            return

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.last_loaded = last_modified

        default_cfg = self.config["strategy_routing"]["default"]
        self.default_strategy = self._instantiate_strategy(default_cfg["class"], default_cfg.get("params", {}))

        self.strategy_cache.clear()
        logger.info("Strategy routing config reloaded.")

    def _instantiate_strategy(self, class_name, params):
        module_name = f"strategies.{class_name.lower().replace('strategy','')}_strategy"
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**params)

    def get(self, symbol, regime):
        if self.hot_reload:
            self._load_config()

        cache_key = f"{symbol}_{regime}"
        if cache_key in self.strategy_cache:
            return self.strategy_cache[cache_key]

        symbol_cfg = self.config["strategy_routing"].get("overrides", {}).get("symbols", {}).get(symbol)
        if symbol_cfg:
            strat = self._instantiate_strategy(symbol_cfg["class"], symbol_cfg.get("params", {}))
            self.strategy_cache[cache_key] = strat
            return strat

        regime_cfg = self.config["strategy_routing"].get("overrides", {}).get("regimes", {}).get(regime)
        if regime_cfg:
            strat = self._instantiate_strategy(regime_cfg["class"], regime_cfg.get("params", {}))
            self.strategy_cache[cache_key] = strat
            return strat

        self.strategy_cache[cache_key] = self.default_strategy
        return self.default_strategy

class TradeLogicManager:
    def __init__(self, config_path="config.yaml", hot_reload=True):
        self.config_path = config_path
        self.hot_reload = hot_reload
        self.logic_cache = {}
        self.config = None
        self.default_logic = None
        self.last_loaded = 0
        self._load_config(force=True)

    def _load_config(self, force=False):
        try:
            last_modified = os.path.getmtime(self.config_path)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return

        if not force and last_modified == self.last_loaded:
            return

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.last_loaded = last_modified

        default_cfg = self.config["trade_logic"]["default"]
        self.default_logic = self._instantiate_logic(default_cfg["class"], default_cfg.get("params", {}))

        self.logic_cache.clear()
        logger.info("Trade logic config reloaded.")

    def _instantiate_logic(self, class_name, params):
        module_name = f"core.trade_logic_{class_name.lower().replace('logic','')}"
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**params)

    def get(self, symbol, strategy):
        if self.hot_reload:
            self._load_config()

        cache_key = f"{symbol}_{strategy}"
        if cache_key in self.logic_cache:
            return self.logic_cache[cache_key]

        symbol_cfg = self.config["trade_logic"].get("overrides", {}).get("symbols", {}).get(symbol)
        if symbol_cfg:
            logic = self._instantiate_logic(symbol_cfg["class"], symbol_cfg.get("params", {}))
            self.logic_cache[cache_key] = logic
            return logic

        strategy_cfg = self.config["trade_logic"].get("overrides", {}).get("strategies", {}).get(strategy)
        if strategy_cfg:
            logic = self._instantiate_logic(strategy_cfg["class"], strategy_cfg.get("params", {}))
            self.logic_cache[cache_key] = logic
            return logic

        self.logic_cache[cache_key] = self.default_logic
        return self.default_logic

class ExecutionEngine:
    def __init__(self, broker, sizer, performance_tracker, trade_logic_manager):
        self.broker = broker
        self.sizer = sizer
        self.performance_tracker = performance_tracker
        self.trade_logic_manager = trade_logic_manager

    def handle_signal(self, symbol, state, signal, price, atr, regime, strategy_name=None):
        state.strategy_name = strategy_name
        trade_logic = self.trade_logic_manager.get(symbol, strategy_name)
        trade_logic.execute(
            symbol, state, signal, price, atr, regime,
            self.broker, self.sizer, self.performance_tracker
        )

class TradeLogicBase(ABC):
    @abstractmethod
    def execute(self, symbol, state, signal, price, atr, regime, broker, sizer, performance_tracker):
        pass

class DefaultTradeLogic(TradeLogicBase):
    def __init__(self, max_pyramid_layers=5, exit_fraction=0.25):
        self.max_pyramid_layers = max_pyramid_layers
        self.exit_fraction = exit_fraction

    def execute(self, symbol, state, signal, price, atr, regime, broker, sizer, performance_tracker):
        TP_MULTIPLIERS = {"low_volatility": 1.03, "normal": 1.05, "high_volatility": 1.10}
        tp_mult = TP_MULTIPLIERS.get(regime, 1.05)

        if state.side:
            state.bars_held += 1
            state.update_excursions(price)

        qty = sizer.calculate_position_size(price, price - atr * 2 if signal == 1 else price + atr * 2, state.portfolio_value, regime, signal)

        if state.side is None:
            if signal == 1 and broker.submit_market_order(symbol, qty, "buy"):
                self._enter_long(state, qty, price, atr, tp_mult)
            elif signal == -1 and broker.submit_market_order(symbol, qty, "sell"):
                self._enter_short(state, qty, price, atr, tp_mult)
        elif state.side == "long" and (signal == -1 or price >= state.take_profit or price <= state.stop_loss):
            self._close_trade(symbol, state, price, regime, broker, performance_tracker)
        elif state.side == "short" and (signal == 1 or price <= state.take_profit or price >= state.stop_loss):
            self._close_trade(symbol, state, price, regime, broker, performance_tracker)

    def _enter_long(self, state, qty, price, atr, tp_mult):
        state.side = "long"
        state.qty = qty
        state.entry_price = price
        state.stop_loss = price - atr * 2
        state.take_profit = price * tp_mult
        state.pyramid_layer = 1
        state.bars_held = 0
        state.max_favorable_excursion = None
        state.max_adverse_excursion = None

    def _enter_short(self, state, qty, price, atr, tp_mult):
        state.side = "short"
        state.qty = qty
        state.entry_price = price
        state.stop_loss = price + atr * 2
        state.take_profit = price * (2 - tp_mult)
        state.pyramid_layer = 1
        state.bars_held = 0
        state.max_favorable_excursion = None
        state.max_adverse_excursion = None

    def _close_trade(self, symbol, state, price, regime, broker, performance_tracker):
        if broker.submit_market_order(symbol, state.qty, "sell" if state.side == "long" else "buy"):
            performance_tracker.record_trade(symbol, state.strategy_name, regime, state.side, state.entry_price, price, state.qty, state.bars_held, state.max_favorable_excursion, state.max_adverse_excursion)
            state.side = None
            state.qty = 0
            state.pyramid_layer = 0
            state.bars_held = 0
            state.max_favorable_excursion = None
            state.max_adverse_excursion = None

def _map_side(side: str):
    return {"buy": OrderSide.BUY, "sell": OrderSide.SELL}[side.lower()]

class BrokerInterface(ABC):
    @abstractmethod
    def get_next_bar(self):
        """Retrieve the next bar of data (e.g., from stream or simulation)."""
        pass

    @abstractmethod
    def execute_trade(self, trade):
        """Execute a trade order with the broker."""
        pass

    @abstractmethod
    def get_account_info(self):
        """Fetch current account information (e.g., cash balance)."""
        pass

    @abstractmethod
    def get_open_position(self, symbol):
        """Get current position details for a given symbol."""
        pass

    @abstractmethod
    def close_position(self, symbol):
        """Close an open position for the given symbol."""
        pass

class OrderSideEnum(Enum):
    BUY = "buy"
    SELL = "sell"

def map_order_side(side: str, broker: str = "alpaca"):
    side_enum = OrderSideEnum(side.strip().lower())

    if broker == "alpaca":
        return {
            OrderSideEnum.BUY: AlpacaOrderSide.BUY,
            OrderSideEnum.SELL: AlpacaOrderSide.SELL,
        }[side_enum]

    raise NotImplementedError(f"Broker mapping not implemented for '{broker}'")

def _map_side(side: str) -> OrderSide:
    return OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

logger = logging()

class AlpacaBroker(BrokerInterface):
    def __init__(self, api_key, api_secret, paper=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.trading_client = None
        self.stream = None
        self.bar_callback = None

    def connect(self):
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
        self.stream = StockDataStream(self.api_key, self.api_secret)
        logger.info("✅ Connected to Alpaca.")

    async def start_stream(self):
        if self.stream:
            await self.stream.run()

    def subscribe_bars(self, callback, symbol):
        self.bar_callback = callback
        self.stream.subscribe_bars(callback, symbol)

    def get_next_bar(self):
        # This is handled async via callback — return None here or design a buffer queue
        return None

    def execute_trade(self, trade):
        return self.submit_market_order(
            symbol=trade["symbol"],
            qty=trade["qty"],
            side=trade["side"]
        )

    def get_account_info(self):
        return self.trading_client.get_account()

    def get_open_position(self, symbol):
        try:
            return self.trading_client.get_open_position(symbol)
        except Exception:
            return None

    def close_position(self, symbol):
        try:
            self.trading_client.close_position(symbol)
            return True
        except Exception as e:
            logger.warning(f"Failed to close position for {symbol}: {e}")
            return False

    def submit_market_order(self, symbol, qty, side: str):
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=_map_side(side),
                time_in_force=TimeInForce.DAY
            )
            self.trading_client.submit_order(order)
            return True
        except Exception as e:
            logger.error(f"Order failed for {symbol}: {e}")
            return False