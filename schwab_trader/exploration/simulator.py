#--------------------------------------------------------------------------------------------------------------------------#
# system level stuff to make sure we get the right root and can import the stuff we want
import sys
from pathlib import Path
project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#--------------------------------------------------------------------------------------------------------------------------#
# mock streamer emits bars to the system and we can hadle them via executor shows how we can deal with them just random changes in price
import asyncio
import nest_asyncio
import logging
import random
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, UTC
import math
from pytz import UTC
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from indicators.atr import ATRIndicator
from strategies.strategy_registry.stochastic_strategy import StochasticStrategy
from strategies.strategy_registry.rsi_strategy import RSIStrategy
from strategies.strategy_registry.macd_strategy import MACDStrategy
from strategies.strategy_registry.momentum_strategy import MomentumStrategy
from core.position_sizer import DynamicPositionSizer
from core.eventhandler import EventHandler

# ---------------------------- Logger Setup ----------------------------
# Create logs directory
os.makedirs("logs", exist_ok=True)

# Create timestamped log filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/simulation_{timestamp}.log"

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Capture all messages

# Clear existing handlers to avoid duplicate logs (especially in notebooks or reloads)
if logger.hasHandlers():
    logger.handlers.clear()

# Create file handler
file_handler = logging.FileHandler(log_filename, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
))

# Create console (stream) handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
))

# Add both handlers to root logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Optionally get named logger if needed
live_logger = logging.getLogger("SimRunner")
live_logger.debug("Logger initialized successfully.")

# ---------------------------- Historical Loader ------------------------------------------------------#
class HistoricalBarLoader:
    def __init__(self, path):
        self.path = path

    def load_last_n_bars(self, symbol: str, n=99):
        file_path = Path(self.path) / f"proc_{symbol}_file.json"
        if not file_path.exists():
            logger.warning(f"No historical file found for {symbol} at {file_path}")
            return []
        with open(file_path, "r") as f:
            data = json.load(f)
        return data[-n:]
    
    def get_latest_close_price(self, symbol: str) -> float:
        bars = self.load_last_n_bars(symbol, n=1)
        if bars:
            return bars[0].get("Close") or bars[0].get("close") or 300.0
        return 300.0  # fallback if missing
#--------------------------------------- Live Plotter -------------------------------------------------#
class LivePlotter:
    def __init__(self, symbols, window=100, draw_interval=5):
        self.symbols = symbols
        self.window = window

        self.price_buffers = {s: deque(maxlen=window) for s in symbols}
        self.timestamps = {s: deque(maxlen=window) for s in symbols}
        self.volume_buffers = {s: deque(maxlen=window) for s in symbols}
        self.signal_buffers = {s: deque(maxlen=window) for s in symbols}
        self.pnl_buffers = {s: deque(maxlen=window) for s in symbols}
        self.trade_markers = {s: [] for s in symbols}  # list of (timestamp, price, type)
        self.draw_interval = draw_interval

        plt.ion()
        self.fig, self.axes = plt.subplots(len(symbols), 1, figsize=(10, 3.5 * len(symbols)), sharex=True)
        if len(symbols) == 1:
            self.axes = [self.axes]

    def update_bar(self, symbol, bar):
        self.price_buffers[symbol].append(bar["close"])
        self.timestamps[symbol].append(bar["timestamp"])
        self.volume_buffers[symbol].append(bar.get("volume", 0))

    def record_signal(self, symbol, timestamp, signal):
        self.signal_buffers[symbol].append((timestamp, signal))

    def record_trade(self, symbol, timestamp, price, side):
        self.trade_markers[symbol].append((timestamp, price, side))  # side = "BUY" or "SELL"

    def record_pnl(self, symbol, timestamp, pnl):
        self.pnl_buffers[symbol].append((timestamp, pnl))

    def draw(self):
        self.frame_count += 1
        if self.frame_count % self.draw_interval != 0:
            return  # skip draw to reduce CPU load
        for i, symbol in enumerate(self.symbols):
            ax = self.axes[i]
            ax.clear()

            # Plot close price
            ax.plot(self.timestamps[symbol], self.price_buffers[symbol], label="Price", color="black")

            # Plot trades
            for ts, px, side in self.trade_markers[symbol]:
                if ts >= self.timestamps[symbol][0]:
                    color = "green" if side == "BUY" else "red"
                    ax.scatter(ts, px, color=color, marker="^" if side == "BUY" else "v", s=50, zorder=5)

            # Plot signals (optional triangles)
            for ts, sig in self.signal_buffers[symbol]:
                if ts >= self.timestamps[symbol][0]:
                    if sig == 1:
                        ax.plot(ts, self.price_buffers[symbol][-1], "^", color="blue", alpha=0.3)
                    elif sig == -1:
                        ax.plot(ts, self.price_buffers[symbol][-1], "v", color="blue", alpha=0.3)

            # Plot volume on secondary Y axis
            ax2 = ax.twinx()
            ax2.bar(self.timestamps[symbol], self.volume_buffers[symbol], color="gray", alpha=0.2, width=0.001)
            ax2.set_yticks([])

            # Plot cumulative PnL (optional)
            pnl_vals = self.pnl_buffers[symbol]
            if pnl_vals:
                pnl_ts, pnl_series = zip(*pnl_vals)
                ax.plot(pnl_ts, pnl_series, label="PnL", color="orange", linestyle="--")

            ax.set_title(symbol)
            ax.grid(True)
            ax.legend()

        self.fig.tight_layout()
        plt.pause(0.1)
# ---------------------------- GBM Simulator ----------------------------------------------------------#
class GBMSimulator:
    """
    Simulates Geometric Brownian Motion (GBM) with cyclical drift and occasional price shocks.
    Adds bounding logic to prevent extreme price drops that cause unrealistic drawdowns.
    """

    def __init__(self, symbols, base_price=300.0, log_prices=False):
        self.symbols = symbols
        self.price_state = {
            symbol: base_price[symbol] if isinstance(base_price, dict) else base_price + random.uniform(-50, 50)
            for symbol in symbols
        }
        self.volatility = {
            symbol: random.uniform(0.01, 0.03) for symbol in symbols
        }
        self.t = {symbol: 0 for symbol in symbols}
        self.cycle_length = 2000
        self.max_drift = 0.00005
        self.dt = 1 / 390  # 1-minute bar
        self.log_prices = log_prices
        self.logger = logging.getLogger("MockStream")
        self.logger.setLevel(logging.DEBUG)

        # Shock settings
        self.shock_probability = 0.002
        self.shock_magnitude_range = (0.01, 0.03)
        self.shock_vol_boost = 1.25

        # Guardrails
        self.max_bar_change_pct = 0.007  # 7% cap up/down
        self.max_log_return = 3  # Max Z-score to clamp log return
        self.min_price_floor = 1.00  # Never below $1.00

    def maybe_apply_shock(self, symbol, price):
        if random.random() < self.shock_probability:
            direction = random.choice([-1, 1])
            magnitude = random.uniform(*self.shock_magnitude_range)
            shock_factor = 1 + direction * magnitude
            shocked_price = max(self.min_price_floor, price * shock_factor)
            self.volatility[symbol] *= self.shock_vol_boost
            self.logger.warning(f"[{symbol}] *** PRICE SHOCK *** | {'UP' if direction > 0 else 'DOWN'} {magnitude:.2%} -> {shocked_price:.2f}")
            return shocked_price
        return price

    def generate_bar(self, symbol):
        prev_price = self.price_state[symbol]
        sigma = self.volatility[symbol]
        Z = np.random.normal(0, 1)

        # Clamp extreme Z to prevent tail blowout
        Z = np.clip(Z, -self.max_log_return, self.max_log_return)

        self.t[symbol] += 1
        mu = self.max_drift * math.sin(2 * math.pi * self.t[symbol] / self.cycle_length)

        # GBM step
        change = (mu - 0.5 * sigma ** 2) * self.dt + sigma * math.sqrt(self.dt) * Z
        raw_price = prev_price * math.exp(change)

        # Cap max percentage move
        capped_price = np.clip(raw_price, prev_price * (1 - self.max_bar_change_pct), prev_price * (1 + self.max_bar_change_pct))
        new_price = max(self.min_price_floor, capped_price)

        # Possibly apply a shock
        close = self.maybe_apply_shock(symbol, new_price)

        # OHLCV construction
        open_ = prev_price
        wick_range = abs(np.random.normal(0, 0.002))
        high = max(open_, close) * (1 + wick_range)
        low = min(open_, close) * (1 - wick_range)
        volume = random.randint(500, 5000)

        self.price_state[symbol] = close

        bar = {
            "timestamp": datetime.now(UTC),
            "symbol": symbol,
            "open": round(open_, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": volume
        }

        if self.log_prices:
            self.logger.info(f"[{symbol}] Bar - O: {bar['open']} | H: {bar['high']} | L: {bar['low']} | C: {bar['close']} | Vol: {bar['volume']}")

        return bar

    def update_all(self):
        return {symbol: self.generate_bar(symbol) for symbol in self.symbols}
# ---------------------------- Strategy Router --------------------------------------------------------#
class StrategyRouter:
    """
    Maps each stock symbol to its preferred trading strategy instance.
    Allows you to run multiple strategies concurrently across different stocks.
    """
    def __init__(self, default_strategy, custom_strategies=None):
        self.default_strategy = default_strategy
        self.custom_strategies = custom_strategies or {}  # e.g., {"AAPL": RSIStrategy(), "TSLA": StochasticStrategy()}

    def get_strategy(self, symbol):
        return self.custom_strategies.get(symbol, self.default_strategy)
# ---------------------------- Drawdown Monitor -------------------------------------------------------#
class DrawdownMonitor:
    """
    Tracks per-symbol drawdowns and disables trading if drawdown exceeds a threshold.
    Adds cooldown support after unlocking to prevent immediate re-entry.
    """

    def __init__(self, max_drawdown=0.35, cooldown_seconds=20):
        self.max_drawdown = max_drawdown
        self.cooldown_seconds = cooldown_seconds
        self.peak = {}
        self.locked = defaultdict(lambda: False)
        self.last_unlock_time = {}  # symbol → datetime
        self.logger = logging.getLogger("DrawdownMonitor")
        self.logger.setLevel(logging.DEBUG)

    def update(self, symbol, portfolio_value):
        now = datetime.now(UTC)

        if symbol not in self.peak:
            self.peak[symbol] = portfolio_value
            self.logger.debug(f"[InitPeak] {symbol} initialized at {portfolio_value:.2f}")
            return True

        # Cooldown logic: if symbol is unlocked but still in cooldown
        if not self.locked[symbol] and symbol in self.last_unlock_time:
            elapsed = (now - self.last_unlock_time[symbol]).total_seconds()
            if elapsed < self.cooldown_seconds:
                self.logger.warning(f"[COOLDOWN] {symbol} in cooldown ({elapsed:.1f}s elapsed). Trading still disabled.")
                return False  # block trade during cooldown

        # Recovery unlock logic
        if self.locked[symbol]:
            drawdown = (portfolio_value - self.peak[symbol]) / self.peak[symbol]
            if portfolio_value >= 0.85 * self.peak[symbol]:
                self.locked[symbol] = False
                self.last_unlock_time[symbol] = now
                self.logger.info(f"[UNLOCKED] {symbol} trading re-enabled after drawdown recovery. Cooldown started.")
                return False  # still in cooldown
            return False  # still locked

        # Update peak if new high
        if portfolio_value > self.peak[symbol]:
            self.peak[symbol] = portfolio_value

        # Drawdown check
        drawdown = max(((portfolio_value - self.peak[symbol]) / self.peak[symbol]), -1.0)
        if drawdown < -self.max_drawdown:
            self.locked[symbol] = True
            self.logger.warning(f"[LOCKED] {symbol} drawdown exceeded: {drawdown:.2%}")
            return False

        self.logger.debug(
            f"[DrawdownCheck] {symbol} | Peak: {self.peak[symbol]:.2f} | Current: {portfolio_value:.2f} | Drawdown: {drawdown:.2%} | Locked: {self.locked[symbol]}"
        )
        return True

    def is_locked(self, symbol):
        return self.locked[symbol]

    def is_in_cooldown(self, symbol):
        if symbol not in self.last_unlock_time:
            return False
        elapsed = (datetime.now(UTC) - self.last_unlock_time[symbol]).total_seconds()
        return elapsed < self.cooldown_seconds

    def reset(self, symbol):
        self.logger.info(f"[RESET] {symbol} lock and peak reset.")
        self.locked[symbol] = False
        self.peak[symbol] = 0.0
        if symbol in self.last_unlock_time:
            del self.last_unlock_time[symbol]

    def unlock(self, symbol):
        if self.locked[symbol]:
            self.locked[symbol] = False
            self.last_unlock_time[symbol] = datetime.now(UTC)
            self.logger.info(f"[UNLOCKED] {symbol} manually unlocked. Cooldown started.")

    def record_drawdown(self, symbol, drawdown):
        if self.locked[symbol]:
            return
        if drawdown < -self.max_drawdown:
            self.locked[symbol] = True
            self.logger.warning(f"[LOCKED] {symbol} drawdown exceeded: {drawdown:.2%}")
# ---------------------------- Mock Executor ----------------------------------------------------------#
class MockExecutor:
    def __init__(self):
        self.logger = logging.getLogger("MockExecutor")
        self.logger.setLevel(logging.DEBUG)
        self.peak_portfolio_value = defaultdict(lambda: 0)
        self.portfolio_history = defaultdict(list)
        self.cash = 100_000  # Shared pool across all symbols
        self.position = defaultdict(int)
        self.total_fees = defaultdict(float)
        self.trailing_stop = defaultdict(lambda: None)
        self.sizer = DynamicPositionSizer(risk_percentage=0.01)
        self.MIN_TRADE_QTY = 1
        self.MIN_FORCE_EXIT_QTY = 10
        self.MAX_PARTIAL_EXITS = 5
        self.partial_exit_counts = defaultdict(int)
        self.drawdown_monitor = DrawdownMonitor(max_drawdown=0.3)
        self.entry_prices = defaultdict(float)
        self.realized_pnl = defaultdict(float)
        self.unrealized_pnl = defaultdict(float)
        self.market_regimes = defaultdict(lambda: "unknown")
        self.last_price = {}
        self.take_profit = defaultdict(lambda: None)
        # Adaptive TP multipliers per regime
        self.TP_MULTIPLIERS = {
            "low_volatility": 1.03,
            "medium_volatility": 1.05,
            "high_volatility": 1.08,
            "unknown": 1.04
        }

    def log_trade_details(self, symbol, action_type, qty, price, sl=None, ts=None, fee=0.0, pnl=None, cash_before=None, cash_after=None, pos_before=None, pos_after=None, atr=None, regime=None):
        now = datetime.now(UTC)
        self.logger.info(
            f"[{now}] [{symbol}] {action_type} | "
            f"Qty: {qty:.1f} | "
            f"Price: ${price:.2f} | "
            f"{f'Stop Loss: ${sl:.2f} | ' if sl else ''}"
            f"{f'Trail Stop: ${ts:.2f} | ' if ts else ''}"
            f"Cash: ${cash_before:.2f} -> ${cash_after:.2f} | "
            f"Fee: ${fee:.2f} | "
            f"Pos: {pos_before:.1f} -> {pos_after:.1f} | "
            f"{f'PnL: ${pnl:.2f} | ' if pnl is not None else ''}"
            f"ATR: {atr:.2f} | "
            f"Regime: {regime}"
        )

    def force_exit_position(self, symbol, price, atr_value, stop_loss_price, market_conditions):
        qty = abs(self.position[symbol])
        if qty < self.MIN_TRADE_QTY:
            self.logger.debug(f"[{symbol}] FORCE EXIT SKIPPED: Qty={qty} < MIN_TRADE_QTY={self.MIN_TRADE_QTY}")
            return

        cash_before = self.cash
        pos_before = self.position[symbol]
        trade_fee = 0.001 * price * qty

        if self.position[symbol] > 0:
            proceeds = price * qty - trade_fee
            entry_price = stop_loss_price + (2 * atr_value)
            pnl = proceeds - (entry_price * qty)
            self.cash += proceeds
            self.position[symbol] = 0
            action = "FORCED SELL"
        else:
            cost = price * qty + trade_fee
            entry_price = stop_loss_price - (2 * atr_value)
            pnl = -(cost - (entry_price * qty))
            self.cash -= cost
            self.position[symbol] = 0
            action = "FORCED COVER"

        self.total_fees[symbol] += trade_fee
        self.realized_pnl[symbol] += pnl
        self.log_trade_details(symbol, action, qty, price, fee=trade_fee, cash_before=cash_before, cash_after=self.cash,
                               pos_before=pos_before, pos_after=0, pnl=pnl, atr=atr_value, regime=market_conditions)
        self.partial_exit_counts[symbol] = 0
    
    def can_trade(self, symbol, qty, price, max_alloc_pct=0.25, min_cash=1000):
        """
        Ensure a trade won't violate allocation or cash constraints.
        - max_alloc_pct: Max % of total portfolio allowed per position
        - min_cash: Minimum required cash to initiate new trades
        """
        position_value = abs(qty * price)

        # Total portfolio value = cash + all current positions (mark-to-market)
        total_value = self.cash + sum(
            abs(self.position[sym] * self.last_price.get(sym, 0.0))
            for sym in self.position
        )

        if total_value <= 0:
            self.logger.warning("[PORTFOLIO] Total value is zero or negative. Blocking all trades.")
            return False

        # Check allocation limit
        alloc_ratio = position_value / total_value
        if alloc_ratio > max_alloc_pct:
            self.logger.warning(
                f"[{symbol}] SKIPPED: Position would be {alloc_ratio:.2%} of portfolio "
                f"(limit {max_alloc_pct:.0%}). Value: ${position_value:.2f}, Total: ${total_value:.2f}"
            )
            return False

        # Check for minimum available cash
        if self.cash < min_cash:
            self.logger.warning(
                f"[{symbol}] SKIPPED: Not enough cash (${self.cash:.2f} < ${min_cash}) to enter new position."
            )
            return False

        return True

    def update_unrealized_pnl(self, latest_prices: dict):
        for symbol in self.positions:
            qty = self.positions[symbol]
            entry_price = self.entry_prices.get(symbol, 0)
            current_price = latest_prices.get(symbol, entry_price)
            
            if qty != 0:
                direction = 1 if qty > 0 else -1
                pnl = (current_price - entry_price) * abs(qty) * direction
                self.unrealized_pnl[symbol] = pnl
            else:
                self.unrealized_pnl[symbol] = 0.0
    
    def log_portfolio_status(self, symbol, price):
        self.last_price[symbol] = price
        pos = self.position[symbol]
        entry_price = self.entry_prices.get(symbol, price)

        # === Unrealized PnL ===
        if pos > 0:
            unrealized_pnl = (price - entry_price) * pos
        elif pos < 0:
            unrealized_pnl = (entry_price - price) * abs(pos)
        else:
            unrealized_pnl = 0.0
        self.unrealized_pnl[symbol] = unrealized_pnl

        # === Realized PnL ===
        realized = self.realized_pnl.get(symbol, 0.0)

        # === Portfolio Value (symbol-specific) ===
        portfolio_value = self.cash + abs(pos) * price

        # === Drawdown BEFORE updating peak ===
        prev_peak = self.peak_portfolio_value[symbol]
        drawdown = (portfolio_value - prev_peak) / prev_peak if prev_peak != 0 else 0.0
        self.peak_portfolio_value[symbol] = max(prev_peak, portfolio_value)

        # === Total portfolio value ===
        total_value = self.cash
        for sym, qty in self.position.items():
            current_price = self.last_price.get(sym, price)
            total_value += abs(qty) * current_price

        total_unrealized = sum(self.unrealized_pnl.values())
        total_realized = sum(self.realized_pnl.values())

        # === Record history ===
        self.portfolio_history[symbol].append({
            "Date": datetime.now(UTC),
            "Portfolio_Value": portfolio_value,
            "Cash": self.cash,
            "Position": pos,
            "Price": price,
            "Drawdown": drawdown,
            "Fees": self.total_fees[symbol],
            "Unrealized_PnL": unrealized_pnl,
            "Realized_PnL": realized,
            "Entry_Price": entry_price
        })

        # === Drawdown Monitoring ===
        self.drawdown_monitor.record_drawdown(symbol, drawdown)

        # === Unlock only if flat and recovered enough ===
        if self.drawdown_monitor.is_locked(symbol) and pos == 0:
            # Estimate what the account had when the drawdown was triggered
            peak = self.drawdown_monitor.peak.get(symbol, 0)
            recovery_threshold = 0.85 * peak

            # Approximate symbol-specific recovery using realized PnL
            last_exit_value = self.realized_pnl[symbol] + self.unrealized_pnl[symbol]
            recovered_value = self.cash + last_exit_value  # imperfect, but better than using global portfolio

            if recovered_value >= recovery_threshold:
                self.drawdown_monitor.unlock(symbol)
                self.logger.info(
                    f"[{symbol}] UNLOCKED: Recovered to ${portfolio_value:.2f} (>= 85% of peak ${peak:.2f} for {symbol})"
                )
        # === Logging ===
        self.logger.info(
            f"[{symbol}] STATUS | Port: ${portfolio_value:,.2f} | Total: ${total_value:,.2f} | "
            f"Cash: ${self.cash:,.2f} | Pos: {pos:.1f} | Px: ${price:.2f} | "
            f"U-PnL: ${unrealized_pnl:,.2f} | R-PnL: ${realized:,.2f} | DD: {drawdown:.2%} | "
            f"Total U: ${total_unrealized:,.2f} | Total R: ${total_realized:,.2f}"
        )

    def execute(self, symbol, df, signal, price, atr_value):
        if self.drawdown_monitor.is_locked(symbol):
            pos = self.position[symbol]

            # Try unlocking if flat and recovered
            self.log_portfolio_status(symbol, price)  # This now runs every time
            if self.drawdown_monitor.is_locked(symbol):  # still locked after check?
                if pos != 0:
                    self.logger.warning(f"[{symbol}] LOCKED: Force exiting open position due to drawdown.")
                    self.force_exit_position(
                        symbol=symbol,
                        price=price,
                        atr_value=atr_value,
                        stop_loss_price=price - atr_value * 2 if pos > 0 else price + atr_value * 2,
                        market_conditions=self.market_regimes.get(symbol, "unknown")
                    )
                else:
                    self.logger.warning(f"[{symbol}] SKIPPED: Trading is locked and no position is open.")
                return  #  Return only *after* trying to unlock
        # New cooldown enforcement (prevents re-entry after unlocking)
        if self.drawdown_monitor.is_in_cooldown(symbol):
            self.logger.warning(f"[{symbol}] SKIPPED: Cooldown period active. Trade blocked.")
            return
        # cash check
        if self.cash < 0:
            logging.warning("Capital exhausted. Trading suspended.")
            return

        exit_fraction = 0.5  # or whatever your partial exit ratio is
        market_conditions = self.market_regimes.get(symbol, "unknown")

        stop_loss_price = price - (atr_value * 2) if signal == 1 else price + (atr_value * 2)

        quantity = self.sizer.calculate_position_size(
            stock_price=price,
            stop_loss_price=stop_loss_price,
            current_cash=self.cash,
            market_conditions=market_conditions,
            signal=signal
        )

        trade_fee = 0.001 * price * quantity
        max_affordable_qty = int(self.cash // (price + trade_fee)) if (price + trade_fee) > 0 else 0

        if max_affordable_qty <= 0:
            self.logger.debug(f"[{symbol}] SKIPPED: Insufficient cash: ${self.cash:.2f}")
            return

        quantity = min(quantity, max_affordable_qty)
        quantity = max(1, quantity)  # Ensure at least 1 if affordable

        if quantity < self.MIN_TRADE_QTY:
            self.logger.debug(f"[{symbol}] SKIPPED: Qty={quantity} < MIN_TRADE_QTY={self.MIN_TRADE_QTY}")
            return

        if quantity < self.MIN_TRADE_QTY:
            self.logger.debug(f"[{symbol}] SKIPPED: Qty={quantity:.0f} < MIN_TRADE_QTY={self.MIN_TRADE_QTY}")
            return

        now = datetime.now(UTC)
        cash_before = self.cash
        pos_before = self.position[symbol]

        # === OPEN LONG ===
        if signal == 1 and self.position[symbol] == 0 and self.can_trade(symbol=symbol, qty=quantity, price=price):
            self.cash -= (price * quantity + trade_fee)
            self.position[symbol] += quantity
            self.total_fees[symbol] += trade_fee
            self.trailing_stop[symbol] = price * 0.97
            self.partial_exit_counts[symbol] = 0
            self.entry_prices[symbol] = price

            tp_mult = self.TP_MULTIPLIERS.get(market_conditions, 1.05)
            self.take_profit[symbol] = price * tp_mult

            self.log_trade_details(symbol, "BUY", quantity, price, sl=stop_loss_price, ts=self.trailing_stop[symbol],
                                fee=trade_fee, cash_before=cash_before, cash_after=self.cash,
                                pos_before=pos_before, pos_after=self.position[symbol],
                                atr=atr_value, regime=market_conditions)

        # === OPEN SHORT ===
        elif signal == -1 and self.position[symbol] == 0 and self.can_trade(symbol=symbol, qty=quantity, price=price):
            self.cash += (price * quantity - trade_fee)
            self.position[symbol] -= quantity
            self.total_fees[symbol] += trade_fee
            self.trailing_stop[symbol] = price * 1.03
            self.partial_exit_counts[symbol] = 0
            self.entry_prices[symbol] = price

            tp_mult = self.TP_MULTIPLIERS.get(market_conditions, 1.05)
            self.take_profit[symbol] = price * (2 - tp_mult)

            self.log_trade_details(symbol, "SHORT SELL", quantity, price, sl=stop_loss_price, ts=self.trailing_stop[symbol],
                                fee=trade_fee, cash_before=cash_before, cash_after=self.cash,
                                pos_before=pos_before, pos_after=self.position[symbol],
                                atr=atr_value, regime=market_conditions)

        # === TAKE PROFIT HIT - LONG ===
        if self.position[symbol] > 0 and self.take_profit[symbol] and price >= self.take_profit[symbol]:
            qty = int(self.position[symbol] * exit_fraction)
            if qty >= self.MIN_TRADE_QTY:
                trade_fee = 0.001 * price * qty
                entry_price = self.entry_prices[symbol]
                pnl = (price - entry_price) * qty - trade_fee

                self.cash += price * qty - trade_fee
                self.position[symbol] -= qty
                self.total_fees[symbol] += trade_fee
                self.partial_exit_counts[symbol] += 1
                self.realized_pnl[symbol] += pnl

                self.log_trade_details(symbol, "TAKE PROFIT - SELL (PARTIAL)", qty, price,
                                    fee=trade_fee, cash_before=cash_before, cash_after=self.cash,
                                    pos_before=pos_before, pos_after=self.position[symbol],
                                    pnl=pnl, atr=atr_value, regime=market_conditions)

                if abs(self.position[symbol]) <= self.MIN_FORCE_EXIT_QTY or self.partial_exit_counts[symbol] >= self.MAX_PARTIAL_EXITS:
                    self.force_exit_position(symbol, price, atr_value, stop_loss_price, market_conditions)

        # === TAKE PROFIT HIT - SHORT ===
        elif self.position[symbol] < 0 and self.take_profit[symbol] and price <= self.take_profit[symbol]:
            qty = int(abs(self.position[symbol]) * exit_fraction)
            if qty >= self.MIN_TRADE_QTY:
                trade_fee = 0.001 * price * qty
                entry_price = self.entry_prices[symbol]
                pnl = (entry_price - price) * qty - trade_fee

                self.cash -= price * qty + trade_fee
                self.position[symbol] += qty
                self.total_fees[symbol] += trade_fee
                self.partial_exit_counts[symbol] += 1
                self.realized_pnl[symbol] += pnl

                self.log_trade_details(symbol, "TAKE PROFIT - COVER SHORT (PARTIAL)", qty, price,
                                    fee=trade_fee, cash_before=cash_before, cash_after=self.cash,
                                    pos_before=pos_before, pos_after=self.position[symbol],
                                    pnl=pnl, atr=atr_value, regime=market_conditions)

                if abs(self.position[symbol]) <= self.MIN_FORCE_EXIT_QTY or self.partial_exit_counts[symbol] >= self.MAX_PARTIAL_EXITS:
                    self.force_exit_position(symbol, price, atr_value, stop_loss_price, market_conditions)

        # === SIGNAL FLIP - PARTIAL EXIT ===
        elif (signal == -1 and self.position[symbol] > 0) or (signal == 1 and self.position[symbol] < 0):
            qty = int(abs(self.position[symbol]) * exit_fraction)
            if qty < self.MIN_TRADE_QTY:
                self.logger.debug(f"[{symbol}] SKIPPED: Partial exit qty={qty} < MIN_TRADE_QTY={self.MIN_TRADE_QTY}")
                return

            trade_fee = 0.001 * price * qty
            entry_price = self.entry_prices[symbol]
            pnl = (price - entry_price) * qty - trade_fee if self.position[symbol] > 0 else (entry_price - price) * qty - trade_fee

            self.cash += price * qty - trade_fee if self.position[symbol] > 0 else - (price * qty + trade_fee)
            self.position[symbol] += qty if self.position[symbol] < 0 else -qty
            self.total_fees[symbol] += trade_fee
            self.partial_exit_counts[symbol] += 1
            self.realized_pnl[symbol] += pnl

            action = "SELL (PARTIAL)" if self.position[symbol] > 0 else "COVER SHORT (PARTIAL)"
            self.log_trade_details(symbol, action, qty, price, fee=trade_fee,
                                cash_before=cash_before, cash_after=self.cash,
                                pos_before=pos_before, pos_after=self.position[symbol],
                                pnl=pnl, atr=atr_value, regime=market_conditions)

            if abs(self.position[symbol]) <= self.MIN_FORCE_EXIT_QTY or self.partial_exit_counts[symbol] >= self.MAX_PARTIAL_EXITS:
                self.force_exit_position(symbol, price, atr_value, stop_loss_price, market_conditions)

        # === TRAILING STOP HIT - LONG ===
        elif self.position[symbol] > 0 and price < self.trailing_stop[symbol]:
            qty = int(self.position[symbol] * exit_fraction)
            if qty < self.MIN_TRADE_QTY:
                return

            trade_fee = 0.001 * price * qty
            entry_price = self.entry_prices[symbol]
            pnl = (price - entry_price) * qty - trade_fee

            self.cash += price * qty - trade_fee
            self.position[symbol] -= qty
            self.total_fees[symbol] += trade_fee
            self.partial_exit_counts[symbol] += 1
            self.realized_pnl[symbol] += pnl

            self.log_trade_details(symbol, "TRAIL STOP HIT - SELL (PARTIAL)", qty, price, ts=self.trailing_stop[symbol],
                                fee=trade_fee, cash_before=cash_before, cash_after=self.cash,
                                pos_before=pos_before, pos_after=self.position[symbol],
                                pnl=pnl, atr=atr_value, regime=market_conditions)

            if abs(self.position[symbol]) <= self.MIN_FORCE_EXIT_QTY or self.partial_exit_counts[symbol] >= self.MAX_PARTIAL_EXITS:
                self.force_exit_position(symbol, price, atr_value, stop_loss_price, market_conditions)

        # === TRAILING STOP HIT - SHORT ===
        elif self.position[symbol] < 0 and price > self.trailing_stop[symbol]:
            qty = int(abs(self.position[symbol]) * exit_fraction)
            if qty < self.MIN_TRADE_QTY:
                return

            trade_fee = 0.001 * price * qty
            entry_price = self.entry_prices[symbol]
            pnl = (entry_price - price) * qty - trade_fee

            self.cash -= price * qty + trade_fee
            self.position[symbol] += qty
            self.total_fees[symbol] += trade_fee
            self.partial_exit_counts[symbol] += 1
            self.realized_pnl[symbol] += pnl

            self.log_trade_details(symbol, "TRAIL STOP HIT - COVER (PARTIAL)", qty, price, ts=self.trailing_stop[symbol],
                                fee=trade_fee, cash_before=cash_before, cash_after=self.cash,
                                pos_before=pos_before, pos_after=self.position[symbol],
                                pnl=pnl, atr=atr_value, regime=market_conditions)

            if abs(self.position[symbol]) <= self.MIN_FORCE_EXIT_QTY or self.partial_exit_counts[symbol] >= self.MAX_PARTIAL_EXITS:
                self.force_exit_position(symbol, price, atr_value, stop_loss_price, market_conditions)
        
        # === PYRAMIDING - Add to winning position ===
        max_pyramid_multiplier = 2  # Allow up to 2x original entry size
        if signal == 1 and self.position[symbol] > 0:
            # Block pyramiding if in drawdown lock or cooldown
            if self.drawdown_monitor.is_locked(symbol) or self.drawdown_monitor.is_in_cooldown(symbol):
                self.logger.debug(f"[{symbol}] SKIPPED: Pyramiding blocked due to lock or cooldown.")
                return
            entry_price = self.entry_prices[symbol]
            if price > entry_price * 1.01 and self.position[symbol] < max_pyramid_multiplier * quantity:
                # Reinvest (pyramid)
                reinvest_qty = int(quantity * 0.5)
                if reinvest_qty >= self.MIN_TRADE_QTY and self.can_trade(symbol, reinvest_qty, price):
                    reinvest_fee = 0.001 * price * reinvest_qty
                    total_cost = price * reinvest_qty + reinvest_fee
                    if self.cash >= total_cost:
                        self.cash -= total_cost
                        self.position[symbol] += reinvest_qty
                        self.total_fees[symbol] += reinvest_fee
                        self.log_trade_details(symbol, "PYRAMID BUY", reinvest_qty, price,
                            fee=reinvest_fee, cash_before=cash_before, cash_after=self.cash,
                            pos_before=pos_before, pos_after=self.position[symbol],
                            atr=atr_value, regime=market_conditions)
        
        # === Update Trailing Stop ===
        if self.position[symbol] > 0:
            self.trailing_stop[symbol] = max(self.trailing_stop[symbol], price * 0.97)
        elif self.position[symbol] < 0:
            self.trailing_stop[symbol] = min(self.trailing_stop[symbol], price * 1.03)
        self.market_regimes[symbol] = market_conditions
        self.log_portfolio_status(symbol, price)
# ---------------------------- Event Handler ----------------------------------------------------------#
class Event:
    def __init__(self, name, payload):
        self.name = name
        self.payload = payload
class EventHandler:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventHandler, cls).__new__(cls)
            cls._instance.listeners = defaultdict(list)
            cls._instance.bar_windows = defaultdict(lambda: deque(maxlen=100))
            cls._instance.current_day_bar = defaultdict(lambda: None)
            cls._instance.logger = logging.getLogger("EventHandler")
        return cls._instance

    def subscribe(self, event_name, callback):
        self.logger.debug(f"Subscribed to '{event_name}'")
        self.listeners[event_name].append(callback)

    def emit(self, event_name, payload):
        event = Event(event_name, payload)
        self.logger.debug(f"Emitting '{event_name}' with payload: {payload}")
        for callback in self.listeners[event_name]:
            callback(event)
#----------------------------- classify regime --------------------------------------------------------#
def classify_regime(price: float, atr: float) -> str:
    ratio = atr / price
    if ratio < 0.01:
        return "low_volatility"
    elif ratio > 0.03:
        return "high_volatility"
    else:
        return "medium_volatility"
# ---------------------------- Strategy Callback ------------------------------------------------------#
def strategy_on_bar(event, strategy_map, executor, handler, plotter: LivePlotter = None):
    """
    Handles incoming bars for a symbol, updates the intraday aggregate bar,
    computes indicators, and executes trades based on strategy signals.

    Supports per-symbol strategy from the strategy_map.
    """
    data = event.payload
    symbol = data["symbol"]
    bar = data["bar"]

    # Check if the symbol has an assigned strategy
    if symbol not in strategy_map:
        executor.logger.warning(f"[{symbol}] No strategy configured. Skipping.")
        return

    strategy = strategy_map[symbol]

    # Check if we need to start a new intraday bar
    current = handler.current_day_bar[symbol]
    bar_date = bar["timestamp"].date()

    if current is None or current["Date"].date() != bar_date:
        if current:
            handler.bar_windows[symbol].append(current)
        handler.current_day_bar[symbol] = {
            "Date": bar["timestamp"],
            "Open": bar["open"],
            "High": bar["high"],
            "Low": bar["low"],
            "Close": bar["close"],
            "Volume": bar["volume"]
        }
    else:
        # Update intraday bar
        cb = handler.current_day_bar[symbol]
        cb["High"] = max(cb["High"], bar["high"])
        cb["Low"] = min(cb["Low"], bar["low"])
        cb["Close"] = bar["close"]
        cb["Volume"] += bar["volume"]

    # Combine bars into DataFrame for indicator computation
    full_window = list(handler.bar_windows[symbol]) + [handler.current_day_bar[symbol]]
    df = pd.DataFrame(full_window)

    if len(df) < 20:
        return  # Not enough data

    df = ATRIndicator(df).compute()
    df = strategy.generate_signal(df)

    latest = df.iloc[-1]
    signal = latest.get("Signal", None)
    price = latest["Close"]
    atr = latest["ATR"]
    if atr is not None and price > 0:
        regime = classify_regime(price, atr)
    else:
        regime = "unknown"

    executor.market_regimes[symbol] = regime
    if plotter:
        plotter.record_signal(symbol, bar["timestamp"], bar.get("Signal", 0))
    
    
    # ⬇️ Always log portfolio status — even if there's no trade signal
    #executor.log_portfolio_status(symbol, price)

    #executor.logger.debug(f"[DEBUG] {symbol} strategy output:\n{df.tail(3)}")
    executor.execute(symbol, df, signal, price, atr)
# ---------------------------- Main Bootstrap ---------------------------------------------------------#
if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "TSLA", "AMD", "META", "AMZN"]
    hist_path = r'/home/kwasi/Projects/trader/schwab_trader/data/data_storage/proc_data'

    # Setup components
    #drawdown_monitor = DrawdownMonitor(max_drawdown=0.20)
    executor = MockExecutor()
    handler = EventHandler()
    loader = HistoricalBarLoader(hist_path)
    plotter = LivePlotter(symbols, window=100)
    last_prices = {
            symbol: loader.get_latest_close_price(symbol) for symbol in symbols
        }

    # Define strategies per symbol
    strategy_map = {
        "AAPL": MomentumStrategy(),
        "MSFT": MomentumStrategy(),
        "TSLA": MomentumStrategy(),
        "AMD": MomentumStrategy(),
        "META": MomentumStrategy(),
        "AMZN": MomentumStrategy()
    }

    # Load historical bars
    for symbol in symbols:
        history = loader.load_last_n_bars(symbol, n=99)
        for bar in history:
            bar["Date"] = pd.to_datetime(bar["Date"], unit='ms') if isinstance(bar["Date"], (int, float)) else pd.to_datetime(bar["Date"])
            handler.bar_windows[symbol].append(bar)
    
    handler.subscribe("BAR_CREATED", lambda event: strategy_on_bar(event, strategy_map, executor, handler, plotter))

    # Simulate bar emission (to be replaced with actual streamer)

    async def mock_stream(symbols, handler: EventHandler, base_price=300.0, interval_sec=0.5, plotter: LivePlotter = None):
        """
        Asynchronously emits synthetic OHLC bars using GBMSimulator.
        Includes realistic drift, volatility, and price shocks.
        """
        logger = logging.getLogger("MockStream")
        logger.setLevel(logging.DEBUG)

        sim = GBMSimulator(symbols, base_price=last_prices, log_prices=True)

        while True:
            bars = sim.update_all()
            for symbol, bar in bars.items():
                handler.emit("BAR_CREATED", {"symbol": symbol, "bar": bar})
                if plotter:
                    plotter.update_bar(symbol, bar)
            if plotter:
                plotter.draw()
            await asyncio.sleep(interval_sec)


    nest_asyncio.apply()
    asyncio.run(mock_stream(symbols, handler))