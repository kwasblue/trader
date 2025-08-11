from loggers.logger import Logger
import random
import numpy as np
import math
from datetime import datetime, UTC


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
        self.logger = Logger(log_file = 'app.log',logger_name='GBM Simulator')

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