import asyncio
import math
import random
from datetime import UTC, datetime

import numpy as np

from core.contracts.events import EVENT_NEW_BAR, EVENT_PRICE_UPDATE
from core.events.eventhandler import get_event_handler
from loggers.logger import Logger


class GBMSimulator:
    """
    Geometric Brownian Motion (GBM) simulator with multi-scale drift:
    - Fast oscillations for short-term sentiment
    - Slow bias cycles for regime shifts (bullish/bearish phases)
    - Occasional random price shocks for realism
    """

    def __init__(
        self,
        symbols,
        base_price=300.0,
        log_prices=False,
        *,
        drift_scale: float = 1.0,
        vol_scale: float = 1.0,
        dt: float = 1 / 390,
        bias_period: int = 10_000,  # how fast the regime bias oscillates
        bias_strength: float = 1.0,  # amplitude of bias drift
    ):
        self.symbols = symbols
        self.price_state = {
            s: base_price[s] if isinstance(base_price, dict) else base_price + random.uniform(-50, 50) for s in symbols
        }

        self.volatility = {s: random.uniform(0.01, 0.03) * vol_scale for s in symbols}
        self.t = {s: 0 for s in symbols}

        # --- Drift & noise scales ---
        self.cycle_length = 2000  # fast oscillation (intraday)
        self.max_drift = 0.00005 * drift_scale
        self.bias_period = bias_period  # long-term regime cycle
        self.bias_strength = bias_strength  # amplitude for oscillating bias
        self.dt = dt
        self.log_prices = log_prices
        # Logger - own file with propagation to app.log
        self.logger = Logger(log_file="gbm_simulator.log", logger_name="GBMSimulator", propagate=True).get_logger()

        self.logger.info(f"GBMSimulator initialized for symbols: {symbols}")

        # --- Shocks ---
        self.shock_probability = 0.002
        self.shock_magnitude_range = (0.01, 0.03)
        self.shock_vol_boost = 1.25

        # --- Guardrails ---
        self.max_bar_change_pct = 0.007
        self.max_log_return = 3
        self.min_price_floor = 1.0
        self.bus = get_event_handler()

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _apply_shock(self, symbol: str, price: float) -> float:
        if random.random() < self.shock_probability:
            direction = random.choice([-1, 1])
            magnitude = random.uniform(*self.shock_magnitude_range)
            shocked_price = max(self.min_price_floor, price * (1 + direction * magnitude))
            self.volatility[symbol] *= self.shock_vol_boost
            self.logger.warning(
                f"[{symbol}] *** SHOCK *** {'UP' if direction > 0 else 'DOWN'} {magnitude:.1%} -> {shocked_price:.2f}"
            )
            return shocked_price
        return price

    # ------------------------------------------------------------
    # Core bar generation
    # ------------------------------------------------------------
    def generate_bar(self, symbol: str) -> dict:
        prev_price = self.price_state[symbol]
        sigma = self.volatility[symbol]
        self.t[symbol] += 1

        # --- Multi-scale drift model ---
        # Short-term cyclical drift (fast oscillation)
        fast_mu = self.max_drift * math.sin(2 * math.pi * self.t[symbol] / self.cycle_length)

        # Long-term oscillating bias (slow regime shifts)
        bias_mu = self.max_drift * self.bias_strength * math.sin(2 * math.pi * self.t[symbol] / self.bias_period)

        # Add small random local drift for micro variation
        noise_mu = np.random.normal(0, self.max_drift * 0.5)

        # Total drift
        mu = fast_mu + bias_mu + noise_mu

        # --- Random Brownian increment ---
        Z = np.random.normal(0, 1)
        Z = np.clip(Z, -self.max_log_return, self.max_log_return)

        # --- GBM step ---
        log_return = (mu - 0.5 * sigma**2) * self.dt + sigma * math.sqrt(self.dt) * Z
        new_price = prev_price * math.exp(log_return)

        # Cap unrealistic jumps
        new_price = np.clip(
            new_price,
            prev_price * (1 - self.max_bar_change_pct),
            prev_price * (1 + self.max_bar_change_pct),
        )
        new_price = max(self.min_price_floor, float(new_price))

        # Possibly apply a shock
        close = self._apply_shock(symbol, new_price)

        # OHLCV generation
        open_ = prev_price
        mid = (open_ + close) / 2
        wick_frac = abs(np.random.normal(0, 0.0015))
        high = max(open_, close, mid * (1 + wick_frac))
        low = min(open_, close, mid * (1 - wick_frac))
        volume = int(abs(np.random.normal(3000, 800)))

        self.price_state[symbol] = close

        bar = {
            "timestamp": datetime.now(UTC),
            "symbol": symbol,
            "open": round(open_, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": volume,
        }
        payload = {
            "symbol": symbol,
            "price": bar["close"],
            "timestamp": bar["timestamp"].isoformat(),
        }
        barpayload = {
            "symbol": symbol,
            "timestamp": bar["timestamp"].isoformat(),
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": bar["close"],
            "volume": bar["volume"],
        }

        # Position update (for dashboard positions)
        # Schedule asynchronously so we don’t block bar generation
        asyncio.create_task(self.bus.emit(EVENT_PRICE_UPDATE, payload))
        asyncio.create_task(self.bus.emit(EVENT_NEW_BAR, barpayload))

        if self.log_prices:
            self.logger.info(
                f"[{symbol}] Bar - O:{bar['open']} | H:{bar['high']} | "
                f"L:{bar['low']} | C:{bar['close']} | Vol:{bar['volume']}"
            )

        return bar

    def update_all(self) -> dict:
        """Generate new bars for all tracked symbols."""
        return {sym: self.generate_bar(sym) for sym in self.symbols}
