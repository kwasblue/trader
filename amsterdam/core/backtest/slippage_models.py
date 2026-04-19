"""
Slippage Models

Provides various slippage models for realistic trade execution simulation.
"""

import numpy as np
from typing import Optional


class SlippageModel:
    """Base class for slippage models."""

    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        side: str,  # 'buy' or 'sell'
        volume: float = None,
        volatility: float = None
    ) -> float:
        """Return the execution price after slippage."""
        raise NotImplementedError


class FixedSlippage(SlippageModel):
    """Fixed percentage slippage."""

    def __init__(self, slippage_pct: float = 0.001):
        self.slippage_pct = slippage_pct

    def calculate_slippage(self, price, quantity, side, volume=None, volatility=None):
        if side == 'buy':
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)


class RandomSlippage(SlippageModel):
    """Random slippage within a range."""

    def __init__(self, min_pct: float = -0.001, max_pct: float = 0.001):
        self.min_pct = min_pct
        self.max_pct = max_pct

    def calculate_slippage(self, price, quantity, side, volume=None, volatility=None):
        slippage = np.random.uniform(self.min_pct, self.max_pct)
        if side == 'buy':
            slippage = abs(slippage)  # Always adverse for buys
        else:
            slippage = -abs(slippage)  # Always adverse for sells
        return price * (1 + slippage)


class VolumeBasedSlippage(SlippageModel):
    """
    Slippage based on order size relative to volume.

    Larger orders relative to volume have more slippage.
    """

    def __init__(
        self,
        base_slippage: float = 0.0001,
        volume_impact: float = 0.1,
        max_slippage: float = 0.02
    ):
        self.base_slippage = base_slippage
        self.volume_impact = volume_impact
        self.max_slippage = max_slippage

    def calculate_slippage(self, price, quantity, side, volume=None, volatility=None):
        if volume is None or volume <= 0:
            volume = quantity * 100  # Assume we're 1% of volume

        # Order size as fraction of volume
        participation = quantity / volume

        # Slippage increases with participation rate
        slippage = self.base_slippage + (participation * self.volume_impact)
        slippage = min(slippage, self.max_slippage)

        if side == 'buy':
            return price * (1 + slippage)
        return price * (1 - slippage)


class VolatilityAdjustedSlippage(SlippageModel):
    """
    Slippage adjusted for volatility.

    Higher volatility = more slippage.
    """

    def __init__(
        self,
        base_slippage: float = 0.0005,
        volatility_multiplier: float = 2.0,
        max_slippage: float = 0.03
    ):
        self.base_slippage = base_slippage
        self.volatility_multiplier = volatility_multiplier
        self.max_slippage = max_slippage

    def calculate_slippage(self, price, quantity, side, volume=None, volatility=None):
        if volatility is None:
            volatility = 0.02  # Default 2% daily volatility

        # Slippage scales with volatility
        slippage = self.base_slippage + (volatility * self.volatility_multiplier / 100)
        slippage = min(slippage, self.max_slippage)

        if side == 'buy':
            return price * (1 + slippage)
        return price * (1 - slippage)


class CompositeSlippage(SlippageModel):
    """
    Combines multiple slippage models.

    Useful for creating more realistic slippage by combining
    volume-based and volatility-based effects.
    """

    def __init__(self, models: list, weights: list = None):
        """
        Args:
            models: List of SlippageModel instances
            weights: Optional weights for each model (default: equal)
        """
        self.models = models
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights

    def calculate_slippage(self, price, quantity, side, volume=None, volatility=None):
        total_slippage_pct = 0.0

        for model, weight in zip(self.models, self.weights):
            exec_price = model.calculate_slippage(price, quantity, side, volume, volatility)
            slippage_pct = (exec_price - price) / price
            total_slippage_pct += slippage_pct * weight

        return price * (1 + total_slippage_pct)
