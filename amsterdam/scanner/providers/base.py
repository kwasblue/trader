"""
Abstract base class for scanner data providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProviderData:
    """Standardized output from a data provider."""

    symbol: str
    provider_name: str
    data: dict[str, Any] = field(default_factory=dict)
    dataframe: pd.DataFrame | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseDataProvider(ABC):
    """Abstract base for all scanner data providers."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fetch(self, symbol: str) -> ProviderData: ...

    def fetch_batch(self, symbols: list[str]) -> dict[str, ProviderData]:
        """Fetch data for multiple symbols. Override for concurrent implementations."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch(symbol)
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to fetch {symbol}: {e}")
        return results

    @abstractmethod
    def is_available(self) -> bool: ...
