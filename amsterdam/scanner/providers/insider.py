"""
Insider activity data provider — stub for future implementation.
"""
from __future__ import annotations

from scanner.providers.base import BaseDataProvider, ProviderData


class InsiderProvider(BaseDataProvider):
    """Stub — will integrate with insider transaction data APIs."""

    @property
    def name(self) -> str:
        return "insider"

    def fetch(self, symbol: str) -> ProviderData:
        return ProviderData(symbol=symbol, provider_name=self.name)

    def is_available(self) -> bool:
        return False
