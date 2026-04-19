"""
Fundamental data provider — stub for future implementation.
"""

from __future__ import annotations

from scanner.providers.base import BaseDataProvider, ProviderData


class FundamentalProvider(BaseDataProvider):
    """Stub — will integrate with yfinance or similar for P/E, market cap, earnings."""

    @property
    def name(self) -> str:
        return "fundamental"

    def fetch(self, symbol: str) -> ProviderData:
        return ProviderData(symbol=symbol, provider_name=self.name)

    def is_available(self) -> bool:
        return False
