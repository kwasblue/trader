"""
Sentiment data provider — stub for future implementation.
"""

from __future__ import annotations

from scanner.providers.base import BaseDataProvider, ProviderData


class SentimentProvider(BaseDataProvider):
    """Stub — will integrate with news/sentiment APIs."""

    @property
    def name(self) -> str:
        return "sentiment"

    def fetch(self, symbol: str) -> ProviderData:
        return ProviderData(symbol=symbol, provider_name=self.name)

    def is_available(self) -> bool:
        return False
