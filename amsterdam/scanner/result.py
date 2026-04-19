"""
Scanner result dataclasses for stock screening output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SymbolScore:
    """Scoring output for a single symbol."""

    symbol: str
    total_score: float  # 0.0 - 1.0 weighted composite
    criteria_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    recommendation: str = "skip"  # "trade", "watch", or "skip"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScanReport:
    """Full scan result containing ranked symbol scores."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    universe_size: int = 0
    results: list[SymbolScore] = field(default_factory=list)
    criteria_used: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def top_n(self, n: int) -> list[SymbolScore]:
        return self.results[:n]

    def trade_candidates(self) -> list[str]:
        return [r.symbol for r in self.results if r.recommendation == "trade"]

    def watch_candidates(self) -> list[str]:
        return [r.symbol for r in self.results if r.recommendation == "watch"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "universe_size": self.universe_size,
            "results": [r.to_dict() for r in self.results],
            "criteria_used": self.criteria_used,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }
