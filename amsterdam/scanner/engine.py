"""
ScannerEngine — orchestrates the stock screening pipeline.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from scanner.result import ScanReport, SymbolScore
from scanner.scorer import SymbolScorer
from scanner.universe import get_universe
from scanner.providers.base import BaseDataProvider, ProviderData
from scanner.providers.technical import TechnicalProvider
from scanner.providers.fundamental import FundamentalProvider
from scanner.providers.sentiment import SentimentProvider
from scanner.providers.insider import InsiderProvider
from scanner.criteria.base import BaseCriterion, create_criterion

logger = logging.getLogger(__name__)


class ScannerEngine:
    """Central orchestrator for the stock scanning pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._providers: Dict[str, BaseDataProvider] = {}
        self._criteria: List[BaseCriterion] = []
        self._scorer: Optional[SymbolScorer] = None
        self._last_report: Optional[ScanReport] = None

        self._init_providers()
        self._init_criteria()
        self._init_scorer()

    def _init_providers(self):
        providers_cfg = self._config.get("providers", {})
        lookback = providers_cfg.get("lookback_bars", 200)

        if providers_cfg.get("technical_enabled", True):
            tech = TechnicalProvider(lookback_bars=lookback)
            if tech.is_available():
                self._providers["technical"] = tech
            else:
                logger.warning("Technical provider data path not found")

        if providers_cfg.get("fundamental_enabled", False):
            fund = FundamentalProvider()
            if fund.is_available():
                self._providers["fundamental"] = fund

        if providers_cfg.get("sentiment_enabled", False):
            sent = SentimentProvider()
            if sent.is_available():
                self._providers["sentiment"] = sent

        if providers_cfg.get("insider_enabled", False):
            ins = InsiderProvider()
            if ins.is_available():
                self._providers["insider"] = ins

        logger.info(f"Scanner providers: {list(self._providers.keys())}")

    def _init_criteria(self):
        # Import criteria modules to trigger registration
        import scanner.criteria.momentum  # noqa: F401
        import scanner.criteria.trend  # noqa: F401

        criteria_configs = self._config.get("criteria", [
            {"name": "momentum_breakout", "params": {}},
            {"name": "relative_strength", "params": {"benchmark": "SPY", "period": 20}},
            {"name": "volume_surge", "params": {"surge_threshold": 1.5}},
            {"name": "trend_following", "params": {}},
        ])

        for cfg in criteria_configs:
            try:
                criterion = create_criterion(cfg)
                self._criteria.append(criterion)
            except ValueError as e:
                logger.warning(f"Skipping criterion: {e}")

        logger.info(f"Scanner criteria: {[c.name for c in self._criteria]}")

    def _init_scorer(self):
        scoring_cfg = self._config.get("scoring", {})
        weights = scoring_cfg.get("weights", {
            "momentum_breakout": 0.30,
            "relative_strength": 0.25,
            "volume_surge": 0.20,
            "trend_following": 0.25,
        })
        self._scorer = SymbolScorer(
            weights=weights,
            trade_threshold=scoring_cfg.get("trade_threshold", 0.7),
            watch_threshold=scoring_cfg.get("watch_threshold", 0.4),
            max_trade_symbols=scoring_cfg.get("max_trade_symbols", 10),
            max_watch_symbols=scoring_cfg.get("max_watch_symbols", 20),
        )

    def scan(
        self,
        symbols: Optional[List[str]] = None,
        update_lists: bool = False,
    ) -> ScanReport:
        """
        Run the full scanning pipeline.

        Args:
            symbols: Override universe with specific symbols
            update_lists: If True, update SymbolListManager trade/watch lists

        Returns:
            ScanReport with ranked results
        """
        start_time = time.time()
        errors: List[str] = []

        # 1. Resolve universe
        if symbols is not None:
            universe = [s.upper().strip() for s in symbols]
        else:
            source = self._config.get("universe_source", "sp500")
            custom = self._config.get("universe", [])
            universe = get_universe(source, custom)

        logger.info(f"Scanning {len(universe)} symbols")

        # 2. Collect benchmark symbols needed by criteria
        extra_symbols = set()
        for criterion in self._criteria:
            if hasattr(criterion, "_benchmark"):
                extra_symbols.add(criterion._benchmark)
        all_fetch_symbols = list(set(universe) | extra_symbols)

        # 3. Fetch data from all providers
        provider_data: Dict[str, Dict[str, ProviderData]] = {}
        for name, provider in self._providers.items():
            try:
                batch = provider.fetch_batch(all_fetch_symbols)
                provider_data[name] = batch
            except Exception as e:
                errors.append(f"Provider {name} failed: {e}")
                logger.error(f"Provider {name} failed: {e}")

        # 4. Score each symbol
        scores: List[SymbolScore] = []
        for symbol in universe:
            try:
                symbol_score = self._score_symbol(symbol, provider_data)
                scores.append(symbol_score)
            except Exception as e:
                errors.append(f"Scoring {symbol} failed: {e}")
                logger.warning(f"Scoring {symbol} failed: {e}")

        # 5. Rank
        ranked = self._scorer.rank(scores) if self._scorer else scores

        # 6. Build report
        report = ScanReport(
            timestamp=datetime.now(timezone.utc),
            universe_size=len(universe),
            results=ranked,
            criteria_used=[c.name for c in self._criteria],
            duration_seconds=round(time.time() - start_time, 2),
            errors=errors,
        )

        self._last_report = report

        # 7. Optionally update lists
        if update_lists:
            self._apply_to_lists(report)

        return report

    def _score_symbol(
        self,
        symbol: str,
        provider_data: Dict[str, Dict[str, ProviderData]],
    ) -> SymbolScore:
        """Run all criteria against a symbol and compute composite score."""
        # Build per-symbol provider data view
        symbol_providers: Dict[str, ProviderData] = {}
        for provider_name, batch in provider_data.items():
            if symbol in batch:
                symbol_providers[provider_name] = batch[symbol]

            # Also add benchmark data keyed as "technical_SPY" etc.
            for criterion in self._criteria:
                if hasattr(criterion, "_benchmark"):
                    bench = criterion._benchmark
                    if bench in batch:
                        symbol_providers[f"{provider_name}_{bench}"] = batch[bench]

        # Evaluate each criterion
        criteria_scores: Dict[str, float] = {}
        for criterion in self._criteria:
            try:
                score = criterion.evaluate(symbol, symbol_providers)
                criteria_scores[criterion.name] = round(score, 4)
            except Exception as e:
                logger.warning(f"Criterion {criterion.name} failed for {symbol}: {e}")
                criteria_scores[criterion.name] = 0.0

        # Extract metadata for display
        metadata: Dict[str, Any] = {"symbol": symbol}
        tech_data = symbol_providers.get("technical")
        if tech_data:
            for key in ["close", "RSI", "volume", "avg_volume_20", "MACD", "SMA", "ATR"]:
                if key in tech_data.data:
                    metadata[key] = tech_data.data[key]

        return self._scorer.score(symbol, criteria_scores, metadata)

    def _apply_to_lists(self, report: ScanReport) -> Dict[str, str]:
        """Apply scan results to SymbolListManager. Additive only — never removes."""
        from core.symbol_list_manager import get_list_manager

        manager = get_list_manager()
        actions: Dict[str, str] = {}

        for result in report.results:
            if result.recommendation == "skip":
                continue

            existing_type = manager.get_list_type(result.symbol)
            note = f"scanner score={result.total_score} @ {report.timestamp.isoformat()}"

            if result.recommendation == "trade":
                if existing_type == "trade":
                    continue  # Already in trade list
                manager.add_to_trade_list(result.symbol, notes=note)
                actions[result.symbol] = "added to trade"
                logger.info(f"Scanner: {result.symbol} -> trade list (score={result.total_score})")

            elif result.recommendation == "watch":
                if existing_type in ("trade", "watch"):
                    continue  # Don't downgrade trade, don't duplicate watch
                manager.add_to_watch_list(result.symbol, notes=note)
                actions[result.symbol] = "added to watch"
                logger.info(f"Scanner: {result.symbol} -> watch list (score={result.total_score})")

        return actions

    @property
    def last_report(self) -> Optional[ScanReport]:
        return self._last_report


# --- Singleton ---

_scanner_engine: Optional[ScannerEngine] = None


def get_scanner(config: Optional[Dict[str, Any]] = None) -> ScannerEngine:
    """Get or create the singleton ScannerEngine."""
    global _scanner_engine
    if _scanner_engine is None or config is not None:
        _scanner_engine = ScannerEngine(config)
    return _scanner_engine
