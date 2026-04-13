"""
Test suite for the stock scanner module.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from scanner.result import SymbolScore, ScanReport
from scanner.scorer import SymbolScorer
from scanner.providers.base import BaseDataProvider, ProviderData
from scanner.criteria.base import BaseCriterion, create_criterion, CRITERION_REGISTRY
from scanner.criteria.momentum import MomentumBreakout, RelativeStrength, VolumeSurge
from scanner.criteria.trend import TrendFollowing, MovingAverageCross
from scanner.universe import get_universe, FALLBACK_SYMBOLS


# =============================================================================
# HELPERS
# =============================================================================

def _make_ohlcv_df(n=200, base_close=100.0, trend=0.001, volume=100000, lowercase=True):
    """Create a synthetic OHLCV DataFrame for testing.

    By default returns lowercase columns (for criteria that expect post-provider data).
    Set lowercase=False for capitalized columns (for passing to TechnicalIndicators).
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = [base_close]
    for i in range(1, n):
        closes.append(closes[-1] * (1 + trend + np.random.normal(0, 0.01)))
    closes = np.array(closes)

    if lowercase:
        df = pd.DataFrame({
            "timestamp": dates,
            "open": closes * 0.999,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.random.randint(int(volume * 0.5), int(volume * 1.5), n),
        })
    else:
        df = pd.DataFrame({
            "timestamp": dates,
            "Open": closes * 0.999,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.random.randint(int(volume * 0.5), int(volume * 1.5), n),
        })
    return df


def _make_provider_data(symbol, df=None, summary=None):
    """Create a ProviderData instance for testing."""
    if df is None:
        df = _make_ohlcv_df()
    if summary is None:
        summary = {}
    return ProviderData(symbol=symbol, provider_name="technical", data=summary, dataframe=df)


def _apply_indicators(df):
    """Apply indicators to a test DataFrame. Handles column casing."""
    from indicators.technical_indicators import TechnicalIndicators
    # Indicators expect capitalized columns
    cap_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    low_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    work = df.rename(columns={k: v for k, v in cap_map.items() if k in df.columns})
    work = TechnicalIndicators(work).apply_all()
    work = work.rename(columns={k: v for k, v in low_map.items() if k in work.columns})
    return work


# =============================================================================
# RESULT TESTS
# =============================================================================

class TestSymbolScore(unittest.TestCase):
    def test_creation(self):
        s = SymbolScore(symbol="AAPL", total_score=0.85, recommendation="trade")
        self.assertEqual(s.symbol, "AAPL")
        self.assertEqual(s.total_score, 0.85)
        self.assertEqual(s.recommendation, "trade")

    def test_to_dict(self):
        s = SymbolScore(symbol="MSFT", total_score=0.5, criteria_scores={"momentum": 0.6})
        d = s.to_dict()
        self.assertEqual(d["symbol"], "MSFT")
        self.assertIn("momentum", d["criteria_scores"])


class TestScanReport(unittest.TestCase):
    def setUp(self):
        self.report = ScanReport(
            universe_size=100,
            results=[
                SymbolScore("AAPL", 0.9, recommendation="trade"),
                SymbolScore("MSFT", 0.75, recommendation="trade"),
                SymbolScore("GOOGL", 0.5, recommendation="watch"),
                SymbolScore("IBM", 0.2, recommendation="skip"),
            ],
            criteria_used=["momentum_breakout"],
        )

    def test_top_n(self):
        top = self.report.top_n(2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0].symbol, "AAPL")

    def test_trade_candidates(self):
        trades = self.report.trade_candidates()
        self.assertEqual(trades, ["AAPL", "MSFT"])

    def test_watch_candidates(self):
        watches = self.report.watch_candidates()
        self.assertEqual(watches, ["GOOGL"])

    def test_to_dict(self):
        d = self.report.to_dict()
        self.assertEqual(d["universe_size"], 100)
        self.assertEqual(len(d["results"]), 4)


# =============================================================================
# SCORER TESTS
# =============================================================================

class TestSymbolScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = SymbolScorer(
            weights={"momentum_breakout": 0.5, "volume_surge": 0.5},
            trade_threshold=0.7,
            watch_threshold=0.4,
            max_trade_symbols=2,
            max_watch_symbols=3,
        )

    def test_score_trade(self):
        result = self.scorer.score("AAPL", {"momentum_breakout": 0.9, "volume_surge": 0.8}, {})
        self.assertEqual(result.recommendation, "trade")
        self.assertAlmostEqual(result.total_score, 0.85, places=2)

    def test_score_watch(self):
        result = self.scorer.score("AAPL", {"momentum_breakout": 0.5, "volume_surge": 0.5}, {})
        self.assertEqual(result.recommendation, "watch")

    def test_score_skip(self):
        result = self.scorer.score("AAPL", {"momentum_breakout": 0.1, "volume_surge": 0.1}, {})
        self.assertEqual(result.recommendation, "skip")

    def test_score_unknown_criterion_ignored(self):
        result = self.scorer.score("AAPL", {"unknown_criterion": 1.0}, {})
        self.assertEqual(result.total_score, 0.0)

    def test_rank_enforces_max_trade(self):
        scores = [
            SymbolScore("A", 0.9, recommendation="trade"),
            SymbolScore("B", 0.85, recommendation="trade"),
            SymbolScore("C", 0.8, recommendation="trade"),
        ]
        ranked = self.scorer.rank(scores)
        trade_count = sum(1 for s in ranked if s.recommendation == "trade")
        self.assertEqual(trade_count, 2)  # max_trade_symbols=2
        self.assertEqual(ranked[2].recommendation, "watch")

    def test_rank_enforces_max_watch(self):
        scores = [
            SymbolScore("A", 0.6, recommendation="watch"),
            SymbolScore("B", 0.55, recommendation="watch"),
            SymbolScore("C", 0.5, recommendation="watch"),
            SymbolScore("D", 0.45, recommendation="watch"),
        ]
        ranked = self.scorer.rank(scores)
        watch_count = sum(1 for s in ranked if s.recommendation == "watch")
        self.assertEqual(watch_count, 3)  # max_watch_symbols=3
        self.assertEqual(ranked[3].recommendation, "skip")

    def test_rank_sorted_descending(self):
        scores = [
            SymbolScore("B", 0.5, recommendation="watch"),
            SymbolScore("A", 0.9, recommendation="trade"),
            SymbolScore("C", 0.3, recommendation="skip"),
        ]
        ranked = self.scorer.rank(scores)
        self.assertEqual(ranked[0].symbol, "A")
        self.assertEqual(ranked[1].symbol, "B")
        self.assertEqual(ranked[2].symbol, "C")


# =============================================================================
# CRITERIA TESTS
# =============================================================================

class TestMomentumBreakout(unittest.TestCase):
    def setUp(self):
        self.criterion = MomentumBreakout()

    def test_name(self):
        self.assertEqual(self.criterion.name, "momentum_breakout")

    def test_empty_data_returns_zero(self):
        score = self.criterion.evaluate("AAPL", {})
        self.assertEqual(score, 0.0)

    def test_no_dataframe_returns_zero(self):
        pd_data = ProviderData(symbol="AAPL", provider_name="technical")
        score = self.criterion.evaluate("AAPL", {"technical": pd_data})
        self.assertEqual(score, 0.0)

    def test_scores_breakout(self):
        df = _make_ohlcv_df(200, base_close=100, trend=0.005)
        df = _apply_indicators(df)
        latest = df.iloc[-1]
        summary = {
            "close": float(latest["close"]),
            "BB_upper": float(latest["close"]) * 0.98,  # Close above BB upper
            "RSI": 60.0,  # In momentum zone
            "MACD": 1.0,
            "MACD_signal": 0.5,  # MACD above signal
        }
        pd_data = ProviderData(symbol="AAPL", provider_name="technical", data=summary, dataframe=df)
        score = self.criterion.evaluate("AAPL", {"technical": pd_data})
        self.assertGreater(score, 0.0)

    def test_score_capped_at_one(self):
        df = _make_ohlcv_df(200, base_close=100, trend=0.01)
        df = _apply_indicators(df)
        summary = {
            "close": 200.0,
            "BB_upper": 100.0,
            "RSI": 60.0,
            "MACD": 5.0,
            "MACD_signal": 1.0,
        }
        pd_data = ProviderData(symbol="TEST", provider_name="technical", data=summary, dataframe=df)
        score = self.criterion.evaluate("TEST", {"technical": pd_data})
        self.assertLessEqual(score, 1.0)


class TestRelativeStrength(unittest.TestCase):
    def setUp(self):
        self.criterion = RelativeStrength(benchmark="SPY", period=20)

    def test_name(self):
        self.assertEqual(self.criterion.name, "relative_strength")

    def test_outperformer_scores_high(self):
        # Symbol: +10% over 20 days
        sym_df = _make_ohlcv_df(50, base_close=100, trend=0.005)
        # SPY: flat
        spy_df = _make_ohlcv_df(50, base_close=100, trend=0.0)
        sym_data = ProviderData("AAPL", "technical", {}, sym_df)
        spy_data = ProviderData("SPY", "technical", {}, spy_df)
        score = self.criterion.evaluate("AAPL", {"technical": sym_data, "technical_SPY": spy_data})
        self.assertGreater(score, 0.5)

    def test_underperformer_scores_low(self):
        # Symbol: -5% over 20 days
        sym_df = _make_ohlcv_df(50, base_close=100, trend=-0.003)
        # SPY: +5%
        spy_df = _make_ohlcv_df(50, base_close=100, trend=0.003)
        sym_data = ProviderData("WEAK", "technical", {}, sym_df)
        spy_data = ProviderData("SPY", "technical", {}, spy_df)
        score = self.criterion.evaluate("WEAK", {"technical": sym_data, "technical_SPY": spy_data})
        self.assertLess(score, 0.5)

    def test_no_benchmark_defaults_to_zero(self):
        sym_df = _make_ohlcv_df(50, base_close=100, trend=0.005)
        sym_data = ProviderData("AAPL", "technical", {}, sym_df)
        score = self.criterion.evaluate("AAPL", {"technical": sym_data})
        # Still gets a score based on absolute return vs 0
        self.assertIsInstance(score, float)


class TestVolumeSurge(unittest.TestCase):
    def setUp(self):
        self.criterion = VolumeSurge(surge_threshold=1.5)

    def test_name(self):
        self.assertEqual(self.criterion.name, "volume_surge")

    def test_no_volume_returns_zero(self):
        pd_data = ProviderData("AAPL", "technical", data={})
        score = self.criterion.evaluate("AAPL", {"technical": pd_data})
        self.assertEqual(score, 0.0)

    def test_low_volume_returns_zero(self):
        pd_data = ProviderData("AAPL", "technical", data={"volume": 100, "avg_volume_20": 100})
        score = self.criterion.evaluate("AAPL", {"technical": pd_data})
        self.assertEqual(score, 0.0)

    def test_moderate_surge(self):
        pd_data = ProviderData("AAPL", "technical", data={"volume": 200, "avg_volume_20": 100})
        score = self.criterion.evaluate("AAPL", {"technical": pd_data})
        self.assertEqual(score, 0.5)  # 2x = 0.5

    def test_high_surge(self):
        pd_data = ProviderData("AAPL", "technical", data={"volume": 500, "avg_volume_20": 100})
        score = self.criterion.evaluate("AAPL", {"technical": pd_data})
        self.assertEqual(score, 1.0)  # 5x+ = 1.0


class TestTrendFollowing(unittest.TestCase):
    def setUp(self):
        self.criterion = TrendFollowing()

    def test_name(self):
        self.assertEqual(self.criterion.name, "trend_following")

    def test_strong_uptrend(self):
        summary = {
            "close": 150.0,
            "SMA": 140.0,       # close > SMA
            "SMA_50": 130.0,    # close > SMA50
            "MACD": 2.0,        # positive
            "MACD_signal": 1.0, # above signal
            "PSAR": 135.0,      # below price
        }
        pd_data = ProviderData("AAPL", "technical", data=summary)
        score = self.criterion.evaluate("AAPL", {"technical": pd_data})
        self.assertEqual(score, 1.0)

    def test_no_trend(self):
        summary = {
            "close": 100.0,
            "SMA": 110.0,       # close < SMA
            "SMA_50": 120.0,    # close < SMA50
            "MACD": -1.0,       # negative
            "MACD_signal": 0.5, # below signal
            "PSAR": 115.0,      # above price
        }
        pd_data = ProviderData("AAPL", "technical", data=summary)
        score = self.criterion.evaluate("AAPL", {"technical": pd_data})
        self.assertEqual(score, 0.0)

    def test_mixed_signals(self):
        summary = {
            "close": 105.0,
            "SMA": 100.0,       # close > SMA
            "MACD": -0.5,       # negative
            "MACD_signal": -0.3,
            "PSAR": 103.0,      # below price
        }
        pd_data = ProviderData("AAPL", "technical", data=summary)
        score = self.criterion.evaluate("AAPL", {"technical": pd_data})
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class TestMovingAverageCross(unittest.TestCase):
    def setUp(self):
        self.criterion = MovingAverageCross(lookback=10)

    def test_name(self):
        self.assertEqual(self.criterion.name, "ma_crossover")

    def test_insufficient_data(self):
        df = _make_ohlcv_df(30)
        pd_data = ProviderData("AAPL", "technical", {}, df)
        score = self.criterion.evaluate("AAPL", {"technical": pd_data})
        self.assertEqual(score, 0.0)


# =============================================================================
# CRITERION REGISTRY TESTS
# =============================================================================

class TestCriterionRegistry(unittest.TestCase):
    def test_registry_populated(self):
        # Import criteria to trigger registration
        import scanner.criteria.momentum
        import scanner.criteria.trend
        self.assertGreater(len(CRITERION_REGISTRY), 0)

    def test_create_momentum_breakout(self):
        c = create_criterion({"name": "momentum_breakout", "params": {}})
        self.assertEqual(c.name, "momentum_breakout")

    def test_create_volume_surge_with_params(self):
        c = create_criterion({"name": "volume_surge", "params": {"surge_threshold": 2.0}})
        self.assertEqual(c.name, "volume_surge")
        self.assertEqual(c._surge_threshold, 2.0)

    def test_create_unknown_raises(self):
        with self.assertRaises(ValueError):
            create_criterion({"name": "nonexistent_criterion", "params": {}})


# =============================================================================
# UNIVERSE TESTS
# =============================================================================

class TestUniverse(unittest.TestCase):
    def test_sp500_returns_list(self):
        symbols = get_universe("sp500")
        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 10)
        self.assertIn("AAPL", symbols)

    def test_custom_returns_provided(self):
        symbols = get_universe("custom", custom_symbols=["AAPL", "msft"])
        self.assertEqual(symbols, ["AAPL", "MSFT"])

    def test_unknown_falls_back_to_sp500(self):
        symbols = get_universe("bogus_source")
        self.assertGreater(len(symbols), 0)

    def test_fallback_symbols_exist(self):
        self.assertGreater(len(FALLBACK_SYMBOLS), 10)


# =============================================================================
# ENGINE TESTS
# =============================================================================

class TestScannerEngine(unittest.TestCase):
    def test_engine_init_with_defaults(self):
        """Engine should initialize without errors using default config."""
        from scanner.engine import ScannerEngine
        engine = ScannerEngine({
            "providers": {"technical_enabled": False},
            "criteria": [],
        })
        self.assertIsNotNone(engine)

    def test_engine_scan_empty_universe(self):
        from scanner.engine import ScannerEngine
        engine = ScannerEngine({
            "providers": {"technical_enabled": False},
            "criteria": [],
        })
        report = engine.scan(symbols=[])
        self.assertEqual(report.universe_size, 0)
        self.assertEqual(len(report.results), 0)

    @patch("scanner.providers.technical.TechnicalProvider.fetch")
    @patch("scanner.providers.technical.TechnicalProvider.is_available", return_value=True)
    def test_engine_scan_with_mock_provider(self, mock_available, mock_fetch):
        """Engine should score symbols when provider returns data."""
        from scanner.engine import ScannerEngine

        df = _make_ohlcv_df(200, trend=0.005)
        df = _apply_indicators(df)
        latest = df.iloc[-1]
        summary = {
            "close": float(latest["close"]),
            "RSI": 60.0,
            "MACD": 1.0,
            "MACD_signal": 0.5,
            "SMA": float(latest.get("SMA", 0)),
            "volume": 150000,
            "avg_volume_20": 100000,
        }
        mock_fetch.return_value = ProviderData("AAPL", "technical", summary, df)

        engine = ScannerEngine({
            "providers": {"technical_enabled": True},
            "criteria": [{"name": "momentum_breakout", "params": {}}],
            "scoring": {"weights": {"momentum_breakout": 1.0}},
        })

        report = engine.scan(symbols=["AAPL"])
        self.assertEqual(report.universe_size, 1)
        self.assertEqual(len(report.results), 1)
        self.assertEqual(report.results[0].symbol, "AAPL")
        self.assertGreater(report.results[0].total_score, 0.0)

    def test_engine_last_report(self):
        from scanner.engine import ScannerEngine
        engine = ScannerEngine({
            "providers": {"technical_enabled": False},
            "criteria": [],
        })
        self.assertIsNone(engine.last_report)
        engine.scan(symbols=["AAPL"])
        self.assertIsNotNone(engine.last_report)


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestScannerConfig(unittest.TestCase):
    def test_config_loads_scanner_section(self):
        from core.config_loader import load_config, ScannerConfig
        config = load_config()
        self.assertIsInstance(config.scanner, ScannerConfig)
        self.assertTrue(config.scanner.enabled)

    def test_to_engine_config(self):
        from core.config_loader import ScannerConfig
        cfg = ScannerConfig()
        d = cfg.to_engine_config()
        self.assertIn("universe_source", d)
        self.assertIn("providers", d)
        self.assertIn("scoring", d)
        self.assertIn("criteria", d)

    def test_default_criteria_present(self):
        from core.config_loader import ScannerConfig
        cfg = ScannerConfig()
        self.assertEqual(len(cfg.criteria), 4)
        names = [c["name"] for c in cfg.criteria]
        self.assertIn("momentum_breakout", names)
        self.assertIn("relative_strength", names)


# =============================================================================
# PROVIDER TESTS
# =============================================================================

class TestProviderStubs(unittest.TestCase):
    def test_fundamental_not_available(self):
        from scanner.providers.fundamental import FundamentalProvider
        p = FundamentalProvider()
        self.assertFalse(p.is_available())
        self.assertEqual(p.name, "fundamental")

    def test_sentiment_not_available(self):
        from scanner.providers.sentiment import SentimentProvider
        p = SentimentProvider()
        self.assertFalse(p.is_available())

    def test_insider_not_available(self):
        from scanner.providers.insider import InsiderProvider
        p = InsiderProvider()
        self.assertFalse(p.is_available())


class TestBaseDataProvider(unittest.TestCase):
    def test_fetch_batch_handles_errors(self):
        class FailingProvider(BaseDataProvider):
            @property
            def name(self):
                return "failing"
            def fetch(self, symbol):
                raise ValueError("fail")
            def is_available(self):
                return True

        p = FailingProvider()
        results = p.fetch_batch(["AAPL", "MSFT"])
        self.assertEqual(len(results), 0)  # Both failed


# =============================================================================
# OPTIMIZATION TESTS
# =============================================================================

class TestOptimizeStrategies(unittest.TestCase):
    def test_optimize_no_symbols_returns_empty(self):
        from scanner.engine import ScannerEngine
        engine = ScannerEngine({
            "providers": {"technical_enabled": False},
            "criteria": [],
        })
        result = engine.optimize_strategies(symbols=[])
        self.assertEqual(result, {})

    def test_optimize_no_report_no_symbols_returns_empty(self):
        from scanner.engine import ScannerEngine
        engine = ScannerEngine({
            "providers": {"technical_enabled": False},
            "criteria": [],
        })
        # No scan run, no symbols passed
        result = engine.optimize_strategies()
        self.assertEqual(result, {})

    @patch("core.backtest.regime_backtest.RegimeBacktester")
    @patch("core.unified_data_pipeline.UnifiedDataPipeline")
    def test_optimize_calls_regime_backtester(self, MockPipeline, MockBacktester):
        """optimize_strategies should call RegimeBacktester for each symbol."""
        from scanner.engine import ScannerEngine

        # Mock pipeline
        mock_pipeline = MockPipeline.return_value
        mock_df = _make_ohlcv_df(200, lowercase=False)
        mock_pipeline.load_symbol_data.return_value = mock_df

        # Mock backtester
        mock_tester = MockBacktester.return_value
        mock_result = MagicMock()
        mock_result.best_strategies = {"low_volatility": "sma", "normal": "momentum", "high_volatility": "rsi"}
        mock_tester.run_regime_analysis.return_value = mock_result

        engine = ScannerEngine({
            "providers": {"technical_enabled": False},
            "criteria": [],
        })

        result = engine.optimize_strategies(symbols=["AAPL", "MSFT"], days=365)

        self.assertEqual(len(result), 2)
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)
        self.assertEqual(mock_tester.run_regime_analysis.call_count, 2)
        self.assertEqual(mock_tester.save_routing_config.call_count, 2)

    @patch("core.backtest.regime_backtest.RegimeBacktester")
    @patch("core.unified_data_pipeline.UnifiedDataPipeline")
    def test_optimize_uses_last_report_candidates(self, MockPipeline, MockBacktester):
        """optimize_strategies with no symbols should use last scan's trade candidates."""
        from scanner.engine import ScannerEngine
        from scanner.result import ScanReport, SymbolScore

        mock_pipeline = MockPipeline.return_value
        mock_pipeline.load_symbol_data.return_value = _make_ohlcv_df(200, lowercase=False)

        mock_tester = MockBacktester.return_value
        mock_result = MagicMock()
        mock_result.best_strategies = {"normal": "sma"}
        mock_tester.run_regime_analysis.return_value = mock_result

        engine = ScannerEngine({
            "providers": {"technical_enabled": False},
            "criteria": [],
        })

        # Simulate a previous scan with trade candidates
        engine._last_report = ScanReport(
            results=[
                SymbolScore("FDX", 0.9, recommendation="trade"),
                SymbolScore("BWA", 0.85, recommendation="trade"),
                SymbolScore("GOOGL", 0.5, recommendation="watch"),
            ]
        )

        result = engine.optimize_strategies()

        # Should only optimize trade candidates (FDX, BWA), not watch (GOOGL)
        self.assertEqual(len(result), 2)
        self.assertIn("FDX", result)
        self.assertIn("BWA", result)
        self.assertNotIn("GOOGL", result)

    @patch("core.backtest.regime_backtest.RegimeBacktester")
    @patch("core.unified_data_pipeline.UnifiedDataPipeline")
    def test_optimize_handles_missing_data(self, MockPipeline, MockBacktester):
        """optimize_strategies should skip symbols with no data."""
        from scanner.engine import ScannerEngine

        mock_pipeline = MockPipeline.return_value
        mock_pipeline.load_symbol_data.return_value = None  # No data

        engine = ScannerEngine({
            "providers": {"technical_enabled": False},
            "criteria": [],
        })

        result = engine.optimize_strategies(symbols=["NODATA"])
        self.assertEqual(result, {})
        MockBacktester.assert_not_called()

    @patch("core.backtest.regime_backtest.RegimeBacktester")
    @patch("core.unified_data_pipeline.UnifiedDataPipeline")
    def test_optimize_handles_backtester_error(self, MockPipeline, MockBacktester):
        """optimize_strategies should continue if one symbol fails."""
        from scanner.engine import ScannerEngine

        mock_pipeline = MockPipeline.return_value
        mock_pipeline.load_symbol_data.return_value = _make_ohlcv_df(200, lowercase=False)

        mock_tester = MockBacktester.return_value
        mock_tester.run_regime_analysis.side_effect = [
            Exception("backtest failed"),  # First symbol fails
            MagicMock(best_strategies={"normal": "sma"}),  # Second succeeds
        ]

        engine = ScannerEngine({
            "providers": {"technical_enabled": False},
            "criteria": [],
        })

        result = engine.optimize_strategies(symbols=["FAIL", "OK"])
        self.assertEqual(len(result), 1)
        self.assertIn("OK", result)


class TestScannerConfigSchedule(unittest.TestCase):
    def test_default_schedule_has_optimize_fields(self):
        from core.config_loader import ScannerConfig
        cfg = ScannerConfig()
        self.assertTrue(cfg.schedule["optimize_after_scan"])
        self.assertEqual(cfg.schedule["optimization_days"], 365)
        self.assertEqual(cfg.schedule["optimization_metric"], "sharpe_ratio")


if __name__ == "__main__":
    unittest.main()
