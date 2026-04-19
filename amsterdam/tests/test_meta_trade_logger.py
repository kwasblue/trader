"""
Tests for MetaTradeLogger and meta trade types.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone

from core.contracts.meta_types import TradeEntryContext, TradeExitContext
from loggers.meta_trade_logger import MetaTradeLogger, generate_trade_id


class TestTradeEntryContext(unittest.TestCase):
    """Tests for TradeEntryContext dataclass."""

    def test_create_entry_context(self):
        """Test creating a TradeEntryContext."""
        entry = TradeEntryContext(
            trade_id="20240224_AAPL_001",
            timestamp=datetime(2024, 2, 24, 15, 30, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            side="buy",
            qty=50,
            price=150.25,
            strategy="momentum",
            regime="normal",
            atr=2.34,
            atr_percentile=0.72,
            drawdown_portfolio_pct=-0.02,
            drawdown_symbol_pct=-0.01,
            position_size_pct=0.05,
            hour_of_day=15,
            day_of_week=3,
            minutes_since_open=360,
            bars_in_regime=12,
            hours_since_last_trade=4.5,
            signal_strength=1,
        )

        self.assertEqual(entry.trade_id, "20240224_AAPL_001")
        self.assertEqual(entry.symbol, "AAPL")
        self.assertEqual(entry.side, "buy")
        self.assertEqual(entry.qty, 50)
        self.assertEqual(entry.price, 150.25)
        self.assertEqual(entry.atr_percentile, 0.72)

    def test_entry_context_to_dict(self):
        """Test TradeEntryContext.to_dict() method."""
        ts = datetime(2024, 2, 24, 15, 30, 0, tzinfo=timezone.utc)
        entry = TradeEntryContext(
            trade_id="test_id",
            timestamp=ts,
            symbol="AAPL",
            side="buy",
            qty=100,
            price=150.0,
            strategy="test",
            regime="normal",
            atr=2.0,
            atr_percentile=0.5,
            drawdown_portfolio_pct=0.0,
            drawdown_symbol_pct=0.0,
            position_size_pct=0.05,
            hour_of_day=15,
            day_of_week=3,
            minutes_since_open=360,
            bars_in_regime=10,
            hours_since_last_trade=5.0,
            signal_strength=1,
        )

        d = entry.to_dict()
        self.assertEqual(d["trade_id"], "test_id")
        self.assertEqual(d["timestamp"], ts.isoformat())
        self.assertEqual(d["symbol"], "AAPL")


class TestTradeExitContext(unittest.TestCase):
    """Tests for TradeExitContext dataclass."""

    def test_create_exit_context(self):
        """Test creating a TradeExitContext."""
        exit_ctx = TradeExitContext(
            trade_id="20240224_AAPL_001",
            timestamp=datetime(2024, 2, 24, 16, 45, 0, tzinfo=timezone.utc),
            price=152.10,
            pnl_dollars=92.50,
            pnl_percent=0.0123,
            hold_bars=75,
            mae_percent=-0.008,
            mfe_percent=0.018,
            exit_reason="take_profit",
        )

        self.assertEqual(exit_ctx.trade_id, "20240224_AAPL_001")
        self.assertEqual(exit_ctx.price, 152.10)
        self.assertEqual(exit_ctx.pnl_dollars, 92.50)
        self.assertEqual(exit_ctx.exit_reason, "take_profit")

    def test_exit_context_to_dict(self):
        """Test TradeExitContext.to_dict() method."""
        ts = datetime(2024, 2, 24, 16, 45, 0, tzinfo=timezone.utc)
        exit_ctx = TradeExitContext(
            trade_id="test_id",
            timestamp=ts,
            price=152.0,
            pnl_dollars=100.0,
            pnl_percent=0.01,
            hold_bars=50,
            mae_percent=-0.005,
            mfe_percent=0.015,
            exit_reason="signal_exit",
        )

        d = exit_ctx.to_dict()
        self.assertEqual(d["trade_id"], "test_id")
        self.assertEqual(d["timestamp"], ts.isoformat())
        self.assertEqual(d["pnl_dollars"], 100.0)


class TestGenerateTradeId(unittest.TestCase):
    """Tests for generate_trade_id function."""

    def test_generate_trade_id_format(self):
        """Test trade ID format."""
        ts = datetime(2024, 2, 24, 15, 30, 45, 123456, tzinfo=timezone.utc)
        trade_id = generate_trade_id("AAPL", ts)

        self.assertTrue(trade_id.startswith("20240224_AAPL_"))
        self.assertIn("153045", trade_id)

    def test_generate_trade_id_unique(self):
        """Test that trade IDs are unique."""
        ids = set()
        for _ in range(100):
            trade_id = generate_trade_id("AAPL")
            ids.add(trade_id)

        # All IDs should be unique
        self.assertEqual(len(ids), 100)


class TestMetaTradeLogger(unittest.TestCase):
    """Tests for MetaTradeLogger."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = MetaTradeLogger(
            log_file="test_meta_trades.jsonl",
            log_dir=self.temp_dir,
        )

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_entry(self):
        """Test logging a trade entry."""
        entry = TradeEntryContext(
            trade_id="test_entry_001",
            timestamp=datetime(2024, 2, 24, 15, 30, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            side="buy",
            qty=50,
            price=150.25,
            strategy="momentum",
            regime="normal",
            atr=2.34,
            atr_percentile=0.72,
            drawdown_portfolio_pct=-0.02,
            drawdown_symbol_pct=-0.01,
            position_size_pct=0.05,
            hour_of_day=15,
            day_of_week=3,
            minutes_since_open=360,
            bars_in_regime=12,
            hours_since_last_trade=4.5,
            signal_strength=1,
        )

        self.logger.log_entry(entry)

        # Read and verify
        with open(self.logger.log_path) as f:
            line = f.readline()
            event = json.loads(line)

        self.assertEqual(event["event"], "entry")
        self.assertEqual(event["trade_id"], "test_entry_001")
        self.assertEqual(event["symbol"], "AAPL")
        self.assertEqual(event["side"], "buy")
        self.assertEqual(event["qty"], 50)
        self.assertEqual(event["price"], 150.25)
        self.assertEqual(event["features"]["strategy"], "momentum")
        self.assertEqual(event["features"]["atr_percentile"], 0.72)

    def test_log_exit(self):
        """Test logging a trade exit."""
        exit_ctx = TradeExitContext(
            trade_id="test_exit_001",
            timestamp=datetime(2024, 2, 24, 16, 45, 0, tzinfo=timezone.utc),
            price=152.10,
            pnl_dollars=92.50,
            pnl_percent=0.0123,
            hold_bars=75,
            mae_percent=-0.008,
            mfe_percent=0.018,
            exit_reason="take_profit",
        )

        self.logger.log_exit(exit_ctx)

        # Read and verify
        with open(self.logger.log_path) as f:
            line = f.readline()
            event = json.loads(line)

        self.assertEqual(event["event"], "exit")
        self.assertEqual(event["trade_id"], "test_exit_001")
        self.assertEqual(event["price"], 152.10)
        self.assertEqual(event["outcome"]["pnl_dollars"], 92.50)
        self.assertEqual(event["outcome"]["exit_reason"], "take_profit")

    def test_log_entry_and_exit(self):
        """Test logging both entry and exit for a trade."""
        trade_id = "test_full_trade_001"

        entry = TradeEntryContext(
            trade_id=trade_id,
            timestamp=datetime(2024, 2, 24, 15, 30, 0, tzinfo=timezone.utc),
            symbol="MSFT",
            side="buy",
            qty=25,
            price=400.00,
            strategy="breakout",
            regime="trending",
            atr=5.0,
            atr_percentile=0.8,
            drawdown_portfolio_pct=-0.01,
            drawdown_symbol_pct=0.0,
            position_size_pct=0.10,
            hour_of_day=15,
            day_of_week=0,
            minutes_since_open=360,
            bars_in_regime=20,
            hours_since_last_trade=24.0,
            signal_strength=1,
        )

        exit_ctx = TradeExitContext(
            trade_id=trade_id,
            timestamp=datetime(2024, 2, 24, 16, 00, 0, tzinfo=timezone.utc),
            price=405.00,
            pnl_dollars=125.00,
            pnl_percent=0.0125,
            hold_bars=30,
            mae_percent=-0.005,
            mfe_percent=0.015,
            exit_reason="take_profit",
        )

        self.logger.log_entry(entry)
        self.logger.log_exit(exit_ctx)

        # Read and verify both lines
        with open(self.logger.log_path) as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)

        entry_event = json.loads(lines[0])
        exit_event = json.loads(lines[1])

        self.assertEqual(entry_event["event"], "entry")
        self.assertEqual(entry_event["trade_id"], trade_id)

        self.assertEqual(exit_event["event"], "exit")
        self.assertEqual(exit_event["trade_id"], trade_id)

    def test_jsonl_format(self):
        """Test that output is valid JSONL (one JSON object per line)."""
        for i in range(5):
            entry = TradeEntryContext(
                trade_id=f"trade_{i:03d}",
                timestamp=datetime.now(timezone.utc),
                symbol="TEST",
                side="buy",
                qty=10,
                price=100.0,
                strategy="test",
                regime="normal",
                atr=1.0,
                atr_percentile=0.5,
                drawdown_portfolio_pct=0.0,
                drawdown_symbol_pct=0.0,
                position_size_pct=0.01,
                hour_of_day=10,
                day_of_week=1,
                minutes_since_open=30,
                bars_in_regime=5,
                hours_since_last_trade=1.0,
                signal_strength=1,
            )
            self.logger.log_entry(entry)

        # Verify each line is valid JSON
        with open(self.logger.log_path) as f:
            for i, line in enumerate(f):
                try:
                    event = json.loads(line)
                    self.assertEqual(event["trade_id"], f"trade_{i:03d}")
                except json.JSONDecodeError:
                    self.fail(f"Line {i} is not valid JSON: {line}")


if __name__ == "__main__":
    unittest.main()
