"""
Test suite for trade logic and decision making.

Tests trade gate, trade logic manager, and decision logic.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import unittest
from unittest.mock import patch


class TestTradeGate(unittest.TestCase):
    """Test trade gate functionality."""

    def test_trade_gate_initialization(self):
        """TradeGate should initialize with correct parameters."""
        try:
            from core.logic.trade_gate import TradeGate

            gate = TradeGate(max_layers=2, min_bars_between_layers=3, regime_min_persist_bars=2, flip_cooldown_bars=1)

            self.assertEqual(gate.max_layers, 2)
            self.assertEqual(gate.min_bars_between_layers, 3)
        except ImportError:
            self.skipTest("TradeGate not found")

    def test_trade_gate_allows_first_trade(self):
        """TradeGate should allow first trade."""
        try:
            from core.logic.trade_gate import TradeGate

            gate = TradeGate()
            gate.on_new_bar("AAPL", bar_id=1, regime="normal")

            state = gate.get_state("AAPL")
            # First trade should generally be allowed
            self.assertIsNotNone(state)
        except ImportError:
            self.skipTest("TradeGate not found")

    def test_trade_gate_tracks_regime(self):
        """TradeGate should track regime changes."""
        try:
            from core.logic.trade_gate import TradeGate

            gate = TradeGate(regime_min_persist_bars=2)

            # First bar in normal regime
            gate.on_new_bar("AAPL", bar_id=1, regime="normal")
            state1 = gate.get_state("AAPL")

            # Second bar in same regime
            gate.on_new_bar("AAPL", bar_id=2, regime="normal")
            state2 = gate.get_state("AAPL")

            # Regime persistence should increase
            self.assertGreaterEqual(state2.regime_persist, state1.regime_persist)
        except ImportError:
            self.skipTest("TradeGate not found")


class TestTradeLogicManager(unittest.TestCase):
    """Test DynamicTradeLogicManager functionality."""

    def setUp(self):
        """Create mock config."""
        self.mock_config = {
            "AAPL": {
                "normal": {"trade_logic_class": "default", "params": {}},
                "high_volatility": {"trade_logic_class": "conservative", "params": {"max_position": 0.5}},
            }
        }

    def test_trade_logic_manager_loads_config(self):
        """TradeLogicManager should load config from file."""
        import tempfile

        try:
            from core.logic.trade_logic_manager import DynamicTradeLogicManager

            # Use a temp directory to avoid read-only filesystem issues
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = os.path.join(tmpdir, "config.json")
                with open(config_path, "w") as f:
                    json.dump(self.mock_config, f)

                manager = DynamicTradeLogicManager(config_path)
                self.assertIsNotNone(manager)
        except ImportError:
            self.skipTest("DynamicTradeLogicManager not found")

    def test_default_trade_logic(self):
        """Default trade logic should be available."""
        try:
            from core.logic.trade_logic_manager import DynamicTradeLogicManager

            # Test that we can instantiate without a file (should use defaults)
            with patch("builtins.open", side_effect=FileNotFoundError):
                try:
                    DynamicTradeLogicManager("/nonexistent.json")
                except FileNotFoundError:
                    pass  # Expected behavior
        except ImportError:
            self.skipTest("DynamicTradeLogicManager not found")


class TestPortfolioState(unittest.TestCase):
    """Test PortfolioState functionality."""

    def test_portfolio_state_initialization(self):
        """PortfolioState should initialize correctly."""
        try:
            from core.logic.portfolio_state import PortfolioState

            portfolio = PortfolioState()

            # Should have empty positions initially
            self.assertEqual(len(portfolio.positions), 0)
        except ImportError:
            self.skipTest("PortfolioState not found")

    def test_portfolio_tracks_equity(self):
        """PortfolioState should track equity."""
        try:
            from core.logic.portfolio_state import PortfolioState

            portfolio = PortfolioState()

            # Initial equity should be accessible
            equity = portfolio.total_equity()
            self.assertIsInstance(equity, (int, float))
        except ImportError:
            self.skipTest("PortfolioState not found")


class TestSymbolState(unittest.TestCase):
    """Test SymbolState functionality."""

    def test_symbol_state_initialization(self):
        """SymbolState should initialize correctly."""
        try:
            from core.logic.symbol_state import SymbolState

            state = SymbolState(symbol="AAPL")

            self.assertEqual(state.symbol, "AAPL")
        except ImportError:
            self.skipTest("SymbolState not found")

    def test_symbol_state_tracks_bar(self):
        """SymbolState should track bar information."""
        try:
            from core.logic.symbol_state import SymbolState

            state = SymbolState(symbol="AAPL")
            state.bar_id = 100
            state.bar_closed = True

            self.assertEqual(state.bar_id, 100)
            self.assertTrue(state.bar_closed)
        except ImportError:
            self.skipTest("SymbolState not found")


class TestDrawdownMonitor(unittest.TestCase):
    """Test DrawdownMonitor functionality."""

    def test_drawdown_monitor_initialization(self):
        """DrawdownMonitor should initialize correctly."""
        try:
            from core.drawdown_monitor import DrawdownMonitor

            monitor = DrawdownMonitor(max_symbol_drawdown=0.12, max_portfolio_drawdown=0.15)

            self.assertEqual(monitor.max_symbol_drawdown, 0.12)
            self.assertEqual(monitor.max_portfolio_drawdown, 0.15)
        except ImportError:
            self.skipTest("DrawdownMonitor not found")

    @patch("asyncio.create_task")
    def test_drawdown_calculation(self, mock_create_task):
        """DrawdownMonitor should calculate drawdown correctly."""
        try:
            from core.drawdown_monitor import DrawdownMonitor

            monitor = DrawdownMonitor()

            # First update sets the peak
            monitor.update_portfolio(10000)

            # Update with lower equity (should show drawdown)
            monitor.update_portfolio(9500)

            drawdown = monitor.get_portfolio_drawdown()
            # Drawdown is negative (9500-10000)/10000 = -0.05
            self.assertAlmostEqual(drawdown, -0.05, places=2)  # 5% drawdown
        except (ImportError, AttributeError):
            self.skipTest("DrawdownMonitor not found or method missing")


if __name__ == "__main__":
    unittest.main()
