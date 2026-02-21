"""
Tests for Simulation Module

Coverage:
- compute_atr helper function
- classify_regime helper function
- SimConfig dataclass
- SimulationRunner initialization and methods
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.simulator.simulation import compute_atr, classify_regime, SimConfig


class TestComputeATR:
    """Tests for compute_atr function."""

    def test_compute_atr_basic(self):
        """Test basic ATR computation."""
        # Create OHLC data
        data = {
            'High': [102, 104, 103, 105, 106, 107, 108, 109, 110, 111,
                     112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
            'Low': [98, 99, 98, 100, 101, 102, 103, 104, 105, 106,
                    107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
            'Close': [100, 102, 101, 103, 104, 105, 106, 107, 108, 109,
                      110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
        }
        df = pd.DataFrame(data)

        atr = compute_atr(df, period=14)

        assert atr is not None
        assert atr > 0
        assert isinstance(atr, float)

    def test_compute_atr_empty_df(self):
        """Test ATR with empty DataFrame."""
        df = pd.DataFrame()

        atr = compute_atr(df)

        assert atr is None

    def test_compute_atr_insufficient_data(self):
        """Test ATR with insufficient data points."""
        data = {
            'High': [102, 104, 103],
            'Low': [98, 99, 98],
            'Close': [100, 102, 101],
        }
        df = pd.DataFrame(data)

        atr = compute_atr(df, period=14)

        assert atr is None

    def test_compute_atr_exact_period(self):
        """Test ATR with exactly period+1 data points."""
        n = 16  # period=14 + 1 for TR calculation + 1
        data = {
            'High': [100 + i for i in range(n)],
            'Low': [95 + i for i in range(n)],
            'Close': [98 + i for i in range(n)],
        }
        df = pd.DataFrame(data)

        atr = compute_atr(df, period=14)

        assert atr is not None

    def test_compute_atr_lowercase_columns(self):
        """Test ATR with lowercase column names."""
        data = {
            'high': [102, 104, 103, 105, 106, 107, 108, 109, 110, 111,
                     112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
            'low': [98, 99, 98, 100, 101, 102, 103, 104, 105, 106,
                    107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
            'close': [100, 102, 101, 103, 104, 105, 106, 107, 108, 109,
                      110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
        }
        df = pd.DataFrame(data)

        atr = compute_atr(df, period=14)

        assert atr is not None

    def test_compute_atr_custom_period(self):
        """Test ATR with custom period."""
        n = 30
        data = {
            'High': [100 + i for i in range(n)],
            'Low': [95 + i for i in range(n)],
            'Close': [98 + i for i in range(n)],
        }
        df = pd.DataFrame(data)

        atr_10 = compute_atr(df, period=10)
        atr_20 = compute_atr(df, period=20)

        assert atr_10 is not None
        assert atr_20 is not None
        # Different periods should give different values (usually)
        # Note: In this linear case they may be similar


class TestClassifyRegime:
    """Tests for classify_regime function."""

    def test_classify_regime_high_volatility(self):
        """Test classification of high volatility regime."""
        # Build ATR history with mean around 2.0
        atr_history = [2.0] * 30
        current_atr = 4.0  # Well above mean + std

        regime = classify_regime(current_atr, atr_history)

        assert regime == "high_volatility"

    def test_classify_regime_low_volatility(self):
        """Test classification of low volatility regime."""
        # Build ATR history with mean around 2.0, std around 0.2
        atr_history = [2.0 + 0.1 * (i % 3 - 1) for i in range(30)]
        current_atr = 1.0  # Well below mean - 0.5*std

        regime = classify_regime(current_atr, atr_history)

        assert regime == "low_volatility"

    def test_classify_regime_normal(self):
        """Test classification of normal regime."""
        atr_history = [2.0] * 30
        current_atr = 2.0  # At the mean

        regime = classify_regime(current_atr, atr_history)

        assert regime == "normal"

    def test_classify_regime_none_atr(self):
        """Test classification with None ATR."""
        atr_history = [2.0] * 30

        regime = classify_regime(None, atr_history)

        assert regime == "normal"

    def test_classify_regime_insufficient_history(self):
        """Test classification with insufficient history."""
        atr_history = [2.0] * 10  # Less than 20

        regime = classify_regime(2.0, atr_history)

        assert regime == "normal"

    def test_classify_regime_empty_history(self):
        """Test classification with empty history."""
        regime = classify_regime(2.0, [])

        assert regime == "normal"


class TestSimConfig:
    """Tests for SimConfig dataclass."""

    def test_simconfig_defaults(self):
        """Test SimConfig with default values."""
        with patch('core.simulator.simulation.Path.mkdir'):
            config = SimConfig(symbols=["AAPL", "MSFT"])

        assert config.symbols == ["AAPL", "MSFT"]
        assert config.steps == 600
        assert config.starting_cash == 100_000.0
        assert config.bar_sleep == 0.05

    def test_simconfig_custom_values(self):
        """Test SimConfig with custom values."""
        with patch('core.simulator.simulation.Path.mkdir'):
            config = SimConfig(
                symbols=["GOOGL"],
                steps=100,
                starting_cash=50_000.0,
                bar_sleep=0.01,
            )

        assert config.symbols == ["GOOGL"]
        assert config.steps == 100
        assert config.starting_cash == 50_000.0
        assert config.bar_sleep == 0.01

    def test_simconfig_empty_symbols_raises(self):
        """Test that empty symbols list raises error."""
        with pytest.raises(ValueError, match="symbols list cannot be empty"):
            with patch('core.simulator.simulation.Path.mkdir'):
                SimConfig(symbols=[])

    def test_simconfig_trade_log_path(self):
        """Test trade_log_path property."""
        with patch('core.simulator.simulation.Path.mkdir'):
            config = SimConfig(
                symbols=["AAPL"],
                trade_log_file="test_trades.csv",
                trade_log_dir="test_logs"
            )

        assert config.trade_log_path == "test_logs/test_trades.csv"

    def test_simconfig_from_config_file(self):
        """Test creating SimConfig from config file."""
        mock_config = MagicMock()
        mock_config.general.default_symbols = ["AAPL"]
        mock_config.simulation.steps = 500
        mock_config.simulation.bar_sleep = 0.02
        mock_config.simulation.starting_cash = 75000.0
        mock_config.simulation.warmup_bars = 100
        mock_config.risk.risk_per_trade = 0.02
        mock_config.position_sizer.max_trade_pct = 0.15
        mock_config.position_sizer.max_holding_pct = 0.30
        mock_config.drawdown_monitor.enabled = True
        mock_config.drawdown_monitor.max_symbol_drawdown = 0.25
        mock_config.drawdown_monitor.max_symbol_daily_drawdown = 0.10
        mock_config.drawdown_monitor.max_portfolio_drawdown = 0.20
        mock_config.drawdown_monitor.max_portfolio_daily_drawdown = 0.08
        mock_config.indicators.atr_period = 14

        with patch('core.simulator.simulation.get_config', return_value=mock_config):
            with patch('core.simulator.simulation.Path.mkdir'):
                config = SimConfig.from_config_file()

        assert config.symbols == ["AAPL"]
        assert config.steps == 500


class TestSimulationRunnerInit:
    """Tests for SimulationRunner initialization."""

    @pytest.fixture
    def sim_config(self):
        """Create a SimConfig for testing."""
        with patch('core.simulator.simulation.Path.mkdir'):
            return SimConfig(
                symbols=["AAPL"],
                steps=10,
                bar_sleep=0.001,
                starting_cash=100000.0,
            )

    def test_simulation_runner_init(self, sim_config):
        """Test SimulationRunner initialization."""
        with patch('core.events.eventhandler.get_event_handler') as mock_eh, \
             patch('core.simulator.simulation.Logger'), \
             patch('core.simulator.simulation.FileTradeLogger'), \
             patch('core.simulator.simulation.GBMSimulator'), \
             patch('core.simulator.simulation.StrategyRoutingManager'), \
             patch('core.simulator.simulation.DynamicTradeLogicManager'), \
             patch('core.simulator.simulation.MockBroker'), \
             patch('core.simulator.simulation.MockExecutor'), \
             patch('core.simulator.simulation.KellyPositionSizer'), \
             patch('core.simulator.simulation.MockExecutionEngine'):

            mock_eh.return_value = MagicMock()

            from core.simulator.simulation import SimulationRunner
            runner = SimulationRunner(sim_config)

            assert runner.cfg == sim_config
            assert "AAPL" in runner.symbol_states
            assert "AAPL" in runner.history


class TestSimulationHelpers:
    """Tests for simulation helper methods."""

    def test_df_from_history_empty(self):
        """Test _df_from_history with empty history."""
        with patch('core.simulator.simulation.Path.mkdir'):
            config = SimConfig(symbols=["AAPL"], steps=1)

        with patch('core.events.eventhandler.get_event_handler') as mock_eh, \
             patch('core.simulator.simulation.Logger'), \
             patch('core.simulator.simulation.FileTradeLogger'), \
             patch('core.simulator.simulation.GBMSimulator'), \
             patch('core.simulator.simulation.StrategyRoutingManager'), \
             patch('core.simulator.simulation.DynamicTradeLogicManager'), \
             patch('core.simulator.simulation.MockBroker'), \
             patch('core.simulator.simulation.MockExecutor'), \
             patch('core.simulator.simulation.KellyPositionSizer'), \
             patch('core.simulator.simulation.MockExecutionEngine'):

            mock_eh.return_value = MagicMock()

            from core.simulator.simulation import SimulationRunner
            runner = SimulationRunner(config)

            # Clear history
            runner.history["AAPL"].clear()

            df = runner._df_from_history("AAPL")

            assert df.empty

    def test_df_from_history_with_data(self):
        """Test _df_from_history with data."""
        with patch('core.simulator.simulation.Path.mkdir'):
            config = SimConfig(symbols=["AAPL"], steps=1)

        with patch('core.events.eventhandler.get_event_handler') as mock_eh, \
             patch('core.simulator.simulation.Logger'), \
             patch('core.simulator.simulation.FileTradeLogger'), \
             patch('core.simulator.simulation.GBMSimulator'), \
             patch('core.simulator.simulation.StrategyRoutingManager'), \
             patch('core.simulator.simulation.DynamicTradeLogicManager'), \
             patch('core.simulator.simulation.MockBroker'), \
             patch('core.simulator.simulation.MockExecutor'), \
             patch('core.simulator.simulation.KellyPositionSizer'), \
             patch('core.simulator.simulation.MockExecutionEngine'):

            mock_eh.return_value = MagicMock()

            from core.simulator.simulation import SimulationRunner
            runner = SimulationRunner(config)

            # Add test data
            runner.history["AAPL"].append({
                "timestamp": datetime.now(timezone.utc),
                "Open": 150.0,
                "High": 152.0,
                "Low": 149.0,
                "Close": 151.0,
                "Volume": 1000000,
            })

            df = runner._df_from_history("AAPL")

            assert not df.empty
            assert "Open" in df.columns
            assert "Close" in df.columns


class TestClassifyRegimeMethod:
    """Tests for SimulationRunner._classify_regime method."""

    def test_classify_regime_unknown_missing_data(self):
        """Test regime classification with missing indicator data."""
        with patch('core.simulator.simulation.Path.mkdir'):
            config = SimConfig(symbols=["AAPL"], steps=1)

        with patch('core.events.eventhandler.get_event_handler') as mock_eh, \
             patch('core.simulator.simulation.Logger'), \
             patch('core.simulator.simulation.FileTradeLogger'), \
             patch('core.simulator.simulation.GBMSimulator'), \
             patch('core.simulator.simulation.StrategyRoutingManager'), \
             patch('core.simulator.simulation.DynamicTradeLogicManager'), \
             patch('core.simulator.simulation.MockBroker'), \
             patch('core.simulator.simulation.MockExecutor'), \
             patch('core.simulator.simulation.KellyPositionSizer'), \
             patch('core.simulator.simulation.MockExecutionEngine'):

            mock_eh.return_value = MagicMock()

            from core.simulator.simulation import SimulationRunner
            runner = SimulationRunner(config)

            # Indicators with NaN values
            indicators_row = {
                'ATR': np.nan,
                'Close': 150.0,
            }

            regime = runner._classify_regime(indicators_row, [])

            assert regime == "UNKNOWN"

    def test_classify_regime_high_volatility_method(self):
        """Test HIGH_VOLATILITY regime classification."""
        with patch('core.simulator.simulation.Path.mkdir'):
            config = SimConfig(symbols=["AAPL"], steps=1)

        with patch('core.events.eventhandler.get_event_handler') as mock_eh, \
             patch('core.simulator.simulation.Logger'), \
             patch('core.simulator.simulation.FileTradeLogger'), \
             patch('core.simulator.simulation.GBMSimulator'), \
             patch('core.simulator.simulation.StrategyRoutingManager'), \
             patch('core.simulator.simulation.DynamicTradeLogicManager'), \
             patch('core.simulator.simulation.MockBroker'), \
             patch('core.simulator.simulation.MockExecutor'), \
             patch('core.simulator.simulation.KellyPositionSizer'), \
             patch('core.simulator.simulation.MockExecutionEngine'):

            mock_eh.return_value = MagicMock()

            from core.simulator.simulation import SimulationRunner
            runner = SimulationRunner(config)

            # ATR history with stable values
            atr_history = [2.0] * 25
            runner.atr_hist["AAPL"].extend(atr_history)

            # High ATR current value
            indicators_row = {
                'ATR': 5.0,  # Way above mean + std
                'Close': 150.0,
                'RSI': 50.0,
            }

            regime = runner._classify_regime(indicators_row, runner.atr_hist["AAPL"])

            assert regime == "HIGH_VOLATILITY"


class TestSimulationStop:
    """Tests for simulation stop functionality."""

    def test_stop_sets_flag(self):
        """Test that stop() sets the stop flag."""
        with patch('core.simulator.simulation.Path.mkdir'):
            config = SimConfig(symbols=["AAPL"], steps=1)

        with patch('core.events.eventhandler.get_event_handler') as mock_eh, \
             patch('core.simulator.simulation.Logger'), \
             patch('core.simulator.simulation.FileTradeLogger'), \
             patch('core.simulator.simulation.GBMSimulator'), \
             patch('core.simulator.simulation.StrategyRoutingManager'), \
             patch('core.simulator.simulation.DynamicTradeLogicManager'), \
             patch('core.simulator.simulation.MockBroker'), \
             patch('core.simulator.simulation.MockExecutor'), \
             patch('core.simulator.simulation.KellyPositionSizer'), \
             patch('core.simulator.simulation.MockExecutionEngine'):

            mock_eh.return_value = MagicMock()

            from core.simulator.simulation import SimulationRunner
            runner = SimulationRunner(config)

            assert runner._stop_requested is False

            runner.stop()

            assert runner._stop_requested is True
