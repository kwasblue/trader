"""
Tests for DrawdownMonitor

Coverage:
- Symbol-level drawdown tracking
- Portfolio-level drawdown tracking
- Cooldown management
- Trading permission checks
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.drawdown_monitor import DrawdownMonitor


@pytest.fixture(autouse=True)
def mock_async_create_task(monkeypatch):
    """Mock asyncio.create_task to prevent 'no running event loop' errors."""

    def mock_create_task(coro, **kwargs):
        # Close the coroutine to avoid warnings
        coro.close()
        return MagicMock()

    monkeypatch.setattr(asyncio, "create_task", mock_create_task)


class TestDrawdownMonitorInit:
    """Tests for DrawdownMonitor initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        monitor = DrawdownMonitor()
        assert monitor.max_symbol_drawdown == 0.30
        assert monitor.max_symbol_daily_drawdown == 0.15
        assert monitor.max_portfolio_drawdown == 0.25
        assert monitor.max_portfolio_daily_drawdown == 0.10

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        monitor = DrawdownMonitor(
            max_symbol_drawdown=0.20,
            max_symbol_daily_drawdown=0.10,
            symbol_cooldown_seconds=300,
            max_portfolio_drawdown=0.15,
            max_portfolio_daily_drawdown=0.08,
            portfolio_cooldown_seconds=600,
        )
        assert monitor.max_symbol_drawdown == 0.20
        assert monitor.max_symbol_daily_drawdown == 0.10
        assert monitor.symbol_cooldown_seconds == 300
        assert monitor.max_portfolio_drawdown == 0.15


class TestSymbolDrawdown:
    """Tests for symbol-level drawdown tracking."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor(max_symbol_drawdown=0.20, max_symbol_daily_drawdown=0.10, symbol_cooldown_seconds=60)

    def test_update_symbol_returns_bool(self, monitor):
        """Test update_symbol returns boolean."""
        result = monitor.update_symbol("AAPL", 150.0)
        assert isinstance(result, bool)

    def test_update_symbol_allows_trading_initially(self, monitor):
        """Test trading is allowed initially."""
        result = monitor.update_symbol("AAPL", 150.0)
        assert result is True

    def test_symbol_blocked_on_max_drawdown(self, monitor):
        """Test symbol gets blocked on exceeding max drawdown."""
        # Set initial value
        monitor.update_symbol("AAPL", 100.0)
        # Large drop (25% > 20% max)
        result = monitor.update_symbol("AAPL", 75.0)
        assert result is False

    def test_symbol_allowed_within_limits(self, monitor):
        """Test symbol allowed within drawdown limits."""
        monitor.update_symbol("AAPL", 100.0)
        # Small drop (5% < 20% max)
        result = monitor.update_symbol("AAPL", 95.0)
        assert result is True


class TestPortfolioDrawdown:
    """Tests for portfolio-level drawdown tracking."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor(
            max_portfolio_drawdown=0.25, max_portfolio_daily_drawdown=0.10, portfolio_cooldown_seconds=120
        )

    def test_update_portfolio_returns_bool(self, monitor):
        """Test update_portfolio returns boolean."""
        result = monitor.update_portfolio(100000.0)
        assert isinstance(result, bool)

    def test_update_portfolio_allows_trading_initially(self, monitor):
        """Test trading is allowed initially."""
        result = monitor.update_portfolio(100000.0)
        assert result is True

    def test_portfolio_blocked_on_max_drawdown(self, monitor):
        """Test portfolio blocked on exceeding max drawdown."""
        monitor.update_portfolio(100000.0)
        # Large drop (30% > 25% max)
        result = monitor.update_portfolio(70000.0)
        assert result is False

    def test_portfolio_allowed_within_limits(self, monitor):
        """Test portfolio allowed within limits."""
        monitor.update_portfolio(100000.0)
        # Small drop (10% < 25% max)
        result = monitor.update_portfolio(90000.0)
        assert result is True


class TestCanTrade:
    """Tests for combined trading permission."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor(max_symbol_drawdown=0.20, max_portfolio_drawdown=0.25)

    def test_can_trade_when_all_ok(self, monitor):
        """Test can_trade returns True when all limits OK."""
        monitor.update_portfolio(100000.0)
        monitor.update_symbol("AAPL", 100.0)

        result = monitor.can_trade("AAPL")
        assert result is True

    def test_can_trade_blocked_by_symbol(self, monitor):
        """Test can_trade blocked when symbol exceeds limit."""
        monitor.update_portfolio(100000.0)
        monitor.update_symbol("AAPL", 100.0)
        monitor.update_symbol("AAPL", 75.0)  # 25% drawdown

        result = monitor.can_trade("AAPL")
        assert result is False

    def test_can_trade_blocked_by_portfolio(self, monitor):
        """Test can_trade blocked when portfolio exceeds limit."""
        monitor.update_portfolio(100000.0)
        monitor.update_portfolio(70000.0)  # 30% drawdown
        monitor.update_symbol("AAPL", 100.0)

        result = monitor.can_trade("AAPL")
        assert result is False


class TestIsBlocked:
    """Tests for blocked state checking."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor()

    def test_is_symbol_blocked_initially_false(self, monitor):
        """Test symbol is not blocked initially."""
        assert monitor.is_symbol_blocked("AAPL") is False

    def test_is_portfolio_blocked_initially_false(self, monitor):
        """Test portfolio is not blocked initially."""
        assert monitor.is_portfolio_blocked() is False

    def test_is_symbol_blocked_after_breach(self, monitor):
        """Test symbol is blocked after breach."""
        monitor.update_symbol("AAPL", 100.0)
        monitor.update_symbol("AAPL", 60.0)  # 40% drawdown

        assert monitor.is_symbol_blocked("AAPL") is True

    def test_is_portfolio_blocked_after_breach(self, monitor):
        """Test portfolio is blocked after breach."""
        monitor.update_portfolio(100000.0)
        monitor.update_portfolio(60000.0)  # 40% drawdown

        assert monitor.is_portfolio_blocked() is True


class TestMultipleSymbols:
    """Tests for handling multiple symbols."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor(max_symbol_drawdown=0.20)

    def test_symbols_tracked_independently(self, monitor):
        """Test symbols are tracked independently."""
        monitor.update_symbol("AAPL", 100.0)
        monitor.update_symbol("MSFT", 200.0)

        # AAPL has drawdown
        monitor.update_symbol("AAPL", 75.0)

        # MSFT should still be tradeable
        assert monitor.can_trade("MSFT") is True

    def test_blocked_symbol_doesnt_affect_others(self, monitor):
        """Test blocked symbol doesn't affect other symbols."""
        monitor.update_portfolio(100000.0)

        monitor.update_symbol("AAPL", 100.0)
        monitor.update_symbol("AAPL", 75.0)  # Block AAPL

        monitor.update_symbol("MSFT", 200.0)

        assert monitor.can_trade("AAPL") is False
        assert monitor.can_trade("MSFT") is True


class TestStartNewDay:
    """Tests for daily reset functionality."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor()

    def test_start_new_day_sets_portfolio(self, monitor):
        """Test start_new_day sets portfolio daily start."""
        monitor.start_new_day(portfolio_equity=100000.0)

        assert monitor.portfolio_daily_start == 100000.0

    def test_start_new_day_sets_symbols(self, monitor):
        """Test start_new_day sets per-symbol daily starts."""
        monitor.start_new_day(per_symbol_equity={"AAPL": 5000.0, "MSFT": 3000.0})

        assert monitor.symbol_daily_start["AAPL"] == 5000.0
        assert monitor.symbol_daily_start["MSFT"] == 3000.0

    def test_start_new_day_both(self, monitor):
        """Test start_new_day sets both portfolio and symbols."""
        monitor.start_new_day(portfolio_equity=100000.0, per_symbol_equity={"AAPL": 5000.0})

        assert monitor.portfolio_daily_start == 100000.0
        assert monitor.symbol_daily_start["AAPL"] == 5000.0


class TestDrawdownValues:
    """Tests for drawdown value retrieval."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor()

    def test_get_portfolio_drawdown(self, monitor):
        """Test get_portfolio_drawdown returns correct value."""
        monitor.update_portfolio(100000.0)
        monitor.update_portfolio(95000.0)

        dd = monitor.get_portfolio_drawdown()

        assert abs(dd - (-0.05)) < 0.001

    def test_get_portfolio_daily_drawdown(self, monitor):
        """Test get_portfolio_daily_drawdown returns correct value."""
        monitor.start_new_day(portfolio_equity=100000.0)
        monitor.update_portfolio(97000.0)

        dd = monitor.get_portfolio_daily_drawdown()

        assert abs(dd - (-0.03)) < 0.001

    def test_get_symbol_drawdown(self, monitor):
        """Test get_symbol_drawdown returns correct value."""
        monitor.update_symbol("AAPL", 100.0)
        monitor.update_symbol("AAPL", 90.0)

        dd = monitor.get_symbol_drawdown("AAPL")

        assert abs(dd - (-0.10)) < 0.001

    def test_get_symbol_daily_drawdown(self, monitor):
        """Test get_symbol_daily_drawdown returns correct value."""
        monitor.start_new_day(per_symbol_equity={"AAPL": 100.0})
        monitor.update_symbol("AAPL", 92.0)

        dd = monitor.get_symbol_daily_drawdown("AAPL")

        assert abs(dd - (-0.08)) < 0.001

    def test_get_symbol_drawdown_unknown(self, monitor):
        """Test get_symbol_drawdown for unknown symbol."""
        dd = monitor.get_symbol_drawdown("UNKNOWN")

        assert dd == 0.0


class TestAdminControls:
    """Tests for admin/manual control methods."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor(symbol_cooldown_seconds=60)

    def test_unlock_symbol(self, monitor):
        """Test unlock_symbol unlocks a locked symbol."""
        # First lock the symbol
        monitor.update_symbol("AAPL", 100.0)
        monitor.update_symbol("AAPL", 60.0)  # 40% drawdown, should lock

        assert monitor.symbol_locked["AAPL"] is True

        monitor.unlock_symbol("AAPL")

        assert monitor.symbol_locked["AAPL"] is False
        assert "AAPL" in monitor.symbol_last_unlock_time

    def test_reset_symbol(self, monitor):
        """Test reset_symbol clears all symbol state."""
        monitor.update_symbol("AAPL", 100.0)
        monitor.update_symbol("AAPL", 60.0)

        monitor.reset_symbol("AAPL")

        assert monitor.symbol_locked["AAPL"] is False
        assert monitor.symbol_peak["AAPL"] == 0.0
        assert monitor.symbol_daily_start["AAPL"] == 0.0

    def test_unlock_portfolio(self, monitor):
        """Test unlock_portfolio unlocks a locked portfolio."""
        monitor.update_portfolio(100000.0)
        monitor.update_portfolio(60000.0)  # 40% drawdown

        assert monitor.portfolio_locked is True

        monitor.unlock_portfolio()

        assert monitor.portfolio_locked is False
        assert monitor.portfolio_last_unlock_time is not None

    def test_reset_portfolio(self, monitor):
        """Test reset_portfolio clears all portfolio state."""
        monitor.update_portfolio(100000.0)
        monitor.update_portfolio(60000.0)

        monitor.reset_portfolio()

        assert monitor.portfolio_locked is False
        assert monitor.portfolio_peak is None
        assert monitor.portfolio_daily_start is None


class TestCooldown:
    """Tests for cooldown behavior."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor(symbol_cooldown_seconds=60, portfolio_cooldown_seconds=120)

    def test_is_symbol_in_cooldown_true(self, monitor):
        """Test is_symbol_in_cooldown when in cooldown."""
        monitor.symbol_last_unlock_time["AAPL"] = datetime.now(timezone.utc)

        assert monitor.is_symbol_in_cooldown("AAPL") is True

    def test_is_symbol_in_cooldown_false(self, monitor):
        """Test is_symbol_in_cooldown when cooldown expired."""
        monitor.symbol_last_unlock_time["AAPL"] = datetime.now(timezone.utc) - timedelta(seconds=120)

        assert monitor.is_symbol_in_cooldown("AAPL") is False

    def test_is_symbol_in_cooldown_no_unlock(self, monitor):
        """Test is_symbol_in_cooldown when never unlocked."""
        assert monitor.is_symbol_in_cooldown("AAPL") is False

    def test_is_portfolio_in_cooldown_true(self, monitor):
        """Test is_portfolio_in_cooldown when in cooldown."""
        monitor.portfolio_last_unlock_time = datetime.now(timezone.utc)

        assert monitor.is_portfolio_in_cooldown() is True

    def test_is_portfolio_in_cooldown_false(self, monitor):
        """Test is_portfolio_in_cooldown when cooldown expired."""
        monitor.portfolio_last_unlock_time = datetime.now(timezone.utc) - timedelta(seconds=200)

        assert monitor.is_portfolio_in_cooldown() is False

    def test_is_portfolio_in_cooldown_no_unlock(self, monitor):
        """Test is_portfolio_in_cooldown when never unlocked."""
        assert monitor.is_portfolio_in_cooldown() is False


class TestPersistence:
    """Tests for state persistence."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor()

    def test_serialize_state(self, monitor):
        """Test _serialize_state creates valid dict."""
        monitor.portfolio_peak = 100000.0
        monitor.portfolio_locked = True
        monitor.symbol_peak["AAPL"] = 10000.0

        state = monitor._serialize_state()

        assert "timestamp" in state
        assert "portfolio" in state
        assert "symbols" in state
        assert state["portfolio"]["peak"] == 100000.0

    def test_deserialize_state(self, monitor):
        """Test _deserialize_state restores state."""
        state = {
            "portfolio": {
                "peak": 100000.0,
                "daily_start": 98000.0,
                "locked": True,
                "current_dd": -0.05,
                "current_daily_dd": -0.02,
            },
            "symbols": {
                "AAPL": {
                    "peak": 10000.0,
                    "daily_start": 9500.0,
                    "locked": True,
                    "current_dd": -0.05,
                    "current_daily_dd": -0.05,
                }
            },
        }

        monitor._deserialize_state(state)

        assert monitor.portfolio_peak == 100000.0
        assert monitor.portfolio_locked is True
        assert monitor.symbol_peak["AAPL"] == 10000.0

    def test_set_persistence_path(self, monitor):
        """Test set_persistence_path sets the path."""
        from pathlib import Path

        path = Path("/tmp/test_dd.json")

        monitor.set_persistence_path(path)

        assert monitor._persistence_path == path

    def test_save_state(self, monitor):
        """Test save_state persists to disk."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            monitor.set_persistence_path(temp_path)
            monitor.portfolio_peak = 100000.0
            monitor.save_state()

            assert temp_path.exists()
        finally:
            if temp_path.exists():
                os.unlink(temp_path)


class TestAsyncWrappers:
    """Tests for async wrapper methods."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor()

    @pytest.mark.asyncio
    async def test_can_trade_async(self, monitor):
        """Test async can_trade wrapper."""
        result = await monitor.can_trade_async("AAPL")

        assert result is True

    @pytest.mark.asyncio
    async def test_update_portfolio_async(self, monitor):
        """Test async update_portfolio wrapper."""
        result = await monitor.update_portfolio_async(100000.0)

        assert result is True
        assert monitor.portfolio_peak == 100000.0

    @pytest.mark.asyncio
    async def test_update_symbol_async(self, monitor):
        """Test async update_symbol wrapper."""
        result = await monitor.update_symbol_async("AAPL", 10000.0)

        assert result is True
        assert monitor.symbol_peak["AAPL"] == 10000.0


class TestPeakTracking:
    """Tests for peak value tracking."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor()

    def test_portfolio_peak_updates_on_new_high(self, monitor):
        """Test portfolio peak updates when new high reached."""
        monitor.update_portfolio(100000.0)
        monitor.update_portfolio(110000.0)
        monitor.update_portfolio(105000.0)

        assert monitor.portfolio_peak == 110000.0

    def test_symbol_peak_updates_on_new_high(self, monitor):
        """Test symbol peak updates when new high reached."""
        monitor.update_symbol("AAPL", 100.0)
        monitor.update_symbol("AAPL", 120.0)
        monitor.update_symbol("AAPL", 115.0)

        assert monitor.symbol_peak["AAPL"] == 120.0

    def test_peak_does_not_decrease(self, monitor):
        """Test peak doesn't decrease on drawdown."""
        monitor.update_portfolio(100000.0)
        monitor.update_portfolio(90000.0)

        assert monitor.portfolio_peak == 100000.0


class TestDailyDrawdownBreaches:
    """Tests for daily drawdown limit breaches."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor(max_symbol_daily_drawdown=0.10, max_portfolio_daily_drawdown=0.08)

    def test_portfolio_daily_breach_locks(self, monitor):
        """Test portfolio locks on daily drawdown breach."""
        monitor.start_new_day(portfolio_equity=100000.0)
        result = monitor.update_portfolio(90000.0)  # 10% daily drawdown > 8%

        assert result is False
        assert monitor.portfolio_locked is True

    def test_symbol_daily_breach_locks(self, monitor):
        """Test symbol locks on daily drawdown breach."""
        monitor.start_new_day(per_symbol_equity={"AAPL": 100.0})
        result = monitor.update_symbol("AAPL", 85.0)  # 15% daily drawdown > 10%

        assert result is False
        assert monitor.symbol_locked["AAPL"] is True
