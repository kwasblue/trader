"""
Tests for DrawdownMonitor

Coverage:
- Symbol-level drawdown tracking
- Portfolio-level drawdown tracking
- Cooldown management
- Trading permission checks
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
import asyncio

import sys
from pathlib import Path
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
    monkeypatch.setattr(asyncio, 'create_task', mock_create_task)


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
            portfolio_cooldown_seconds=600
        )
        assert monitor.max_symbol_drawdown == 0.20
        assert monitor.max_symbol_daily_drawdown == 0.10
        assert monitor.symbol_cooldown_seconds == 300
        assert monitor.max_portfolio_drawdown == 0.15


class TestSymbolDrawdown:
    """Tests for symbol-level drawdown tracking."""

    @pytest.fixture
    def monitor(self):
        return DrawdownMonitor(
            max_symbol_drawdown=0.20,
            max_symbol_daily_drawdown=0.10,
            symbol_cooldown_seconds=60
        )

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
            max_portfolio_drawdown=0.25,
            max_portfolio_daily_drawdown=0.10,
            portfolio_cooldown_seconds=120
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
        return DrawdownMonitor(
            max_symbol_drawdown=0.20,
            max_portfolio_drawdown=0.25
        )

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
