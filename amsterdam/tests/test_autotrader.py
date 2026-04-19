"""
Tests for AutoTrader and MarketScheduler

Coverage:
- Market schedule calculations
- Trading day detection
- Holiday handling
- State machine transitions
- Daily cycle orchestration
"""

import sys
from datetime import datetime, time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.daemon import AutoTrader, AutoTraderState, MarketScheduler

ET = ZoneInfo("America/New_York")


class TestMarketScheduler:
    """Tests for MarketScheduler."""

    @pytest.fixture
    def scheduler(self):
        return MarketScheduler()

    def test_is_trading_day_weekday(self, scheduler):
        """Test that weekdays are trading days."""
        # Monday
        monday = datetime(2024, 1, 8, 10, 0, tzinfo=ET)
        assert scheduler.is_trading_day(monday) is True

    def test_is_trading_day_weekend(self, scheduler):
        """Test that weekends are not trading days."""
        # Saturday
        saturday = datetime(2024, 1, 6, 10, 0, tzinfo=ET)
        assert scheduler.is_trading_day(saturday) is False

        # Sunday
        sunday = datetime(2024, 1, 7, 10, 0, tzinfo=ET)
        assert scheduler.is_trading_day(sunday) is False

    def test_is_trading_day_holiday(self, scheduler):
        """Test that holidays are not trading days."""
        # Christmas 2024
        christmas = datetime(2024, 12, 25, 10, 0, tzinfo=ET)
        assert scheduler.is_trading_day(christmas) is False

        # New Year's 2025
        new_year = datetime(2025, 1, 1, 10, 0, tzinfo=ET)
        assert scheduler.is_trading_day(new_year) is False

    def test_is_market_open_during_hours(self, scheduler):
        """Test market open detection during trading hours."""
        with patch.object(scheduler, "now_et") as mock_now:
            # 10:30 AM on a Monday
            mock_now.return_value = datetime(2024, 1, 8, 10, 30, tzinfo=ET)
            assert scheduler.is_market_open() is True

    def test_is_market_open_before_hours(self, scheduler):
        """Test market closed before trading hours."""
        with patch.object(scheduler, "now_et") as mock_now:
            # 8:00 AM on a Monday
            mock_now.return_value = datetime(2024, 1, 8, 8, 0, tzinfo=ET)
            assert scheduler.is_market_open() is False

    def test_is_market_open_after_hours(self, scheduler):
        """Test market closed after trading hours."""
        with patch.object(scheduler, "now_et") as mock_now:
            # 5:00 PM on a Monday
            mock_now.return_value = datetime(2024, 1, 8, 17, 0, tzinfo=ET)
            assert scheduler.is_market_open() is False

    def test_is_market_open_weekend(self, scheduler):
        """Test market closed on weekends."""
        with patch.object(scheduler, "now_et") as mock_now:
            # Saturday at 10:30 AM
            mock_now.return_value = datetime(2024, 1, 6, 10, 30, tzinfo=ET)
            assert scheduler.is_market_open() is False

    def test_get_next_market_open_same_day(self, scheduler):
        """Test next market open calculation for same day."""
        with patch.object(scheduler, "now_et") as mock_now:
            # 8:00 AM on Monday
            mock_now.return_value = datetime(2024, 1, 8, 8, 0, tzinfo=ET)
            next_open = scheduler.get_next_market_open()
            assert next_open.date() == datetime(2024, 1, 8).date()
            assert next_open.time() == time(9, 30)

    def test_get_next_market_open_next_day(self, scheduler):
        """Test next market open calculation for next day."""
        with patch.object(scheduler, "now_et") as mock_now:
            # 5:00 PM on Monday
            mock_now.return_value = datetime(2024, 1, 8, 17, 0, tzinfo=ET)
            next_open = scheduler.get_next_market_open()
            assert next_open.date() == datetime(2024, 1, 9).date()

    def test_get_next_market_open_over_weekend(self, scheduler):
        """Test next market open skips weekends."""
        with patch.object(scheduler, "now_et") as mock_now:
            # Friday 5:00 PM
            mock_now.return_value = datetime(2024, 1, 5, 17, 0, tzinfo=ET)
            next_open = scheduler.get_next_market_open()
            # Should be Monday
            assert next_open.weekday() == 0  # Monday
            assert next_open.date() == datetime(2024, 1, 8).date()

    def test_get_market_close_today(self, scheduler):
        """Test getting today's market close time."""
        with patch.object(scheduler, "now_et") as mock_now:
            mock_now.return_value = datetime(2024, 1, 8, 10, 0, tzinfo=ET)
            close = scheduler.get_market_close_today()
            assert close.time() == time(16, 0)
            assert close.date() == datetime(2024, 1, 8).date()

    def test_seconds_until_market_open(self, scheduler):
        """Test seconds calculation until market open."""
        with patch.object(scheduler, "now_et") as mock_now:
            # 9:00 AM - 30 minutes before open
            mock_now.return_value = datetime(2024, 1, 8, 9, 0, tzinfo=ET)
            seconds = scheduler.seconds_until_market_open()
            assert seconds == 30 * 60  # 30 minutes

    def test_seconds_until_market_close(self, scheduler):
        """Test seconds calculation until market close."""
        with patch.object(scheduler, "now_et") as mock_now:
            # 3:00 PM - 1 hour before close
            mock_now.return_value = datetime(2024, 1, 8, 15, 0, tzinfo=ET)
            seconds = scheduler.seconds_until_market_close()
            assert seconds == 60 * 60  # 1 hour


class TestAutoTrader:
    """Tests for AutoTrader."""

    @pytest.fixture
    def autotrader(self):
        """Create AutoTrader with mocked AppContext (canonical path)."""
        import logging

        from app.bootstrap import AppContext

        # Create mock config
        mock_config = MagicMock(
            autotrader=MagicMock(
                pre_market_buffer_minutes=15,
                post_market_delay_minutes=5,
                preflight_max_retries=3,
                preflight_retry_delay=1,
            )
        )

        # Create mock AppContext (canonical path)
        mock_ctx = AppContext(
            config=mock_config,
            logger=logging.getLogger("test"),
            mode="daemon",
            symbols=["AAPL", "MSFT"],
            root_path=ROOT,
            metadata={"broker": "alpaca"},
        )

        # Create AutoTrader with canonical AppContext
        trader = AutoTrader(ctx=mock_ctx, dry_run=True, update_data_days=5)
        return trader

    def test_init(self, autotrader):
        """Test AutoTrader initialization."""
        assert autotrader.symbols == ["AAPL", "MSFT"]
        assert autotrader.broker == "alpaca"
        assert autotrader.dry_run is True
        assert autotrader.state == AutoTraderState.INITIALIZING

    def test_set_state(self, autotrader):
        """Test state transitions."""
        autotrader._set_state(AutoTraderState.WAITING_FOR_MARKET)
        assert autotrader.state == AutoTraderState.WAITING_FOR_MARKET

        autotrader._set_state(AutoTraderState.TRADING)
        assert autotrader.state == AutoTraderState.TRADING

    def test_stop(self, autotrader):
        """Test stop signal."""
        autotrader.running = True
        autotrader.stop()
        assert autotrader.running is False

    def test_get_status(self, autotrader):
        """Test status reporting."""
        status = autotrader.get_status()
        assert "state" in status
        assert "running" in status
        assert "symbols" in status
        assert "broker" in status
        assert "dry_run" in status
        assert "stats" in status
        assert status["symbols"] == ["AAPL", "MSFT"]

    @pytest.mark.asyncio
    async def test_run_preflight_success(self, autotrader):
        """Test successful preflight check."""
        # Set running flag so network check loop runs
        autotrader.running = True

        # Mock the PreFlightChecker import
        mock_checker = MagicMock()
        mock_checker.run_all_checks = AsyncMock(return_value=True)

        with (
            patch.dict("sys.modules", {"preflight": MagicMock(PreFlightChecker=MagicMock(return_value=mock_checker))}),
            patch.object(autotrader, "_check_network_connectivity_sync", return_value=True),
        ):
            result = await autotrader._run_preflight()
            assert result is True

    @pytest.mark.asyncio
    async def test_run_preflight_failure(self, autotrader):
        """Test failed preflight check."""
        # Set running flag so network check loop runs
        autotrader.running = True

        with (
            patch("preflight.PreFlightChecker") as MockChecker,
            patch.object(autotrader, "_check_network_connectivity_sync", return_value=True),
        ):
            mock_checker = MagicMock()
            mock_checker.run_all_checks = AsyncMock(return_value=False)
            MockChecker.return_value = mock_checker

            result = await autotrader._run_preflight()
            assert result is False

    @pytest.mark.asyncio
    async def test_run_post_market(self, autotrader):
        """Test post-market data update."""
        autotrader.running = True
        autotrader.post_market_delay = 0  # Skip delay for test

        with patch("core.unified_data_pipeline.UnifiedDataPipeline") as MockPipeline:
            mock_pipeline = MagicMock()
            mock_pipeline.update_symbols = AsyncMock(return_value={"AAPL": 100, "MSFT": 100})
            MockPipeline.return_value = mock_pipeline

            with patch("asyncio.sleep", new_callable=AsyncMock):
                await autotrader._run_post_market()

            assert autotrader.stats["last_data_update"] is not None


class TestAutoTraderIntegration:
    """Integration tests for AutoTrader."""

    @pytest.mark.asyncio
    async def test_dry_run_session(self):
        """Test a complete dry run session."""
        import logging

        from app.bootstrap import AppContext

        # Create mock config
        mock_config = MagicMock(autotrader=MagicMock(pre_market_buffer_minutes=0, post_market_delay_minutes=0))

        # Create mock AppContext (canonical path)
        mock_ctx = AppContext(
            config=mock_config,
            logger=logging.getLogger("test"),
            mode="daemon",
            symbols=["AAPL"],
            root_path=ROOT,
            metadata={"broker": "alpaca"},
        )

        trader = AutoTrader(ctx=mock_ctx, dry_run=True)

        # Mock scheduler to say market is open briefly then closes
        call_count = [0]

        def mock_is_open():
            call_count[0] += 1
            return call_count[0] < 3

        trader.scheduler.is_market_open = mock_is_open
        trader.running = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await trader._run_trading_session()

        assert trader.stats["sessions_completed"] == 1
