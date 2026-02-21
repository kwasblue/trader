"""
Tests for StateReconciler

Coverage:
- Full sync from broker
- Position check and mismatch detection
- Reconciliation with auto-correct
- Order verification
- Periodic reconciliation
- Halt management
- Cash sync
- Severity classification
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import asyncio

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.state_reconciler import (
    StateReconciler,
    ReconcilerConfig,
    ReconcileResult,
    PositionMismatch,
    ReconcileAction,
)
from core.app_types import BrokerSnapshot, PositionView


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = AsyncMock()
    broker.get_account_info = AsyncMock()
    broker.get_order_status = AsyncMock()
    return broker


@pytest.fixture
def mock_portfolio():
    """Create a mock portfolio."""
    portfolio = MagicMock()
    portfolio.positions = {}
    portfolio.cash = 100000.0
    portfolio.sync_from_snapshot = MagicMock()
    return portfolio


@pytest.fixture
def reconciler(mock_broker, mock_portfolio):
    """Create a StateReconciler for testing."""
    with patch('core.state_reconciler.get_event_handler') as mock_eh:
        mock_eh.return_value = AsyncMock()
        return StateReconciler(
            broker=mock_broker,
            portfolio=mock_portfolio,
            config=ReconcilerConfig(),
        )


class TestReconcilerConfig:
    """Tests for ReconcilerConfig defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReconcilerConfig()

        assert config.qty_tolerance == 0.001
        assert config.price_tolerance == 0.01
        assert config.cash_tolerance == 1.0
        assert config.reconcile_interval == 15
        assert config.halt_on_critical is True
        assert config.auto_correct_minor is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReconcilerConfig(
            qty_tolerance=0.01,
            reconcile_interval=30,
            halt_on_critical=False,
        )

        assert config.qty_tolerance == 0.01
        assert config.reconcile_interval == 30
        assert config.halt_on_critical is False


class TestPositionMismatch:
    """Tests for PositionMismatch dataclass."""

    def test_mismatch_creation(self):
        """Test creating a position mismatch."""
        mismatch = PositionMismatch(
            symbol="AAPL",
            local_qty=100,
            broker_qty=95,
            local_avg_price=150.0,
            broker_avg_price=150.0,
            qty_diff=5.0,
            price_diff=0.0,
            timestamp="2024-01-01T12:00:00Z",
            severity="minor",
        )

        assert mismatch.symbol == "AAPL"
        assert mismatch.qty_diff == 5.0
        assert mismatch.severity == "minor"

    def test_mismatch_to_dict(self):
        """Test mismatch serialization."""
        mismatch = PositionMismatch(
            symbol="AAPL",
            local_qty=100,
            broker_qty=95,
            local_avg_price=150.0,
            broker_avg_price=150.0,
            qty_diff=5.0,
            price_diff=0.0,
            timestamp="2024-01-01T12:00:00Z",
            severity="minor",
        )

        d = mismatch.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["qty_diff"] == 5.0


class TestReconcileResult:
    """Tests for ReconcileResult dataclass."""

    def test_result_creation(self):
        """Test creating a reconcile result."""
        result = ReconcileResult(
            success=True,
            timestamp="2024-01-01T12:00:00Z",
            positions_checked=5,
            broker_positions=5,
            local_positions=5,
        )

        assert result.success is True
        assert result.positions_checked == 5

    def test_result_to_dict(self):
        """Test result serialization."""
        result = ReconcileResult(
            success=True,
            timestamp="2024-01-01T12:00:00Z",
            positions_checked=5,
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["positions_checked"] == 5


class TestFullSync:
    """Tests for full_sync method."""

    @pytest.mark.asyncio
    async def test_full_sync_success(self, reconciler, mock_broker, mock_portfolio):
        """Test successful full sync from broker."""
        snapshot = BrokerSnapshot(
            cash=100000.0,
            equity=150000.0,
            positions={"AAPL": PositionView(symbol="AAPL", qty=100, avg_entry_price=150.0)},
        )
        mock_broker.get_account_info.return_value = snapshot

        result = await reconciler.full_sync()

        assert result.success is True
        assert result.action_taken == "full_sync"
        mock_portfolio.sync_from_snapshot.assert_called_once_with(snapshot)

    @pytest.mark.asyncio
    async def test_full_sync_failure_no_snapshot(self, reconciler, mock_broker):
        """Test full sync when broker returns None."""
        mock_broker.get_account_info.return_value = None

        result = await reconciler.full_sync()

        assert result.success is False
        assert "sync_failed" in result.action_taken

    @pytest.mark.asyncio
    async def test_full_sync_exception(self, reconciler, mock_broker):
        """Test full sync when broker throws exception."""
        mock_broker.get_account_info.side_effect = Exception("Connection error")

        result = await reconciler.full_sync()

        assert result.success is False
        assert "sync_error" in result.action_taken


class TestCheckPositions:
    """Tests for check_positions method."""

    @pytest.mark.asyncio
    async def test_check_positions_match(self, reconciler, mock_broker, mock_portfolio):
        """Test check when positions match."""
        # Local position
        local_pos = MagicMock()
        local_pos.qty = 100
        local_pos.avg_price = 150.0
        mock_portfolio.positions = {"AAPL": local_pos}
        mock_portfolio.cash = 100000.0

        # Broker position
        snapshot = BrokerSnapshot(
            cash=100000.0,
            equity=115000.0,
            positions={"AAPL": PositionView(symbol="AAPL", qty=100, avg_entry_price=150.0)},
        )
        mock_broker.get_account_info.return_value = snapshot

        result = await reconciler.check_positions()

        assert result.success is True
        assert len(result.mismatches) == 0
        assert result.cash_match is True

    @pytest.mark.asyncio
    async def test_check_positions_qty_mismatch(self, reconciler, mock_broker, mock_portfolio):
        """Test check when quantities don't match."""
        # Local position: 100 shares
        local_pos = MagicMock()
        local_pos.qty = 100
        local_pos.avg_price = 150.0
        mock_portfolio.positions = {"AAPL": local_pos}

        # Broker position: 90 shares (10 share difference)
        snapshot = BrokerSnapshot(
            cash=100000.0,
            equity=113500.0,
            positions={"AAPL": PositionView(symbol="AAPL", qty=90, avg_entry_price=150.0)},
        )
        mock_broker.get_account_info.return_value = snapshot

        result = await reconciler.check_positions()

        assert result.success is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0].symbol == "AAPL"
        assert result.mismatches[0].qty_diff == 10.0

    @pytest.mark.asyncio
    async def test_check_positions_cash_mismatch(self, reconciler, mock_broker, mock_portfolio):
        """Test check when cash doesn't match."""
        mock_portfolio.positions = {}
        mock_portfolio.cash = 100000.0

        snapshot = BrokerSnapshot(
            cash=95000.0,  # $5000 difference
            equity=95000.0,
            positions={},
        )
        mock_broker.get_account_info.return_value = snapshot

        result = await reconciler.check_positions()

        assert result.cash_match is False

    @pytest.mark.asyncio
    async def test_check_positions_extra_broker_position(self, reconciler, mock_broker, mock_portfolio):
        """Test check when broker has position not in local."""
        mock_portfolio.positions = {}

        snapshot = BrokerSnapshot(
            cash=100000.0,
            equity=115000.0,
            positions={"AAPL": PositionView(symbol="AAPL", qty=100, avg_entry_price=150.0)},
        )
        mock_broker.get_account_info.return_value = snapshot

        result = await reconciler.check_positions()

        assert result.success is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0].local_qty == 0
        assert result.mismatches[0].broker_qty == 100

    @pytest.mark.asyncio
    async def test_check_positions_extra_local_position(self, reconciler, mock_broker, mock_portfolio):
        """Test check when local has position not at broker."""
        local_pos = MagicMock()
        local_pos.qty = 100
        local_pos.avg_price = 150.0
        mock_portfolio.positions = {"AAPL": local_pos}

        snapshot = BrokerSnapshot(
            cash=100000.0,
            equity=100000.0,
            positions={},
        )
        mock_broker.get_account_info.return_value = snapshot

        result = await reconciler.check_positions()

        assert result.success is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0].local_qty == 100
        assert result.mismatches[0].broker_qty == 0


class TestReconcile:
    """Tests for reconcile method."""

    @pytest.mark.asyncio
    async def test_reconcile_no_mismatches(self, reconciler, mock_broker, mock_portfolio):
        """Test reconcile when no mismatches."""
        mock_portfolio.positions = {}
        mock_portfolio.cash = 100000.0

        snapshot = BrokerSnapshot(
            cash=100000.0,
            equity=100000.0,
            positions={},
        )
        mock_broker.get_account_info.return_value = snapshot

        result = await reconciler.reconcile()

        assert result.success is True
        assert result.action_taken == "none_needed"

    @pytest.mark.asyncio
    async def test_reconcile_auto_correct_minor(self, reconciler, mock_broker, mock_portfolio):
        """Test reconcile auto-corrects minor mismatches."""
        reconciler.config.auto_correct_minor = True

        # Local position with small difference
        local_pos = MagicMock()
        local_pos.qty = 100
        local_pos.avg_price = 150.0
        mock_portfolio.positions = {"AAPL": local_pos}
        mock_portfolio.cash = 100000.0

        # Broker has slight difference (minor)
        snapshot = BrokerSnapshot(
            cash=100000.0,
            equity=114925.0,
            positions={"AAPL": PositionView(symbol="AAPL", qty=99.5, avg_entry_price=150.0)},
        )
        mock_broker.get_account_info.return_value = snapshot

        result = await reconciler.reconcile()

        assert "auto_corrected" in result.action_taken

    @pytest.mark.asyncio
    async def test_reconcile_halt_on_critical(self, mock_broker, mock_portfolio):
        """Test reconcile halts on critical mismatch."""
        with patch('core.state_reconciler.get_event_handler') as mock_eh:
            mock_eh.return_value = AsyncMock()

            halt_callback = MagicMock()
            reconciler = StateReconciler(
                broker=mock_broker,
                portfolio=mock_portfolio,
                config=ReconcilerConfig(halt_on_critical=True, major_qty_diff=10, minor_qty_diff=1),
                on_halt=halt_callback,
            )

            # Local position
            local_pos = MagicMock()
            local_pos.qty = 100
            local_pos.avg_price = 150.0
            mock_portfolio.positions = {"AAPL": local_pos}
            mock_portfolio.cash = 100000.0

            # Broker has huge difference (critical - above major_qty_diff of 10)
            snapshot = BrokerSnapshot(
                cash=100000.0,
                equity=107500.0,
                positions={"AAPL": PositionView(symbol="AAPL", qty=50, avg_entry_price=150.0)},
            )
            mock_broker.get_account_info.return_value = snapshot

            result = await reconciler.reconcile()

            assert reconciler.is_halted is True
            assert "halted" in result.action_taken
            halt_callback.assert_called_once()


class TestOrderVerification:
    """Tests for verify_order method."""

    @pytest.mark.asyncio
    async def test_verify_order_filled(self, reconciler, mock_broker):
        """Test verifying a filled order."""
        order_result = MagicMock()
        order_result.status = "filled"
        order_result.filled_qty = 100
        mock_broker.get_order_status.return_value = order_result

        result = await reconciler.verify_order(
            order_id="ORD123",
            symbol="AAPL",
            expected_qty=100,
            expected_side="buy",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_verify_order_qty_mismatch(self, reconciler, mock_broker):
        """Test verifying order with wrong quantity."""
        order_result = MagicMock()
        order_result.status = "filled"
        order_result.filled_qty = 90  # Expected 100
        mock_broker.get_order_status.return_value = order_result

        result = await reconciler.verify_order(
            order_id="ORD123",
            symbol="AAPL",
            expected_qty=100,
            expected_side="buy",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_order_cancelled(self, reconciler, mock_broker):
        """Test verifying a cancelled order."""
        order_result = MagicMock()
        order_result.status = "cancelled"
        order_result.filled_qty = 0
        mock_broker.get_order_status.return_value = order_result

        result = await reconciler.verify_order(
            order_id="ORD123",
            symbol="AAPL",
            expected_qty=100,
            expected_side="buy",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_order_not_found(self, reconciler, mock_broker):
        """Test verifying when order not found."""
        mock_broker.get_order_status.return_value = None
        reconciler.config.order_verify_retries = 1
        reconciler.config.order_verify_timeout = 0.1

        result = await reconciler.verify_order(
            order_id="ORD123",
            symbol="AAPL",
            expected_qty=100,
            expected_side="buy",
        )

        assert result is False


class TestPeriodicReconciliation:
    """Tests for periodic reconciliation."""

    @pytest.mark.asyncio
    async def test_start_periodic(self, reconciler):
        """Test starting periodic reconciliation."""
        await reconciler.start_periodic(interval_seconds=1)

        assert reconciler._running is True
        assert reconciler._task is not None

        await reconciler.stop_periodic()

    @pytest.mark.asyncio
    async def test_stop_periodic(self, reconciler):
        """Test stopping periodic reconciliation."""
        await reconciler.start_periodic(interval_seconds=1)
        await reconciler.stop_periodic()

        assert reconciler._running is False

    @pytest.mark.asyncio
    async def test_periodic_runs_reconcile(self, reconciler, mock_broker, mock_portfolio):
        """Test that periodic loop runs reconcile."""
        mock_portfolio.positions = {}
        mock_portfolio.cash = 100000.0

        snapshot = BrokerSnapshot(
            cash=100000.0,
            equity=100000.0,
            positions={},
        )
        mock_broker.get_account_info.return_value = snapshot

        await reconciler.start_periodic(interval_seconds=0.1)
        await asyncio.sleep(0.25)
        await reconciler.stop_periodic()

        # Should have called get_account_info at least once
        assert mock_broker.get_account_info.call_count >= 1


class TestHaltManagement:
    """Tests for halt management."""

    @pytest.mark.asyncio
    async def test_clear_halt(self, reconciler):
        """Test clearing halt state."""
        reconciler._halted = True

        reconciler.clear_halt()

        assert reconciler.is_halted is False

    @pytest.mark.asyncio
    async def test_force_sync_and_clear(self, reconciler, mock_broker, mock_portfolio):
        """Test force sync clears halt."""
        reconciler._halted = True

        snapshot = BrokerSnapshot(
            cash=100000.0,
            equity=100000.0,
            positions={},
        )
        mock_broker.get_account_info.return_value = snapshot

        result = await reconciler.force_sync_and_clear()

        assert result.success is True
        assert reconciler.is_halted is False


class TestCashSync:
    """Tests for cash sync."""

    @pytest.mark.asyncio
    async def test_sync_cash_success(self, reconciler, mock_broker, mock_portfolio):
        """Test successful cash sync."""
        mock_portfolio.cash = 100000.0

        snapshot = BrokerSnapshot(
            cash=95000.0,
            equity=95000.0,
            positions={},
        )
        mock_broker.get_account_info.return_value = snapshot

        result = await reconciler.sync_cash()

        assert result is True
        assert mock_portfolio.cash == 95000.0

    @pytest.mark.asyncio
    async def test_sync_cash_failure(self, reconciler, mock_broker):
        """Test cash sync failure."""
        mock_broker.get_account_info.return_value = None

        result = await reconciler.sync_cash()

        assert result is False


class TestSeverityClassification:
    """Tests for severity classification."""

    def test_minor_severity(self, reconciler):
        """Test minor severity classification."""
        reconciler.config.minor_qty_diff = 1.0
        reconciler.config.major_qty_diff = 10.0

        assert reconciler._classify_severity(0.5) == "minor"

    def test_major_severity(self, reconciler):
        """Test major severity classification."""
        reconciler.config.minor_qty_diff = 1.0
        reconciler.config.major_qty_diff = 10.0

        assert reconciler._classify_severity(5.0) == "major"

    def test_critical_severity(self, reconciler):
        """Test critical severity classification."""
        reconciler.config.minor_qty_diff = 1.0
        reconciler.config.major_qty_diff = 10.0

        assert reconciler._classify_severity(50.0) == "critical"


class TestReconcilerStats:
    """Tests for reconciler statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self, reconciler, mock_broker, mock_portfolio):
        """Test getting reconciler stats."""
        mock_portfolio.positions = {}
        mock_portfolio.cash = 100000.0

        snapshot = BrokerSnapshot(
            cash=100000.0,
            equity=100000.0,
            positions={},
        )
        mock_broker.get_account_info.return_value = snapshot

        await reconciler.full_sync()

        stats = reconciler.get_stats()

        assert stats["sync_count"] == 1
        assert stats["is_halted"] is False
        assert stats["is_running"] is False

    def test_last_result_property(self, reconciler):
        """Test last_result property."""
        assert reconciler.last_result is None

        reconciler._last_result = ReconcileResult(
            success=True,
            timestamp="2024-01-01T12:00:00Z",
            positions_checked=0,
        )

        assert reconciler.last_result is not None
        assert reconciler.last_result.success is True
