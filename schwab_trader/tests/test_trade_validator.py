"""
Tests for TradeValidator - Pre-trade validation

Tests cover:
- Halt enforcement
- Conflicting order detection
- Position state validation
- Buying power checks
- Position existence checks for exits
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.trade_validator import TradeValidator, ValidationResult
from core.order_registry import OrderRegistry
from core.enums import PositionState


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        result = ValidationResult.success()
        assert result.valid is True
        assert len(result.errors) == 0

    def test_success_with_warnings(self):
        """Test success result with warnings."""
        result = ValidationResult.success(warnings=["Minor issue"])
        assert result.valid is True
        assert len(result.warnings) == 1

    def test_failure_result(self):
        """Test creating a failure result."""
        result = ValidationResult.failure(errors=["Critical error"])
        assert result.valid is False
        assert len(result.errors) == 1

    def test_add_error(self):
        """Test adding an error."""
        result = ValidationResult(valid=True)
        result.add_error("Something went wrong")

        assert result.valid is False
        assert "Something went wrong" in result.errors

    def test_add_warning(self):
        """Test adding a warning."""
        result = ValidationResult(valid=True)
        result.add_warning("Minor issue")

        assert result.valid is True  # Warnings don't affect validity
        assert "Minor issue" in result.warnings

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = ValidationResult(
            valid=False,
            errors=["Error 1"],
            warnings=["Warning 1"]
        )
        d = result.to_dict()

        assert d["valid"] is False
        assert "Error 1" in d["errors"]
        assert "Warning 1" in d["warnings"]


class TestTradeValidator:
    """Test TradeValidator functionality."""

    @pytest.fixture
    def mock_portfolio(self):
        """Create mock portfolio."""
        portfolio = MagicMock()
        portfolio.cash = 100000.0
        portfolio.positions = {}
        portfolio.get_position_state = MagicMock(return_value=PositionState.NONE)
        return portfolio

    @pytest.fixture
    def mock_registry(self):
        """Create mock order registry."""
        registry = MagicMock(spec=OrderRegistry)
        registry.get_conflicting_orders = AsyncMock(return_value=[])
        return registry

    @pytest.fixture
    def mock_reconciler(self):
        """Create mock reconciler."""
        reconciler = MagicMock()
        reconciler.is_halted = False
        return reconciler

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = MagicMock()
        broker.get_buying_power = MagicMock(return_value=100000.0)
        return broker

    @pytest.fixture
    def validator(self, mock_portfolio, mock_registry, mock_reconciler, mock_broker):
        """Create validator with all mocks."""
        return TradeValidator(
            portfolio=mock_portfolio,
            order_registry=mock_registry,
            reconciler=mock_reconciler,
            broker=mock_broker,
        )

    @pytest.mark.asyncio
    async def test_valid_entry_trade(self, validator):
        """Test validation passes for valid entry trade."""
        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            action_type="entry",
        )

        assert result.valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_halt_enforcement(self, validator, mock_reconciler):
        """Test trading is blocked when halted."""
        mock_reconciler.is_halted = True

        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            action_type="entry",
        )

        assert result.valid is False
        assert any("halted" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_conflicting_orders_warning(self, validator, mock_registry):
        """Test conflicting orders generate warning."""
        # Mock conflicting order
        mock_order = MagicMock()
        mock_order.order_id = "ORD123"
        mock_registry.get_conflicting_orders = AsyncMock(return_value=[mock_order])

        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            action_type="entry",
        )

        # Should still be valid (warning, not error)
        assert result.valid is True
        assert len(result.warnings) > 0
        assert any("cancel" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_position_state_pending_entry(self, validator):
        """Test validation fails when position state doesn't allow orders."""
        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            action_type="entry",
            position_state=PositionState.PENDING_ENTRY,
        )

        assert result.valid is False
        assert any("does not allow" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_position_state_none_allows_entry(self, validator):
        """Test NONE state allows entry orders."""
        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            action_type="entry",
            position_state=PositionState.NONE,
        )

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_position_state_open_allows_exit(self, validator, mock_portfolio):
        """Test OPEN state allows exit orders."""
        # Set up position for exit
        mock_position = MagicMock()
        mock_position.qty = 10
        mock_portfolio.positions = {"AAPL": mock_position}

        result = await validator.validate(
            symbol="AAPL",
            side="sell",
            qty=10,
            price=150.0,
            action_type="exit",
            position_state=PositionState.OPEN,
        )

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_insufficient_buying_power(self, validator, mock_broker):
        """Test validation fails with insufficient buying power."""
        mock_broker.get_buying_power = MagicMock(return_value=100.0)

        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,  # Total = $1500, but only $100 available
            action_type="entry",
        )

        assert result.valid is False
        assert any("buying power" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_exit_without_position(self, validator, mock_portfolio):
        """Test exit validation fails without position."""
        mock_portfolio.positions = {}

        result = await validator.validate(
            symbol="AAPL",
            side="sell",
            qty=10,
            price=150.0,
            action_type="exit",
        )

        assert result.valid is False
        assert any("no position" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_exit_with_position(self, validator, mock_portfolio):
        """Test exit validation passes with position."""
        mock_position = MagicMock()
        mock_position.qty = 10
        mock_portfolio.positions = {"AAPL": mock_position}

        result = await validator.validate(
            symbol="AAPL",
            side="sell",
            qty=10,
            price=150.0,
            action_type="exit",
        )

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_invalid_quantity(self, validator):
        """Test validation fails with invalid quantity."""
        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=0,
            price=150.0,
            action_type="entry",
        )

        assert result.valid is False
        assert any("quantity" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_negative_quantity(self, validator):
        """Test validation fails with negative quantity."""
        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=-10,
            price=150.0,
            action_type="entry",
        )

        assert result.valid is False

    @pytest.mark.asyncio
    async def test_validate_entry_convenience(self, validator):
        """Test validate_entry convenience method."""
        result = await validator.validate_entry(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
        )

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_validate_exit_convenience(self, validator, mock_portfolio):
        """Test validate_exit convenience method."""
        mock_position = MagicMock()
        mock_position.qty = 10
        mock_portfolio.positions = {"AAPL": mock_position}

        result = await validator.validate_exit(
            symbol="AAPL",
            side="sell",
            qty=10,
            price=150.0,
        )

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_broker_error_generates_warning(self, validator, mock_broker):
        """Test broker errors generate warnings, not failures."""
        mock_broker.get_buying_power = MagicMock(side_effect=Exception("API error"))

        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            action_type="entry",
        )

        # Should still be valid (warning about verification)
        assert result.valid is True
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_no_reconciler(self, mock_portfolio, mock_registry, mock_broker):
        """Test validation works without reconciler."""
        validator = TradeValidator(
            portfolio=mock_portfolio,
            order_registry=mock_registry,
            reconciler=None,  # No reconciler
            broker=mock_broker,
        )

        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            action_type="entry",
        )

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_no_broker_uses_portfolio_cash(self, mock_portfolio, mock_registry):
        """Test validation uses portfolio cash when broker not available."""
        mock_portfolio.cash = 100.0  # Low cash

        validator = TradeValidator(
            portfolio=mock_portfolio,
            order_registry=mock_registry,
            reconciler=None,
            broker=None,  # No broker
        )

        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,  # Total $1500, only $100 cash
            action_type="entry",
        )

        assert result.valid is False
        assert any("cash" in e.lower() for e in result.errors)
