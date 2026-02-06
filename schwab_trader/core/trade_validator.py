"""
Trade Validator - Centralized pre-trade validation

Provides a single point for validating trades before execution.
Checks multiple conditions and returns a comprehensive result.

Features:
- Reconciler halt enforcement
- Wash trade prevention (conflicting orders)
- Position state validation
- Buying power verification
- Position existence checks for exits
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from core.enums import PositionState

if TYPE_CHECKING:
    from core.logic.portfolio_state import PortfolioState
    from core.order_registry import OrderRegistry
    from core.state_reconciler import StateReconciler
    from core.base.base_broker_interface import BaseBrokerInterface


@dataclass
class ValidationResult:
    """
    Result of trade validation.

    Attributes:
        valid: Whether the trade passed all validation checks
        errors: List of validation errors (trade should not proceed)
        warnings: List of warnings (trade can proceed but note issues)
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def success(cls, warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(valid=True, warnings=warnings or [])

    @classmethod
    def failure(cls, errors: List[str], warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(valid=False, errors=errors, warnings=warnings or [])

    def add_error(self, error: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning (does not affect validity)."""
        self.warnings.append(warning)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class TradeValidator:
    """
    Centralized trade validation.

    Performs multiple validation checks before trade execution:
    1. Reconciler halt check
    2. Conflicting orders check (wash trade prevention)
    3. Position state check
    4. Buying power check (for buys)
    5. Position existence check (for exits)

    Example:
        validator = TradeValidator(portfolio, order_registry, reconciler, broker)

        result = await validator.validate(
            symbol="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            action_type="entry"
        )

        if not result.valid:
            logger.warning(f"Trade rejected: {result.errors}")
            return None
    """

    def __init__(
        self,
        portfolio: "PortfolioState",
        order_registry: "OrderRegistry",
        reconciler: Optional["StateReconciler"] = None,
        broker: Optional["BaseBrokerInterface"] = None,
    ):
        """
        Initialize trade validator.

        Args:
            portfolio: Portfolio state for position/cash checks
            order_registry: Order registry for pending order checks
            reconciler: State reconciler for halt checks (optional)
            broker: Broker interface for buying power (optional)
        """
        self.portfolio = portfolio
        self.order_registry = order_registry
        self.reconciler = reconciler
        self.broker = broker

    async def validate(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        action_type: str,
        position_state: Optional[PositionState] = None,
    ) -> ValidationResult:
        """
        Validate a trade before execution.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            qty: Quantity to trade
            price: Expected price
            action_type: Type of action ("entry", "exit", "partial_exit", "reversal")
            position_state: Current position state (optional)

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(valid=True)

        # 1. Check reconciler halt
        self._check_halt(result)
        if not result.valid:
            return result

        # 2. Check conflicting orders (wash trade prevention)
        await self._check_conflicting_orders(result, symbol, side)

        # 3. Check position state allows new orders
        self._check_position_state(result, position_state)

        # 4. Check buying power for buy orders
        if side.lower() == "buy" and action_type == "entry":
            await self._check_buying_power(result, qty, price)

        # 5. Check position exists for exits
        if action_type in ("exit", "partial_exit"):
            self._check_position_exists(result, symbol)

        # 6. Basic quantity check
        if qty <= 0:
            result.add_error(f"Invalid quantity: {qty}")

        return result

    def _check_halt(self, result: ValidationResult) -> None:
        """Check if trading is halted by reconciler."""
        if self.reconciler and self.reconciler.is_halted:
            result.add_error("Trading halted by reconciler due to state mismatch")

    async def _check_conflicting_orders(
        self,
        result: ValidationResult,
        symbol: str,
        side: str
    ) -> None:
        """Check for conflicting orders that would cause wash trade."""
        if self.order_registry is None:
            return

        conflicting = await self.order_registry.get_conflicting_orders(symbol, side)
        if conflicting:
            # Note: This is a warning, not error - we'll cancel them
            order_ids = [o.order_id for o in conflicting]
            result.add_warning(
                f"Conflicting orders will be cancelled: {order_ids}"
            )

    def _check_position_state(
        self,
        result: ValidationResult,
        position_state: Optional[PositionState]
    ) -> None:
        """Check if position state allows new orders."""
        if position_state is None:
            return

        if not position_state.allows_new_orders:
            result.add_error(
                f"Position state '{position_state.value}' does not allow new orders"
            )

    async def _check_buying_power(
        self,
        result: ValidationResult,
        qty: int,
        price: float
    ) -> None:
        """Check if sufficient buying power for buy order."""
        required = qty * price

        if self.broker:
            try:
                if hasattr(self.broker, 'get_buying_power'):
                    available = self.broker.get_buying_power()
                else:
                    available = self.broker.get_available_funds()

                if required > available:
                    result.add_error(
                        f"Insufficient buying power: need ${required:,.2f}, "
                        f"have ${available:,.2f}"
                    )
            except Exception as e:
                result.add_warning(f"Could not verify buying power: {e}")
        else:
            # Fall back to portfolio cash
            if required > self.portfolio.cash:
                result.add_error(
                    f"Insufficient cash: need ${required:,.2f}, "
                    f"have ${self.portfolio.cash:,.2f}"
                )

    def _check_position_exists(
        self,
        result: ValidationResult,
        symbol: str
    ) -> None:
        """Check that position exists for exit orders."""
        position = self.portfolio.positions.get(symbol)
        if not position or position.qty == 0:
            result.add_error(f"No position to exit for {symbol}")

    async def validate_entry(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        position_state: Optional[PositionState] = None,
    ) -> ValidationResult:
        """Convenience method for validating entry orders."""
        return await self.validate(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            action_type="entry",
            position_state=position_state,
        )

    async def validate_exit(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        position_state: Optional[PositionState] = None,
    ) -> ValidationResult:
        """Convenience method for validating exit orders."""
        return await self.validate(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            action_type="exit",
            position_state=position_state,
        )
