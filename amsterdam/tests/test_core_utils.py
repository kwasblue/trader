"""
Tests for core utilities: retry, health, logging, persistence, shutdown.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.health import HealthChecker, HealthStatus
from core.utils.logging_utils import (
    StructuredFormatter,
    generate_correlation_id,
    get_correlation_id,
    mask_sensitive_data,
    set_correlation_id,
)
from core.utils.persistence import PersistedOrder, PersistedPosition, StatePersistence
from core.utils.retry import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
    async_retry,
    calculate_delay,
    retry,
)


class TestRetryDecorator(unittest.TestCase):
    """Test retry decorator functionality."""

    def test_successful_function_no_retry(self):
        """Function that succeeds should not retry."""
        call_count = 0

        @retry(max_attempts=3)
        def successful_fn():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_fn()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 1)

    def test_failing_function_retries(self):
        """Function that fails should retry."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def failing_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"

        result = failing_fn()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)

    def test_exhausted_retries_raises(self):
        """Function that always fails should raise after max attempts."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent error")

        with self.assertRaises(ValueError):
            always_fails()

        self.assertEqual(call_count, 3)


class TestAsyncRetry(unittest.TestCase):
    """Test async retry decorator."""

    def test_async_retry_success(self):
        """Async function that succeeds should not retry."""
        call_count = 0

        @async_retry(max_attempts=3)
        async def async_success():
            nonlocal call_count
            call_count += 1
            return "async_success"

        result = asyncio.run(async_success())
        self.assertEqual(result, "async_success")
        self.assertEqual(call_count, 1)

    def test_async_retry_with_failures(self):
        """Async function that fails should retry."""
        call_count = 0

        @async_retry(max_attempts=3, base_delay=0.01)
        async def async_failing():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary")
            return "recovered"

        result = asyncio.run(async_failing())
        self.assertEqual(result, "recovered")
        self.assertEqual(call_count, 2)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality."""

    def test_circuit_starts_closed(self):
        """Circuit should start in closed state."""
        cb = CircuitBreaker(name="test")
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_circuit_opens_on_failures(self):
        """Circuit should open after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(name="test", config=config)

        for _ in range(3):
            cb.record_failure(Exception("fail"))

        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_circuit_rejects_when_open(self):
        """Open circuit should reject requests."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=10.0)
        cb = CircuitBreaker(name="test", config=config)

        cb.record_failure(Exception("fail"))
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb._should_allow_request())

    def test_circuit_closes_on_success(self):
        """Circuit should close after successes in half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, success_threshold=2, timeout=0.01)
        cb = CircuitBreaker(name="test", config=config)

        cb.record_failure(Exception("fail"))
        self.assertEqual(cb.state, CircuitState.OPEN)

        # Wait for timeout
        import time

        time.sleep(0.02)

        # Should transition to half-open
        cb._should_allow_request()
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)

        # Record successes
        cb.record_success()
        cb.record_success()

        self.assertEqual(cb.state, CircuitState.CLOSED)


class TestDelayCalculation(unittest.TestCase):
    """Test exponential backoff delay calculation."""

    def test_exponential_increase(self):
        """Delay should increase exponentially."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        delay_0 = calculate_delay(0, config)
        delay_1 = calculate_delay(1, config)
        delay_2 = calculate_delay(2, config)

        self.assertEqual(delay_0, 1.0)
        self.assertEqual(delay_1, 2.0)
        self.assertEqual(delay_2, 4.0)

    def test_max_delay_cap(self):
        """Delay should not exceed max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=10.0, exponential_base=2.0, jitter=False)

        delay = calculate_delay(10, config)  # Would be 1024 without cap
        self.assertEqual(delay, 10.0)


class TestHealthChecker(unittest.TestCase):
    """Test health checker functionality."""

    def test_register_component(self):
        """Should register health check for component."""
        # Create fresh instance for testing
        checker = HealthChecker.__new__(HealthChecker)
        checker._initialized = False
        checker.__init__()

        async def check():
            return True

        checker.register("test_component", check)
        self.assertIn("test_component", checker._components)

    def test_check_healthy_component(self):
        """Should report healthy for passing check."""
        checker = HealthChecker.__new__(HealthChecker)
        checker._initialized = False
        checker.__init__()

        async def healthy_check():
            return True

        checker.register("healthy", healthy_check)

        async def run_check():
            return await checker.check_component("healthy")

        result = asyncio.run(run_check())
        self.assertEqual(result.status, HealthStatus.HEALTHY)

    def test_check_unhealthy_component(self):
        """Should report unhealthy for failing check."""
        checker = HealthChecker.__new__(HealthChecker)
        checker._initialized = False
        checker.__init__()

        async def failing_check():
            raise Exception("Connection failed")

        checker.register("failing", failing_check)

        async def run_check():
            return await checker.check_component("failing")

        result = asyncio.run(run_check())
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)


class TestMaskSensitiveData(unittest.TestCase):
    """Test sensitive data masking."""

    def test_mask_api_key(self):
        """Should mask API keys."""
        data = {"api_key": "sk_live_abc123xyz"}
        masked = mask_sensitive_data(data)
        self.assertEqual(masked["api_key"], "***MASKED***")

    def test_mask_password(self):
        """Should mask passwords."""
        data = {"password": "super_secret_123"}
        masked = mask_sensitive_data(data)
        self.assertEqual(masked["password"], "***MASKED***")

    def test_mask_nested_sensitive(self):
        """Should mask nested sensitive fields."""
        data = {"config": {"api_secret": "secret_value_here"}}
        masked = mask_sensitive_data(data)
        self.assertEqual(masked["config"]["api_secret"], "***MASKED***")

    def test_preserve_non_sensitive(self):
        """Should preserve non-sensitive fields."""
        data = {"username": "john", "symbol": "AAPL"}
        masked = mask_sensitive_data(data)
        self.assertEqual(masked["username"], "john")
        self.assertEqual(masked["symbol"], "AAPL")


class TestCorrelationId(unittest.TestCase):
    """Test correlation ID management."""

    def test_generate_correlation_id(self):
        """Should generate 8-character correlation ID."""
        cid = generate_correlation_id()
        self.assertEqual(len(cid), 8)

    def test_set_and_get_correlation_id(self):
        """Should set and get correlation ID."""
        set_correlation_id("test1234")
        self.assertEqual(get_correlation_id(), "test1234")


class TestPersistence(unittest.TestCase):
    """Test state persistence."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_trading.db")
        self.persistence = StatePersistence(self.db_path)

    def tearDown(self):
        """Clean up temporary database."""
        self.persistence.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)

    def test_save_and_get_position(self):
        """Should save and retrieve position."""
        position = PersistedPosition(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            side="long",
            opened_at=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        self.persistence.save_position(position)
        retrieved = self.persistence.get_position("AAPL")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.symbol, "AAPL")
        self.assertEqual(retrieved.quantity, 100)
        self.assertEqual(retrieved.avg_price, 150.0)

    def test_save_and_get_order(self):
        """Should save and retrieve order."""
        order = PersistedOrder(
            order_id="order_123",
            symbol="MSFT",
            side="buy",
            quantity=50,
            order_type="market",
            status="pending",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        self.persistence.save_order(order)
        retrieved = self.persistence.get_order("order_123")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.symbol, "MSFT")
        self.assertEqual(retrieved.quantity, 50)

    def test_get_all_positions(self):
        """Should retrieve all open positions."""
        for i, symbol in enumerate(["AAPL", "MSFT", "GOOGL"]):
            position = PersistedPosition(
                symbol=symbol,
                quantity=100 * (i + 1),
                avg_price=100.0,
                side="long",
                opened_at=datetime.now(timezone.utc).isoformat(),
                last_updated=datetime.now(timezone.utc).isoformat(),
            )
            self.persistence.save_position(position)

        positions = self.persistence.get_all_positions()
        self.assertEqual(len(positions), 3)

    def test_record_trade(self):
        """Should record trade history."""
        self.persistence.record_trade(
            order_id="order_456",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            pnl=500.0,
        )

        history = self.persistence.get_trade_history(symbol="AAPL")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["pnl"], 500.0)


class TestStructuredFormatter(unittest.TestCase):
    """Test structured log formatter."""

    def test_format_includes_required_fields(self):
        """Formatted log should include required fields."""
        import json
        import logging

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["logger"], "test_logger")
        self.assertEqual(data["message"], "Test message")
        self.assertIn("timestamp", data)
        self.assertIn("source", data)


if __name__ == "__main__":
    unittest.main()
