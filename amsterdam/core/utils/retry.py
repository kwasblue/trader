"""
Retry utilities with exponential backoff and circuit breaker pattern.

Provides decorators and utilities for resilient API calls.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)
    non_retryable_exceptions: tuple = ()


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # failures before opening
    success_threshold: int = 2  # successes in half-open before closing
    timeout: float = 30.0  # seconds before trying half-open
    excluded_exceptions: tuple = ()  # exceptions that don't count as failures


@dataclass
class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.

    Prevents cascading failures by stopping requests to failing services.
    """

    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit state."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"[CircuitBreaker:{self.name}] Transitioning to HALF_OPEN")
                    return True
            return False

        # HALF_OPEN - allow request to test
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"[CircuitBreaker:{self.name}] Circuit CLOSED after recovery")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def record_failure(self, exc: Exception) -> None:
        """Record a failed request."""
        # Check if exception is excluded
        if isinstance(exc, self.config.excluded_exceptions):
            return

        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # Immediately open on failure in half-open
            self.state = CircuitState.OPEN
            logger.warning(f"[CircuitBreaker:{self.name}] Circuit OPEN after half-open failure")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"[CircuitBreaker:{self.name}] Circuit OPEN after {self.failure_count} failures")


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, breaker_name: str):
        self.breaker_name = breaker_name
        super().__init__(f"Circuit breaker '{breaker_name}' is OPEN")


# Global circuit breaker registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name=name, config=config or CircuitBreakerConfig())
    return _circuit_breakers[name]


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for retry attempt with exponential backoff."""
    delay = config.base_delay * (config.exponential_base**attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add random jitter (0-25% of delay)
        delay = delay * (1 + random.random() * 0.25)

    return delay


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (Exception,),
    circuit_breaker: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for synchronous functions with retry logic.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        retryable_exceptions: Tuple of exceptions to retry on
        circuit_breaker: Optional circuit breaker name
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions,
    )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            breaker = get_circuit_breaker(circuit_breaker) if circuit_breaker else None

            for attempt in range(config.max_attempts):
                # Check circuit breaker
                if breaker and not breaker._should_allow_request():
                    raise CircuitOpenError(circuit_breaker)

                try:
                    result = func(*args, **kwargs)
                    if breaker:
                        breaker.record_success()
                    return result

                except config.retryable_exceptions as e:
                    if breaker:
                        breaker.record_failure(e)

                    if attempt == config.max_attempts - 1:
                        logger.error(f"[Retry] {func.__name__} failed after {config.max_attempts} attempts: {e}")
                        raise

                    delay = calculate_delay(attempt, config)
                    logger.warning(
                        f"[Retry] {func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)

            raise RuntimeError("Unexpected retry loop exit")

        return wrapper

    return decorator


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (Exception,),
    circuit_breaker: str | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for async functions with retry logic.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        retryable_exceptions: Tuple of exceptions to retry on
        circuit_breaker: Optional circuit breaker name
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions,
    )

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            breaker = get_circuit_breaker(circuit_breaker) if circuit_breaker else None

            for attempt in range(config.max_attempts):
                # Check circuit breaker
                if breaker and not breaker._should_allow_request():
                    raise CircuitOpenError(circuit_breaker)

                try:
                    result = await func(*args, **kwargs)
                    if breaker:
                        breaker.record_success()
                    return result

                except config.retryable_exceptions as e:
                    if breaker:
                        breaker.record_failure(e)

                    if attempt == config.max_attempts - 1:
                        logger.error(f"[Retry] {func.__name__} failed after {config.max_attempts} attempts: {e}")
                        raise

                    delay = calculate_delay(attempt, config)
                    logger.warning(
                        f"[Retry] {func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)

            raise RuntimeError("Unexpected retry loop exit")

        return wrapper

    return decorator
