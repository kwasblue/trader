"""
Core utilities for trading system reliability and operations.

Modules:
- retry: Retry logic with exponential backoff and circuit breaker
- health: Health check and monitoring
- logging_utils: Structured logging with correlation IDs
- persistence: State persistence for recovery
- shutdown: Graceful shutdown handling
"""
from core.utils.retry import (
    retry,
    async_retry,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    get_circuit_breaker,
    RetryConfig,
)
from core.utils.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    get_health_checker,
)
from core.utils.logging_utils import (
    get_correlation_id,
    set_correlation_id,
    generate_correlation_id,
    mask_sensitive_data,
    StructuredFormatter,
    ContextLogger,
    get_structured_logger,
    setup_structured_logging,
    with_correlation_id,
    async_with_correlation_id,
)
from core.utils.persistence import (
    StatePersistence,
    PersistedPosition,
    PersistedOrder,
    SystemState,
    get_persistence,
)
from core.utils.shutdown import (
    ShutdownManager,
    ShutdownPhase,
    get_shutdown_manager,
    create_shutdown_handlers,
)

__all__ = [
    # Retry
    'retry',
    'async_retry',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitOpenError',
    'get_circuit_breaker',
    'RetryConfig',
    # Health
    'HealthChecker',
    'HealthStatus',
    'ComponentHealth',
    'SystemHealth',
    'get_health_checker',
    # Logging
    'get_correlation_id',
    'set_correlation_id',
    'generate_correlation_id',
    'mask_sensitive_data',
    'StructuredFormatter',
    'ContextLogger',
    'get_structured_logger',
    'setup_structured_logging',
    'with_correlation_id',
    'async_with_correlation_id',
    # Persistence
    'StatePersistence',
    'PersistedPosition',
    'PersistedOrder',
    'SystemState',
    'get_persistence',
    # Shutdown
    'ShutdownManager',
    'ShutdownPhase',
    'get_shutdown_manager',
    'create_shutdown_handlers',
]
