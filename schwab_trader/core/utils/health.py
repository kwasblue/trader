"""
Health check and monitoring utilities.

Provides health status tracking for system components.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Awaitable
import threading

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "consecutive_failures": self.consecutive_failures,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class SystemHealth:
    """Aggregate health status for the system."""
    status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: datetime
    uptime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "components": {
                name: comp.to_dict() for name, comp in self.components.items()
            },
        }


class HealthChecker:
    """
    Central health check manager for all system components.

    Features:
    - Register health check functions for components
    - Periodic background health checks
    - Aggregate health status
    - Health history tracking
    """

    _instance: Optional["HealthChecker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "HealthChecker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._components: Dict[str, ComponentHealth] = {}
        self._checks: Dict[str, Callable[[], Awaitable[bool]]] = {}
        self._check_interval: float = 30.0  # seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_time = time.monotonic()
        self._initialized = True

    def register(
        self,
        name: str,
        check_func: Callable[[], Awaitable[bool]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a health check for a component.

        Args:
            name: Component name
            check_func: Async function that returns True if healthy
            metadata: Optional metadata about the component
        """
        self._components[name] = ComponentHealth(
            name=name,
            metadata=metadata or {},
        )
        self._checks[name] = check_func
        logger.info(f"[HealthChecker] Registered component: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a component."""
        self._components.pop(name, None)
        self._checks.pop(name, None)

    async def check_component(self, name: str) -> ComponentHealth:
        """Run health check for a specific component."""
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Component not registered",
            )

        component = self._components[name]
        check_func = self._checks[name]

        start_time = time.monotonic()
        try:
            is_healthy = await asyncio.wait_for(check_func(), timeout=10.0)
            latency = (time.monotonic() - start_time) * 1000

            component.last_check = datetime.now(timezone.utc)
            component.latency_ms = latency

            if is_healthy:
                component.status = HealthStatus.HEALTHY
                component.message = "OK"
                component.last_success = component.last_check
                component.consecutive_failures = 0
            else:
                component.consecutive_failures += 1
                if component.consecutive_failures >= 3:
                    component.status = HealthStatus.UNHEALTHY
                else:
                    component.status = HealthStatus.DEGRADED
                component.message = "Check returned false"

        except asyncio.TimeoutError:
            component.status = HealthStatus.UNHEALTHY
            component.message = "Health check timed out"
            component.consecutive_failures += 1
            component.last_check = datetime.now(timezone.utc)

        except Exception as e:
            component.status = HealthStatus.UNHEALTHY
            component.message = f"Check failed: {str(e)[:100]}"
            component.consecutive_failures += 1
            component.last_check = datetime.now(timezone.utc)
            logger.exception(f"[HealthChecker] Check failed for {name}")

        return component

    async def check_all(self) -> SystemHealth:
        """Run health checks for all registered components."""
        tasks = [self.check_component(name) for name in self._checks]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Determine aggregate status
        statuses = [c.status for c in self._components.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            aggregate = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            aggregate = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            aggregate = HealthStatus.DEGRADED
        else:
            aggregate = HealthStatus.UNKNOWN

        return SystemHealth(
            status=aggregate,
            components=dict(self._components),
            timestamp=datetime.now(timezone.utc),
            uptime_seconds=time.monotonic() - self._start_time,
        )

    async def start(self, interval: float = 30.0) -> None:
        """Start periodic health checks."""
        if self._running:
            return

        self._check_interval = interval
        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info(f"[HealthChecker] Started with {interval}s interval")

    async def stop(self) -> None:
        """Stop periodic health checks."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[HealthChecker] Stopped")

    async def _check_loop(self) -> None:
        """Background loop for periodic health checks."""
        while self._running:
            try:
                await self.check_all()
            except Exception as e:
                logger.exception(f"[HealthChecker] Error in check loop: {e}")

            await asyncio.sleep(self._check_interval)

    def get_status(self) -> Dict[str, Any]:
        """Get current health status synchronously (cached)."""
        statuses = [c.status for c in self._components.values()]

        if not statuses:
            aggregate = HealthStatus.UNKNOWN
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            aggregate = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            aggregate = HealthStatus.UNHEALTHY
        else:
            aggregate = HealthStatus.DEGRADED

        return {
            "status": aggregate.value,
            "uptime_seconds": time.monotonic() - self._start_time,
            "components": {
                name: comp.to_dict() for name, comp in self._components.items()
            },
        }


def get_health_checker() -> HealthChecker:
    """Get the singleton health checker instance."""
    return HealthChecker()
