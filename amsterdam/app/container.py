"""
Amsterdam Canonical Path - Composition Root
============================================

This module is the COMPOSITION ROOT for the Amsterdam application.
It lazily initializes components only when needed.

CANONICAL PATH HIERARCHY
------------------------
    bootstrap_app() -> AppContext -> AppContainer -> RunnerFactory -> runner

Usage:
    from app.bootstrap import bootstrap_app
    from app.container import AppContainer

    ctx = bootstrap_app(mode='daemon', broker='alpaca', symbols=['AAPL'])
    container = AppContainer(ctx)
    runner = container.get_runner()  # Lazy-loaded via RunnerFactory
    await runner.run()

This module is part of the canonical path. Do not create runners or other
core components directly - always go through the container.
"""

from typing import TYPE_CHECKING

from app.bootstrap import AppContext

if TYPE_CHECKING:
    from core.base.base_live_runner import BaseLiveRunner
    from core.events.eventhandler import EventHandler


class AppContainer:
    """Dependency injection container for lazy component initialization

    Components are created on-demand and cached for reuse.

    Example:
        >>> from app.bootstrap import bootstrap_app
        >>> from app.container import AppContainer
        >>> ctx = bootstrap_app(mode='daemon', symbols=['AAPL'], broker='schwab')
        >>> container = AppContainer(ctx)
        >>> runner = container.get_runner()  # Lazy-loaded
        >>> event_handler = container.get_event_handler()  # Lazy-loaded
    """

    def __init__(self, context: AppContext):
        """Initialize container with application context

        Args:
            context: AppContext from bootstrap_app()
        """
        self.ctx = context
        self._runner: BaseLiveRunner | None = None
        self._event_handler: EventHandler | None = None

    def get_runner(self) -> "BaseLiveRunner":
        """Lazy-load runner from factory

        Creates runner on first call, returns cached instance on subsequent calls.

        Returns:
            BaseLiveRunner instance configured with broker, symbols, and config
        """
        if self._runner is None:
            from core.runner_factory import RunnerFactory

            broker = self.ctx.metadata.get("broker", "schwab")
            self._runner = RunnerFactory.create(broker=broker, symbols=self.ctx.symbols, config=self.ctx.config)
            self.ctx.logger.info(f"Runner initialized: {broker}")
        return self._runner

    def get_event_handler(self) -> "EventHandler":
        """Get singleton event handler

        Returns:
            EventHandler singleton instance
        """
        if self._event_handler is None:
            from core.events.eventhandler import get_event_handler

            self._event_handler = get_event_handler()
            self.ctx.logger.debug("Event handler initialized")
        return self._event_handler

    def cleanup(self):
        """Cleanup resources before shutdown

        Calls cleanup methods on components if available.
        Should be called before application exit.
        """
        self.ctx.logger.info("Cleaning up container resources")

        if self._runner is not None:
            if hasattr(self._runner, "cleanup"):
                try:
                    self._runner.cleanup()
                    self.ctx.logger.debug("Runner cleanup complete")
                except Exception as e:
                    self.ctx.logger.error(f"Runner cleanup failed: {e}")

        if self._event_handler is not None:
            if hasattr(self._event_handler, "cleanup"):
                try:
                    self._event_handler.cleanup()
                    self.ctx.logger.debug("Event handler cleanup complete")
                except Exception as e:
                    self.ctx.logger.error(f"Event handler cleanup failed: {e}")

        self.ctx.logger.info("Container cleanup complete")
