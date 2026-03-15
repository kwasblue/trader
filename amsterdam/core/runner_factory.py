# core/runner_factory.py
"""
Runner Factory - Registry pattern for live trading runners.

Provides config-driven runner selection without hardcoded imports.
Adding a new broker only requires registering it here.

Usage:
    from core.runner_factory import RunnerFactory

    # Get runner class by name
    runner_cls = RunnerFactory.get("alpaca")
    runner = runner_cls(symbols=["AAPL"], config=config)

    # Or create directly
    runner = RunnerFactory.create("schwab", symbols=["AAPL"], config=config)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Type

if TYPE_CHECKING:
    from core.base.base_live_runner import BaseLiveRunner
    from core.config_loader import TradingConfig


class RunnerFactory:
    """
    Factory for creating live trading runners.

    Supports registration of new runners without modifying existing code.
    All runners must inherit from BaseLiveRunner.
    """

    _registry: Dict[str, Type["BaseLiveRunner"]] = {}
    _loaded: bool = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Lazy-load default runners on first access."""
        if cls._loaded:
            return

        # Import and register default runners
        # These imports are deferred to avoid circular dependencies
        try:
            from core.alpaca_runner import AlpacaLiveRunner
            cls._registry["alpaca"] = AlpacaLiveRunner
        except ImportError:
            pass  # Alpaca dependencies not installed

        try:
            from core.schwab_runner import SchwabLiveRunner
            cls._registry["schwab"] = SchwabLiveRunner
        except ImportError:
            pass  # Schwab dependencies not installed

        try:
            from core.schwab_runner_multitf import SchwabLiveRunnerMultiTF
            cls._registry["schwab_multitf"] = SchwabLiveRunnerMultiTF
            cls._registry["schwab-mtf"] = SchwabLiveRunnerMultiTF  # Alias
        except ImportError:
            pass  # Multi-timeframe dependencies not installed

        cls._loaded = True

    @classmethod
    def register(cls, name: str, runner_class: Type["BaseLiveRunner"]) -> None:
        """
        Register a runner class.

        Args:
            name: Broker name (e.g., "alpaca", "schwab", "ibkr")
            runner_class: Runner class that inherits from BaseLiveRunner

        Example:
            from core.runner_factory import RunnerFactory
            from my_broker.runner import MyBrokerRunner

            RunnerFactory.register("mybroker", MyBrokerRunner)
        """
        cls._registry[name.lower()] = runner_class

    @classmethod
    def get(cls, broker: str) -> Type["BaseLiveRunner"]:
        """
        Get a runner class by broker name.

        Args:
            broker: Broker name (case-insensitive)

        Returns:
            Runner class

        Raises:
            ValueError: If broker not registered
        """
        cls._ensure_loaded()

        broker_lower = broker.lower()
        if broker_lower not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown broker: '{broker}'. "
                f"Available brokers: {available}"
            )

        return cls._registry[broker_lower]

    @classmethod
    def create(
        cls,
        broker: str,
        symbols: List[str],
        config: Optional["TradingConfig"] = None,
        **kwargs
    ) -> "BaseLiveRunner":
        """
        Create a runner instance.

        Args:
            broker: Broker name (case-insensitive)
            symbols: List of symbols to trade
            config: Optional TradingConfig
            **kwargs: Additional arguments passed to runner constructor

        Returns:
            Configured runner instance

        Example:
            runner = RunnerFactory.create(
                "alpaca",
                symbols=["AAPL", "MSFT"],
                config=config
            )
            await runner.run()
        """
        runner_class = cls.get(broker)
        return runner_class(symbols=symbols, config=config, **kwargs)

    @classmethod
    def available_brokers(cls) -> List[str]:
        """
        Get list of registered broker names.

        Returns:
            List of broker names
        """
        cls._ensure_loaded()
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, broker: str) -> bool:
        """
        Check if a broker is registered.

        Args:
            broker: Broker name (case-insensitive)

        Returns:
            True if broker is registered
        """
        cls._ensure_loaded()
        return broker.lower() in cls._registry
