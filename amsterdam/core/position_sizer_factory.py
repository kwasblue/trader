# core/position_sizer_factory.py
"""
Position Sizer Factory - Registry pattern for position sizers.

Provides config-driven sizer selection without hardcoded imports.
Adding a new sizer only requires registering it here.

Usage:
    from core.position_sizer_factory import PositionSizerFactory

    # Get sizer class by name
    sizer_cls = PositionSizerFactory.get("kelly")
    sizer = sizer_cls(risk_percentage=0.02)

    # Or create directly from config
    sizer = PositionSizerFactory.create_from_config(config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.base.position_sizer_base import PositionSizerBase
    from core.config_loader import TradingConfig


class PositionSizerFactory:
    """
    Factory for creating position sizers.

    Supports registration of new sizers without modifying existing code.
    All sizers must inherit from PositionSizerBase.
    """

    _registry: dict[str, type[PositionSizerBase]] = {}
    _loaded: bool = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Lazy-load default sizers on first access."""
        if cls._loaded:
            return

        # Import and register default sizers
        try:
            from core.position_sizer import KellyPositionSizer, SimplePositionSizer

            # Register with multiple aliases for flexibility
            cls._registry["kelly"] = KellyPositionSizer
            cls._registry["dynamic"] = KellyPositionSizer  # Config default
            cls._registry["production"] = KellyPositionSizer

            cls._registry["simple"] = SimplePositionSizer
            cls._registry["basic"] = SimplePositionSizer
            cls._registry["legacy"] = SimplePositionSizer
        except ImportError:
            pass

        cls._loaded = True

    @classmethod
    def register(cls, name: str, sizer_class: type[PositionSizerBase]) -> None:
        """
        Register a sizer class.

        Args:
            name: Sizer name (e.g., "kelly", "simple", "custom")
            sizer_class: Sizer class that inherits from PositionSizerBase

        Example:
            from core.position_sizer_factory import PositionSizerFactory
            from my_sizer import CustomSizer

            PositionSizerFactory.register("custom", CustomSizer)
        """
        cls._registry[name.lower()] = sizer_class

    @classmethod
    def get(cls, sizer_type: str) -> type[PositionSizerBase]:
        """
        Get a sizer class by type name.

        Args:
            sizer_type: Sizer type name (case-insensitive)

        Returns:
            Sizer class

        Raises:
            ValueError: If sizer type not registered
        """
        cls._ensure_loaded()

        type_lower = sizer_type.lower()
        if type_lower not in cls._registry:
            available = ", ".join(sorted(set(cls._registry.keys())))
            raise ValueError(f"Unknown position sizer type: '{sizer_type}'. Available types: {available}")

        return cls._registry[type_lower]

    @classmethod
    def create(cls, sizer_type: str, risk_percentage: float, **kwargs) -> PositionSizerBase:
        """
        Create a sizer instance.

        Args:
            sizer_type: Sizer type name (case-insensitive)
            risk_percentage: Base risk percentage (required by all sizers)
            **kwargs: Additional arguments passed to sizer constructor

        Returns:
            Configured sizer instance

        Example:
            sizer = PositionSizerFactory.create(
                "kelly",
                risk_percentage=0.02,
                max_trade_pct=0.10,
                max_holding_pct=0.25,
            )
        """
        sizer_class = cls.get(sizer_type)
        return sizer_class(risk_percentage=risk_percentage, **kwargs)

    @classmethod
    def create_from_config(cls, config: TradingConfig | None = None) -> PositionSizerBase:
        """
        Create a sizer instance from TradingConfig.

        Args:
            config: TradingConfig instance. Uses global config if not provided.

        Returns:
            Configured sizer instance based on config.position_sizer settings

        Example:
            sizer = PositionSizerFactory.create_from_config(config)
        """
        if config is None:
            from core.config_loader import get_config

            config = get_config()

        ps = config.position_sizer
        sizer_class = cls.get(ps.type)

        # SimplePositionSizer only takes risk_percentage
        if ps.type.lower() in ("simple", "basic", "legacy"):
            return sizer_class(risk_percentage=ps.risk_percentage)

        # KellyPositionSizer takes full config
        return sizer_class(
            risk_percentage=ps.risk_percentage,
            max_trade_pct=ps.max_trade_pct,
            max_holding_pct=ps.max_holding_pct,
            fee_rate=ps.fee_rate,
            allow_fractional=ps.allow_fractional,
            lot_size=ps.lot_size,
        )

    @classmethod
    def available_types(cls) -> list[str]:
        """
        Get list of registered sizer type names.

        Returns:
            List of unique sizer type names
        """
        cls._ensure_loaded()
        # Return unique names (some are aliases)
        return sorted(set(cls._registry.keys()))

    @classmethod
    def is_registered(cls, sizer_type: str) -> bool:
        """
        Check if a sizer type is registered.

        Args:
            sizer_type: Sizer type name (case-insensitive)

        Returns:
            True if sizer type is registered
        """
        cls._ensure_loaded()
        return sizer_type.lower() in cls._registry
