"""
Base criterion and criterion registry for the scanner.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from scanner.providers.base import ProviderData

logger = logging.getLogger(__name__)


class BaseCriterion(ABC):
    """Abstract base for all screening criteria. Returns a score from 0.0 to 1.0."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def required_providers(self) -> list[str]:
        return ["technical"]

    @abstractmethod
    def evaluate(self, symbol: str, provider_data: dict[str, ProviderData]) -> float:
        """Evaluate a symbol and return a score between 0.0 and 1.0."""
        ...


# --- Registry ---

CRITERION_REGISTRY: dict[str, type[BaseCriterion]] = {}


def register_criterion(cls: type[BaseCriterion]) -> type[BaseCriterion]:
    """Decorator to register a criterion class."""
    cls.__new__(cls)
    # Get the name from a temporary instance
    cls_name = cls.__name__
    # We'll register by a snake_case key derived from class name
    key = cls_name
    CRITERION_REGISTRY[key] = cls
    return cls


def create_criterion(config: dict[str, Any]) -> BaseCriterion:
    """Create a criterion from a config dict like {"name": "momentum_breakout", "params": {}}."""
    name = config["name"]
    params = config.get("params", {})

    # Look up by name in registry
    for key, cls in CRITERION_REGISTRY.items():
        # Match against the class's .name property
        try:
            instance = cls(**params)
            if instance.name == name:
                return instance
        except Exception:
            continue

    raise ValueError(f"Unknown criterion: {name}. Available: {list_criteria()}")


def list_criteria() -> list[str]:
    """Return names of all registered criteria."""
    result = []
    for cls in CRITERION_REGISTRY.values():
        try:
            instance = cls.__new__(cls)
            # Call __init__ with no extra args to get the name
            BaseCriterion.__init__(instance)
            result.append(cls.__name__)
        except Exception:
            result.append(cls.__name__)
    return result
