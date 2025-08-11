# strategies/strategy_registry/strategy_registry.py

# strategies/strategy_registry/strategy_registry.py
import os, importlib, inspect, re
from typing import Dict, Type
from core.base.base_strategy import BaseStrategy  # adjust path if needed

STRATEGY_CLASS_REGISTRY: Dict[str, Type[BaseStrategy]] = {}

def _camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def _register_with_aliases(cls: Type[BaseStrategy]) -> None:
    # explicit key on the class wins if present
    explicit = getattr(cls, "strategy_key", None)

    class_name = cls.__name__            # e.g., "SMAStrategy"
    module_tail = cls.__module__.split(".")[-1]  # e.g., "sma_strategy"
    base_snake = _camel_to_snake(class_name.replace("Strategy", ""))  # "sma"

    candidates = [
        module_tail,                      # "sma_strategy" (module filename)
        class_name,                       # "SMAStrategy"
        class_name.lower(),               # "smastrategy"
        base_snake,                       # "sma"
        f"{base_snake}_strategy",         # "sma_strategy"
    ]
    if explicit:
        candidates.insert(0, explicit.lower())

    for key in dict.fromkeys(k.lower() for k in candidates):
        STRATEGY_CLASS_REGISTRY[key] = cls

def load_strategies_from_directory(directory: str) -> None:
    for filename in os.listdir(directory):
        if not filename.endswith(".py") or filename.startswith("__"):
            continue
        mod_name = filename[:-3]
        module_path = f"strategies.strategy_registry.{mod_name}"
        try:
            module = importlib.import_module(module_path)
        except Exception as e:
            print(f"[Loader] Failed to import {module_path}: {e}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                _register_with_aliases(obj)

    # Ensure a default maps to SMA if available
    if "default" not in STRATEGY_CLASS_REGISTRY:
        for k in ("sma_strategy", "sma", "smastrategy", "smastrategy".lower()):
            if k in STRATEGY_CLASS_REGISTRY:
                STRATEGY_CLASS_REGISTRY["default"] = STRATEGY_CLASS_REGISTRY[k]
                break

# Auto-load at import
load_strategies_from_directory(os.path.dirname(__file__))
