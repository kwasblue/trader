import importlib
import inspect
import pkgutil

from core.base.base_strategy import BaseStrategy

STRATEGY_MAP = {}

# Auto-discover all strategy subclasses
for _, module_name, _ in pkgutil.iter_modules(__path__):
    mod = importlib.import_module(f"{__name__}.{module_name}")
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
            key = name.replace("Strategy", "").lower()  # e.g., 'SMAStrategy' → 'sma'
            STRATEGY_MAP[key] = obj


def load_strategy(name, params=None):
    cls = STRATEGY_MAP.get(name)
    if not cls:
        raise ValueError(f"Strategy '{name}' not found")
    # Pass params as keyword arguments for strategies that use named args,
    # or as a positional dict for strategies that use self.params
    if params:
        try:
            return cls(**params)
        except TypeError:
            # Fallback for strategies expecting params as dict
            return cls(params)
    return cls()


def list_strategies():
    return list(STRATEGY_MAP.keys())
