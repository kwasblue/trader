from .sma_strategy import SMAStrategy
from .rsi_strategy import RSIStrategy

STRATEGY_MAP = {
    'sma': SMAStrategy,
    'rsi': RSIStrategy,
}

def load_strategy(name, params=None):
    cls = STRATEGY_MAP.get(name)
    if not cls:
        raise ValueError(f"Strategy '{name}' not found")
    return cls(params)