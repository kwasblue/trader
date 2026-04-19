"""
Universe provider — resolves which symbols the scanner should screen.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# Fallback mega-caps if sp500.json is missing
FALLBACK_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK.B",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "JNJ",
    "WMT",
    "MA",
]


def get_universe(source: str = "sp500", custom_symbols: list[str] | None = None) -> list[str]:
    """
    Resolve the screening universe based on source type.

    Args:
        source: One of "sp500", "custom", "watchlist", "all"
        custom_symbols: Used when source is "custom"

    Returns:
        List of uppercase symbol strings
    """
    if source == "sp500":
        return _load_sp500()
    elif source == "custom":
        return [s.upper().strip() for s in (custom_symbols or [])]
    elif source == "watchlist":
        return _load_from_list_manager("watch")
    elif source == "all":
        return _load_from_list_manager("all")
    else:
        logger.warning(f"Unknown universe source '{source}', falling back to sp500")
        return _load_sp500()


def _load_sp500() -> list[str]:
    sp500_path = DATA_DIR / "sp500.json"
    if sp500_path.exists():
        try:
            with open(sp500_path) as f:
                symbols = json.load(f)
            if isinstance(symbols, list) and symbols:
                return [s.upper().strip() for s in symbols]
        except Exception as e:
            logger.warning(f"Failed to load sp500.json: {e}")

    logger.info("Using fallback mega-cap symbols")
    return FALLBACK_SYMBOLS.copy()


def _load_from_list_manager(list_type: str) -> list[str]:
    try:
        from core.symbol_list_manager import get_list_manager

        manager = get_list_manager()
        if list_type == "watch":
            return manager.get_watch_list()
        elif list_type == "trade":
            return manager.get_trade_list()
        else:
            # "all" = union of trade + watch
            return list(set(manager.get_trade_list() + manager.get_watch_list()))
    except Exception as e:
        logger.warning(f"Failed to load from SymbolListManager: {e}")
        return FALLBACK_SYMBOLS.copy()
