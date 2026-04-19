"""
Stock Scanner — screens and ranks symbols for trading.

Usage:
    from scanner import get_scanner

    scanner = get_scanner(config)
    report = scanner.scan()

    # Or scan specific symbols
    report = scanner.scan(symbols=["AAPL", "MSFT", "GOOGL"])

    # Auto-update trade/watch lists
    report = scanner.scan(update_lists=True)
"""

from scanner.engine import ScannerEngine, get_scanner
from scanner.result import ScanReport, SymbolScore

__all__ = ["ScannerEngine", "get_scanner", "ScanReport", "SymbolScore"]
