# Schwab Trader

A comprehensive algorithmic trading platform with support for Schwab and Alpaca brokers, featuring real-time streaming, backtesting, and a professional monitoring GUI.

## Features

- **Multi-Broker Support**: Trade with Schwab or Alpaca (paper/live)
- **Autonomous Trading (AutoTrader)**: Fully automated trading daemon with market hours awareness
- **Real-Time Streaming**: WebSocket-based price feeds with automatic reconnection
- **Strategy Framework**: Pluggable strategies with regime-based routing
- **Risk Management**: Drawdown monitoring, position sizing, trade gates
- **Professional GUI**: PySide6-based monitoring dashboard with real-time charts
- **Comprehensive Backtesting**: Vectorized backtester with walk-forward analysis and Monte Carlo simulation
- **Event-Driven Architecture**: Async event bus for decoupled components
- **Pre-Flight Checks**: Automated validation before trading sessions
- **Historical Data Management**: Unified data pipeline with multiple sources

