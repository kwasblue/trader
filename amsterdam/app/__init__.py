"""Amsterdam Trading System - Application Bootstrap Module

This module provides unified initialization for all Amsterdam entrypoints.
"""

from app.bootstrap import bootstrap_app, AppContext
from app.container import AppContainer

__all__ = ['bootstrap_app', 'AppContext', 'AppContainer']
