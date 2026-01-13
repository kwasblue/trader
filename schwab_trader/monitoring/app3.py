"""
Production-Ready Trading GUI System - Complete Refactored Version
All files with proper error handling, memory management, and type safety
"""

# ============================================================================
# 1. app_main.py - Entry Point
# ============================================================================
import io
import sys
import asyncio
import logging
from pathlib import Path
from PySide6 import QtWidgets, QtCore
from qasync import QEventLoop

# UTF-8 safe stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Path setup
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TradingGUI")


class TradingApplication:
    """Main application wrapper with proper lifecycle management"""
    
    def __init__(self):
        self.app = None
        self.loop = None
        self.window = None
        self.startup_task = None
        
    async def startup(self):
        """Async startup sequence with error handling"""
        try:
            await asyncio.sleep(0.5)  # Let Qt initialize
            
            if not self.window:
                raise RuntimeError("Window not initialized")
            
            # Start data feeder
            logger.info("Starting data feeder...")
            await self.window.feeder.start_safe()
            self.window._append_log("[INIT] Data feeder started successfully.")
            
            # State aggregator is already wired via signals
            self.window._append_log("[INIT] StateAggregator active and feeding GUI.")
            
            logger.info("Application startup complete")
            
        except Exception as e:
            logger.exception(f"Startup failed: {e}")
            if self.window:
                self.window._append_log(f"[ERROR] Startup failed: {e}")
    
    async def shutdown(self):
        """Graceful shutdown sequence"""
        try:
            logger.info("Shutting down...")
            
            if self.window and hasattr(self.window, 'feeder'):
                await self.window.feeder.stop()
            
            if self.startup_task and not self.startup_task.done():
                self.startup_task.cancel()
                try:
                    await self.startup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def run(self):
        """Main run loop"""
        try:
            # Create Qt application
            self.app = QtWidgets.QApplication(sys.argv)
            
            # Apply theme
            from monitoring.theme import apply_dark_palette
            apply_dark_palette(self.app)
            
            # Setup event loop
            self.loop = QEventLoop(self.app)
            asyncio.set_event_loop(self.loop)
            
            # Create window
            from monitoring.views.main_window2 import MainWindow
            from core.events.eventhandler import EventHandler
            
            self.window = MainWindow()
            self.window.bus = EventHandler()
            self.window.show()
            
            # Wire shutdown
            self.app.aboutToQuit.connect(
                lambda: asyncio.create_task(self.shutdown())
            )
            
            # Schedule startup
            self.startup_task = self.loop.create_task(self.startup())
            
            # Run
            with self.loop:
                self.loop.run_forever()
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.exception(f"Fatal error: {e}")
        finally:
            # Final cleanup
            QtWidgets.QApplication.processEvents()
            sys.exit(0)


def main():
    app = TradingApplication()
    app.run()


if __name__ == "__main__":
    main()
