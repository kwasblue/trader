#%% monitoring/app_main_v2.py
import io, sys, asyncio
from pathlib import Path
from PySide6 import QtWidgets
from qasync import QEventLoop

# --- UTF-8 safe stdout ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- Path setup ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Imports ---
from monitoring.theme import apply_dark_palette
from monitoring.views.main_window2 import MainWindow  # ✅ new version
from core.events.eventhandler import EventHandler


def main():
    # --- Qt App + Theme ---
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)

    # --- Integrated asyncio event loop (Qt + asyncio) ---
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    # --- Global Event Bus ---
    bus = EventHandler()

    # --- Window ---
    win = MainWindow()
    win.bus = bus  # share global event handler
    win.show()

    # --- Graceful shutdown: stop feeder and tasks ---
    app.aboutToQuit.connect(lambda: asyncio.create_task(win.feeder.stop()))

    # --- Startup logic (asynchronous initialization) ---
    async def startup():
        await asyncio.sleep(0.5)  # let Qt initialize fully

        # Feeder starts and connects itself to EventBus internally
        await win.feeder.start_safe()
        win._append_log("[INIT] Data feeder started and subscribed to EventBus.")

        # The StateAggregator in MainWindow will automatically handle
        # incoming events and push merged snapshots to the GUI.
        win._append_log("[INIT] StateAggregator active and feeding GUI snapshots.")

    loop.create_task(startup())

    # --- Run event loop ---
    try:
        with loop:
            loop.run_forever()
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        try:
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
