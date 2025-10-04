#%% monitoring/app_main.py
import io, sys, asyncio
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
from PySide6 import QtWidgets
from qasync import QEventLoop

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.theme import apply_dark_palette
from monitoring.views.main_window import MainWindow
from monitoring.feeds.feeder import DataFeeder
from core.events.eventhandler import EventHandler
from core.events import events


def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)

    # --- Create and register the asyncio–Qt loop FIRST ---
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    # --- Then create window and event bus ---
    bus = EventHandler()
    feeder = DataFeeder()
    win = MainWindow()
    win.bus = bus
    win.show()

    app.aboutToQuit.connect(lambda: asyncio.create_task(feeder.stop()))

    async def startup():
        await asyncio.sleep(1)  # give Qt a moment to settle
        await feeder.start_safe()
        win._append_log("[INIT] Data feeder started")

        await asyncio.gather(
            win.bus.subscribe(events.EVENT_PNL_UPDATE, win._on_pnl_update),
            win.bus.subscribe(events.EVENT_NEW_TRADE, win._on_new_trade),
            win.bus.subscribe(events.EVENT_ORDER_STATUS, win._on_order_status),
            win.bus.subscribe(events.EVENT_ALERT, win._on_alert),
            win.bus.subscribe(events.EVENT_POSITION_UPDATE, win._on_position_update),
            win.bus.subscribe(events.EVENT_PRICE_UPDATE, win._on_price_update),
            win.bus.subscribe(events.EVENT_MANUAL_ORDER, win._on_manual_order),
        )
        win._append_log("[INIT] Backend event subscriptions ready")

    loop.create_task(startup())

    try:
        with loop:
            loop.run_forever()
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        # Do NOT call loop.run_until_complete here; qasync closes the loop on exit.
        # Let aboutToQuit handler stop the feeder; give Qt a chance to flush timers.
        try:
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass
        sys.exit(0)

if __name__ == "__main__":
    main()
