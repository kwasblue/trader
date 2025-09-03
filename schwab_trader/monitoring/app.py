import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # .../schwab_trader
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# =====================================================
# APP ENTRY / SIGNAL WIRING
# =====================================================

from PySide6 import QtWidgets, QtCore, QtGui
from typing import List, Dict, Optional
import sys
import random
import pandas as pd
import numpy as np
import pyqtgraph as pg
from monitoring.theme import apply_dark_palette
from monitoring.views.main_window import MainWindow
from monitoring.feeds.feeder import DataFeeder
import datetime as dt
from monitoring.widgets.candles import Candles

def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)

    win = MainWindow(); win.show()
    feeder = DataFeeder()

    # Connect UI -> Feeder (in your system, connect to your event bus / broker)
    win.ctrl.halt_changed.connect(feeder.set_halted)
    win.ctrl.flatten_all.connect(feeder.do_flatten_all)
    win.ctrl.cancel_all.connect(feeder.do_cancel_all)
    win.ctrl.flatten_symbol.connect(feeder.do_flatten_symbol)
    win.ctrl.manual_order.connect(feeder.do_manual_order)

    # Dashboard wires
    feeder.s.symbols.connect(win.pos_model.replace_rows)
    feeder.s.equity_point.connect(lambda d,v: (win._eq_x.append(len(win._eq_x)), win._eq_y.append(v), win.eq_curve.setData(win._eq_x, win._eq_y)))

    # Guardrail: monitor day PnL vs limit and auto-halt
    win._day_realized_running = 0.0
    def _guardrail_check(day_realized: float):
        win._day_realized_running += float(day_realized)
        limit = float(win.daily_loss_limit_spin.value()) if hasattr(win, 'daily_loss_limit_spin') else 0.0
        if hasattr(win,'guard_status'):
            win.guard_status.setText(f"Auto-halt when day PnL ≤ -${limit:,.2f} (curr: ${win._day_realized_running:,.2f})")
        if limit > 0 and win._day_realized_running <= -limit and not getattr(win, '_halted', False):
            win._append_log("[GUARD] Daily loss limit breached → HALTING")
            QtWidgets.QMessageBox.critical(win, "Guardrail Triggered", f"Daily loss limit breached (P/L ${win._day_realized_running:,.2f} ≤ -${limit:,.2f}). Trading HALTED.")
            win._toggle_panic()
            feeder.s.alerts.emit([{ "id":"guard1","ts":dt.datetime.now(dt.UTC).isoformat()+"Z","level":"error","text":"Daily loss limit breached — trading halted","sticky":True }])

    feeder.s.realized_point.connect(lambda d,day,cum: (win._rz_x.append(len(win._rz_x)), win._rz_daily.append(day), win._rz_cum.append(cum), win.rz_bars.setOpts(x=win._rz_x, height=win._rz_daily), win.rz_line.setData(win._rz_x, win._rz_cum), _guardrail_check(day)))
    feeder.s.risk_stats.connect(lambda u,r,dd: (win._set_kpi(win.unreal_lbl,u, money=True), win._set_kpi(win.realized_lbl,r, money=True), win._set_kpi(win.dd_lbl,dd, pct=True)))

    # Market context
    def on_ohlc(sym, dflike):
        if win.symbol_combo.currentText()!=sym: return
        df = dflike if isinstance(dflike:=dflike, pd.DataFrame) else pd.DataFrame(dflike)
        if win.candle_item: win.price_plot.removeItem(win.candle_item)
        win.candle_item=Candles(df); win.price_plot.addItem(win.candle_item)
        if win.ma20: win.price_plot.removeItem(win.ma20)
        if win.ma50: win.price_plot.removeItem(win.ma50)
        x=np.arange(len(df)); win.ma20=win.price_plot.plot(x, df['ma20'].to_numpy(), pen=pg.mkPen('#93c5fd', width=2)); win.ma50=win.price_plot.plot(x, df['ma50'].to_numpy(), pen=pg.mkPen('#fbbf24', width=2))
        last=float(df.iloc[-1]['close']); win.sl_line.setValue(last*0.97); win.tp_line.setValue(last*1.03)
        entries=[(int(len(df)*0.3), df['close'].iloc[int(len(df)*0.3)]), (int(len(df)*0.7), df['close'].iloc[int(len(df)*0.7)])]
        exits=[(int(len(df)*0.4), df['close'].iloc[int(len(df)*0.4)]), (int(len(df)*0.8), df['close'].iloc[int(len(df)*0.8)])]
        win.entry_marks.setData([e[0] for e in entries],[e[1] for e in entries]); win.exit_marks.setData([e[0] for e in exits],[e[1] for e in exits])
        win.regime_lbl.setText(random.choice(["low_vol","normal_vol","high_vol"]))
        win.trend_lbl.setText(random.choice(["trend","mean-reversion"]))
        win.bull_lbl.setText(random.choice(["bull","bear"]))
    feeder.s.ohlc.connect(on_ohlc)
    win.symbol_combo.currentTextChanged.connect(lambda _: feeder.s.ohlc.emit(win.symbol_combo.currentText(), feeder._ohlc[win.symbol_combo.currentText()]))
    feeder.s.news.connect(lambda items: (win.news_list.clear(), [win.news_list.addItem(f"[{i['ts']}] {i['symbol']}: {i['headline']} ({i['sentiment']})") for i in items]))

    # Performance
    def update_perf_labels():
        if len(win._eq_y) < 10: return
        arr=np.diff(np.log(np.clip(np.array(win._eq_y),1e-6,1e12)))
        sharpe = arr.mean() / (arr.std()+1e-9) * np.sqrt(252)
        sortino = arr.mean() / (np.std(arr[arr<0])+1e-9) * np.sqrt(252)
        maxdd = min(0.0, (min(win._eq_y)-max(win._eq_y))/max(win._eq_y)) if win._eq_y else 0.0
        win._set_kpi(win.sharpe_lbl, sharpe, money=False)
        win._set_kpi(win.sortino_lbl, sortino, money=False)
        win._set_kpi(win.kelly_lbl, min(1.0, max(0.0, sharpe/(sortino+1e-9))), money=False)
        win._set_kpi(win.maxdd_lbl, maxdd, pct=True)
        win._set_kpi(win.hit_lbl, random.uniform(0.4,0.7), pct=True)
        win._set_kpi(win.avwin_lbl, random.uniform(20,120), money=True)
        win._set_kpi(win.avloss_lbl, -random.uniform(20,120), money=True)
    perf_timer=QtCore.QTimer(); perf_timer.timeout.connect(lambda: (update_perf_labels(), _update_perf_charts())); perf_timer.start(1500)

    def _update_perf_charts():
        mat = np.random.rand(6, 6).astype(np.float32)
        win.heat_img.setImage(mat.T)
        win.hist_plot.clear()
        data = np.random.normal(10, 60, size=300)
        y, x = np.histogram(data, bins=30)
        centers = (x[:-1] + x[1:]) / 2.0
        width = (x[1] - x[0]) * 0.9
        bars = pg.BarGraphItem(x=centers, height=y, width=width, brush=pg.mkBrush(100, 100, 255, 150))
        win.hist_plot.addItem(bars)
        win.duration_plot.clear()
        durs = np.random.exponential(scale=60, size=100)
        win.duration_plot.plot(np.sort(durs))
        win.streaks_plot.clear()
        streaks = np.cumsum(np.random.choice([-1, 1], size=50))
        win.streaks_plot.plot(streaks)

    # Execution
    feeder.s.health.connect(lambda h: (win._set_kpi(win.lat_lbl,h['latency_ms'], money=False), win._set_kpi(win.p95_lbl,h['p95_latency_ms'], money=False), win._set_kpi(win.slip_lbl,h['slippage_bps'], money=False), win._set_kpi(win.hb_lbl,h['heartbeat_secs'], money=False), win._set_kpi(win.rec_lbl,h['reconnects'], money=False), win._set_kpi(win.err_lbl,h['api_errors'], money=False)))
    feeder.s.queue.connect(lambda q: (win._set_kpi(win.q_pending,q['pending']), win._set_kpi(win.q_working,q['working']), win._set_kpi(win.q_canceled,q['canceled'])))
    feeder.s.cooldown.connect(lambda c: win.cooldown_banner.setText("COOLDOWN ACTIVE" if c else ""))

    # Alerts
    def render_alerts(items:List[Dict]):
        win.alerts_list.clear()
        for a in items:
            item=QtWidgets.QListWidgetItem(f"[{a['level'].upper()}] {a['ts']}: {a['text']}")
            if a['level']=="error": item.setBackground(QtGui.QColor(80,0,0,120))
            elif a['level']=="warning": item.setBackground(QtGui.QColor(80,40,0,120))
            win.alerts_list.addItem(item)
    feeder.s.alerts.connect(render_alerts)

    # Strategies
    feeder.s.strategy_signals.connect(lambda rows: _fill_strategy_table(win, rows))
    feeder.s.regime_breakdown.connect(lambda d: win.regime_bar.setOpts(x=[0,1,2], height=[d.get('low_vol',0), d.get('normal_vol',0), d.get('high_vol',0)]))

    def _fill_strategy_table(win, rows):
        win.sig_table.setRowCount(0)
        for r in rows:
            row=win.sig_table.rowCount(); win.sig_table.insertRow(row)
            for c,key in enumerate(["strategy","last_signal","confidence","next_eval_ts"]):
                item=QtWidgets.QTableWidgetItem(f"{r.get(key)}"); win.sig_table.setItem(row,c,item)

    # Ops session summary demo refresh
    ops_timer=QtCore.QTimer(); ops_timer.timeout.connect(lambda: (win._set_kpi(win.today_real_lbl, random.uniform(-300, 400), money=True), win._set_kpi(win.trade_count_lbl, random.randint(0,25)), win._set_kpi(win.winrate_lbl, random.uniform(0.3,0.8), pct=True))); ops_timer.start(2000)

    # History
    feeder.s.benchmark_point.connect(lambda d,v: _update_benchmark(win,v))
    def _update_benchmark(win, v):
        if len(win._eq_y)>0:
            x=list(range(len(win._eq_y)))
            win.bench_eq.setData(x, win._eq_y)
            bench = getattr(win, '_bench_series', [])
            bench.append(v)
            win._bench_series = bench
            win.bench_idx.setData(list(range(len(bench))), bench)
        mat=np.random.randn(8,16)
        win.calendar_img.setImage(mat.T)

    # Replay tab
    win.replay_slider.valueChanged.connect(lambda k: win.replay_curve.setData(list(range(k+1)), win._eq_y[:k+1] if len(win._eq_y)>k else win._eq_y))

    # Orders/Trades/Logs
    feeder.s.orders.connect(lambda rows: win.orders_model.replace_rows(rows) if hasattr(win,'orders_model') else None)
    feeder.s.trades.connect(lambda rows: win.trades_model.replace_rows(rows) if hasattr(win,'trades_model') else None)
    feeder.s.log.connect(lambda line: win._append_log(line))

    feeder.start()
    rc = app.exec()
    feeder.stop(); feeder.wait(1500)
    sys.exit(rc)

if __name__ == '__main__':
    main()
