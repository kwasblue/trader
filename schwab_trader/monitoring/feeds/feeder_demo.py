from PySide6 import QtCore
from typing import List, Dict
import pandas as pd
import numpy as np
import random
import datetime as dt


# =====================================================
# DATA FEEDER (background thread with extended signals)
# =====================================================
class FeedSignals(QtCore.QObject):
    symbols = QtCore.Signal(list)
    orders  = QtCore.Signal(list)
    trades  = QtCore.Signal(list)
    equity_point   = QtCore.Signal(str, float)
    realized_point = QtCore.Signal(str, float, float)
    risk_stats     = QtCore.Signal(float, float, float)
    log = QtCore.Signal(str)
    health = QtCore.Signal(dict)          # {latency_ms, p95_latency_ms, slippage_bps, heartbeat_secs, reconnects, api_errors}
    queue = QtCore.Signal(dict)           # {pending, working, canceled}
    cooldown = QtCore.Signal(bool)        # True if cooldown engaged
    alerts = QtCore.Signal(list)          # [{id, ts, level, text, sticky}]
    news = QtCore.Signal(list)            # [{ts, symbol, headline, sentiment}]
    ohlc = QtCore.Signal(str, object)     # symbol, pandas.DataFrame or dict of lists
    benchmark_point = QtCore.Signal(str, float)  # date, benchmark equity/index value
    strategy_signals = QtCore.Signal(list)       # [{strategy, last_signal, confidence, next_eval_ts}]
    regime_breakdown = QtCore.Signal(dict)       # {low_vol: pnl, normal_vol: pnl, high_vol: pnl}

class DataFeeder(QtCore.QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.s = FeedSignals(); self._running=True
        self._equity=100_000.0; self._cum_realized=0.0; self._peak=self._equity; self._max_dd=0.0
        self._bench=100.0
        self._order_id=4
        self._cooldown=False
        self._halted=False
        # seed OHLC per symbol
        self._symbols=["AAPL","MSFT","TSLA","AMD"]
        self._ohlc={sym:self._gen_ohlc() for sym in self._symbols}

    @QtCore.Slot(bool)
    def set_halted(self, h: bool):
        self._halted = h
        self.s.log.emit(f"[SYS] {'HALTED' if h else 'RESUMED'} by operator")

    @QtCore.Slot()
    def do_flatten_all(self):
        self.s.log.emit("[OPS] Flatten All requested")
        self.s.alerts.emit(self._mock_alerts(extra=True))

    @QtCore.Slot()
    def do_cancel_all(self):
        self.s.log.emit("[OPS] Cancel All working orders requested")

    @QtCore.Slot(str)
    def do_flatten_symbol(self, sym: str):
        self.s.log.emit(f"[OPS] Flatten symbol requested: {sym}")

    @QtCore.Slot(dict)
    def do_manual_order(self, order: Dict):
        self._order_id += 1
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {"id": f"o{self._order_id}", "ts": now, "symbol": order["symbol"],
               "side": order["side"].upper(), "qty": int(order["qty"]),
               "price": float(order.get("price") or 0.0), "status": "SUBMITTED", "route": order.get("route","Auto")}
        self.s.orders.emit(self._mock_orders() + [row])
        self.s.log.emit(f"[MANUAL] {row['route']} -> {row['symbol']} {row['side']} {row['qty']} @ {row['price']} ({order.get('type','market').upper()} / {order.get('tif','DAY')})")

    def stop(self): self._running=False

    def _gen_ohlc(self, n=120, start=200.0):
        prices=[start]
        for _ in range(n-1): prices.append(prices[-1]*(1+(random.random()-0.5)*0.01))
        df=pd.DataFrame({"close":prices}); df["open"]=df["close"].shift(1).fillna(df["close"]) ; df["high"]=df[["open","close"]].max(axis=1)*(1+np.random.rand(n)*0.01) ; df["low"]=df[["open","close"]].min(axis=1)*(1-np.random.rand(n)*0.01)
        df["ma20"]=df["close"].rolling(20).mean(); df["ma50"]=df["close"].rolling(50).mean()
        df["ts"]=pd.date_range(end=dt.datetime.now(), periods=n, freq="h")
        df.reset_index(drop=True,inplace=True)
        return df

    def run(self):
        # initial payload
        self.s.orders.emit(self._mock_orders()); self.s.trades.emit(self._mock_trades()); self.s.alerts.emit(self._mock_alerts());
        self.s.strategy_signals.emit(self._mock_strategy_signals())
        self.s.news.emit(self._mock_news())
        self.s.regime_breakdown.emit({"low_vol":500,"normal_vol":-120,"high_vol":820})
        for sym,df in self._ohlc.items(): self.s.ohlc.emit(sym, df)

        while self._running:
            if self._halted:
                health={"latency_ms":0.0,"p95_latency_ms":0.0,"slippage_bps":0.0,"heartbeat_secs":0.0,"reconnects":0,"api_errors":0}
                self.s.health.emit(health)
                self.msleep(400)
                continue

            # symbols snapshot
            symbols=[]
            for sym in self._symbols:
                df=self._ohlc[sym]
                last=float(df.iloc[-1]["close"]) * (1+(random.random()-0.5)*0.002)
                avg=last*random.uniform(0.98,1.02); qty=random.choice([0,10,25,50,100]); side=("flat" if qty==0 else random.choice(["long","short"]))
                unreal=(last-avg)*qty*(1 if side=="long" else -1 if side=="short" else 0)
                realized=random.uniform(-500,800); pnl_pct=(last/avg-1) if avg else 0.0
                atr=last*random.uniform(0.005,0.04); regime=random.choice(["low_vol","normal_vol","high_vol"])
                risk=max(0,min(100,int((atr/max(last,1e-6))*100*2)))
                symbols.append({"symbol":sym,"side":side,"qty":float(qty),"avg":float(avg),"last":float(last),"unreal":float(unreal),"realized":float(realized),"pnl_pct":float(pnl_pct),"atr":float(atr),"regime":regime,"risk":float(risk)})
            self.s.symbols.emit(symbols)

            # equity & realized & benchmark
            shock=(random.random()-0.5)*0.006; self._equity*=(1.0+shock); self._peak=max(self._peak,self._equity); self._max_dd=min(self._max_dd,(self._equity-self._peak)/self._peak) if self._peak else 0.0
            daily_realized=(random.random()-0.48)*120; self._cum_realized+=daily_realized
            self._bench*=(1+(random.random()-0.5)*0.003)
            d=dt.date.today().isoformat(); self.s.equity_point.emit(d,float(self._equity)); self.s.realized_point.emit(d,float(daily_realized),float(self._cum_realized)); self.s.benchmark_point.emit(d,float(self._bench))

            # health + queue
            health={"latency_ms":random.uniform(20,120),"p95_latency_ms":random.uniform(100,300),"slippage_bps":random.uniform(-5,15),"heartbeat_secs":random.uniform(0,3),"reconnects":random.randint(0,1),"api_errors":random.randint(0,1)}
            queue={"pending":random.randint(0,5),"working":random.randint(0,5),"canceled":random.randint(0,2)}
            self.s.health.emit(health); self.s.queue.emit(queue)

            # toggle cooldown randomly
            if random.random()<0.1:
                self._cooldown=not self._cooldown; self.s.cooldown.emit(self._cooldown)
                if self._cooldown: self.s.alerts.emit(self._mock_alerts(extra=True))

            # occasional order + log
            if random.random()<0.25:
                self._order_id+=1
                new_order={"id":f"o{self._order_id}","ts":dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"symbol":random.choice(self._symbols),"side":random.choice(["BUY","SELL","SHORT SELL","COVER"]),"qty":random.choice([5,10,25,50]),"price":round(random.uniform(100,900),2),"status":random.choice(["FILLED","PARTIAL","CANCELED","REJECTED"]) }
                self.s.orders.emit(self._mock_orders()+[new_order])
                self.s.log.emit(f"[{new_order['ts']}] {new_order['symbol']} {new_order['side']} {new_order['qty']} @ {new_order['price']}")

            self.msleep(1000)

    def _mock_orders(self)->List[Dict]:
        return [{"id":"o1","ts":"2025-08-27 14:22:09","symbol":"AAPL","side":"BUY","qty":25,"price":219.10,"status":"FILLED"},{"id":"o2","ts":"2025-08-27 14:05:41","symbol":"TSLA","side":"SELL","qty":10,"price":239.90,"status":"PARTIAL"},{"id":"o3","ts":"2025-08-27 13:02:10","symbol":"MSFT","side":"SHORT SELL","qty":10,"price":422.40,"status":"CANCELED"},{"id":"o4","ts":"2025-08-27 12:41:00","symbol":"NVDA","side":"BUY","qty":5,"price":850.00,"status":"REJECTED"}]
    def _mock_trades(self)->List[Dict]:
        return [{"id":1,"ts":"2025-08-27 13:44:02","symbol":"AAPL","side":"BUY","qty":50,"price":218.70,"fee":0.75,"sl":213.30,"tp":224.20},{"id":2,"ts":"2025-08-27 10:15:10","symbol":"TSLA","side":"SELL","qty":10,"price":240.10,"fee":0.75,"sl":232.50,"tp":252.00},{"id":3,"ts":"2025-08-26 15:20:57","symbol":"MSFT","side":"SHORT SELL","qty":20,"price":423.20,"fee":0.75,"sl":436.00,"tp":412.70}]
    def _mock_alerts(self, extra=False)->List[Dict]:
        base=[{"id":"al1","ts":dt.datetime.now(dt.UTC).isoformat()+"Z","level":"warning","text":"Drawdown exceeded 3% intraday. Cooldown engaged.","sticky":True},{"id":"al2","ts":dt.datetime.now(dt.UTC).isoformat()+"Z","level":"info","text":"AAPL hit TP1. Partial exit executed (25%).","sticky":True}]
        return base+([{ "id":"al3","ts":dt.datetime.now(dt.UTC).isoformat()+"Z","level":"error","text":"Broker throttle / reconnect.","sticky":True}] if extra else [])
    def _mock_strategy_signals(self)->List[Dict]:
        return [
            {"strategy":"Momentum","last_signal":"BUY","confidence":0.72,"next_eval_ts":(dt.datetime.now(dt.UTC)+dt.timedelta(minutes=15)).isoformat()+"Z"},
            {"strategy":"MeanRev","last_signal":"HOLD","confidence":0.41,"next_eval_ts":(dt.datetime.now(dt.UTC)+dt.timedelta(minutes=5)).isoformat()+"Z"},
            {"strategy":"ML","last_signal":"SELL","confidence":0.63,"next_eval_ts":(dt.datetime.now(dt.UTC)+dt.timedelta(minutes=30)).isoformat()+"Z"},
        ]
    def _mock_news(self)->List[Dict]:
        return [
            {"ts":dt.datetime.now(dt.UTC).isoformat()+"Z","symbol":"AAPL","headline":"Apple announces product event date","sentiment":"pos"},
            {"ts":dt.datetime.now(dt.UTC).isoformat()+"Z","symbol":"MSFT","headline":"Earnings beat consensus","sentiment":"pos"},
            {"ts":dt.datetime.now(dt.UTC).isoformat()+"Z","symbol":"TSLA","headline":"Delivery numbers mixed","sentiment":"neu"},
        ]