# tools/replay_from_trades.py
from __future__ import annotations
import csv
from datetime import datetime
from collections import defaultdict

def replay_equity_from_trades(
    csv_path: str,
    starting_cash: float = 100_000.0,
    final_prices: dict[str, float] | None = None,
):
    """
    Rebuild cash/positions/equity by replaying your trade log.

    final_prices: optional dict of last prices per symbol to mark open positions.
                  If None, we use each symbol's last traded price from the log.
    """
    # state
    cash = float(starting_cash)
    pos_qty = defaultdict(int)          # symbol -> signed qty (long>0, short<0)
    pos_avg = defaultdict(float)        # symbol -> avg price
    last_px = defaultdict(float)        # symbol -> last seen trade price

    # load rows (sorted by time)
    rows = []
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            # parse what we need
            row["timestamp"] = datetime.fromisoformat(row["timestamp"])
            row["price"] = float(row["price"])
            row["quantity"] = int(row["quantity"])
            rows.append(row)
    rows.sort(key=lambda x: x["timestamp"])

    def mtm_equity():
        # Correct: cash + Σ(qty * last_price)
        equity = cash
        for sym, qty in pos_qty.items():
            if qty == 0:
                continue
            px = final_prices.get(sym) if final_prices else last_px[sym]
            equity += qty * px
        return equity

    def buy(sym, qty, px):
        nonlocal cash
        cash -= px * qty
        old_qty = pos_qty[sym]
        new_qty = old_qty + qty

        if old_qty < 0:
            # covering a short (maybe flipping)
            if new_qty < 0:
                # still short after partial cover: avg unchanged for remaining short
                pass
            elif new_qty == 0:
                # fully flat: clear avg
                pos_avg[sym] = 0.0
            else:
                # flipped to net long: new long avg at px
                pos_avg[sym] = px
        else:
            # adding/increasing a long: weighted avg
            total_cost = pos_avg[sym] * old_qty + px * qty
            pos_avg[sym] = total_cost / max(new_qty, 1)

        pos_qty[sym] = new_qty
        last_px[sym] = px

    def sell(sym, qty, px):
        nonlocal cash
        cash += px * qty
        old_qty = pos_qty[sym]
        new_qty = old_qty - qty

        if old_qty > 0 and new_qty < 0:
            # flipped from long to short -> new short avg at px
            pos_avg[sym] = px
        elif old_qty <= 0:
            # weighted avg for short (qty negative)
            # treat like adding to short: average on absolute units
            total_units = abs(old_qty) + qty
            if total_units > 0:
                pos_avg[sym] = (pos_avg[sym] * abs(old_qty) + px * qty) / total_units
            else:
                pos_avg[sym] = 0.0
        else:
            # reducing a long; avg unchanged unless fully flat
            if new_qty == 0:
                pos_avg[sym] = 0.0

        pos_qty[sym] = new_qty
        last_px[sym] = px

    # action mapping
    # enter_long = buy, exit_long = sell; enter_short = sell, exit_short = buy
    for row in rows:
        sym   = row["symbol"]
        act   = row["action"]
        px    = row["price"]
        qty   = row["quantity"]
        notes = (row.get("notes") or "").strip().lower()

        if act == "enter_long":
            buy(sym, qty, px)
        elif act == "exit_long":
            sell(sym, qty, px)
        elif act == "enter_short":
            sell(sym, qty, px)
        elif act == "exit_short":
            buy(sym, qty, px)
        elif act == "partial_exit":
            # use notes to decide side if present, else infer from position sign
            side = "long" if "long" in notes else ("short" if "short" in notes else ("long" if pos_qty[sym] > 0 else "short"))
            if side == "long":
                sell(sym, qty, px)
            else:
                buy(sym, qty, px)
        else:
            # ignore non-fill rows
            continue

    # compute final equity marked to last price per symbol (or provided)
    final_eq = mtm_equity()
    snapshot = {
        "cash": cash,
        "positions": {s: {"qty": pos_qty[s], "avg": pos_avg[s], "last": (final_prices or last_px)[s]}
                      for s in pos_qty.keys() if pos_qty[s] != 0},
        "equity": final_eq
    }
    return snapshot
