# core/logic/trade_gate.py
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime, timedelta

Side = Literal["long", "short", None]

@dataclass
class GateState:
    last_bar_id: Optional[int] = None
    did_action_this_bar: bool = False
    layers: int = 0
    side: Side = None
    cooldown_until: Optional[datetime] = None
    last_action_ts: Optional[datetime] = None
    last_regime: Optional[str] = None
    regime_persist: int = 0  # bars persisted in current regime

class TradeGate:
    def __init__(self, max_layers:int=1, min_bars_between_layers:int=2,
                 regime_min_persist_bars:int=3, flip_cooldown_bars:int=1):
        self.state = {}
        self.max_layers = max_layers
        self.min_bars_between_layers = min_bars_between_layers
        self.regime_min_persist_bars = regime_min_persist_bars
        self.flip_cooldown_bars = flip_cooldown_bars
        # per-symbol: bars_since_last_layer
        self._bars_since_layer = {}

    def get_state(self, symbol: str) -> GateState:
        return self.state.setdefault(symbol, GateState())

    def get_regime_persist(self, symbol: str) -> int:
        return self.get_state(symbol).regime_persist

    def on_new_bar(self, symbol:str, bar_id:int, regime:str):
        s = self.state.setdefault(symbol, GateState())
        # s.did_action_this_bar = False       # don't clear on every update

        if s.last_bar_id != bar_id:
            # ✅ only clear when we actually roll to a NEW bar
            s.last_bar_id = bar_id
            s.did_action_this_bar = False

            self._bars_since_layer[symbol] = self._bars_since_layer.get(symbol, 999) + 1

            # regime hysteresis should also update once per new bar
            if s.last_regime == regime:
                s.regime_persist += 1
            else:
                s.last_regime, s.regime_persist = regime, 1
        # else: same bar → DO NOT touch did_action_this_bar or regime_persist

    def can_enter(self, symbol:str, side:Side, ts:datetime, bar_id:int, regime:str,
                  allow_pyramiding:bool=False):
        s = self.state.setdefault(symbol, GateState())
        # Block if we already acted this bar (unless pyramiding path)
        if s.did_action_this_bar and not allow_pyramiding:
            return False, "already_acted_this_bar"

        # Cooldown after flips (reverse) to avoid immediate re-entries
        if s.cooldown_until and ts < s.cooldown_until:
            return False, "cooldown_active"

        # Regime must persist a few bars to avoid mid-bar flip spam
        if s.last_regime != regime:
            # on_new_bar should have updated this; guard anyway
            return False, "regime_mismatch"

        if s.regime_persist < self.regime_min_persist_bars:
            return False, "regime_not_persistent"

        # Pyramiding constraints
        if allow_pyramiding:
            if s.layers >= self.max_layers:
                return False, "max_layers_reached"
            if self._bars_since_layer.get(symbol, 999) < self.min_bars_between_layers:
                return False, "min_bars_between_layers"
        else:
            # fresh entry only if flat
            if s.side is not None:
                return False, "already_in_position"

        return True, "ok"

    def mark_action(self, symbol:str, ts:datetime, bar_id:int, new_side:Side,
                    action:str, flipped:bool=False, pyramided:bool=False):
        s = self.state.setdefault(symbol, GateState())
        s.did_action_this_bar = True
        s.last_action_ts = ts
        if flipped:
            s.side = new_side
            s.layers = 1
            # small cooldown after a flip to avoid thrash
            s.cooldown_until = ts + timedelta(minutes=self.flip_cooldown_bars)
            self._bars_since_layer[symbol] = 0
        elif pyramided:
            s.side = new_side
            s.layers += 1
            self._bars_since_layer[symbol] = 0
        else:
            s.side = new_side
            s.layers = 1
            self._bars_since_layer[symbol] = 0

    def close_position(self, symbol:str, ts:datetime, bar_id:int):
        s = self.state.setdefault(symbol, GateState())
        s.did_action_this_bar = True
        s.side = None
        s.layers = 0
        s.last_action_ts = ts
        self._bars_since_layer[symbol] = 0
