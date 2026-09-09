"""Internal & external market structure, BOS / CHOCH / MSS.

Internal structure  = every confirmed swing, alternating H/L.
External structure  = only swings separated by >= EXTERNAL_STRUCTURE_ATR from
                      the previous external swing of the opposite kind.

The two sequences are kept in SEPARATE objects and are never mixed (spec 8).

Break rules (spec 9/10/11):
  * a break requires a CLOSE beyond the level by >= MIN_BREAK_ATR * ATR;
    a wick through the level is never a break;
  * a break WITH the prevailing direction is a BOS;
  * the first valid break AGAINST it is a CHOCH;
  * a break that follows a liquidity sweep of the opposite side within
    MSS_SWEEP_LOOKBACK bars is additionally flagged as an MSS.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .swings import HIGH, LOW, Swing

BULLISH = "BULLISH"
BEARISH = "BEARISH"
RANGE = "RANGE"
TRANSITION = "TRANSITION"
NONE = "NONE"

BOS = "BOS"
CHOCH = "CHOCH"


class StructureEvent:
    __slots__ = ("kind", "direction", "time", "index", "tf", "scope", "level",
                 "swing", "is_mss", "sweep", "displacement", "close")

    def __init__(self, kind: str, direction: str, time: int, index: int, tf: str,
                 scope: str, level: float, swing: Optional[Swing], close: float):
        self.kind = kind              # BOS | CHOCH
        self.direction = direction    # BULLISH | BEARISH
        self.time = time
        self.index = index
        self.tf = tf
        self.scope = scope            # INTERNAL | EXTERNAL
        self.level = level
        self.swing = swing
        self.close = close
        self.is_mss = False
        self.sweep = None
        self.displacement = 0.0

    def __repr__(self) -> str:  # pragma: no cover
        tag = "MSS/" if self.is_mss else ""
        return f"<{tag}{self.kind} {self.scope} {self.direction} {self.tf} @{self.index}>"


class _Leg:
    """One alternating swing sequence with HH/HL/LH/LL labelling."""

    def __init__(self, separation_atr: float):
        self.separation_atr = separation_atr
        self.swings: List[Swing] = []
        self.direction = NONE
        self.last_high: Optional[Swing] = None
        self.last_low: Optional[Swing] = None
        self.prev_high: Optional[Swing] = None
        self.prev_low: Optional[Swing] = None
        self.sequence: List[str] = []

    def _label(self, sw: Swing) -> str:
        if sw.kind == HIGH:
            ref = self.last_high
            return "HH" if (ref is None or sw.price > ref.price) else "LH"
        ref = self.last_low
        return "HL" if (ref is None or sw.price > ref.price) else "LL"

    def add(self, sw: Swing, atr: float) -> bool:
        """Try to extend the leg with a confirmed swing.  True if accepted."""
        if not self.swings:
            self._accept(sw)
            return True

        last = self.swings[-1]
        if sw.kind == last.kind:
            # same side: keep only the more extreme point of the leg
            more_extreme = (sw.price > last.price) if sw.kind == HIGH else (sw.price < last.price)
            if more_extreme:
                self.swings.pop()
                if self.sequence:
                    self.sequence.pop()
                self._restore_refs()
                self._accept(sw)
                return True
            return False

        # opposite side: needs enough separation to count as its own leg
        if self.separation_atr > 0.0:
            need = self.separation_atr * (atr if atr > 0 else last.atr)
            if need > 0 and abs(sw.price - last.price) < need:
                return False
        self._accept(sw)
        return True

    def _accept(self, sw: Swing) -> None:
        sw.label = self._label(sw)
        sw.is_external = self.separation_atr > 0.0
        self.swings.append(sw)
        self.sequence.append(sw.label)
        if sw.kind == HIGH:
            self.prev_high, self.last_high = self.last_high, sw
        else:
            self.prev_low, self.last_low = self.last_low, sw
        if len(self.sequence) >= 2:
            self._update_direction()
        if len(self.swings) > 400:
            self.swings = self.swings[-200:]
            self.sequence = self.sequence[-200:]

    def _restore_refs(self) -> None:
        highs = [s for s in self.swings if s.kind == HIGH]
        lows = [s for s in self.swings if s.kind == LOW]
        self.last_high = highs[-1] if highs else None
        self.prev_high = highs[-2] if len(highs) > 1 else None
        self.last_low = lows[-1] if lows else None
        self.prev_low = lows[-2] if len(lows) > 1 else None

    def _update_direction(self) -> None:
        tail = self.sequence[-2:]
        if tail == ["HH", "HL"] or tail == ["HL", "HH"]:
            self.direction = BULLISH
        elif tail == ["LL", "LH"] or tail == ["LH", "LL"]:
            self.direction = BEARISH
        # mixed tails leave the direction as set by the last break

    def unbroken_high(self) -> Optional[Swing]:
        for sw in reversed(self.swings):
            if sw.kind == HIGH and not sw.broken:
                return sw
        return None

    def unbroken_low(self) -> Optional[Swing]:
        for sw in reversed(self.swings):
            if sw.kind == LOW and not sw.broken:
                return sw
        return None


class StructureEngine:
    """Per-timeframe structure state."""

    def __init__(self, series, cfg):
        self.s = series
        self.cfg = cfg
        self.internal = _Leg(0.0)
        self.external = _Leg(cfg.EXTERNAL_STRUCTURE_ATR)
        self.events: List[StructureEvent] = []
        self.last_external_break_index: int = -1
        self.regime: str = NONE

    # -- swing intake --------------------------------------------------- #
    def on_swings(self, swings: List[Swing], atr: float) -> None:
        for sw in swings:
            self.internal.add(sw, atr)
            # external leg gets its own Swing copy so labels never collide
            ext = Swing(sw.index, sw.time, sw.price, sw.kind, sw.tf,
                        sw.confirmed_time, sw.atr)
            self.external.add(ext, atr)

    # -- break detection ------------------------------------------------ #
    def on_bar_closed(self, i: int, atr: float) -> List[StructureEvent]:
        s = self.s
        close = s.c[i]
        buf = self.cfg.MIN_BREAK_ATR * atr if atr > 0 else 0.0
        out: List[StructureEvent] = []

        for scope, leg in (("INTERNAL", self.internal), ("EXTERNAL", self.external)):
            hi = leg.unbroken_high()
            if hi is not None and hi.index < i and close > hi.price + buf:
                hi.broken = True
                kind = CHOCH if leg.direction == BEARISH else BOS
                ev = StructureEvent(kind, BULLISH, s.ot[i], i, s.tf, scope,
                                    hi.price, hi, close)
                leg.direction = BULLISH
                out.append(ev)
                if scope == "EXTERNAL":
                    self.last_external_break_index = i

            lo = leg.unbroken_low()
            if lo is not None and lo.index < i and close < lo.price - buf:
                lo.broken = True
                kind = CHOCH if leg.direction == BULLISH else BOS
                ev = StructureEvent(kind, BEARISH, s.ot[i], i, s.tf, scope,
                                    lo.price, lo, close)
                leg.direction = BEARISH
                out.append(ev)
                if scope == "EXTERNAL":
                    self.last_external_break_index = i

        self.events.extend(out)
        if len(self.events) > 600:
            self.events = self.events[-300:]
        return out

    # -- context -------------------------------------------------------- #
    def update_regime(self, i: int) -> str:
        """M15 market regime (spec 47)."""
        seq = self.external.sequence
        bars_since_break = i - self.last_external_break_index
        if self.last_external_break_index < 0 or bars_since_break > self.cfg.RANGE_MODE_M15_BARS:
            self.regime = RANGE
        elif bars_since_break <= 2:
            self.regime = TRANSITION
        elif len(seq) >= 2:
            self.regime = {BULLISH: BULLISH, BEARISH: BEARISH}.get(
                self.external.direction, RANGE)
        else:
            self.regime = RANGE
        return self.regime

    def dealing_range(self):
        """Active external dealing range (low, high, eq) or None."""
        hi = self.external.last_high
        lo = self.external.last_low
        if hi is None or lo is None or hi.price <= lo.price:
            return None
        return lo.price, hi.price, (hi.price + lo.price) / 2.0

    def pd_position(self, price: float):
        """0.0 = external low, 1.0 = external high.  None if no range."""
        dr = self.dealing_range()
        if dr is None:
            return None
        lo, hi, _ = dr
        if hi <= lo:
            return None
        return (price - lo) / (hi - lo)

    def recent_events(self, since_index: int, scope: Optional[str] = None,
                      direction: Optional[str] = None) -> List[StructureEvent]:
        out = []
        for ev in reversed(self.events):
            if ev.index < since_index:
                break
            if scope and ev.scope != scope:
                continue
            if direction and ev.direction != direction:
                continue
            out.append(ev)
        return out
