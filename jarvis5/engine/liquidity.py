"""Liquidity levels, equal-high/low clusters and sweep detection.

Level taxonomy (spec 12):
    PREVIOUS_HIGH / PREVIOUS_LOW      most recent confirmed swing
    INTERNAL_HIGH / INTERNAL_LOW      internal structure swing
    EXTERNAL_HIGH / EXTERNAL_LOW      external structure swing
    EQUAL_HIGH / EQUAL_LOW            >=2 swings within EQUAL_LIQUIDITY_ATR
    SESSION_HIGH / SESSION_LOW        running extreme of the current session

A sweep (spec 14) is NOT a touch.  It requires the wick to trade through the
level and the bar to close back on the original side of it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..core.timeutil import session_of, session_start_ms
from .swings import HIGH, LOW, Swing

BUY_SIDE = "BUY_SIDE"     # resting liquidity above price (highs)
SELL_SIDE = "SELL_SIDE"   # resting liquidity below price (lows)

STRONG = "STRONG"
MEDIUM = "MEDIUM"
WEAK = "WEAK"

_BASE_STRENGTH = {
    "EXTERNAL_HIGH": 3, "EXTERNAL_LOW": 3,
    "EQUAL_HIGH": 3, "EQUAL_LOW": 3,
    "SESSION_HIGH": 2, "SESSION_LOW": 2,
    "INTERNAL_HIGH": 1, "INTERNAL_LOW": 1,
    "PREVIOUS_HIGH": 1, "PREVIOUS_LOW": 1,
}


class LiquidityLevel:
    __slots__ = ("price", "ltype", "side", "tf", "strength", "creation_time",
                 "creation_index", "swept", "swept_time", "cluster_id",
                 "touches", "sources")

    def __init__(self, price: float, ltype: str, side: str, tf: str,
                 creation_time: int, creation_index: int, cluster_id: int = -1):
        self.price = price
        self.ltype = ltype
        self.side = side
        self.tf = tf
        self.creation_time = creation_time
        self.creation_index = creation_index
        self.cluster_id = cluster_id
        self.swept = False
        self.swept_time = 0
        self.touches = 1
        self.sources = 1
        self.strength = _BASE_STRENGTH.get(ltype, 1)

    def age_bars(self, i: int) -> int:
        return i - self.creation_index

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Liq {self.tf} {self.ltype} {self.price:.8g} s={self.strength}>"


class Sweep:
    __slots__ = ("level", "side", "direction", "index", "time", "tf", "penetration",
                 "penetration_atr", "wick_ratio", "return_distance", "volume_ratio",
                 "liquidity_strength", "rejection", "quality", "score",
                 "post_displacement", "structure_reaction", "extreme", "close",
                 "consumed")

    def __init__(self, level: LiquidityLevel, index: int, time: int, tf: str):
        self.level = level
        self.side = level.side
        # a sell-side sweep is a bullish reversal candidate and vice versa
        self.direction = "BULLISH" if level.side == SELL_SIDE else "BEARISH"
        self.index = index
        self.time = time
        self.tf = tf
        self.penetration = 0.0
        self.penetration_atr = 0.0
        self.wick_ratio = 0.0
        self.return_distance = 0.0
        self.volume_ratio = 1.0
        self.liquidity_strength = level.strength
        self.rejection = 0.0
        self.quality = WEAK
        self.score = 0
        self.post_displacement = 0.0
        self.structure_reaction = False
        self.extreme = 0.0
        self.close = 0.0
        self.consumed = False

    def age_bars(self, i: int) -> int:
        return i - self.index

    def as_dict(self) -> dict:
        return {
            "type": self.level.ltype,
            "side": self.side,
            "price": self.level.price,
            "penetration_atr": round(self.penetration_atr, 4),
            "wick_ratio": round(self.wick_ratio, 4),
            "return_distance": round(self.return_distance, 8),
            "volume_ratio": round(self.volume_ratio, 4),
            "liquidity_strength": self.liquidity_strength,
            "rejection": round(self.rejection, 4),
            "quality": self.quality,
            "score": self.score,
            "post_displacement": round(self.post_displacement, 4),
            "structure_reaction": self.structure_reaction,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Sweep {self.tf} {self.side} {self.quality} @{self.index}>"


class LiquidityEngine:
    def __init__(self, series, cfg):
        self.s = series
        self.cfg = cfg
        self.highs: List[LiquidityLevel] = []
        self.lows: List[LiquidityLevel] = []
        self.sweeps: List[Sweep] = []
        self._cluster_seq = 0
        self._session_name = ""
        self._session_high: Optional[LiquidityLevel] = None
        self._session_low: Optional[LiquidityLevel] = None
        self._vol_sum = 0.0
        self._vol_n = 0

    # ------------------------------------------------------------------ #
    # level construction
    # ------------------------------------------------------------------ #
    def on_swings(self, swings: List[Swing], atr: float, i: int,
                  internal_leg, external_leg) -> None:
        for sw in swings:
            if sw.kind == HIGH:
                self._add(sw.price, "PREVIOUS_HIGH", BUY_SIDE, sw.time, sw.index, atr)
                self._add(sw.price, "INTERNAL_HIGH", BUY_SIDE, sw.time, sw.index, atr)
            else:
                self._add(sw.price, "PREVIOUS_LOW", SELL_SIDE, sw.time, sw.index, atr)
                self._add(sw.price, "INTERNAL_LOW", SELL_SIDE, sw.time, sw.index, atr)

        # external levels come from the (separately maintained) external leg
        eh = external_leg.last_high
        if eh is not None:
            self._add(eh.price, "EXTERNAL_HIGH", BUY_SIDE, eh.time, eh.index, atr)
        el = external_leg.last_low
        if el is not None:
            self._add(el.price, "EXTERNAL_LOW", SELL_SIDE, el.time, el.index, atr)

        if swings and atr > 0:
            self._detect_equals(atr)
        self._prune()

    def _add(self, price: float, ltype: str, side: str, time: int, index: int,
             atr: float) -> LiquidityLevel:
        book = self.highs if side == BUY_SIDE else self.lows
        tol = self.cfg.EQUAL_LIQUIDITY_ATR * atr if atr > 0 else 0.0
        for lv in reversed(book[-30:]):
            if lv.ltype == ltype and abs(lv.price - price) <= tol:
                lv.touches += 1
                lv.strength = min(6, lv.strength + 1)
                lv.creation_time = max(lv.creation_time, time)
                return lv
        lv = LiquidityLevel(price, ltype, side, self.s.tf, time, index)
        book.append(lv)
        return lv

    def _detect_equals(self, atr: float) -> None:
        """Group unswept swing levels that sit within EQUAL_LIQUIDITY_ATR."""
        tol = self.cfg.EQUAL_LIQUIDITY_ATR * atr
        radius = self.cfg.LIQUIDITY_CLUSTER_ATR * atr
        if tol <= 0:
            return
        for side, book, eq_type in ((BUY_SIDE, self.highs, "EQUAL_HIGH"),
                                    (SELL_SIDE, self.lows, "EQUAL_LOW")):
            pool = [lv for lv in book[-40:]
                    if not lv.swept and lv.ltype in ("PREVIOUS_HIGH", "PREVIOUS_LOW",
                                                     "INTERNAL_HIGH", "INTERNAL_LOW",
                                                     "EXTERNAL_HIGH", "EXTERNAL_LOW")]
            if len(pool) < 2:
                continue
            pool.sort(key=lambda x: x.price)
            group: List[LiquidityLevel] = []
            for lv in pool:
                if group and abs(lv.price - group[-1].price) <= tol:
                    group.append(lv)
                    continue
                self._emit_equal(group, side, eq_type, radius)
                group = [lv]
            self._emit_equal(group, side, eq_type, radius)

    def _emit_equal(self, group, side, eq_type, radius) -> None:
        if len(group) < 2:
            return
        # cluster price = the extreme of the group (where stops actually rest)
        price = max(g.price for g in group) if side == BUY_SIDE else min(g.price for g in group)
        book = self.highs if side == BUY_SIDE else self.lows
        for lv in reversed(book[-40:]):
            if lv.ltype == eq_type and abs(lv.price - price) <= max(radius, 1e-12):
                lv.sources = max(lv.sources, len(group))
                lv.strength = min(6, _BASE_STRENGTH[eq_type] + len(group) - 2)
                return
        newest = max(group, key=lambda g: g.creation_index)
        self._cluster_seq += 1
        lv = LiquidityLevel(price, eq_type, side, self.s.tf, newest.creation_time,
                            newest.creation_index, self._cluster_seq)
        lv.sources = len(group)
        lv.strength = min(6, _BASE_STRENGTH[eq_type] + len(group) - 2)
        book.append(lv)

    def update_session_levels(self, i: int) -> None:
        s = self.s
        t = s.ot[i]
        name = session_of(t, self.cfg.SESSIONS, self.cfg.TIMEZONE_OFFSET_HOURS)
        if name != self._session_name:
            self._session_name = name
            self._session_high = None
            self._session_low = None
        if name == "OFF":
            return
        if self._session_high is None or s.h[i] > self._session_high.price:
            if self._session_high is not None:
                try:
                    self.highs.remove(self._session_high)
                except ValueError:
                    pass
            self._session_high = LiquidityLevel(s.h[i], "SESSION_HIGH", BUY_SIDE,
                                                s.tf, t, i)
            self.highs.append(self._session_high)
        if self._session_low is None or s.l[i] < self._session_low.price:
            if self._session_low is not None:
                try:
                    self.lows.remove(self._session_low)
                except ValueError:
                    pass
            self._session_low = LiquidityLevel(s.l[i], "SESSION_LOW", SELL_SIDE,
                                               s.tf, t, i)
            self.lows.append(self._session_low)

    def _prune(self) -> None:
        cap = self.cfg.LIQUIDITY_MAX_LEVELS
        if len(self.highs) > cap:
            self.highs = [lv for lv in self.highs if not lv.swept][-cap:] or self.highs[-cap:]
        if len(self.lows) > cap:
            self.lows = [lv for lv in self.lows if not lv.swept][-cap:] or self.lows[-cap:]
        if len(self.sweeps) > 200:
            self.sweeps = self.sweeps[-100:]

    # ------------------------------------------------------------------ #
    # sweep detection
    # ------------------------------------------------------------------ #
    def on_bar_closed(self, i: int, atr: float) -> List[Sweep]:
        s = self.s
        self._vol_sum += s.v[i]
        self._vol_n += 1
        avg_vol = (self._vol_sum / self._vol_n) if self._vol_n else 0.0

        out: List[Sweep] = []
        if atr <= 0:
            return out
        hi, lo, cl, op = s.h[i], s.l[i], s.c[i], s.o[i]
        rng = hi - lo
        min_pen = self.cfg.MIN_SWEEP_PENETRATION_ATR * atr
        max_pen = self.cfg.MAX_SWEEP_PENETRATION_ATR * atr

        # buy-side (highs) : wick above the level, close back below it
        for lv in self.highs:
            if lv.swept or lv.creation_index >= i:
                continue
            if hi <= lv.price:
                continue
            pen = hi - lv.price
            if pen < min_pen or pen > max_pen:
                continue
            if cl >= lv.price:
                continue                      # closed through -> break, not sweep
            sw = self._build(lv, i, atr, pen, rng, avg_vol)
            out.append(sw)

        # sell-side (lows) : wick below the level, close back above it
        for lv in self.lows:
            if lv.swept or lv.creation_index >= i:
                continue
            if lo >= lv.price:
                continue
            pen = lv.price - lo
            if pen < min_pen or pen > max_pen:
                continue
            if cl <= lv.price:
                continue
            sw = self._build(lv, i, atr, pen, rng, avg_vol)
            out.append(sw)

        for sw in out:
            sw.level.swept = True
            sw.level.swept_time = s.ot[i]
        self.sweeps.extend(out)
        return out

    def _build(self, lv, i, atr, pen, rng, avg_vol) -> Sweep:
        s = self.s
        sw = Sweep(lv, i, s.ot[i], s.tf)
        hi, lo, cl, op = s.h[i], s.l[i], s.c[i], s.o[i]
        sw.penetration = pen
        sw.penetration_atr = pen / atr
        sw.extreme = hi if lv.side == BUY_SIDE else lo
        sw.close = cl
        body = abs(cl - op)
        if lv.side == BUY_SIDE:
            wick = hi - max(op, cl)
            sw.return_distance = lv.price - cl
        else:
            wick = min(op, cl) - lo
            sw.return_distance = cl - lv.price
        sw.wick_ratio = (wick / rng) if rng > 0 else 0.0
        sw.rejection = (sw.return_distance / atr) if atr > 0 else 0.0
        sw.volume_ratio = (s.v[i] / avg_vol) if avg_vol > 0 else 1.0

        # ---- deterministic quality score (spec 15) --------------------- #
        score = 0
        st = lv.strength
        score += 2 if st >= 3 else (1 if st >= 2 else 0)
        p = sw.penetration_atr
        score += 2 if 0.05 <= p <= 0.60 else (1 if p <= 1.00 else 0)
        w = sw.wick_ratio
        score += 2 if w >= 0.60 else (1 if w >= 0.35 else 0)
        if sw.volume_ratio >= 1.20:
            score += 1
        if sw.rejection >= 0.05:
            score += 1
        sw.score = score
        if score >= self.cfg.SWEEP_STRONG_SCORE:
            sw.quality = STRONG
        elif score >= self.cfg.SWEEP_MEDIUM_SCORE:
            sw.quality = MEDIUM
        else:
            sw.quality = WEAK
        return sw

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def recent_sweeps(self, i: int, lookback: int, side: Optional[str] = None,
                      unconsumed: bool = False) -> List[Sweep]:
        out = []
        for sw in reversed(self.sweeps):
            if i - sw.index > lookback:
                break
            if side and sw.side != side:
                continue
            if unconsumed and sw.consumed:
                continue
            out.append(sw)
        return out

    def opposing_levels(self, price: float, direction: str) -> List[LiquidityLevel]:
        """Untouched liquidity in front of the trade, nearest first."""
        if direction == "BUY":
            pool = [lv for lv in self.highs if not lv.swept and lv.price > price]
            pool.sort(key=lambda lv: lv.price)
        else:
            pool = [lv for lv in self.lows if not lv.swept and lv.price < price]
            pool.sort(key=lambda lv: -lv.price)
        return pool
