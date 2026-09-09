"""Incremental RSI, volume and price-action detectors (spec 8, 9, 10).

Everything is fed one CLOSED bar at a time and every state it exposes depends
only on bars <= the current index.  Nothing is fitted; each detector is a
threshold rule written out in full.
"""
from __future__ import annotations

from array import array
from collections import deque
from typing import Dict, List, Optional

# ---------------------------------------------------------------------- #
# RSI
# ---------------------------------------------------------------------- #
class RSI:
    """Wilder RSI, incremental."""

    __slots__ = ("period", "values", "_prev", "_ag", "_al", "_count", "n")

    def __init__(self, period: int = 14):
        self.period = period
        self.values = array("d")
        self._prev: Optional[float] = None
        self._ag = 0.0
        self._al = 0.0
        self._count = 0
        self.n = 0

    def update(self, close: float) -> float:
        if self._prev is None:
            self._prev = close
            self.values.append(50.0)
            self.n += 1
            return 50.0
        change = close - self._prev
        self._prev = close
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        p = self.period
        if self._count < p:
            self._ag += gain
            self._al += loss
            self._count += 1
            if self._count == p:
                self._ag /= p
                self._al /= p
                val = self._rsi()
            else:
                val = 50.0
        else:
            self._ag = (self._ag * (p - 1) + gain) / p
            self._al = (self._al * (p - 1) + loss) / p
            val = self._rsi()
        self.values.append(val)
        self.n += 1
        return val

    def _rsi(self) -> float:
        if self._al <= 0:
            return 100.0 if self._ag > 0 else 50.0
        rs = self._ag / self._al
        return 100.0 - (100.0 / (1.0 + rs))

    @property
    def ready(self) -> bool:
        return self._count >= self.period

    @property
    def value(self) -> float:
        return self.values[-1] if self.n else 50.0

    def at(self, i: int) -> float:
        if i < 0 or i >= self.n:
            return 50.0
        return self.values[i]


def rsi_signals(rsi: RSI, i: int, swings, cfg) -> Dict[str, int]:
    """Active RSI conditions -> bars since the condition last occurred.

    Divergences are built from CONFIRMED swings only, so they inherit the
    swing engine's K-bar publication delay and cannot peek forward.
    """
    out: Dict[str, int] = {}
    if not rsi.ready or i < 2:
        return out
    v = rsi.at(i)
    prev = rsi.at(i - 1)

    if v < 20:
        out["RSI_LT_20"] = 0
    if v < 30:
        out["RSI_LT_30"] = 0
    if v > 50:
        out["RSI_GT_50"] = 0
    if v > 70:
        out["RSI_GT_70"] = 0
    if v > 80:
        out["RSI_GT_80"] = 0
    for level in (30, 40, 50):
        if prev < level <= v:
            out[f"RSI_RECLAIM_{level}"] = 0
        if prev > level >= v:
            out[f"RSI_LOSE_{level}"] = 0

    # momentum reversal: RSI turns after an extreme
    lo = max(0, i - 5)
    window = [rsi.at(j) for j in range(lo, i + 1)]
    if len(window) >= 3:
        if min(window) < 30 and v > window[window.index(min(window))] + 5:
            out["RSI_MOMENTUM_REVERSAL_UP"] = i - (lo + window.index(min(window)))
        if max(window) > 70 and v < window[window.index(max(window))] - 5:
            out["RSI_MOMENTUM_REVERSAL_DOWN"] = i - (lo + window.index(max(window)))

    look = cfg.RSI_DIVERGENCE_LOOKBACK
    lows = [s for s in swings.lows if i - s.index <= look and s.index < i][-2:]
    if len(lows) == 2:
        p1, p2 = lows[0], lows[1]
        if p2.price < p1.price and rsi.at(p2.index) > rsi.at(p1.index):
            out["RSI_BULLISH_DIVERGENCE"] = i - p2.index
        # failure swing (bullish): second low holds above 30 after a dip below
        if rsi.at(p1.index) < 30 and rsi.at(p2.index) > rsi.at(p1.index) \
                and p2.price > p1.price:
            out["RSI_FAILURE_SWING_BULL"] = i - p2.index
    highs = [s for s in swings.highs if i - s.index <= look and s.index < i][-2:]
    if len(highs) == 2:
        p1, p2 = highs[0], highs[1]
        if p2.price > p1.price and rsi.at(p2.index) < rsi.at(p1.index):
            out["RSI_BEARISH_DIVERGENCE"] = i - p2.index
        if rsi.at(p1.index) > 70 and rsi.at(p2.index) < rsi.at(p1.index) \
                and p2.price < p1.price:
            out["RSI_FAILURE_SWING_BEAR"] = i - p2.index
    return out


# ---------------------------------------------------------------------- #
# Volume
# ---------------------------------------------------------------------- #
class VolumeTracker:
    __slots__ = ("period", "buf", "total", "ratios", "n")

    def __init__(self, period: int = 50):
        self.period = period
        self.buf: deque = deque(maxlen=period)
        self.total = 0.0
        self.ratios = array("d")
        self.n = 0

    def update(self, volume: float) -> float:
        avg = (self.total / len(self.buf)) if self.buf else 0.0
        ratio = (volume / avg) if avg > 0 else 1.0
        if len(self.buf) == self.period:
            self.total -= self.buf[0]
        self.buf.append(volume)
        self.total += volume
        self.ratios.append(ratio)
        self.n += 1
        return ratio

    @property
    def ready(self) -> bool:
        return len(self.buf) >= self.period

    def at(self, i: int) -> float:
        if i < 0 or i >= self.n:
            return 1.0
        return self.ratios[i]


def volume_signals(series, vol: VolumeTracker, i: int, atr: float,
                   cfg) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not vol.ready or i < 5:
        return out
    r = vol.at(i)
    for lvl in cfg.VOLUME_SPIKE_LEVELS:
        if r >= lvl:
            out[f"VOLUME_GT_{lvl:g}X"] = 0
    if r >= cfg.VOLUME_SPIKE_LEVELS[0]:
        out["VOLUME_SPIKE"] = 0
    recent = [vol.at(j) for j in range(i - 4, i + 1)]
    prior = [vol.at(j) for j in range(max(0, i - 14), i - 4)]
    if prior:
        ra = sum(recent) / len(recent)
        pa = sum(prior) / len(prior)
        if pa > 0 and ra >= 1.5 * pa:
            out["VOLUME_EXPANSION"] = 0
        if pa > 0 and ra <= cfg.VOLUME_CONTRACTION_RATIO * pa:
            out["VOLUME_CONTRACTION"] = 0
    if r >= cfg.WY_VOLUME_CLIMAX and atr > 0 and series.range(i) >= 1.5 * atr:
        out["VOLUME_CLIMAX"] = 0

    rng = series.range(i)
    body = series.body(i)
    if r >= cfg.VOLUME_SPIKE_LEVELS[0] and rng > 0:
        if body / rng < 0.35:
            out["HIGH_VOLUME_REJECTION"] = 0
        else:
            out["HIGH_VOLUME_BREAKOUT"] = 0

    # a breakout candle carried by unusually LOW participation
    lo = max(0, i - 20)
    prior_high = max(series.h[j] for j in range(lo, i)) if i > lo else series.h[i]
    prior_low = min(series.l[j] for j in range(lo, i)) if i > lo else series.l[i]
    broke = series.c[i] > prior_high or series.c[i] < prior_low
    if broke:
        out["VOLUME_BREAKOUT" if r >= 1.5 else "LOW_VOLUME_BREAKOUT"] = 0
    return out


# ---------------------------------------------------------------------- #
# Price action
# ---------------------------------------------------------------------- #
def price_action_signals(series, i: int, atr: float, cfg) -> Dict[str, int]:
    out: Dict[str, int] = {}
    # single-bar and two-bar patterns need only the previous bar; the deeper
    # lookbacks below clamp their own ranges
    if i < 1 or atr <= 0:
        return out
    o, h, l, c = series.o[i], series.h[i], series.l[i], series.c[i]
    po, ph, pl, pc = series.o[i-1], series.h[i-1], series.l[i-1], series.c[i-1]
    rng = h - l
    body = abs(c - o)
    if rng <= 0:
        return out
    upper = h - max(o, c)
    lower = min(o, c) - l
    bull = c > o
    bear = c < o

    if bull and pc < po and c >= po and o <= pc:
        out["PA_BULLISH_ENGULFING"] = 0
    if bear and pc > po and c <= po and o >= pc:
        out["PA_BEARISH_ENGULFING"] = 0

    if lower / rng >= cfg.PIN_BAR_WICK_RATIO:
        out["PA_PIN_BAR"] = 0
        if body / rng <= 0.35:
            out["PA_HAMMER"] = 0
    if upper / rng >= cfg.PIN_BAR_WICK_RATIO:
        out["PA_PIN_BAR"] = 0
        if body / rng <= 0.35:
            out["PA_SHOOTING_STAR"] = 0

    if h <= ph and l >= pl:
        out["PA_INSIDE_BAR"] = 0
    if h > ph and l < pl:
        out["PA_OUTSIDE_BAR"] = 0
    if body >= cfg.LARGE_BODY_ATR * atr:
        out["PA_LARGE_BODY"] = 0
    if body >= cfg.MOMENTUM_CANDLE_ATR * atr:
        out["PA_MOMENTUM_CANDLE_UP" if bull else "PA_MOMENTUM_CANDLE_DOWN"] = 0

    lo = max(0, i - 20)
    if i > lo:
        prior_high = max(series.h[j] for j in range(lo, i))
        prior_low = min(series.l[j] for j in range(lo, i))
        if c > prior_high:
            out["PA_BREAKOUT_CANDLE_UP"] = 0
            out["PA_RANGE_BREAKOUT"] = 0
        if c < prior_low:
            out["PA_BREAKOUT_CANDLE_DOWN"] = 0
            out["PA_RANGE_BREAKDOWN"] = 0
        # false breakout: pierced the extreme, closed back inside
        if h > prior_high and c < prior_high:
            out["PA_FALSE_BREAKOUT_UP"] = 0
        if l < prior_low and c > prior_low:
            out["PA_FALSE_BREAKOUT_DOWN"] = 0

    # breakout + retest: a break happened recently and price has come back to it
    for back in range(2, min(21, i)):
        j = i - back
        lo2 = max(0, j - 20)
        if j <= lo2:
            continue
        ph2 = max(series.h[k] for k in range(lo2, j))
        if series.c[j] > ph2 and l <= ph2 <= h and c > ph2:
            out["PA_BREAKOUT_RETEST_UP"] = back
            break
    for back in range(2, min(21, i)):
        j = i - back
        lo2 = max(0, j - 20)
        if j <= lo2:
            continue
        pl2 = min(series.l[k] for k in range(lo2, j))
        if series.c[j] < pl2 and l <= pl2 <= h and c < pl2:
            out["PA_BREAKOUT_RETEST_DOWN"] = back
            break

    # reversal candle: rejection of the immediately preceding direction
    if bull and pc < po and lower / rng >= 0.4:
        out["PA_REVERSAL_CANDLE_UP"] = 0
    if bear and pc > po and upper / rng >= 0.4:
        out["PA_REVERSAL_CANDLE_DOWN"] = 0
    return out
