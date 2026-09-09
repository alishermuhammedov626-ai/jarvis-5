"""Deterministic Wyckoff phase detection (spec 7).

Wyckoff is usually read by eye; here every element is written as an explicit
threshold rule so that "Spring detected" means the same thing on every bar of
every coin, and so the result is reproducible.

A trading range is the anchor: without one, no accumulation, distribution,
spring or upthrust can be claimed.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple


class Range:
    __slots__ = ("high", "low", "start", "end", "bars", "width_atr",
                 "vol_first_half", "vol_second_half")

    def __init__(self, high, low, start, end, width_atr):
        self.high = high
        self.low = low
        self.start = start
        self.end = end
        self.bars = end - start + 1
        self.width_atr = width_atr
        self.vol_first_half = 0.0
        self.vol_second_half = 0.0

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    def position(self, price: float) -> float:
        span = self.high - self.low
        return (price - self.low) / span if span > 0 else 0.5


def find_range(series, i: int, atr: float, cfg) -> Optional[Range]:
    """The trading range ENDING BEFORE the recent window, if one exists.

    Crucially the range body excludes the last WY_RECENT_BARS bars.  If it did
    not, a spring's own low would define the range low and `low < range_low`
    could never be true -- the detector would be structurally incapable of
    finding the pattern it is named after.

    Grows the window backwards while price stays inside a band narrow enough
    (in ATR terms) to be a range rather than a trend leg.
    """
    if atr <= 0 or i < cfg.WY_MIN_RANGE_BARS + cfg.WY_RECENT_BARS:
        return None
    i = i - cfg.WY_RECENT_BARS
    start = max(0, i - cfg.WY_RANGE_BARS + 1)
    hi = series.h[i]
    lo = series.l[i]
    best: Optional[Tuple[int, float, float]] = None
    for j in range(i, start - 1, -1):
        if series.h[j] > hi:
            hi = series.h[j]
        if series.l[j] < lo:
            lo = series.l[j]
        width = (hi - lo) / atr
        bars = i - j + 1
        if width > cfg.WY_MAX_RANGE_ATR:
            break
        if bars >= cfg.WY_MIN_RANGE_BARS and width >= cfg.WY_MIN_RANGE_ATR:
            best = (j, hi, lo)
    if best is None:
        return None
    j, hi, lo = best
    r = Range(hi, lo, j, i, (hi - lo) / atr)
    mid = (j + i) // 2
    n1 = max(1, mid - j + 1)
    n2 = max(1, i - mid)
    r.vol_first_half = sum(series.v[k] for k in range(j, mid + 1)) / n1
    r.vol_second_half = sum(series.v[k] for k in range(mid + 1, i + 1)) / n2
    return r


def detect(series, i: int, atr: float, vol, cfg) -> Dict[str, int]:
    """Active Wyckoff elements -> how many bars ago each was observed."""
    out: Dict[str, int] = {}
    rng = find_range(series, i, atr, cfg)
    if rng is None:
        return out
    out["WY_TRADING_RANGE"] = i - rng.end

    contracting = rng.vol_second_half < rng.vol_first_half * 0.85
    expanding = rng.vol_second_half > rng.vol_first_half * 1.15

    spring_at = _spring(series, rng, i, vol, cfg)
    if spring_at is not None:
        out["WY_SPRING"] = i - spring_at
    upthrust_at = _upthrust(series, rng, i, vol, cfg)
    if upthrust_at is not None:
        out["WY_UPTHRUST"] = i - upthrust_at

    sos_at = _sos(series, rng, i, atr, vol, cfg)
    if sos_at is not None:
        out["WY_SOS"] = i - sos_at
    sow_at = _sow(series, rng, i, atr, vol, cfg)
    if sow_at is not None:
        out["WY_SOW"] = i - sow_at

    # LPS / LPSY: the retest that holds after a sign of strength / weakness
    if sos_at is not None:
        for j in range(sos_at + 1, i + 1):
            if series.l[j] <= rng.high <= series.h[j] and series.c[j] > rng.high:
                out["WY_LPS"] = i - j
                break
    if sow_at is not None:
        for j in range(sow_at + 1, i + 1):
            if series.l[j] <= rng.low <= series.h[j] and series.c[j] < rng.low:
                out["WY_LPSY"] = i - j
                break

    # breakouts and their failures
    look = min(cfg.WY_BREAKOUT_LOOKBACK, i - rng.end)
    for back in range(0, look + 1):
        j = i - back
        if j <= rng.end:
            break
        if series.c[j] > rng.high:
            out.setdefault("WY_RANGE_BREAKOUT", back)
        if series.c[j] < rng.low:
            out.setdefault("WY_RANGE_BREAKDOWN", back)
    for back in range(0, min(cfg.WY_FALSE_BREAKOUT_BARS, i) + 1):
        j = i - back
        if j <= rng.end:
            break
        if series.h[j] > rng.high and series.c[j] < rng.high:
            out.setdefault("WY_FALSE_BREAKOUT_UP", back)
        if series.l[j] < rng.low and series.c[j] > rng.low:
            out.setdefault("WY_FALSE_BREAKOUT_DOWN", back)

    # phase labels
    higher_lows = _higher_lows(series, rng)
    lower_highs = _lower_highs(series, rng)
    if contracting and (spring_at is not None or higher_lows):
        out["WY_ACCUMULATION"] = 0
    if contracting and (upthrust_at is not None or lower_highs):
        out["WY_DISTRIBUTION"] = 0
    if expanding and spring_at is not None:
        out["WY_ACCUMULATION"] = 0

    prior = _prior_leg(series, rng, cfg)
    if prior == "UP" and "WY_ACCUMULATION" in out:
        out["WY_REACCUMULATION"] = 0
    if prior == "DOWN" and "WY_DISTRIBUTION" in out:
        out["WY_REDISTRIBUTION"] = 0
    return out


def _spring(series, rng: Range, i: int, vol, cfg) -> Optional[int]:
    """Price dips below range low and closes back inside -- stops taken."""
    lo = max(rng.end + 1, i - cfg.WY_SPRING_LOOKBACK)
    for j in range(i, lo - 1, -1):
        if series.l[j] < rng.low and series.c[j] > rng.low:
            return j
    return None


def _upthrust(series, rng: Range, i: int, vol, cfg) -> Optional[int]:
    lo = max(rng.end + 1, i - cfg.WY_SPRING_LOOKBACK)
    for j in range(i, lo - 1, -1):
        if series.h[j] > rng.high and series.c[j] < rng.high:
            return j
    return None


def _sos(series, rng: Range, i: int, atr: float, vol, cfg) -> Optional[int]:
    lo = max(rng.end + 1, i - cfg.WY_BREAKOUT_LOOKBACK)
    for j in range(i, lo - 1, -1):
        if series.c[j] > rng.high and series.body(j) >= atr and vol.at(j) >= 1.3:
            return j
    return None


def _sow(series, rng: Range, i: int, atr: float, vol, cfg) -> Optional[int]:
    lo = max(rng.end + 1, i - cfg.WY_BREAKOUT_LOOKBACK)
    for j in range(i, lo - 1, -1):
        if series.c[j] < rng.low and series.body(j) >= atr and vol.at(j) >= 1.3:
            return j
    return None


def _higher_lows(series, rng: Range) -> bool:
    mid = (rng.start + rng.end) // 2
    if mid <= rng.start or rng.end <= mid:
        return False
    first = min(series.l[j] for j in range(rng.start, mid + 1))
    second = min(series.l[j] for j in range(mid + 1, rng.end + 1))
    return second > first


def _lower_highs(series, rng: Range) -> bool:
    mid = (rng.start + rng.end) // 2
    if mid <= rng.start or rng.end <= mid:
        return False
    first = max(series.h[j] for j in range(rng.start, mid + 1))
    second = max(series.h[j] for j in range(mid + 1, rng.end + 1))
    return second < first


def _prior_leg(series, rng: Range, cfg) -> str:
    """Was the range preceded by a trend leg?  (re-accumulation vs accumulation)"""
    start = max(0, rng.start - cfg.WY_TREND_LEG_BARS)
    if rng.start - start < 20:
        return "NONE"
    before = series.c[start]
    at = series.c[rng.start]
    if before <= 0:
        return "NONE"
    change = (at - before) / before
    if change >= cfg.WY_TREND_LEG_PCT:
        return "UP"
    if change <= -cfg.WY_TREND_LEG_PCT:
        return "DOWN"
    return "NONE"
