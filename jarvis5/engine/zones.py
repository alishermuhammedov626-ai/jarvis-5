"""Order Block / Fair Value Gap zones and their confluence (spec 21-23).

A zone is only created as the by-product of a displacement leg that produced a
structural break -- a naked OB or FVG is never an entry signal on its own.
"""
from __future__ import annotations

from typing import List, Optional

OB = "OB"
FVG = "FVG"
OB_FVG = "OB_FVG"


class Zone:
    __slots__ = ("lower", "upper", "kind", "direction", "tf", "creation_time",
                 "creation_index", "source_ob", "source_fvg", "fresh", "touches",
                 "mitigated", "displacement", "atr", "linked_event", "zone_id")

    _seq = 0

    def __init__(self, lower: float, upper: float, kind: str, direction: str,
                 tf: str, creation_time: int, creation_index: int, atr: float):
        Zone._seq += 1
        self.zone_id = Zone._seq
        self.lower = lower
        self.upper = upper
        self.kind = kind
        self.direction = direction        # BULLISH (demand) | BEARISH (supply)
        self.tf = tf
        self.creation_time = creation_time
        self.creation_index = creation_index
        self.atr = atr
        self.source_ob = None
        self.source_fvg = None
        self.fresh = True
        self.touches = 0
        self.mitigated = False
        self.displacement = 0.0
        self.linked_event = None

    @property
    def mid(self) -> float:
        return (self.lower + self.upper) / 2.0

    @property
    def height(self) -> float:
        return self.upper - self.lower

    def height_atr(self) -> float:
        return self.height / self.atr if self.atr > 0 else 0.0

    def contains(self, price: float, tol: float = 0.0) -> bool:
        return (self.lower - tol) <= price <= (self.upper + tol)

    def entry_edge(self) -> float:
        """The edge price first reaches on a retest."""
        return self.upper if self.direction == "BULLISH" else self.lower

    def far_edge(self) -> float:
        return self.lower if self.direction == "BULLISH" else self.upper

    def age_bars(self, i: int) -> int:
        return i - self.creation_index

    def as_dict(self) -> dict:
        return {
            "zone_id": self.zone_id, "kind": self.kind, "tf": self.tf,
            "direction": self.direction, "lower": self.lower, "upper": self.upper,
            "height_atr": round(self.height_atr(), 4),
            "fresh": self.fresh, "touches": self.touches,
            "mitigated": self.mitigated,
            "displacement": round(self.displacement, 4),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Zone {self.kind} {self.direction} [{self.lower:.8g},{self.upper:.8g}]>"


def find_order_block(series, break_index: int, direction: str, cfg,
                     start_index: Optional[int] = None):
    """Last meaningful opposite-direction candle before the impulse.

    BULLISH -> last bearish candle;  BEARISH -> last bullish candle.
    """
    first = start_index if start_index is not None else break_index
    lo_bound = max(0, first - cfg.OB_LOOKBACK)
    for j in range(first, lo_bound - 1, -1):
        if j > break_index:
            continue
        bull = series.c[j] > series.o[j]
        bear = series.c[j] < series.o[j]
        if direction == "BULLISH" and bear:
            return j
        if direction == "BEARISH" and bull:
            return j
    return None


def build_ob_zone(series, ob_index: int, direction: str, atr: float, cfg) -> Zone:
    if cfg.OB_USE_WICKS:
        lower, upper = series.l[ob_index], series.h[ob_index]
    else:
        lower = min(series.o[ob_index], series.c[ob_index])
        upper = max(series.o[ob_index], series.c[ob_index])
    z = Zone(lower, upper, OB, direction, series.tf, series.ot[ob_index],
             ob_index, atr)
    z.source_ob = ob_index
    return z


def find_fvg(series, i: int, direction: str, atr: float, cfg):
    """3-candle FVG ending at bar i (needs i >= 2)."""
    if i < 2 or atr <= 0:
        return None
    a, c = i - 2, i
    if direction == "BULLISH":
        if series.h[a] < series.l[c]:
            size = series.l[c] - series.h[a]
            if size >= cfg.MIN_FVG_ATR * atr:
                return series.h[a], series.l[c], size
    else:
        if series.l[a] > series.h[c]:
            size = series.l[a] - series.h[c]
            if size >= cfg.MIN_FVG_ATR * atr:
                return series.h[c], series.l[a], size
    return None


def build_zone(series, break_index: int, direction: str, atr: float, cfg,
               leg_start: Optional[int] = None):
    """Build the best available entry zone for an impulse ending at break_index.

    Order of preference: OB+FVG confluence > OB > FVG.
    """
    ob_zone = None
    ob_index = find_order_block(series, break_index, direction, cfg,
                                start_index=(leg_start - 1) if leg_start else None)
    if ob_index is not None:
        ob_zone = build_ob_zone(series, ob_index, direction, atr, cfg)

    # an FVG created by the impulse itself
    fvg_zone = None
    for j in range(max(2, (leg_start or break_index)), break_index + 1):
        got = find_fvg(series, j, direction, atr, cfg)
        if got:
            lower, upper, size = got
            fz = Zone(lower, upper, FVG, direction, series.tf, series.ot[j], j, atr)
            fz.source_fvg = j
            fvg_zone = fz
            break

    if ob_zone and fvg_zone:
        tol = cfg.ZONE_CONFLUENCE_ATR * atr
        overlap = (ob_zone.lower - tol) <= fvg_zone.upper and \
                  (fvg_zone.lower - tol) <= ob_zone.upper
        if overlap:
            z = Zone(min(ob_zone.lower, fvg_zone.lower),
                     max(ob_zone.upper, fvg_zone.upper),
                     OB_FVG, direction, series.tf, ob_zone.creation_time,
                     ob_zone.creation_index, atr)
            z.source_ob = ob_zone.source_ob
            z.source_fvg = fvg_zone.source_fvg
            if z.height_atr() <= cfg.MAX_ZONE_ATR:
                return z
    for z in (ob_zone, fvg_zone):
        if z is not None and z.height_atr() <= cfg.MAX_ZONE_ATR and z.height > 0:
            return z
    return None
