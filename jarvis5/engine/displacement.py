"""Displacement measurement, ATR-normalised (spec 20).

A single candle body, or a 1..DISPLACEMENT_MAX_LEG_BARS same-direction leg,
divided by the ATR known at that bar.

    STRONG  >= STRONG_DISPLACEMENT_ATR   (1.5)
    MEDIUM  >= MIN_DISPLACEMENT_ATR      (1.0)
    WEAK    <  MIN_DISPLACEMENT_ATR
"""
from __future__ import annotations

STRONG = "STRONG"
MEDIUM = "MEDIUM"
WEAK = "WEAK"


class Displacement:
    __slots__ = ("index", "direction", "ratio", "grade", "start_index",
                 "start_price", "end_price")

    def __init__(self, index: int, direction: str, ratio: float, grade: str,
                 start_index: int, start_price: float, end_price: float):
        self.index = index
        self.direction = direction
        self.ratio = ratio
        self.grade = grade
        self.start_index = start_index
        self.start_price = start_price
        self.end_price = end_price

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Disp {self.direction} {self.ratio:.2f}ATR {self.grade} @{self.index}>"


def grade(ratio: float, cfg) -> str:
    if ratio >= cfg.STRONG_DISPLACEMENT_ATR:
        return STRONG
    if ratio >= cfg.MIN_DISPLACEMENT_ATR:
        return MEDIUM
    return WEAK


def measure(series, i: int, atr: float, cfg, direction: str = None):
    """Best displacement leg ending at bar i.  None if ATR is not ready."""
    if atr <= 0 or i < 0:
        return None
    best = None
    max_leg = min(cfg.DISPLACEMENT_MAX_LEG_BARS, i + 1)
    for legs in range(1, max_leg + 1):
        start = i - legs + 1
        if start < 0:
            break
        # every bar of the leg must push the same way
        ok = True
        for j in range(start, i + 1):
            if series.c[j] > series.o[j]:
                d = "BULLISH"
            elif series.c[j] < series.o[j]:
                d = "BEARISH"
            else:
                ok = False
                break
            if j == start:
                leg_dir = d
            elif d != leg_dir:
                ok = False
                break
        if not ok:
            continue
        if direction and leg_dir != direction:
            continue
        move = abs(series.c[i] - series.o[start])
        ratio = move / atr
        if best is None or ratio > best.ratio:
            best = Displacement(i, leg_dir, ratio, grade(ratio, cfg), start,
                                series.o[start], series.c[i])
    return best
