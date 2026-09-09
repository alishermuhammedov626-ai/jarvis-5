"""Deterministic synthetic M1 data.

Not a market model and not used for any reported result -- it exists so the
whole pipeline (aggregation, structure, setups, execution, reporting) can be
exercised and regression-tested without a network connection.
"""
from __future__ import annotations

import math
import random
import zlib
from typing import Optional

from ..core.series import Series

MINUTE_MS = 60_000


def make_series(symbol: str, start_ms: int, minutes: int, price: float = 0.20,
                seed: int = 7, vol_bps: float = 6.0,
                trend_bars: int = 720) -> Series:
    # zlib.crc32 is stable across processes; hash() is NOT (PYTHONHASHSEED)
    rng = random.Random(seed ^ zlib.crc32(symbol.encode()))
    s = Series(symbol, "M1")
    p = price
    drift = 0.0
    for i in range(minutes):
        if i % trend_bars == 0:
            drift = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) * vol_bps / 40_000.0
        shock = 0.0
        if rng.random() < 0.004:                     # impulse / displacement leg
            shock = rng.choice([-1, 1]) * vol_bps * rng.uniform(3.0, 9.0) / 10_000.0
        step = rng.gauss(drift, vol_bps / 10_000.0) + shock
        o = p
        c = max(1e-9, p * (1.0 + step))
        # wick is expressed relatively and only then scaled by price, so the
        # generated series has the same character at any price level
        wick = (abs(step) * rng.uniform(0.2, 1.6) + vol_bps / 40_000.0) * p
        h = max(o, c) + wick * rng.uniform(0.0, 1.0)
        l = min(o, c) - wick * rng.uniform(0.0, 1.0)
        l = max(l, 1e-9)
        v = rng.uniform(1_000, 50_000) * (2.5 if shock else 1.0)
        s.append(start_ms + i * MINUTE_MS, o, h, l, c, v, v * c)
        p = c
    return s
