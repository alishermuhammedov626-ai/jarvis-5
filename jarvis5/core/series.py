"""Memory-compact OHLCV series backed by array.array.

A 1-year M1 history for 6 symbols is ~3.1M candles; python objects would cost
gigabytes, so every column is a flat typed array instead.
"""
from __future__ import annotations

from array import array
from typing import Iterable, List, Optional, Sequence

MINUTE_MS = 60_000
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15}


class Series:
    """Append-only OHLCV columns for a single symbol / timeframe."""

    __slots__ = ("symbol", "tf", "ot", "o", "h", "l", "c", "v", "qv", "n")

    def __init__(self, symbol: str, tf: str):
        self.symbol = symbol
        self.tf = tf
        self.ot = array("q")     # open time, ms epoch
        self.o = array("d")
        self.h = array("d")
        self.l = array("d")
        self.c = array("d")
        self.v = array("d")
        self.qv = array("d")     # quote volume (0.0 when unavailable)
        self.n = 0

    # -- construction -------------------------------------------------- #
    def append(self, ot: int, o: float, h: float, l: float, c: float,
               v: float, qv: float = 0.0) -> None:
        self.ot.append(int(ot))
        self.o.append(float(o))
        self.h.append(float(h))
        self.l.append(float(l))
        self.c.append(float(c))
        self.v.append(float(v))
        self.qv.append(float(qv))
        self.n += 1

    def __len__(self) -> int:
        return self.n

    # -- time ---------------------------------------------------------- #
    @property
    def step_ms(self) -> int:
        return TF_MINUTES[self.tf] * MINUTE_MS

    def close_time(self, i: int) -> int:
        """Exclusive close timestamp of bar i (= open time of bar i+1)."""
        return self.ot[i] + self.step_ms

    def range(self, i: int) -> float:
        return self.h[i] - self.l[i]

    def body(self, i: int) -> float:
        return abs(self.c[i] - self.o[i])

    def is_bull(self, i: int) -> bool:
        return self.c[i] > self.o[i]

    def is_bear(self, i: int) -> bool:
        return self.c[i] < self.o[i]

    def slice_dict(self, i: int) -> dict:
        return {
            "open_time": self.ot[i], "open": self.o[i], "high": self.h[i],
            "low": self.l[i], "close": self.c[i], "volume": self.v[i],
        }

    def slice(self, start_ms: Optional[int] = None,
              end_ms: Optional[int] = None) -> "Series":
        """Chronological sub-series [start_ms, end_ms).  Used by walk-forward
        splits -- never by the engine, which only ever moves forward."""
        out = Series(self.symbol, self.tf)
        for i in range(self.n):
            t = self.ot[i]
            if start_ms is not None and t < start_ms:
                continue
            if end_ms is not None and t >= end_ms:
                break
            out.append(t, self.o[i], self.h[i], self.l[i], self.c[i],
                       self.v[i], self.qv[i])
        return out

    # -- integrity ----------------------------------------------------- #
    def check_continuity(self) -> List[int]:
        """Return indices where the expected bar spacing is violated.

        Gaps are reported, never silently repaired: a corrupted candle
        sequence is a hard rejection reason (INVALID_DATA), not something the
        backtest is allowed to paper over.
        """
        step = self.step_ms
        bad = []
        for i in range(1, self.n):
            if self.ot[i] - self.ot[i - 1] != step:
                bad.append(i)
        return bad

    def check_sanity(self) -> List[int]:
        bad = []
        for i in range(self.n):
            hi, lo, op, cl = self.h[i], self.l[i], self.o[i], self.c[i]
            if not (hi >= lo and hi >= op and hi >= cl and lo <= op and lo <= cl):
                bad.append(i)
            elif lo <= 0.0:
                bad.append(i)
        return bad


def aggregate(m1: Series, tf: str) -> Series:
    """Deterministically aggregate M1 -> M5 / M15.

    Buckets are aligned to the exchange clock (epoch ms is already aligned to
    UTC midnight, and 5 / 15 divide the hour evenly), so bucket_start =
    open_time - open_time % step.  A bucket is emitted ONLY when every one of
    its constituent M1 candles is present -- a partial bucket would otherwise
    publish a bar whose high/low are unknowable at that point in time.
    """
    step = TF_MINUTES[tf] * MINUTE_MS
    need = TF_MINUTES[tf]
    out = Series(m1.symbol, tf)

    i = 0
    n = m1.n
    while i < n:
        start = m1.ot[i] - (m1.ot[i] % step)
        j = i
        o = m1.o[i]
        hi = m1.h[i]
        lo = m1.l[i]
        cl = m1.c[i]
        vol = 0.0
        qvol = 0.0
        count = 0
        while j < n and m1.ot[j] < start + step:
            if m1.h[j] > hi:
                hi = m1.h[j]
            if m1.l[j] < lo:
                lo = m1.l[j]
            cl = m1.c[j]
            vol += m1.v[j]
            qvol += m1.qv[j]
            count += 1
            j += 1
        if count == need:
            out.append(start, o, hi, lo, cl, vol, qvol)
        i = j
    return out
