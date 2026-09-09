"""Incremental Wilder ATR.

Fed one CLOSED candle at a time.  Value at bar i only ever depends on bars
<= i, which is what makes it safe for the no-look-ahead contract.
"""
from __future__ import annotations

from array import array


class ATR:
    __slots__ = ("period", "values", "_prev_close", "_sum", "_count", "_atr", "n")

    def __init__(self, period: int = 14):
        self.period = int(period)
        self.values = array("d")     # values[i] = ATR after bar i (0.0 while warming up)
        self._prev_close = None
        self._sum = 0.0
        self._count = 0
        self._atr = 0.0
        self.n = 0

    def update(self, high: float, low: float, close: float) -> float:
        if self._prev_close is None:
            tr = high - low
        else:
            pc = self._prev_close
            tr = max(high - low, abs(high - pc), abs(low - pc))
        self._prev_close = close

        if self._count < self.period:
            self._sum += tr
            self._count += 1
            self._atr = self._sum / self._count if self._count == self.period else 0.0
            if self._count == self.period:
                self._atr = self._sum / self.period
        else:
            self._atr = (self._atr * (self.period - 1) + tr) / self.period

        self.values.append(self._atr)
        self.n += 1
        return self._atr

    @property
    def value(self) -> float:
        return self._atr

    @property
    def ready(self) -> bool:
        return self._count >= self.period and self._atr > 0.0

    def at(self, i: int) -> float:
        """ATR as it was known after bar i."""
        if i < 0 or i >= self.n:
            return 0.0
        return self.values[i]
