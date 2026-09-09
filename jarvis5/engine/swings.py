"""Fractal swing engine (K bars each side) with strict confirmation delay.

A swing at index p is only published after bar p+K has CLOSED.  Nothing
downstream ever sees a swing before the market has actually printed the bars
that define it -- this is the single most important no-look-ahead guarantee in
the system.
"""
from __future__ import annotations

from typing import List, Optional

HIGH = "HIGH"
LOW = "LOW"


class Swing:
    __slots__ = ("index", "time", "price", "kind", "tf", "confirmed_time",
                 "atr", "broken", "swept", "label", "is_external", "strength")

    def __init__(self, index: int, time: int, price: float, kind: str, tf: str,
                 confirmed_time: int, atr: float):
        self.index = index
        self.time = time                  # open time of the swing bar
        self.price = price
        self.kind = kind                  # HIGH | LOW
        self.tf = tf
        self.confirmed_time = confirmed_time
        self.atr = atr
        self.broken = False
        self.swept = False
        self.label = ""                   # HH | HL | LH | LL
        self.is_external = False
        self.strength = 1

    def age_bars(self, current_index: int) -> int:
        return current_index - self.index

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Swing {self.tf} {self.kind} {self.price:.8g} @{self.index} {self.label}>"


class SwingEngine:
    """Feeds on a Series and emits confirmed swings."""

    def __init__(self, series, k: int, atr):
        self.s = series
        self.k = int(k)
        self.atr = atr
        self.highs: List[Swing] = []
        self.lows: List[Swing] = []
        self.all: List[Swing] = []
        self._last_checked = -1

    def on_bar_closed(self, i: int) -> List[Swing]:
        """Called after bar i closed.  Returns swings confirmed by this bar."""
        k = self.k
        p = i - k
        out: List[Swing] = []
        if p - k < 0:
            return out
        s = self.s
        hp = s.h[p]
        lp = s.l[p]

        is_high = True
        for j in range(p - k, p):
            if not (hp > s.h[j]):
                is_high = False
                break
        if is_high:
            for j in range(p + 1, p + k + 1):
                if not (hp >= s.h[j]):
                    is_high = False
                    break
        if is_high:
            sw = Swing(p, s.ot[p], hp, HIGH, s.tf, s.close_time(i), self.atr.at(p))
            self.highs.append(sw)
            self.all.append(sw)
            out.append(sw)

        is_low = True
        for j in range(p - k, p):
            if not (lp < s.l[j]):
                is_low = False
                break
        if is_low:
            for j in range(p + 1, p + k + 1):
                if not (lp <= s.l[j]):
                    is_low = False
                    break
        if is_low:
            sw = Swing(p, s.ot[p], lp, LOW, s.tf, s.close_time(i), self.atr.at(p))
            self.lows.append(sw)
            self.all.append(sw)
            out.append(sw)

        return out

    # -- queries -------------------------------------------------------- #
    def last_high(self, unbroken: bool = False) -> Optional[Swing]:
        for sw in reversed(self.highs):
            if not unbroken or not sw.broken:
                return sw
        return None

    def last_low(self, unbroken: bool = False) -> Optional[Swing]:
        for sw in reversed(self.lows):
            if not unbroken or not sw.broken:
                return sw
        return None

    def recent(self, kind: str, count: int) -> List[Swing]:
        src = self.highs if kind == HIGH else self.lows
        return src[-count:] if count else []
