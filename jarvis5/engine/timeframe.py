"""One bundled per-timeframe state machine.

Everything that can be known about a timeframe at bar i is produced by a single
call to `on_bar_closed(i)` -- and that call only ever touches bars <= i.
"""
from __future__ import annotations

from typing import List, Optional

from . import displacement as disp
from .atr import ATR
from .liquidity import LiquidityEngine
from .structure import StructureEngine
from .swings import SwingEngine


class TimeframeState:
    def __init__(self, series, cfg):
        self.s = series
        self.cfg = cfg
        self.tf = series.tf
        self.atr = ATR(cfg.ATR_PERIOD)
        self.swings = SwingEngine(series, cfg.SWING_K, self.atr)
        self.structure = StructureEngine(series, cfg)
        self.liquidity = LiquidityEngine(series, cfg)
        self.i = -1                       # last closed bar index
        self.last_events = []
        self.last_sweeps = []
        self.last_displacement = None
        self.displacements = {}           # index -> Displacement

    @property
    def ready(self) -> bool:
        return self.atr.ready and self.i >= self.cfg.ATR_PERIOD + 2 * self.cfg.SWING_K

    def on_bar_closed(self, i: int):
        """Advance the timeframe by exactly one CLOSED bar."""
        assert i == self.i + 1, f"non-sequential bar {i} after {self.i}"
        self.i = i
        s = self.s
        a = self.atr.update(s.h[i], s.l[i], s.c[i])

        new_swings = self.swings.on_bar_closed(i)
        if new_swings:
            self.structure.on_swings(new_swings, a)
            self.liquidity.on_swings(new_swings, a, i,
                                     self.structure.internal, self.structure.external)
        self.liquidity.update_session_levels(i)

        sweeps = self.liquidity.on_bar_closed(i, a)
        events = self.structure.on_bar_closed(i, a)

        d = disp.measure(s, i, a, self.cfg)
        self.last_displacement = d
        if d is not None and d.grade != disp.WEAK:
            self.displacements[i] = d
            if len(self.displacements) > 400:
                for k in sorted(self.displacements)[:200]:
                    del self.displacements[k]

        # link a break to the sweep that preceded it -> MSS (spec 11)
        for ev in events:
            look = self.cfg.MSS_SWEEP_LOOKBACK
            want_side = "SELL_SIDE" if ev.direction == "BULLISH" else "BUY_SIDE"
            for sw in reversed(self.liquidity.sweeps):
                if i - sw.index > look:
                    break
                if sw.side != want_side:
                    continue
                if sw.index > ev.index:
                    continue
                ev.is_mss = True
                ev.sweep = sw
                sw.structure_reaction = True
                break
            if d is not None and d.direction == ev.direction:
                ev.displacement = d.ratio

        # keep the post-sweep displacement statistic current
        for sw in self.liquidity.sweeps[-20:]:
            if sw.index <= i and i - sw.index <= self.cfg.MSS_SWEEP_LOOKBACK:
                if d is not None and d.direction == sw.direction:
                    if d.ratio > sw.post_displacement:
                        sw.post_displacement = d.ratio

        self.last_events = events
        self.last_sweeps = sweeps
        return events, sweeps

    # -- convenience ---------------------------------------------------- #
    def atr_now(self) -> float:
        return self.atr.value

    def price(self) -> float:
        return self.s.c[self.i] if self.i >= 0 else 0.0

    def regime(self) -> str:
        return self.structure.update_regime(self.i)
