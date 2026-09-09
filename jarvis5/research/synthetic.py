"""Synthetic market with DELIBERATELY INJECTED patterns (spec 30).

The research engine has to be validated before it is pointed at real money
data, and the only way to validate a detector is to hand it data whose ground
truth you already know.

Ten scenario builders are concatenated in a deterministic order; each records
what it injected and where.  A detector that cannot find a sweep in the
`SWEEP_MSS_DISPLACEMENT` segments, or that finds just as many in the `RANDOM`
segments, is broken -- and the synthetic test says so.

These prices are NOT a market model.  No performance number from them means
anything; they exist to test the instrument, not the market.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from ..core.series import Series

MINUTE_MS = 60_000

SCENARIOS = [
    "SWEEP_MSS_DISPLACEMENT_LONG",
    "SWEEP_FVG_LONG",
    "WYCKOFF_SPRING_LONG",
    "RSI_REVERSAL_LONG",
    "VOLUME_BREAKOUT_LONG",
    "FALSE_BREAKOUT_SHORT",
    "TREND_CONTINUATION_LONG",
    "RANGE_BREAKOUT_LONG",
    "RANDOM",
    "NOISE",
]


class Builder:
    """Appends bars to a Series with a controllable noise floor."""

    def __init__(self, symbol: str, start_ms: int, price: float, seed: int,
                 tick_bps: float = 6.0):
        self.s = Series(symbol, "M1")
        self.t = start_ms
        self.p = price
        self.rng = random.Random(seed)
        self.tick = tick_bps / 10_000.0
        self.truth: List[dict] = []
        self.markers: Dict[str, int] = {}

    # -- primitives ----------------------------------------------------- #
    def bar(self, o: float, h: float, l: float, c: float, v: float) -> None:
        h = max(h, o, c)
        l = min(l, o, c)
        self.s.append(self.t, o, h, l, c, v, v * c)
        self.t += MINUTE_MS
        self.p = c

    def noise(self, n: int, vol: float = 1.0, vmul: float = 1.0) -> None:
        for _ in range(n):
            o = self.p
            step = self.rng.gauss(0.0, self.tick * vol)
            c = o * (1.0 + step)
            w = abs(step) * self.rng.uniform(0.3, 1.2) + self.tick * 0.5 * vol
            h = max(o, c) * (1.0 + w * self.rng.uniform(0.0, 1.0))
            l = min(o, c) * (1.0 - w * self.rng.uniform(0.0, 1.0))
            self.bar(o, h, l, c, self.rng.uniform(800, 1600) * vmul)

    def band(self, n: int, width_pct: float, vmul: float = 1.0) -> Tuple[float, float]:
        """Oscillate inside a fixed band; returns (low, high) of the band."""
        centre = self.p
        lo = centre * (1 - width_pct / 2)
        hi = centre * (1 + width_pct / 2)
        for k in range(n):
            o = self.p
            # deterministic oscillation plus a little noise
            phase = (k % 12) / 12.0
            target = lo + (hi - lo) * (0.5 + 0.42 * (1 if (k // 6) % 2 else -1) * phase)
            c = o + (target - o) * 0.5
            c = min(max(c, lo * 1.0005), hi * 0.9995)
            w = self.tick * 0.6
            h = max(o, c) * (1 + w * self.rng.uniform(0.0, 1.0))
            l = min(o, c) * (1 - w * self.rng.uniform(0.0, 1.0))
            self.bar(o, min(h, hi), max(l, lo), c, self.rng.uniform(700, 1400) * vmul)
        return lo, hi

    def wick(self, depth_pct: float, down: bool = True, vmul: float = 2.5,
             close_back: float = 0.6) -> None:
        """One bar that pierces a level and closes back inside it."""
        o = self.p
        if down:
            l = o * (1 - depth_pct)
            c = o * (1 - depth_pct * (1 - close_back))
            h = o * (1 + self.tick * 0.3)
        else:
            h = o * (1 + depth_pct)
            c = o * (1 + depth_pct * (1 - close_back))
            l = o * (1 - self.tick * 0.3)
        self.bar(o, h, l, c, self.rng.uniform(2500, 4000) * vmul)

    def spike_through(self, level: float, beyond_pct: float, down: bool = True,
                      vmul: float = 3.0, close_back: float = 0.55) -> None:
        """Trade decisively THROUGH a known level and close back inside it.

        Takes the level as an argument rather than a percentage of the current
        price, so the injected spring/sweep provably breaks the level it is
        meant to break instead of merely coming close to it.
        """
        o = self.p
        if down:
            l = level * (1 - beyond_pct)
            c = level * (1 + beyond_pct * close_back)
            h = max(o, c) * (1 + self.tick * 0.3)
        else:
            h = level * (1 + beyond_pct)
            c = level * (1 - beyond_pct * close_back)
            l = min(o, c) * (1 - self.tick * 0.3)
        self.bar(o, h, l, c, self.rng.uniform(2500, 4000) * vmul)

    def recent_low(self, bars: int, exclude_recent: int = 6) -> float:
        """Lowest low of an ESTABLISHED region.

        The last few bars are excluded because a K-bar fractal has not been
        confirmed there yet, so no liquidity level exists at that low for a
        sweep to take -- injecting one would test nothing.
        """
        n = self.s.n - exclude_recent
        lo = max(0, n - bars)
        return min(self.s.l[j] for j in range(lo, n)) if n > lo else self.p

    def recent_high(self, bars: int, exclude_recent: int = 6) -> float:
        n = self.s.n - exclude_recent
        lo = max(0, n - bars)
        return max(self.s.h[j] for j in range(lo, n)) if n > lo else self.p

    def impulse(self, total_pct: float, bars: int, up: bool = True,
                vmul: float = 2.0) -> None:
        """A run of same-direction bodies (a displacement leg)."""
        per = (1.0 + total_pct) ** (1.0 / bars) - 1.0
        for _ in range(bars):
            o = self.p
            c = o * (1 + per) if up else o * (1 - per)
            if up:
                h = c * (1 + self.tick * 0.2)
                l = o * (1 - self.tick * 0.2)
            else:
                l = c * (1 - self.tick * 0.2)
                h = o * (1 + self.tick * 0.2)
            self.bar(o, h, l, c, self.rng.uniform(2000, 3500) * vmul)

    def gap_fvg(self, size_pct: float, up: bool = True) -> None:
        """Three bars that leave an unfilled gap (a fair value gap)."""
        o = self.p
        if up:
            self.bar(o, o * (1 + self.tick * 0.3), o * (1 - self.tick * 0.3),
                     o * (1 + self.tick * 0.2), self.rng.uniform(900, 1500))
            a = self.p
            self.bar(a, a * (1 + size_pct * 1.6), a * (1 - self.tick * 0.1),
                     a * (1 + size_pct * 1.4), self.rng.uniform(3000, 4500))
            b = self.p
            self.bar(b, b * (1 + size_pct * 0.4), b * (1 - self.tick * 0.05),
                     b * (1 + size_pct * 0.2), self.rng.uniform(1500, 2500))
        else:
            self.bar(o, o * (1 + self.tick * 0.3), o * (1 - self.tick * 0.3),
                     o * (1 - self.tick * 0.2), self.rng.uniform(900, 1500))
            a = self.p
            self.bar(a, a * (1 + self.tick * 0.1), a * (1 - size_pct * 1.6),
                     a * (1 - size_pct * 1.4), self.rng.uniform(3000, 4500))
            b = self.p
            self.bar(b, b * (1 + self.tick * 0.05), b * (1 - size_pct * 0.4),
                     b * (1 - size_pct * 0.2), self.rng.uniform(1500, 2500))

    def drift(self, n: int, total_pct: float, vmul: float = 1.0) -> None:
        per = (1.0 + total_pct) ** (1.0 / max(1, n)) - 1.0
        for _ in range(n):
            o = self.p
            step = per + self.rng.gauss(0.0, self.tick * 0.6)
            c = o * (1 + step)
            w = self.tick * 0.7
            h = max(o, c) * (1 + w * self.rng.uniform(0.0, 1.0))
            l = min(o, c) * (1 - w * self.rng.uniform(0.0, 1.0))
            self.bar(o, h, l, c, self.rng.uniform(900, 1800) * vmul)

    def note(self, key: str) -> None:
        """Record the bar index of an element we are deliberately injecting."""
        self.markers[key] = self.s.n - 1

    def mark(self, scenario: str, start: int, move_at: int, direction: str) -> None:
        self.truth.append({
            "scenario": scenario,
            "start_index": start,
            "end_index": self.s.n - 1,
            "move_index": move_at,
            "direction": direction,
            "markers": dict(self.markers),
        })
        self.markers = {}


# ---------------------------------------------------------------------- #
# scenario builders
# ---------------------------------------------------------------------- #
def _sweep_mss_displacement_long(b: Builder) -> None:
    start = b.s.n
    lo, hi = b.band(70, 0.0040)
    b.noise(6)
    b.spike_through(b.recent_low(80), 0.0018, down=True)   # sell-side sweep
    b.note("sweep_low")
    b.noise(4)
    b.impulse(0.0035, 4, up=True)          # displacement through internal highs
    b.note("displacement_up")
    b.drift(6, -0.0012)                    # correction back into the zone
    move_at = b.s.n
    b.impulse(0.0130, 10, up=True)         # the +1 % move itself
    b.noise(30)
    b.mark("SWEEP_MSS_DISPLACEMENT_LONG", start, move_at, "LONG")


def _sweep_fvg_long(b: Builder) -> None:
    start = b.s.n
    b.band(60, 0.0035)
    b.spike_through(b.recent_low(70), 0.0018, down=True)
    b.note("sweep_low")
    b.noise(3)
    b.gap_fvg(0.0022, up=True)
    b.note("fvg_up")
    b.drift(5, -0.0008)
    move_at = b.s.n
    b.impulse(0.0125, 9, up=True)
    b.noise(30)
    b.mark("SWEEP_FVG_LONG", start, move_at, "LONG")


def _wyckoff_spring_long(b: Builder) -> None:
    start = b.s.n
    b.band(60, 0.0045, vmul=1.4)           # high-volume early range
    b.band(60, 0.0045, vmul=0.5)           # volume dries up  (contraction)
    floor = b.recent_low(140)              # the range low the market has built
    b.spike_through(floor, 0.0022, down=True, vmul=3.0)   # the spring
    b.note("spring")
    b.noise(8, vmul=0.7)
    move_at = b.s.n
    b.impulse(0.0140, 12, up=True, vmul=2.5)
    b.noise(30)
    b.mark("WYCKOFF_SPRING_LONG", start, move_at, "LONG")


def _rsi_reversal_long(b: Builder) -> None:
    start = b.s.n
    b.drift(45, -0.0170)                   # sustained decline -> RSI oversold
    b.note("rsi_oversold")
    b.noise(6)
    move_at = b.s.n
    b.impulse(0.0135, 11, up=True)
    b.noise(30)
    b.mark("RSI_REVERSAL_LONG", start, move_at, "LONG")


def _volume_breakout_long(b: Builder) -> None:
    start = b.s.n
    lo, hi = b.band(70, 0.0035, vmul=0.6)
    move_at = b.s.n
    b.impulse(0.0135, 10, up=True, vmul=4.0)   # breakout on 4x volume
    b.note("volume_breakout")
    b.noise(30)
    b.mark("VOLUME_BREAKOUT_LONG", start, move_at, "LONG")


def _false_breakout_short(b: Builder) -> None:
    start = b.s.n
    b.band(70, 0.0040)
    b.spike_through(b.recent_high(80), 0.0018, down=False)  # pokes above, closes back
    b.note("sweep_high")
    b.noise(4)
    move_at = b.s.n
    b.impulse(0.0130, 10, up=False)
    b.noise(30)
    b.mark("FALSE_BREAKOUT_SHORT", start, move_at, "SHORT")


def _trend_continuation_long(b: Builder) -> None:
    start = b.s.n
    b.drift(40, 0.0110)                    # established uptrend
    b.drift(14, -0.0038)                   # pullback
    b.drift(6, 0.0010)                     # small bounce, so the pullback low
    #                                        becomes an established level below price
    b.spike_through(b.recent_low(20), 0.0012, down=True)  # stop-run below it
    b.note("sweep_low")
    move_at = b.s.n
    b.impulse(0.0125, 10, up=True)
    b.noise(30)
    b.mark("TREND_CONTINUATION_LONG", start, move_at, "LONG")


def _range_breakout_long(b: Builder) -> None:
    start = b.s.n
    b.band(80, 0.0038)
    move_at = b.s.n
    b.impulse(0.0130, 9, up=True)
    b.note("range_breakout")
    b.noise(30)
    b.mark("RANGE_BREAKOUT_LONG", start, move_at, "LONG")


def _random(b: Builder) -> None:
    start = b.s.n
    b.noise(150)
    b.mark("RANDOM", start, -1, "NONE")


def _noise_patterns(b: Builder) -> None:
    """Pattern-shaped bars with NO follow-through: the false-positive trap."""
    start = b.s.n
    b.band(50, 0.0035)
    b.wick(0.0025, down=True)              # a sweep that leads nowhere
    b.noise(10)
    b.wick(0.0025, down=False)
    b.noise(10)
    b.gap_fvg(0.0015, up=True)             # a gap that is immediately filled
    b.drift(10, -0.0020)
    b.noise(40)
    b.mark("NOISE", start, -1, "NONE")


BUILDERS = {
    "SWEEP_MSS_DISPLACEMENT_LONG": _sweep_mss_displacement_long,
    "SWEEP_FVG_LONG": _sweep_fvg_long,
    "WYCKOFF_SPRING_LONG": _wyckoff_spring_long,
    "RSI_REVERSAL_LONG": _rsi_reversal_long,
    "VOLUME_BREAKOUT_LONG": _volume_breakout_long,
    "FALSE_BREAKOUT_SHORT": _false_breakout_short,
    "TREND_CONTINUATION_LONG": _trend_continuation_long,
    "RANGE_BREAKOUT_LONG": _range_breakout_long,
    "RANDOM": _random,
    "NOISE": _noise_patterns,
}


def make_market(symbol: str, start_ms: int, cycles: int = 12,
                price: float = 1.0, seed: int = 11,
                tick_bps: float = 6.0):
    """Build a synthetic market and return (series, ground_truth).

    Scenarios are separated by a COOL-DOWN that retraces most of the injected
    move.  Without it each impulse would begin inside the previous one's
    horizon with no meaningful reset, and the event clusterer would correctly
    merge them into one long advance -- making the injected moves impossible to
    score individually.
    """
    b = Builder(symbol, start_ms, price, seed, tick_bps)
    b.noise(400)                            # warm-up before anything is injected
    for _cyc in range(cycles):
        for name in SCENARIOS:
            p0 = b.p
            BUILDERS[name](b)
            move = b.p / p0 - 1.0
            if abs(move) > 0.001:
                b.drift(50, -0.7 * move)    # give most of it back
            b.noise(40)
    b.noise(400)
    return b.s, b.truth


def truth_index(truth: List[dict], n: int) -> List[str]:
    """Bar index -> scenario label, for scoring detection by segment."""
    labels = ["WARMUP"] * n
    for t in truth:
        for i in range(t["start_index"], min(t["end_index"] + 1, n)):
            labels[i] = t["scenario"]
    return labels


if __name__ == "__main__":          # python -m jarvis5.research.synthetic
    import sys

    from .cli import main_synthetic

    sys.exit(main_synthetic())
