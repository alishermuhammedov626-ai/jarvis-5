"""Zone retest + M1 micro confirmation (spec 24, 25, 51).

M1 never selects a direction; it only decides WHEN a direction that M5 already
established is allowed to be executed:

    price returns into the OB/FVG
      -> M1 micro liquidity sweep (opposite side)
      -> M1 CHOCH / BOS in the setup direction
      -> M1 displacement
    -> entry at that confirmation bar's close.

Each of the three legs is individually switchable so the cost of the rule can
be measured instead of assumed.
"""
from __future__ import annotations

from typing import Optional

from ..engine.liquidity import BUY_SIDE, SELL_SIDE
from ..engine.structure import BEARISH, BULLISH, CHOCH
from ..engine import displacement as dsp


class Confirmation:
    __slots__ = ("ok", "reason", "micro_sweep", "micro_event", "micro_disp",
                 "invalidation_price", "chase_atr", "price")

    def __init__(self, ok: bool, reason: str = ""):
        self.ok = ok
        self.reason = reason
        self.micro_sweep = None
        self.micro_event = None
        self.micro_disp = None
        self.invalidation_price = 0.0
        self.chase_atr = 0.0
        self.price = 0.0

    def as_dict(self) -> dict:
        return {
            "m1_micro_sweep": self.micro_sweep.level.ltype if self.micro_sweep else "",
            "m1_micro_sweep_quality": self.micro_sweep.quality if self.micro_sweep else "",
            "m1_micro_break": self.micro_event.kind if self.micro_event else "",
            "m1_micro_break_scope": self.micro_event.scope if self.micro_event else "",
            "m1_micro_displacement": round(self.micro_disp.ratio, 4) if self.micro_disp else 0.0,
            "chase_atr": round(self.chase_atr, 4),
        }


def check_retest(ps, m1, i: int, cfg) -> bool:
    """Has price traded back into the zone on the just-closed M1 bar?"""
    z = ps.zone
    tol = cfg.ZONE_RETEST_TOLERANCE_ATR * (z.atr if z.atr > 0 else 0.0)
    lo, hi = m1.s.l[i], m1.s.h[i]
    touched = (lo <= z.upper + tol) and (hi >= z.lower - tol)
    if not touched:
        return False
    if not ps.retested:
        ps.retested = True
        ps.retest_time = m1.s.ot[i]
        z.touches += 1
        if z.touches > 1:
            z.fresh = False
        # correction depth = retracement of the impulse leg, 0..1
        d = ps.displacement
        span = abs(d.end_price - d.start_price)
        if span > 0:
            if ps.direction == BULLISH:
                ps.correction_depth = max(0.0, (d.end_price - lo) / span)
            else:
                ps.correction_depth = max(0.0, (hi - d.end_price) / span)
    return True


def check_m1(ps, m1, i: int, cfg) -> Confirmation:
    """Evaluate the M1 confirmation stack for a retested setup."""
    direction = ps.direction
    atr1 = m1.atr_now()
    look = cfg.M1_CONFIRMATION_LOOKBACK
    res = Confirmation(False)
    res.price = m1.s.c[i]

    # --- chase protection (spec 24) -------------------------------------- #
    z = ps.zone
    ref_atr = z.atr if z.atr > 0 else atr1
    if ref_atr > 0:
        if direction == BULLISH:
            dist = max(0.0, m1.s.c[i] - z.upper)
        else:
            dist = max(0.0, z.lower - m1.s.c[i])
        res.chase_atr = dist / ref_atr
        if res.chase_atr > cfg.MAX_CHASE_ATR:
            res.reason = "CHASE"
            return res

    # --- micro liquidity sweep ------------------------------------------- #
    want_side = SELL_SIDE if direction == BULLISH else BUY_SIDE
    sweep = None
    for sw in reversed(m1.liquidity.sweeps):
        if i - sw.index > look:
            break
        if sw.side == want_side:
            sweep = sw
            break
    if cfg.M1_REQUIRE_SWEEP and sweep is None:
        res.reason = "NO_M1_SWEEP"
        return res
    res.micro_sweep = sweep

    # --- micro structure shift ------------------------------------------- #
    event = None
    floor_index = sweep.index if sweep is not None else i - look
    for ev in reversed(m1.structure.events):
        if i - ev.index > look:
            break
        if ev.direction != direction or ev.index < floor_index:
            continue
        event = ev
        break
    if cfg.M1_REQUIRE_STRUCTURE and event is None:
        res.reason = "NO_M1_STRUCTURE"
        return res
    res.micro_event = event

    # --- micro displacement ---------------------------------------------- #
    best = None
    for j in range(max(0, i - 2), i + 1):
        d = dsp.measure(m1.s, j, atr1, cfg, direction=direction)
        if d is not None and (best is None or d.ratio > best.ratio):
            best = d
    if cfg.M1_REQUIRE_DISPLACEMENT:
        if best is None or best.ratio < cfg.M1_MIN_DISPLACEMENT_ATR:
            res.reason = "NO_M1_DISPLACEMENT"
            return res
    res.micro_disp = best

    # --- structural invalidation level for the stop ---------------------- #
    res.invalidation_price = _invalidation(ps, m1, i, sweep, cfg)
    if res.invalidation_price <= 0:
        res.reason = "SL_IMPOSSIBLE"
        return res

    res.ok = True
    return res


def _invalidation(ps, m1, i, sweep, cfg) -> float:
    """Structure-based invalidation price on M1 (spec 29).

    BUY  -> the lowest of: confirmed M1 swing low, micro sweep low, zone floor
    SELL -> the highest of the mirrored levels
    """
    look = cfg.M1_CONFIRMATION_LOOKBACK
    if ps.direction == BULLISH:
        cands = []
        for sw in reversed(m1.swings.lows):
            if i - sw.index > look * 2:
                break
            cands.append(sw.price)
        if sweep is not None:
            cands.append(sweep.extreme)
        cands.append(ps.zone.lower)
        cands = [c for c in cands if c < m1.s.c[i]]
        return max(cands) if cands else 0.0
    cands = []
    for sw in reversed(m1.swings.highs):
        if i - sw.index > look * 2:
            break
        cands.append(sw.price)
    if sweep is not None:
        cands.append(sweep.extreme)
    cands.append(ps.zone.upper)
    cands = [c for c in cands if c > m1.s.c[i]]
    return min(cands) if cands else 0.0
