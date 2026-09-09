"""M15 context -> M5 setup formation (spec 16-19, 26, 48-50).

Nothing here predicts anything.  Each rule is a yes/no test against structure
that has ALREADY printed.

Flow, all on closed M5 bars:

    M5 liquidity sweep            -> seed a Candidate
      + M5 CHOCH / BOS / MSS      in the direction opposite the swept side
      + M5 displacement           >= MIN_DISPLACEMENT_ATR
      + OB / FVG zone             produced by that impulse
    -> PendingSetup, which then waits for a retest and M1 confirmation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..engine import displacement as dsp
from ..engine.liquidity import BUY_SIDE, SELL_SIDE
from ..engine.structure import BEARISH, BULLISH, CHOCH, RANGE, TRANSITION
from ..engine.zones import build_zone
from . import journal as J
from .journal import Candidate

SETUP_A = "A_REVERSAL"
SETUP_B = "B_RANGE_REVERSAL"
SETUP_C = "C_TREND_CONTINUATION"
SETUP_D = "D_DOUBLE_LIQUIDITY"

# state machine labels (spec 40)
S_WAIT = "WAIT"
S_M15_CONTEXT = "M15_CONTEXT"
S_LIQUIDITY_FOUND = "LIQUIDITY_FOUND"
S_M5_SWEEP = "M5_SWEEP"
S_M5_MSS_CHOCH = "M5_MSS_CHOCH"
S_M5_BOS = "M5_BOS"
S_DISPLACEMENT = "DISPLACEMENT"
S_OB_FVG_FOUND = "OB_FVG_FOUND"
S_CORRECTION = "CORRECTION"
S_ZONE_RETEST = "ZONE_RETEST"
S_M1_CONFIRMATION = "M1_CONFIRMATION"
S_RISK_CHECK = "RISK_CHECK"
S_COST_CHECK = "COST_CHECK"
S_ENTRY = "ENTRY"


class PendingSetup:
    __slots__ = ("symbol", "direction", "setup_type", "zone", "sweep", "event",
                 "displacement", "created_time", "created_index", "expiry_index",
                 "invalidation_price", "state", "retested", "retest_time",
                 "candidate", "context", "signal_id", "dead", "m15_sweep",
                 "correction_depth", "m5_atr")

    def __init__(self, symbol, direction, setup_type, zone, sweep, event,
                 displacement, created_time, created_index, expiry_index,
                 invalidation_price, candidate, context, m5_atr):
        self.symbol = symbol
        self.direction = direction            # BULLISH | BEARISH
        self.setup_type = setup_type
        self.zone = zone
        self.sweep = sweep
        self.event = event
        self.displacement = displacement
        self.created_time = created_time
        self.created_index = created_index
        self.expiry_index = expiry_index
        self.invalidation_price = invalidation_price
        self.state = S_OB_FVG_FOUND
        self.retested = False
        self.retest_time = 0
        self.candidate = candidate
        self.context = context
        self.m15_sweep = None
        self.correction_depth = 0.0
        self.m5_atr = m5_atr
        self.dead = False
        self.signal_id = candidate.signal_id

    @property
    def trade_direction(self) -> str:
        return "BUY" if self.direction == BULLISH else "SELL"


def _classify(direction: str, regime: str, m15_sweep, event) -> str:
    if m15_sweep is not None and event is not None and event.is_mss:
        return SETUP_D
    if regime == RANGE:
        return SETUP_B
    if (direction == BULLISH and regime == BULLISH) or \
       (direction == BEARISH and regime == BEARISH):
        return SETUP_C
    return SETUP_A


class SetupFactory:
    """Per-symbol M5 setup formation."""

    def __init__(self, symbol: str, cfg, m15, m5, journal: J.Journal):
        self.symbol = symbol
        self.cfg = cfg
        self.m15 = m15
        self.m5 = m5
        self.journal = journal
        self.tracking: List[Dict] = []     # sweeps waiting for a structural break
        self.pending: List[PendingSetup] = []

    # ------------------------------------------------------------------ #
    def on_m5_closed(self, i: int) -> List[PendingSetup]:
        cfg = self.cfg
        m5, m15 = self.m5, self.m15
        atr5 = m5.atr_now()
        created: List[PendingSetup] = []

        # 1. seed a candidate for every fresh M5 sweep -------------------- #
        for sw in m5.last_sweeps:
            direction = BULLISH if sw.side == SELL_SIDE else BEARISH
            cand = Candidate(self.symbol, m5.s.ot[i], direction)
            cand.signal_id = "{}|{}|{}@{:.10g}".format(
                self.symbol, direction, sw.level.ltype, sw.level.price)
            cand.transition(S_LIQUIDITY_FOUND, m5.s.ot[i],
                            f"{sw.level.ltype} strength={sw.level.strength}")
            cand.transition(S_M5_SWEEP, m5.s.ot[i],
                            f"{sw.quality} pen={sw.penetration_atr:.2f}ATR")
            cand.data.update({
                "m5_sweep_type": sw.level.ltype,
                "m5_sweep_quality": sw.quality,
                "m5_sweep_score": sw.score,
                "m5_sweep_penetration_atr": round(sw.penetration_atr, 4),
                "m5_sweep_wick_ratio": round(sw.wick_ratio, 4),
                "m5_sweep_volume_ratio": round(sw.volume_ratio, 4),
                "liquidity_price": sw.level.price,
                "liquidity_strength": sw.level.strength,
            })
            if not m15.ready:
                cand.reject(J.NO_M15_CONTEXT)
                self.journal.add(cand)
                continue
            if sw.quality == "WEAK" and not cfg.ALLOW_WEAK_SWEEP:
                cand.reject(J.NO_M5_SWEEP)
                self.journal.add(cand)
                continue
            self.tracking.append({"sweep": sw, "cand": cand, "dir": direction,
                                  "seed": i})

        # 2. advance tracked sweeps towards a structural break ------------ #
        still: List[Dict] = []
        for tr in self.tracking:
            sw, cand, direction = tr["sweep"], tr["cand"], tr["dir"]
            age = i - sw.index
            if age > cfg.SETUP_SWEEP_LOOKBACK_M5:
                cand.reject(J.NO_MSS)
                self.journal.add(cand)
                continue
            # invalidated: closed beyond the sweep extreme against the setup
            if direction == BULLISH and m5.s.c[i] < sw.extreme:
                cand.reject(J.SETUP_INVALIDATED)
                self.journal.add(cand)
                continue
            if direction == BEARISH and m5.s.c[i] > sw.extreme:
                cand.reject(J.SETUP_INVALIDATED)
                self.journal.add(cand)
                continue

            event = self._find_break(i, sw, direction)
            if event is None:
                still.append(tr)
                continue

            cand.transition(S_M5_MSS_CHOCH if event.is_mss or event.kind == CHOCH
                            else S_M5_BOS, m5.s.ot[i],
                            f"{event.kind} {event.scope} mss={event.is_mss}")
            cand.data.update({
                "m5_break_kind": event.kind,
                "m5_break_scope": event.scope,
                "m5_mss": event.is_mss,
            })

            d = self._leg_displacement(sw.index, event.index, direction, atr5)
            if d is None or d.ratio < cfg.MIN_DISPLACEMENT_ATR:
                cand.data["m5_displacement"] = round(d.ratio, 4) if d else 0.0
                cand.reject(J.NO_DISPLACEMENT)
                self.journal.add(cand)
                continue
            cand.transition(S_DISPLACEMENT, m5.s.ot[i],
                            f"{d.ratio:.2f}ATR {d.grade}")
            cand.data["m5_displacement"] = round(d.ratio, 4)
            cand.data["m5_displacement_grade"] = d.grade

            zone = build_zone(m5.s, event.index, direction, atr5, cfg,
                              leg_start=d.start_index)
            if zone is None:
                cand.reject(J.NO_ZONE)
                cand.reject(J.NO_OB)
                cand.reject(J.NO_FVG)
                self.journal.add(cand)
                continue
            zone.displacement = d.ratio
            zone.linked_event = event
            cand.transition(S_OB_FVG_FOUND, m5.s.ot[i],
                            f"{zone.kind} h={zone.height_atr():.2f}ATR")
            cand.data.update(zone.as_dict())
            cand.signal_id += f"|z{zone.zone_id}"

            ctx = self._m15_context(direction)
            cand.data.update(ctx)
            if ctx.get("pd_conflict") and cfg.PD_HARD_FILTER:
                cand.reject(J.PREMIUM_DISCOUNT)
                self.journal.add(cand)
                continue

            m15_sweep = ctx.pop("_m15_sweep", None)
            setup_type = _classify(direction, ctx["m15_regime"], m15_sweep, event)
            cand.setup_type = setup_type
            cand.data["setup_type"] = setup_type

            invalidation = sw.extreme if direction == BULLISH else sw.extreme
            ps = PendingSetup(self.symbol, direction, setup_type, zone, sw, event,
                              d, m5.s.ot[i], i, i + cfg.SETUP_EXPIRY_M5_BARS,
                              invalidation, cand, ctx, atr5)
            ps.m15_sweep = m15_sweep
            ps.state = S_CORRECTION
            cand.transition(S_CORRECTION, m5.s.ot[i], "waiting for retest")
            sw.consumed = True
            self.pending.append(ps)
            created.append(ps)

        self.tracking = still

        # 3. expire / invalidate pending setups --------------------------- #
        alive: List[PendingSetup] = []
        for ps in self.pending:
            if ps.dead:
                continue
            if i > ps.expiry_index:
                ps.candidate.reject(J.NO_RETEST if not ps.retested
                                    else J.NO_M1_CONFIRMATION)
                ps.candidate.reject(J.SETUP_EXPIRED)
                self.journal.add(ps.candidate)
                continue
            if ps.direction == BULLISH and m5.s.c[i] < ps.invalidation_price:
                ps.candidate.reject(J.SETUP_INVALIDATED)
                self.journal.add(ps.candidate)
                continue
            if ps.direction == BEARISH and m5.s.c[i] > ps.invalidation_price:
                ps.candidate.reject(J.SETUP_INVALIDATED)
                self.journal.add(ps.candidate)
                continue
            alive.append(ps)
        self.pending = alive
        return created

    # ------------------------------------------------------------------ #
    def _find_break(self, i, sweep, direction):
        """Most recent M5 break in `direction` that happened after the sweep."""
        best = None
        for ev in reversed(self.m5.structure.events):
            if ev.index < sweep.index:
                break
            if ev.direction != direction or ev.index > i:
                continue
            if i - ev.index > self.cfg.SETUP_SWEEP_LOOKBACK_M5:
                continue
            # prefer MSS, then CHOCH, then BOS
            rank = 3 if ev.is_mss else (2 if ev.kind == CHOCH else 1)
            if best is None or rank > best[0]:
                best = (rank, ev)
        return best[1] if best else None

    def _leg_displacement(self, from_index, to_index, direction, atr):
        best = None
        lo = max(0, from_index)
        for j in range(lo, to_index + 1):
            d = dsp.measure(self.m5.s, j, atr, self.cfg, direction=direction)
            if d is None:
                continue
            if best is None or d.ratio > best.ratio:
                best = d
        return best

    def _m15_context(self, direction) -> Dict:
        m15 = self.m15
        regime = m15.regime()
        price = m15.price()
        pd = m15.structure.pd_position(price)
        ctx: Dict = {
            "m15_regime": regime,
            "m15_external_direction": m15.structure.external.direction,
            "m15_internal_direction": m15.structure.internal.direction,
            "m15_external_sequence": ",".join(m15.structure.external.sequence[-4:]),
            "m15_internal_sequence": ",".join(m15.structure.internal.sequence[-4:]),
            "pd_position": round(pd, 4) if pd is not None else None,
        }
        conflict = False
        if pd is not None and self.cfg.PD_ENABLED:
            tol = self.cfg.PD_TOLERANCE
            if direction == BULLISH and pd > self.cfg.PD_EQ + tol:
                conflict = True
            if direction == BEARISH and pd < self.cfg.PD_EQ - tol:
                conflict = True
        ctx["pd_conflict"] = conflict

        want = SELL_SIDE if direction == BULLISH else BUY_SIDE
        m15_sw = None
        for sw in reversed(m15.liquidity.sweeps):
            if m15.i - sw.index > self.cfg.DOUBLE_LIQUIDITY_M15_WINDOW:
                break
            if sw.side == want:
                m15_sw = sw
                break
        ctx["m15_sweep"] = m15_sw.level.ltype if m15_sw else ""
        ctx["m15_sweep_quality"] = m15_sw.quality if m15_sw else ""
        ctx["_m15_sweep"] = m15_sw
        return ctx
