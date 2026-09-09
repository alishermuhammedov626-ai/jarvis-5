"""Pattern extraction at a point in time (spec 6, 11).

The extractor REUSES the trading engine's structure / liquidity / displacement
/ zone detectors read-only.  Nothing in jarvis5/engine or jarvis5/strategy is
modified: a pattern called `M5_MSS_UP` here is the same object the trading
engine would have seen, which is the whole point of measuring it.

Ages are reported in MINUTES, so an M5 or M15 pattern and an M1 pattern land on
the same 5 / 10 / 20 / 30 / 60 / 120 window scale.

No look-ahead: the replay walks forward one closed bar at a time and a snapshot
at bar i can only read state produced by bars <= i.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..engine import displacement as dsp
from ..engine.liquidity import BUY_SIDE, SELL_SIDE
from ..engine.structure import BEARISH, BULLISH, CHOCH, RANGE, TRANSITION
from ..engine.timeframe import TimeframeState
from ..engine.zones import find_fvg, find_order_block
from . import wyckoff
from .indicators import RSI, VolumeTracker, price_action_signals, rsi_signals, \
    volume_signals

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15}

# features describing a persistent STATE rather than a dated occurrence
STATE_SUFFIXES = (
    "_STRUCT_BULLISH", "_STRUCT_BEARISH", "_STRUCT_RANGE", "_STRUCT_TRANSITION",
    "_PREMIUM", "_DISCOUNT", "_EQUILIBRIUM",
    "_UPTREND", "_DOWNTREND", "_RANGING",
    "_HH", "_HL", "_LH", "_LL",
    "_EXPANSION", "_COMPRESSION",
    "_INTERNAL_BULLISH", "_INTERNAL_BEARISH",
)


def is_state(name: str) -> bool:
    return name.endswith(STATE_SUFFIXES)


class TFReplay:
    """One timeframe's engine plus the research-only indicators."""

    def __init__(self, series, cfg, tcfg):
        self.s = series
        self.cfg = cfg
        self.tf = series.tf
        self.minutes = TF_MINUTES[series.tf]
        self.state = TimeframeState(series, tcfg)
        self.rsi = RSI(cfg.RSI_PERIOD)
        self.vol = VolumeTracker(cfg.VOLUME_AVG_PERIOD)
        self.i = -1
        self.atr_hist: List[float] = []

    def advance(self, i: int) -> None:
        self.state.on_bar_closed(i)
        self.rsi.update(self.s.c[i])
        self.vol.update(self.s.v[i])
        self.i = i
        a = self.state.atr_now()
        self.atr_hist.append(a)
        if len(self.atr_hist) > 200:
            del self.atr_hist[:100]

    @property
    def ready(self) -> bool:
        return self.state.ready and self.rsi.ready

    def max_lookback_bars(self, minutes: int) -> int:
        return max(1, minutes // self.minutes)


def _add(out: Dict[str, int], name: str, age_minutes: int) -> None:
    """Keep the most RECENT occurrence of a pattern."""
    prev = out.get(name)
    if prev is None or age_minutes < prev:
        out[name] = age_minutes


def extract_timeframe(r: TFReplay, i: int, max_minutes: int,
                      cfg) -> Dict[str, int]:
    """All SMC/structure features visible on this timeframe at bar i."""
    out: Dict[str, int] = {}
    if not r.ready:
        return out
    tf = r.tf
    m = r.minutes
    st = r.state
    s = r.s
    atr = st.atr_now()
    if atr <= 0:
        return out
    look = r.max_lookback_bars(max_minutes)
    price = s.c[i]

    # ---- liquidity sweeps ------------------------------------------------ #
    for sw in reversed(st.liquidity.sweeps):
        back = i - sw.index
        if back > look:
            break
        if back < 0:
            continue
        age = back * m
        side = "SELLSIDE" if sw.side == SELL_SIDE else "BUYSIDE"
        _add(out, f"{tf}_SWEEP_{side}", age)
        _add(out, f"{tf}_SWEEP_{side}_{sw.quality}", age)
        _add(out, f"{tf}_SWEEP_TYPE_{sw.level.ltype}", age)
        _add(out, f"{tf}_LIQUIDITY_REJECTION", age)
        if sw.penetration_atr > 0:
            _add(out, f"{tf}_LIQUIDITY_PENETRATION", age)
        _add(out, f"{tf}_CLOSE_BACK_INSIDE_LEVEL", age)

    # ---- structure events ------------------------------------------------ #
    for ev in reversed(st.structure.events):
        back = i - ev.index
        if back > look:
            break
        if back < 0:
            continue
        age = back * m
        d = "UP" if ev.direction == BULLISH else "DOWN"
        scope = ev.scope                       # INTERNAL | EXTERNAL
        _add(out, f"{tf}_{ev.kind}_{d}", age)
        _add(out, f"{tf}_{scope}_{ev.kind}_{d}", age)
        if ev.is_mss:
            _add(out, f"{tf}_MSS_{d}", age)
            _add(out, f"{tf}_{scope}_MSS_{d}", age)

    # ---- displacement ---------------------------------------------------- #
    for j in range(i, max(-1, i - look), -1):
        d = st.displacements.get(j)
        if d is None:
            continue
        age = (i - j) * m
        dd = "UP" if d.direction == BULLISH else "DOWN"
        _add(out, f"{tf}_DISPLACEMENT_{dd}", age)
        _add(out, f"{tf}_DISPLACEMENT_{dd}_{d.grade}", age)

    # ---- FVG ------------------------------------------------------------- #
    for j in range(i, max(1, i - look), -1):
        for direction, tag in ((BULLISH, "BULL"), (BEARISH, "BEAR")):
            got = find_fvg(s, j, direction, atr, r.state.cfg)
            if not got:
                continue
            lower, upper, _size = got
            age = (i - j) * m
            _add(out, f"{tf}_FVG_{tag}", age)
            if lower <= price <= upper:
                _add(out, f"{tf}_IN_FVG_{tag}", age)
            elif tag == "BULL" and price > upper:
                _add(out, f"{tf}_FVG_{tag}_UNFILLED", age)
            elif tag == "BEAR" and price < lower:
                _add(out, f"{tf}_FVG_{tag}_UNFILLED", age)

    # ---- order blocks, breakers, mitigation ------------------------------ #
    for ev in reversed(st.structure.events):
        back = i - ev.index
        if back > look:
            break
        if back < 0:
            continue
        direction = ev.direction
        ob = find_order_block(s, ev.index, direction, r.state.cfg)
        if ob is None:
            continue
        tag = "BULL" if direction == BULLISH else "BEAR"
        lower, upper = s.l[ob], s.h[ob]
        age = (i - ob) * m
        _add(out, f"{tf}_OB_{tag}", age)
        if lower <= price <= upper:
            _add(out, f"{tf}_IN_OB_{tag}", age)
        # touched at least once since creation -> mitigation zone
        touched = any(s.l[k] <= upper and s.h[k] >= lower
                      for k in range(ob + 1, i + 1))
        if touched:
            _add(out, f"{tf}_MITIGATION_ZONE_{tag}", age)
        # violated by a close beyond its far edge -> breaker of the other side
        if tag == "BULL" and any(s.c[k] < lower for k in range(ob + 1, i + 1)):
            _add(out, f"{tf}_BREAKER_BEAR", age)
        if tag == "BEAR" and any(s.c[k] > upper for k in range(ob + 1, i + 1)):
            _add(out, f"{tf}_BREAKER_BULL", age)

    # ---- premium / discount / equilibrium -------------------------------- #
    pd = st.structure.pd_position(price)
    if pd is not None:
        if pd > 0.55:
            out[f"{tf}_PREMIUM"] = 0
        elif pd < 0.45:
            out[f"{tf}_DISCOUNT"] = 0
        else:
            out[f"{tf}_EQUILIBRIUM"] = 0

    # ---- structure state -------------------------------------------------- #
    regime = st.structure.update_regime(i)
    out[f"{tf}_STRUCT_{regime}"] = 0
    ext = st.structure.external
    inn = st.structure.internal
    if ext.direction == BULLISH:
        out[f"{tf}_UPTREND"] = 0
    elif ext.direction == BEARISH:
        out[f"{tf}_DOWNTREND"] = 0
    else:
        out[f"{tf}_RANGING"] = 0
    if inn.direction == BULLISH:
        out[f"{tf}_INTERNAL_BULLISH"] = 0
    elif inn.direction == BEARISH:
        out[f"{tf}_INTERNAL_BEARISH"] = 0
    if ext.sequence:
        out[f"{tf}_EXTERNAL_{ext.sequence[-1]}"] = 0
    if inn.sequence:
        out[f"{tf}_INTERNAL_{inn.sequence[-1]}"] = 0
    if len(ext.sequence) >= 2:
        tail = "".join(ext.sequence[-2:])
        if tail in ("HHHL", "HLHH"):
            out[f"{tf}_HH_HL_SEQUENCE"] = 0
        if tail in ("LLLH", "LHLL"):
            out[f"{tf}_LH_LL_SEQUENCE"] = 0

    # ---- liquidity in front of price -------------------------------------- #
    near = cfg.NEAR_LIQUIDITY_ATR * atr
    for book, side in ((st.liquidity.highs, "BUYSIDE"),
                       (st.liquidity.lows, "SELLSIDE")):
        for lv in book[-40:]:
            if lv.swept:
                continue
            if abs(lv.price - price) <= near:
                out[f"{tf}_NEAR_{lv.ltype}"] = 0
                out[f"{tf}_NEAR_{side}_LIQUIDITY"] = 0

    # ---- pullback / expansion / compression --------------------------------- #
    _trend_context(out, r, i, atr, cfg)
    return out


def _trend_context(out: Dict[str, int], r: TFReplay, i: int, atr: float,
                   cfg) -> None:
    tf = r.tf
    st = r.state
    s = r.s
    ext = st.structure.external
    price = s.c[i]

    hi = ext.last_high
    lo = ext.last_low
    if hi is not None and lo is not None and hi.price > lo.price:
        span = hi.price - lo.price
        if ext.direction == BULLISH:
            depth = (hi.price - price) / span
            if 0.15 <= depth <= 0.50:
                out[f"{tf}_PULLBACK"] = 0
            elif depth > 0.50:
                out[f"{tf}_DEEP_PULLBACK"] = 0
            elif depth < 0.0:
                out[f"{tf}_TREND_CONTINUATION_UP"] = 0
        elif ext.direction == BEARISH:
            depth = (price - lo.price) / span
            if 0.15 <= depth <= 0.50:
                out[f"{tf}_PULLBACK"] = 0
            elif depth > 0.50:
                out[f"{tf}_DEEP_PULLBACK"] = 0
            elif depth < 0.0:
                out[f"{tf}_TREND_CONTINUATION_DOWN"] = 0

    # trend reversal: the most recent event is a CHOCH against the old direction
    for ev in reversed(st.structure.events[-6:]):
        back = i - ev.index
        if back > r.max_lookback_bars(120):
            break
        if ev.kind == CHOCH:
            out[f"{tf}_TREND_REVERSAL"] = back * r.minutes
            break

    # breakout / breakout failure over the recent range
    look = r.max_lookback_bars(120)
    lo_i = max(0, i - look)
    if i > lo_i:
        rh = max(s.h[j] for j in range(lo_i, i))
        rl = min(s.l[j] for j in range(lo_i, i))
        if price > rh:
            out[f"{tf}_BREAKOUT_UP"] = 0
        if price < rl:
            out[f"{tf}_BREAKOUT_DOWN"] = 0
        for back in range(1, min(10, i - lo_i)):
            j = i - back
            rh2 = max(s.h[k] for k in range(max(0, j - look), j))
            if s.c[j] > rh2 and price < rh2:
                out[f"{tf}_BREAKOUT_FAILURE_UP"] = back * r.minutes
                break

    # volatility regime
    hist = r.atr_hist
    if len(hist) >= 60:
        recent = hist[-1]
        base = sum(hist[-60:-10]) / 50.0
        if base > 0:
            if recent >= 1.3 * base:
                out[f"{tf}_EXPANSION"] = 0
            elif recent <= 0.75 * base:
                out[f"{tf}_COMPRESSION"] = 0


class SymbolReplay:
    """Walks M1 forward, feeding M5/M15 only when their bars have closed."""

    def __init__(self, symbol: str, m1, m5, m15, cfg):
        self.symbol = symbol
        self.cfg = cfg
        tcfg = cfg.to_trading_config()
        self.tcfg = tcfg
        self.m1 = TFReplay(m1, cfg, tcfg)
        self.m5 = TFReplay(m5, cfg, tcfg)
        self.m15 = TFReplay(m15, cfg, tcfg)
        self.p5 = 0
        self.p15 = 0
        self.i = -1

    def advance(self, i: int) -> None:
        """Advance to the close of M1 bar i."""
        now_close = self.m1.s.ot[i] + 60_000
        while self.p15 < self.m15.s.n and self.m15.s.close_time(self.p15) <= now_close:
            self.m15.advance(self.p15)
            self.p15 += 1
        while self.p5 < self.m5.s.n and self.m5.s.close_time(self.p5) <= now_close:
            self.m5.advance(self.p5)
            self.p5 += 1
        self.m1.advance(i)
        self.i = i

    @property
    def ready(self) -> bool:
        return self.m1.ready and self.m5.ready and self.m15.ready

    def snapshot(self, i: int) -> Dict[str, int]:
        """All patterns visible at the close of bar i, name -> age in minutes."""
        assert i == self.i, "snapshot must be taken at the current bar"
        cfg = self.cfg
        max_minutes = max(cfg.WINDOWS)
        out: Dict[str, int] = {}
        out.update(extract_timeframe(self.m1, i, max_minutes, cfg))
        if self.p5 > 0:
            out.update(extract_timeframe(self.m5, self.p5 - 1, max_minutes, cfg))
        if self.p15 > 0:
            out.update(extract_timeframe(self.m15, self.p15 - 1, max_minutes, cfg))

        atr1 = self.m1.state.atr_now()
        for name, age in rsi_signals(self.m1.rsi, i, self.m1.state.swings, cfg).items():
            _add(out, name, age)
        for name, age in volume_signals(self.m1.s, self.m1.vol, i, atr1, cfg).items():
            _add(out, name, age)
        for name, age in price_action_signals(self.m1.s, i, atr1, cfg).items():
            _add(out, f"{name}", age)
        for name, age in wyckoff.detect(self.m1.s, i, atr1, self.m1.vol, cfg).items():
            _add(out, name, age)
        if self.p5 > 0:
            j5 = self.p5 - 1
            atr5 = self.m5.state.atr_now()
            for name, age in wyckoff.detect(self.m5.s, j5, atr5, self.m5.vol,
                                            cfg).items():
                _add(out, f"M5_{name}", age * 5)
            for name, age in rsi_signals(self.m5.rsi, j5, self.m5.state.swings,
                                         cfg).items():
                _add(out, f"M5_{name}", age * 5)
        return out
