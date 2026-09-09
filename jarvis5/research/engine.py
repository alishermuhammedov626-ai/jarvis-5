"""Research pipeline: events -> replay -> patterns -> edge statistics.

Two passes per symbol, and the separation between them is the whole
no-look-ahead argument:

  PASS 1 (outcomes)  may look forward.  It answers "did a 1 % move follow?"
  PASS 2 (patterns)  may not.  It replays the market bar by bar and records
                     only what was visible at that instant.

The two are joined by the bar index, never by shared state, so a pattern can
never be computed from the outcome it is supposed to precede.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

from ..core.series import aggregate
from . import combos, events, simulate
from .features import SymbolReplay
from .stats import Aggregator, Label

LONG = "LONG"
SHORT = "SHORT"


def window_extreme_forward(arr, n: int, horizon: int, want_max: bool) -> List[float]:
    """w[i] = max/min of arr[i+1 .. i+horizon]  (O(n) monotonic deque)."""
    out = [0.0] * n
    if n == 0:
        return out
    rev = [arr[n - 1 - k] for k in range(n)]
    dq: deque = deque()
    for k in range(n):
        v = rev[k]
        if want_max:
            while dq and rev[dq[-1]] <= v:
                dq.pop()
        else:
            while dq and rev[dq[-1]] >= v:
                dq.pop()
        dq.append(k)
        while dq[0] < k - horizon + 1:
            dq.popleft()
        i = n - 2 - k
        if 0 <= i < n:
            out[i] = rev[dq[0]]
    out[n - 1] = arr[n - 1]
    return out


class SymbolResult:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.long_events: List = []
        self.short_events: List = []
        self.bars = 0
        self.sampled = 0
        self.event_patterns: Counter = Counter()
        self.event_rows: List[dict] = []
        self.data_quality: Dict[str, object] = {}
        self.elapsed = 0.0


class ResearchEngine:
    def __init__(self, cfg, funding=None, progress=True):
        cfg.validate()
        self.cfg = cfg
        self.agg = Aggregator(cfg)
        self.funding = funding
        self.progress = progress
        self.results: Dict[str, SymbolResult] = {}
        self.target_keys = [f"{t*100:g}pct" for t in cfg.TARGETS]

    # ------------------------------------------------------------------ #
    def segments(self, n: int) -> List[Tuple[str, int, int]]:
        c = self.cfg
        a = int(n * c.SPLIT_DISCOVERY)
        b = a + int(n * c.SPLIT_VALIDATION)
        return [("DISCOVERY", 0, a), ("VALIDATION", a, b), ("OOS", b, n)]

    def _segment_of(self, i: int, segs) -> str:
        for name, lo, hi in segs:
            if lo <= i < hi:
                return name
        return "OOS"

    # ------------------------------------------------------------------ #
    def run_symbol(self, symbol: str, m1) -> SymbolResult:
        cfg = self.cfg
        t0 = time.time()
        res = SymbolResult(symbol)
        n = m1.n
        res.bars = n
        gaps = m1.check_continuity()
        bad = m1.check_sanity()
        res.data_quality = {
            "symbol": symbol,
            "m1_bars": n,
            "first_time": m1.ot[0] if n else 0,
            "last_time": m1.ot[-1] if n else 0,
            "days": round((m1.ot[-1] - m1.ot[0]) / 86_400_000, 2) if n > 1 else 0,
            "missing_candles": len(gaps),
            "malformed_candles": len(bad),
            "duplicate_timestamps": 0,
            "funding_available": bool(self.funding and self.funding.is_available(symbol)),
        }
        if n < cfg.WARMUP_BARS + cfg.EVENT_HORIZON_BARS:
            res.data_quality["note"] = "insufficient history"
            res.elapsed = time.time() - t0
            self.results[symbol] = res
            return res

        m5 = aggregate(m1, "M5")
        m15 = aggregate(m1, "M15")
        res.data_quality["m5_bars"] = m5.n
        res.data_quality["m15_bars"] = m15.n

        # ---- PASS 1: outcomes (allowed to look forward) ----------------- #
        longs, shorts, tt_long, tt_short = events.detect(m1, symbol, cfg)
        res.long_events = longs
        res.short_events = shorts

        H = cfg.EVENT_HORIZON_BARS
        wmax = window_extreme_forward(m1.h, n, H, True)
        wmin = window_extreme_forward(m1.l, n, H, False)

        segs = self.segments(n)
        for ev in longs + shorts:
            ev.segment = self._segment_of(ev.origin_index, segs)

        event_origins = {ev.origin_index: ev for ev in longs}
        event_origins_short = {ev.origin_index: ev for ev in shorts}

        # ---- PASS 2: pattern replay (no look-ahead) --------------------- #
        replay = SymbolReplay(symbol, m1, m5, m15, cfg)
        stride = cfg.BASELINE_STRIDE
        warm = cfg.WARMUP_BARS
        last_sample = n - H - 1          # outcomes must be fully observable

        for i in range(n):
            replay.advance(i)
            if i < warm or i > last_sample or not replay.ready:
                continue
            is_sample = (i % stride == 0)
            ev_long = event_origins.get(i)
            ev_short = event_origins_short.get(i)
            if not is_sample and ev_long is None and ev_short is None:
                continue

            feats = replay.snapshot(i)
            feats.update(combos.build(feats))

            if ev_long is not None:
                self._tag_event(res, ev_long, feats)
            if ev_short is not None:
                self._tag_event(res, ev_short, feats)

            if not is_sample:
                continue          # event bars must NOT bias the population

            labels = self._labels(m1, i, tt_long, tt_short, wmax, wmin)
            sims = {}
            if cfg.SIMULATE:
                for d in (LONG, SHORT):
                    sim = simulate.simulate(replay, i, d, cfg, self.funding, symbol)
                    if sim.ok:
                        sims[d] = sim
                    else:
                        self.agg.rejections[sim.reason] += 1
            scopes = ["ALL", f"SYM:{symbol}", f"SEG:{self._segment_of(i, segs)}"]
            self.agg.observe_bar(scopes, feats, labels, sims)
            res.sampled += 1

        res.elapsed = time.time() - t0
        self.results[symbol] = res
        return res

    # ------------------------------------------------------------------ #
    def _labels(self, m1, i, tt_long, tt_short, wmax, wmin) -> Dict[str, Label]:
        c = m1.c[i]
        out = {}
        hits_l = {}
        hits_s = {}
        for k, key in enumerate(self.target_keys):
            hits_l[key] = tt_long[k][i] >= 0
            hits_s[key] = tt_short[k][i] >= 0
        out[LONG] = Label(LONG, hits_l, (wmax[i] - c) / c, (c - wmin[i]) / c,
                          tt_long[0][i])
        out[SHORT] = Label(SHORT, hits_s, (c - wmin[i]) / c, (wmax[i] - c) / c,
                           tt_short[0][i])
        return out

    def _tag_event(self, res: SymbolResult, ev, feats: Dict[str, int]) -> None:
        cfg = self.cfg
        primary = cfg.PRIMARY_WINDOW
        active = sorted(k for k, age in feats.items() if age <= primary)
        for name in active:
            res.event_patterns[f"{ev.direction}|{name}"] += 1
        ev.features = {k: v for k, v in feats.items() if v <= primary}
        if len(res.event_rows) < cfg.MAX_EVENT_ROWS:
            row = ev.as_row(cfg.TARGETS)
            row["patterns"] = ";".join(active[:60])
            row["pattern_count"] = len(active)
            top = sorted(((v, k) for k, v in ev.features.items()))[:8]
            row["nearest_patterns"] = ";".join(f"{k}@{v}m" for v, k in top)
            res.event_rows.append(row)

    # ------------------------------------------------------------------ #
    def run(self, data: Dict[str, object]) -> "ResearchEngine":
        total = len(data)
        for idx, (symbol, m1) in enumerate(data.items(), 1):
            if self.progress:
                print(f"[{idx}/{total}] {symbol}", flush=True)
            r = self.run_symbol(symbol, m1)
            if self.progress:
                self._print_symbol(r)
        return self

    def _print_symbol(self, r: SymbolResult) -> None:
        tk = self.target_keys
        l2 = sum(1 for e in r.long_events if e.targets_hit.get(tk[2]))
        s2 = sum(1 for e in r.short_events if e.targets_hit.get(tk[2]))
        print(f"      bars={r.bars}  sampled={r.sampled}  "
              f"+1% events={len(r.long_events)}  -1% events={len(r.short_events)}  "
              f"+2%={l2}  -2%={s2}  ({r.elapsed:.1f}s)")
        top = r.event_patterns.most_common(5)
        if top:
            shown = ", ".join(f"{k.split('|',1)[1]}({v})" for k, v in top)
            print(f"      most frequent pre-event patterns: {shown}")
