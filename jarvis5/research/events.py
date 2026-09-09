"""1 %+ MOVE EVENT detection and clustering (spec 3, 18).

Every M1 candle is treated as a potential origin: from close[i], does price
reach +1 % (long) or -1 % (short) within EVENT_HORIZON_BARS?

Naive scanning is O(n x horizon).  Instead each target uses one heap keyed by
the target PRICE: at bar j every pending origin whose target price is at or
below high[j] resolves simultaneously.  Each origin is therefore resolved at
the FIRST bar that reaches its target -- exactly the definition -- in
O(n log n) overall.

Hundreds of origins inside one impulse are then merged by `cluster()` into a
single MOVE EVENT, so one push is counted once rather than three hundred times.
"""
from __future__ import annotations

import heapq
from array import array
from typing import Dict, List, Optional, Tuple

LONG = "LONG"
SHORT = "SHORT"


def time_to_targets(series, targets: List[float], direction: str,
                    horizon: int) -> List[array]:
    """bars-to-target for every origin, per target.  -1 = never within horizon.

    Only bars strictly AFTER the origin are considered: an origin cannot be
    resolved by its own candle.
    """
    n = series.n
    out = [array("i", bytes(4 * n)) for _ in targets]
    for arr in out:
        for i in range(n):
            arr[i] = -1

    hi, lo, cl = series.h, series.l, series.c
    for k, tgt in enumerate(targets):
        heap: List[Tuple[float, int]] = []
        res = out[k]
        if direction == LONG:
            factor = 1.0 + tgt
            for j in range(n):
                price = hi[j]
                while heap and heap[0][0] <= price:
                    _t, origin = heapq.heappop(heap)
                    if j - origin <= horizon:
                        res[origin] = j - origin
                    # else: the first touch arrived too late -> stays -1
                heapq.heappush(heap, (cl[j] * factor, j))
        else:
            factor = 1.0 - tgt
            for j in range(n):
                price = -lo[j]           # negate so the same min-heap works
                while heap and heap[0][0] <= price:
                    _t, origin = heapq.heappop(heap)
                    if j - origin <= horizon:
                        res[origin] = j - origin
                heapq.heappush(heap, (-cl[j] * factor, j))
    return out


class MoveEvent:
    __slots__ = ("symbol", "direction", "origin_index", "event_start",
                 "event_end", "start_price", "extreme_price", "mfe_pct",
                 "mae_pct", "time_to", "origins", "duration_bars",
                 "start_time", "end_time", "segment", "features", "targets_hit",
                 "impulse_start", "impulse_time", "impulse_price")

    def __init__(self, symbol: str, direction: str, origin_index: int):
        self.symbol = symbol
        self.direction = direction
        self.origin_index = origin_index
        self.event_start = origin_index
        self.event_end = origin_index
        self.start_price = 0.0
        self.extreme_price = 0.0
        self.mfe_pct = 0.0
        self.mae_pct = 0.0
        self.time_to: Dict[str, int] = {}
        self.targets_hit: Dict[str, bool] = {}
        self.origins = 1
        self.duration_bars = 0
        self.start_time = 0
        self.end_time = 0
        self.segment = ""
        self.features: Dict[str, int] = {}
        # The bar the directional leg actually launched from (lowest low for a
        # long, highest high for a short) between the first qualifying origin
        # and the move's extreme.  Patterns are read HERE: it is the bar a
        # trader would have had to act on, and it is still strictly historical.
        self.impulse_start = origin_index
        self.impulse_time = 0
        self.impulse_price = 0.0

    def as_row(self, targets: List[float]) -> dict:
        row = {
            "symbol": self.symbol,
            "direction": self.direction,
            "event_start": self.start_time,
            "event_end": self.end_time,
            "origin_index": self.origin_index,
            "duration_bars": self.duration_bars,
            "merged_origins": self.origins,
            "start_price": self.start_price,
            "impulse_start": self.impulse_time,
            "impulse_start_index": self.impulse_start,
            "impulse_price": self.impulse_price,
            "bars_origin_to_impulse": self.impulse_start - self.origin_index,
            "extreme_price": self.extreme_price,
            "maximum_favorable_excursion": round(self.mfe_pct, 6),
            "maximum_adverse_excursion": round(self.mae_pct, 6),
            "segment": self.segment,
        }
        for t in targets:
            key = f"{t*100:g}pct"
            row[f"time_to_{key}"] = self.time_to.get(key, -1)
            row[f"hit_{key}"] = int(self.targets_hit.get(key, False))
        return row


def _measure(series, ev: MoveEvent, horizon: int, targets: List[float],
             tt: List[array]) -> None:
    """Fill excursions and target times by direct forward scan.

    This runs only on CLUSTERED events (a few thousand per symbol), so the
    per-event scan is cheap.  It uses future bars deliberately: an event's
    OUTCOME is allowed to see the future -- only its PATTERNS are not.
    """
    i = ev.origin_index
    entry = series.c[i]
    ev.start_price = entry
    ev.start_time = series.ot[i]
    end = min(series.n - 1, i + horizon)

    best = entry
    worst = entry
    best_idx = i
    for j in range(i + 1, end + 1):
        if ev.direction == LONG:
            if series.h[j] > best:
                best = series.h[j]
                best_idx = j
            if series.l[j] < worst:
                worst = series.l[j]
        else:
            if series.l[j] < best:
                best = series.l[j]
                best_idx = j
            if series.h[j] > worst:
                worst = series.h[j]

    ev.extreme_price = best
    # anchor the event on the launch bar of the leg
    launch = i
    if best_idx > i:
        if ev.direction == LONG:
            lowest = series.l[i]
            for j in range(i, best_idx + 1):
                if series.l[j] <= lowest:
                    lowest = series.l[j]
                    launch = j
        else:
            highest = series.h[i]
            for j in range(i, best_idx + 1):
                if series.h[j] >= highest:
                    highest = series.h[j]
                    launch = j
    ev.impulse_start = launch
    ev.impulse_time = series.ot[launch]
    ev.impulse_price = series.c[launch]

    if ev.direction == LONG:
        ev.mfe_pct = (best - entry) / entry
        ev.mae_pct = (entry - worst) / entry
    else:
        ev.mfe_pct = (entry - best) / entry
        ev.mae_pct = (worst - entry) / entry

    ev.event_end = max(ev.event_end, best_idx)
    ev.end_time = series.ot[ev.event_end]
    ev.duration_bars = ev.event_end - ev.event_start
    for k, t in enumerate(targets):
        key = f"{t*100:g}pct"
        v = tt[k][i]
        ev.time_to[key] = v
        ev.targets_hit[key] = v >= 0


def cluster(series, symbol: str, direction: str, tt: List[array],
            targets: List[float], cfg) -> List[MoveEvent]:
    """Merge the run of origins belonging to one impulse into one MoveEvent.

    An origin extends the current event while the impulse is still running
    (it appears no later than the event's own target hit, plus a small gap) AND
    price has not meaningfully reset against the move.

    The reset test only arms once the move has actually EXTENDED
    (CLUSTER_MIN_EXTENSION_PCT).  Without that guard, the ordinary oscillation
    of a pre-impulse range trips "50 % retracement" every few bars and shatters
    one impulse into a dozen phantom events.
    """
    first = tt[0]                      # the 1 % target defines an event
    horizon = cfg.EVENT_HORIZON_BARS
    gap = cfg.CLUSTER_GAP_BARS
    min_ext = getattr(cfg, "CLUSTER_MIN_EXTENSION_PCT", 0.005)
    events: List[MoveEvent] = []
    n = series.n
    is_long = direction == LONG

    current: Optional[MoveEvent] = None
    cur_reach = -1
    cur_extreme = 0.0
    cur_start_price = 0.0

    for i in range(n):
        if current is not None:                 # running extreme, O(1) per bar
            if is_long:
                if series.h[i] > cur_extreme:
                    cur_extreme = series.h[i]
            else:
                if series.l[i] < cur_extreme:
                    cur_extreme = series.l[i]

        if first[i] < 0:
            continue
        reach = i + first[i]

        if current is not None:
            if is_long:
                span = cur_extreme - cur_start_price
                retrace = cur_extreme - series.c[i]
            else:
                span = cur_start_price - cur_extreme
                retrace = series.c[i] - cur_extreme
            extended = span >= min_ext * cur_start_price
            reset = extended and retrace >= cfg.CLUSTER_RESET_PCT * span
            inside = i <= cur_reach + gap
            in_horizon = (i - current.event_start) <= horizon
            if inside and in_horizon and not reset:
                current.origins += 1
                current.event_end = max(current.event_end, reach)
                if reach > cur_reach:
                    cur_reach = reach
                continue
            events.append(current)

        current = MoveEvent(symbol, direction, i)
        current.event_end = reach
        cur_reach = reach
        cur_start_price = series.c[i]
        cur_extreme = series.h[i] if is_long else series.l[i]
    if current is not None:
        events.append(current)

    # a genuinely new event needs real separation from the previous one
    spaced: List[MoveEvent] = []
    for ev in events:
        if spaced and ev.event_start - spaced[-1].event_start < cfg.MIN_EVENT_SEPARATION_BARS:
            spaced[-1].origins += ev.origins
            spaced[-1].event_end = max(spaced[-1].event_end, ev.event_end)
            continue
        spaced.append(ev)

    for ev in spaced:
        _measure(series, ev, horizon, targets, tt)
    return spaced


def detect(series, symbol: str, cfg):
    """Returns (long_events, short_events, tt_long, tt_short)."""
    tt_long = time_to_targets(series, cfg.TARGETS, LONG, cfg.EVENT_HORIZON_BARS)
    tt_short = time_to_targets(series, cfg.TARGETS, SHORT, cfg.EVENT_HORIZON_BARS)
    longs = cluster(series, symbol, LONG, tt_long, cfg.TARGETS, cfg)
    shorts = cluster(series, symbol, SHORT, tt_short, cfg.TARGETS, cfg)
    return longs, shorts, tt_long, tt_short
