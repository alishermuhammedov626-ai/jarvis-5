"""Conditional edge measurement and honest classification (spec 14-16, 23-26).

The central question is NOT "how often does this pattern appear before a 1 %
move".  It is:

    P(1 % move | pattern)   vs   P(1 % move | no pattern)

A pattern that shows up before 70 % of moves while also showing up before 65 %
of everything else has an edge of 5 points, not of 70.  The control group here
is every OTHER sampled bar, so the comparison is exact rather than sampled.

Nothing is fitted and nothing is optimised.  A pattern that measures badly is
reported as WEAK; a pattern with 12 occurrences is reported as INSUFFICIENT
SAMPLE and never rescued by pooling it with something else.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# classification verdicts
ROBUST = "ROBUST"
PROMISING = "PROMISING"
WEAK = "WEAK"
INSUFFICIENT = "INSUFFICIENT SAMPLE"
OVERFIT_RISK = "OVERFIT RISK"
FAILED_OOS = "OVERFIT / FAILED OOS"

DISCOVERY = "DISCOVERY"
VALIDATION = "VALIDATION"
OOS = "OOS"


class Hist:
    """Fixed-width histogram: medians without keeping every observation."""

    __slots__ = ("width", "buckets", "n", "total", "minv", "maxv")

    def __init__(self, width: float):
        self.width = width
        self.buckets: Dict[int, int] = {}
        self.n = 0
        self.total = 0.0
        self.minv = 0.0
        self.maxv = 0.0

    def add(self, value: float) -> None:
        b = int(math.floor(value / self.width))
        self.buckets[b] = self.buckets.get(b, 0) + 1
        if self.n == 0:
            self.minv = self.maxv = value
        else:
            self.minv = min(self.minv, value)
            self.maxv = max(self.maxv, value)
        self.n += 1
        self.total += value

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    def quantile(self, q: float) -> float:
        if not self.n:
            return 0.0
        target = q * self.n
        seen = 0
        for b in sorted(self.buckets):
            seen += self.buckets[b]
            if seen >= target:
                return (b + 0.5) * self.width
        return self.maxv

    @property
    def median(self) -> float:
        return self.quantile(0.5)

    def merge(self, other: "Hist") -> None:
        if other is None or other.n == 0:
            return
        for b, c in other.buckets.items():
            self.buckets[b] = self.buckets.get(b, 0) + c
        if self.n == 0:
            self.minv, self.maxv = other.minv, other.maxv
        else:
            self.minv = min(self.minv, other.minv)
            self.maxv = max(self.maxv, other.maxv)
        self.n += other.n
        self.total += other.total


class Cell:
    """Counters for one (scope, feature, window, direction)."""

    __slots__ = ("n", "hits", "mfe", "mae", "t1", "sim_n", "sim_wins",
                 "sim_r", "sim_r_gross", "sim_pnl", "sim_exits",
                 "trail_n", "trail_r", "trail_pnl",
                 "sim_r_pos", "sim_r_neg", "r_run", "r_peak", "r_maxdd",
                 "full", "mfe_sum", "mae_sum", "t1_sum", "t1_n")

    def __init__(self, targets: List[str], full: bool = True):
        self.n = 0
        self.full = full
        self.hits: Dict[str, int] = {t: 0 for t in targets}
        # Histograms give medians without keeping every observation, but they
        # are the hot path.  Only the ALL scope needs medians, so per-coin and
        # per-segment cells keep running sums instead.
        self.mfe = Hist(0.001) if full else None
        self.mae = Hist(0.001) if full else None
        self.t1 = Hist(1.0) if full else None
        self.mfe_sum = 0.0
        self.mae_sum = 0.0
        self.t1_sum = 0.0
        self.t1_n = 0
        self.sim_n: Dict[str, int] = {}
        self.sim_wins: Dict[str, int] = {}
        self.sim_r: Dict[str, float] = {}
        self.sim_r_gross: Dict[str, float] = {}
        self.sim_pnl: Dict[str, float] = {}
        self.sim_exits: Dict[str, Dict[str, int]] = {}
        self.trail_n = 0
        self.trail_r = 0.0
        self.trail_pnl = 0.0
        self.sim_r_pos: Dict[str, float] = {}
        self.sim_r_neg: Dict[str, float] = {}
        # running R-drawdown over the sampled trade stream (symbol-ordered,
        # not globally time-ordered -- labelled as such in the reports)
        self.r_run = 0.0
        self.r_peak = 0.0
        self.r_maxdd = 0.0

    def observe(self, label) -> None:
        self.n += 1
        hits = self.hits
        for k in label.hit_keys:
            hits[k] += 1
        self.mfe_sum += label.mfe_pct
        self.mae_sum += label.mae_pct
        if label.time_to_1pct >= 0:
            self.t1_sum += label.time_to_1pct
            self.t1_n += 1
        if self.full:
            self.mfe.add(label.mfe_pct)
            self.mae.add(label.mae_pct)
            if label.time_to_1pct >= 0:
                self.t1.add(float(label.time_to_1pct))

    # -- derived means, valid in both modes ----------------------------- #
    @property
    def mfe_mean(self) -> float:
        return self.mfe_sum / self.n if self.n else 0.0

    @property
    def mae_mean(self) -> float:
        return self.mae_sum / self.n if self.n else 0.0

    @property
    def t1_mean(self) -> float:
        return self.t1_sum / self.t1_n if self.t1_n else 0.0

    def mfe_median(self) -> float:
        return self.mfe.median if self.full else self.mfe_mean

    def mae_median(self) -> float:
        return self.mae.median if self.full else self.mae_mean

    def t1_median(self) -> float:
        return self.t1.median if self.full else self.t1_mean

    def merge(self, other: "Cell") -> None:
        """Combine a disjoint age bin into a cumulative window total."""
        if other is None or other.n == 0:
            return
        self.n += other.n
        for k, v in other.hits.items():
            self.hits[k] = self.hits.get(k, 0) + v
        self.mfe_sum += other.mfe_sum
        self.mae_sum += other.mae_sum
        self.t1_sum += other.t1_sum
        self.t1_n += other.t1_n
        if self.full and other.full:
            self.mfe.merge(other.mfe)
            self.mae.merge(other.mae)
            self.t1.merge(other.t1)
        for src, dst in ((other.sim_n, self.sim_n), (other.sim_wins, self.sim_wins)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v
        for src, dst in ((other.sim_r, self.sim_r),
                         (other.sim_r_gross, self.sim_r_gross),
                         (other.sim_pnl, self.sim_pnl),
                         (other.sim_r_pos, self.sim_r_pos),
                         (other.sim_r_neg, self.sim_r_neg)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0.0) + v
        for k, ex in other.sim_exits.items():
            tgt = self.sim_exits.setdefault(k, {})
            for e, v in ex.items():
                tgt[e] = tgt.get(e, 0) + v
        self.trail_n += other.trail_n
        self.trail_r += other.trail_r
        self.trail_pnl += other.trail_pnl
        self.r_maxdd = min(self.r_maxdd, other.r_maxdd)

    def observe_sim(self, sim) -> None:
        for key, o in sim.outcomes.items():
            self.sim_n[key] = self.sim_n.get(key, 0) + 1
            if o["r_net"] > 0:
                self.sim_wins[key] = self.sim_wins.get(key, 0) + 1
            self.sim_r[key] = self.sim_r.get(key, 0.0) + o["r_net"]
            self.sim_r_gross[key] = self.sim_r_gross.get(key, 0.0) + o["r_gross"]
            self.sim_pnl[key] = self.sim_pnl.get(key, 0.0) + o["pnl_net"]
            if o["r_net"] >= 0:
                self.sim_r_pos[key] = self.sim_r_pos.get(key, 0.0) + o["r_net"]
            else:
                self.sim_r_neg[key] = self.sim_r_neg.get(key, 0.0) + o["r_net"]
            ex = self.sim_exits.setdefault(key, {})
            ex[o["exit"]] = ex.get(o["exit"], 0) + 1
        primary = next(iter(sim.outcomes), None)
        if primary is not None:
            self.r_run += sim.outcomes[primary]["r_net"]
            self.r_peak = max(self.r_peak, self.r_run)
            dd = self.r_run - self.r_peak
            if dd < self.r_maxdd:
                self.r_maxdd = dd
        tr = sim.trailing.get("ALL")
        if tr:
            self.trail_n += 1
            self.trail_r += tr["r_net"]
            self.trail_pnl += tr["pnl_net"]


class Label:
    """The forward outcome of one sampled bar in one direction."""

    __slots__ = ("direction", "hits", "hit_keys", "mfe_pct", "mae_pct",
                 "time_to_1pct")

    def __init__(self, direction: str, hits: Dict[str, bool], mfe: float,
                 mae: float, t1: int):
        self.direction = direction
        self.hits = hits
        # only the targets actually reached, so the hot loop is short
        self.hit_keys = tuple(k for k, v in hits.items() if v)
        self.mfe_pct = mfe
        self.mae_pct = mae
        self.time_to_1pct = t1


class Aggregator:
    """Streaming accumulation -- no per-bar snapshots are ever stored."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.target_keys = [f"{t*100:g}pct" for t in cfg.TARGETS]
        self.cells: Dict[Tuple[str, str, str, str], Cell] = {}
        self.totals: Dict[Tuple[str, str], Cell] = {}
        self.rejections: Dict[str, int] = defaultdict(int)
        self.sampled = 0

    def _cell(self, scope: str, feature: str, window: str, direction: str) -> Cell:
        key = (scope, feature, window, direction)
        c = self.cells.get(key)
        if c is None:
            c = Cell(self.target_keys, full=(scope == "ALL"))
            self.cells[key] = c
        return c

    def _total(self, scope: str, direction: str) -> Cell:
        key = (scope, direction)
        c = self.totals.get(key)
        if c is None:
            c = Cell(self.target_keys, full=(scope == "ALL"))
            self.totals[key] = c
        return c

    def bin_for(self, age: int) -> Optional[str]:
        """Smallest configured window that contains this age (a disjoint bin)."""
        for w in self.cfg.WINDOWS:
            if age <= w:
                return str(w)
        return None

    def observe_bar(self, scopes: List[str], features: Dict[str, int],
                    labels: Dict[str, Label], sims: Dict[str, object]) -> None:
        """One sampled bar: update baselines, then every active pattern."""
        self.sampled += 1
        for scope in scopes:
            for direction, label in labels.items():
                t = self._total(scope, direction)
                t.observe(label)
                sim = sims.get(direction)
                if sim is not None and sim.ok:
                    t.observe_sim(sim)

        # Each observation lands in exactly ONE disjoint age bin.  Cumulative
        # windows (age <= 5, <= 10, ... <= 120) are reconstructed at report time
        # by summing bins, which is identical arithmetic at a quarter of the
        # work -- windows are nested, so counting into all of them is redundant.
        pairs = list(labels.items())
        for name, age in features.items():
            bucket = "STATE" if (age == 0 and _is_state(name)) else self.bin_for(age)
            if bucket is None:
                continue
            for scope in scopes:
                for direction, label in pairs:
                    c = self._cell(scope, name, bucket, direction)
                    c.observe(label)
                    sim = sims.get(direction)
                    if sim is not None and sim.ok:
                        c.observe_sim(sim)

    def window_cell(self, scope: str, feature: str, window: str,
                    direction: str) -> Optional[Cell]:
        """Cumulative cell for a window, summed from its disjoint age bins."""
        if window == "STATE":
            return self.cells.get((scope, feature, "STATE", direction))
        target = int(window)
        out: Optional[Cell] = None
        for w in self.cfg.WINDOWS:
            if w > target:
                break
            c = self.cells.get((scope, feature, str(w), direction))
            if c is None:
                continue
            if out is None:
                out = Cell(self.target_keys, full=(scope == "ALL"))
            out.merge(c)
        return out

    def feature_windows(self, scope: str) -> Dict:
        """(feature, direction) -> the window labels that have any data."""
        out: Dict = {}
        for (sc, feature, bucket, direction) in self.cells:
            if sc != scope:
                continue
            out.setdefault((feature, direction), set()).add(bucket)
        return out

    def baseline(self, scope: str, direction: str, target: str) -> Tuple[int, int]:
        t = self.totals.get((scope, direction))
        if t is None:
            return 0, 0
        return t.n, t.hits.get(target, 0)


_STATE_CACHE: Dict[str, bool] = {}


def _is_state(name: str) -> bool:
    v = _STATE_CACHE.get(name)
    if v is None:
        from .features import is_state
        v = is_state(name)
        _STATE_CACHE[name] = v
    return v


# ---------------------------------------------------------------------- #
# edge maths
# ---------------------------------------------------------------------- #
def edge_stats(n: int, hits: int, total_n: int, total_hits: int) -> Dict[str, float]:
    """Conditional rate vs the control group (every OTHER sampled bar)."""
    other_n = total_n - n
    other_hits = total_hits - hits
    p_pattern = hits / n if n else 0.0
    p_control = other_hits / other_n if other_n else 0.0
    edge = p_pattern - p_control
    lift = (p_pattern / p_control) if p_control > 0 else 0.0

    a, b = hits, n - hits                      # with pattern: move / no move
    c, d = other_hits, other_n - other_hits
    if b > 0 and c > 0 and d > 0 and a > 0:
        odds = (a / b) / (c / d)
    else:                                       # Haldane-Anscombe correction
        odds = ((a + 0.5) / (b + 0.5)) / ((c + 0.5) / (d + 0.5)) \
            if (b + c + d) > 0 else 0.0
    return {
        "conditional_rate": round(p_pattern, 6),
        "baseline_rate": round(p_control, 6),
        "edge": round(edge, 6),
        "relative_lift": round(lift, 6),
        "odds_ratio": round(odds, 6),
    }


def classify(n: int, edge: float, lift: float, cfg,
             validation: Optional[Dict] = None,
             oos: Optional[Dict] = None) -> str:
    """Verdict for one pattern.  Never rescued, never rounded up."""
    if n < cfg.MIN_SAMPLE:
        return INSUFFICIENT
    strong = edge >= cfg.MIN_EDGE and lift >= cfg.MIN_RELATIVE_LIFT
    if not strong:
        return WEAK
    if validation is not None and validation.get("n", 0) >= cfg.MIN_SAMPLE:
        if validation["edge"] <= 0:
            return FAILED_OOS
    if oos is not None and oos.get("n", 0) >= cfg.MIN_SAMPLE:
        if oos["edge"] <= 0:
            return FAILED_OOS
        if oos["edge"] >= cfg.MIN_EDGE * 0.5 and n >= cfg.EXPLORATORY_SAMPLE:
            return ROBUST
        return PROMISING
    if n < cfg.EXPLORATORY_SAMPLE:
        return OVERFIT_RISK          # strong-looking but exploratory sample
    return PROMISING


def coin_consistency(per_coin: Dict[str, Dict], cfg) -> Dict[str, object]:
    """Cross-coin consistent vs coin-specific (spec 24)."""
    voting = {k: v for k, v in per_coin.items()
              if v.get("n", 0) >= cfg.COIN_CONSISTENCY_MIN_SAMPLE}
    if not voting:
        return {"status": "NO_COIN_SAMPLE", "positive_coins": 0,
                "voting_coins": 0, "detail": {}}
    positive = [k for k, v in voting.items() if v["edge"] > 0]
    if len(voting) < 2:
        # one coin cannot be evidence for or against cross-coin consistency
        return {"status": "SINGLE_COIN", "positive_coins": len(positive),
                "voting_coins": len(voting), "coins": sorted(positive),
                "detail": {k: round(v["edge"], 4) for k, v in sorted(voting.items())}}
    status = "CROSS_COIN_CONSISTENT" \
        if len(positive) >= cfg.COIN_CONSISTENCY_MIN_COINS else (
            "COIN_SPECIFIC" if len(positive) <= 2 else "PARTIAL")
    return {
        "status": status,
        "positive_coins": len(positive),
        "voting_coins": len(voting),
        "coins": sorted(positive),
        "detail": {k: round(v["edge"], 4) for k, v in sorted(voting.items())},
    }


def rank_key(row: Dict) -> tuple:
    """Ranking (spec 23): edge first, then how well it actually paid.

    Sample size gates the row rather than boosting it, so a lucky n=31 pattern
    cannot outrank a solid n=3000 one on noise alone.
    """
    verdict_rank = {ROBUST: 0, PROMISING: 1, OVERFIT_RISK: 2, WEAK: 3,
                    FAILED_OOS: 4, INSUFFICIENT: 5}
    return (
        verdict_rank.get(row.get("verdict", WEAK), 5),
        -row.get("edge", 0.0),
        -row.get("expectancy_r_1pct", 0.0),
        -row.get("n", 0),
    )
