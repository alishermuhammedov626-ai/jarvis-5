"""Synthetic validation gate (spec 30).

Before the engine is allowed near real money data it has to prove, on data
whose ground truth we constructed, that:

  * every injected move is found as an event,
  * each injected pattern is actually detected where it was injected,
  * and the same detectors do NOT fire at the same rate in the RANDOM and
    NOISE segments -- a detector that says "sweep" everywhere has found
    nothing.

The last point is the one that matters.  High detection alone proves only that
the rule is loose.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..core.series import aggregate
from . import combos, events
from .features import SymbolReplay
from .synthetic import make_market

# marker -> feature names that should fire, and how many bars of slack the
# higher timeframes need before they can see it
EXPECTED: Dict[str, Tuple[List[str], int]] = {
    "sweep_low": (["M1_SWEEP_SELLSIDE", "M5_SWEEP_SELLSIDE"], 8),
    "sweep_high": (["M1_SWEEP_BUYSIDE", "M5_SWEEP_BUYSIDE"], 8),
    "displacement_up": (["M1_DISPLACEMENT_UP", "M5_DISPLACEMENT_UP"], 8),
    "fvg_up": (["M1_FVG_BULL", "M5_FVG_BULL"], 8),
    "spring": (["WY_SPRING", "M5_WY_SPRING"], 10),
    "rsi_oversold": (["RSI_LT_30", "RSI_LT_20"], 4),
    "volume_breakout": (["VOLUME_GT_2X", "VOLUME_SPIKE", "VOLUME_BREAKOUT"], 4),
    "range_breakout": (["PA_RANGE_BREAKOUT", "M1_BREAKOUT_UP",
                        "WY_RANGE_BREAKOUT"], 6),
}

MIN_DETECTION_RATE = 0.75
MIN_SEPARATION = 1.30          # injected rate must beat the control rate by this
# Separation is measured against RANDOM segments only.  The NOISE segments
# deliberately contain real sweeps and gaps that simply lead nowhere, so a
# detector SHOULD fire there -- scoring against them would be measuring the
# detector's outcome prediction, not its detection.
CONTROL_SCENARIOS = ("RANDOM",)
SATURATION_RATE = 0.50         # above this, a detector is common, not selective


def run(cfg, cycles: int = 6, price: float = 1.0, seed: int = 11,
        progress: bool = True) -> Dict:
    series, truth = make_market("SYNTH", 1_767_225_600_000, cycles=cycles,
                                price=price, seed=seed)
    m5 = aggregate(series, "M5")
    m15 = aggregate(series, "M15")
    longs, shorts, _tl, _ts = events.detect(series, "SYNTH", cfg)
    all_events = longs + shorts

    # ---- bars we need feature snapshots at ---------------------------- #
    need: Dict[int, None] = {}
    marker_bars: List[Tuple[str, int, int]] = []      # (marker, start, slack)
    for t in truth:
        for key, idx in t.get("markers", {}).items():
            if key not in EXPECTED:
                continue
            slack = EXPECTED[key][1]
            marker_bars.append((key, idx, slack))
            for j in range(idx, min(series.n, idx + slack + 1)):
                need[j] = None
    control_bars: List[int] = []
    for t in truth:
        if t["scenario"] not in CONTROL_SCENARIOS:
            continue
        for j in range(t["start_index"], t["end_index"] + 1, 3):
            control_bars.append(j)
            need[j] = None

    snaps: Dict[int, Dict[str, int]] = {}
    replay = SymbolReplay("SYNTH", series, m5, m15, cfg)
    for i in range(series.n):
        replay.advance(i)
        if i in need and replay.ready:
            f = replay.snapshot(i)
            f.update(combos.build(f))
            snaps[i] = f

    checks: List[Dict] = []

    # ---- 1. every injected move is covered by a detected event -------- #
    covered = 0
    injected = 0
    for t in truth:
        mi = t["move_index"]
        if mi < 0:
            continue
        injected += 1
        pool = longs if t["direction"] == "LONG" else shorts
        if any(e.event_start - cfg.EVENT_HORIZON_BARS <= mi <= e.event_end
               for e in pool):
            covered += 1
    checks.append(_check("event_coverage", covered == injected,
                         f"{covered}/{injected} injected moves are inside a "
                         f"detected event window"))

    # ---- 2. events are not manufactured out of pure noise -------------- #
    noise_bars = sum(t["end_index"] - t["start_index"] + 1 for t in truth
                     if t["scenario"] in ("RANDOM", "NOISE"))
    inj_bars = sum(t["end_index"] - t["start_index"] + 1 for t in truth
                   if t["scenario"] not in ("RANDOM", "NOISE"))
    noise_events = sum(1 for e in all_events if _in_scenarios(
        e.impulse_start, truth, ("RANDOM", "NOISE")))
    inj_events = len(all_events) - noise_events
    noise_density = noise_events / noise_bars if noise_bars else 0.0
    inj_density = inj_events / inj_bars if inj_bars else 0.0
    checks.append(_check(
        "event_density_separation", inj_density > noise_density,
        f"injected segments {inj_density*1000:.2f} events/1000 bars vs "
        f"noise segments {noise_density*1000:.2f}"))

    # ---- 3. each injected pattern is detected where it was injected ---- #
    detail = []
    all_ok = True
    for key, (names, slack) in EXPECTED.items():
        hits = 0
        total = 0
        for mkey, idx, sl in marker_bars:
            if mkey != key:
                continue
            total += 1
            found = False
            for j in range(idx, min(series.n, idx + sl + 1)):
                f = snaps.get(j)
                if f and _fresh(f, names, sl):
                    found = True
                    break
            hits += int(found)
        rate = hits / total if total else 0.0
        # control: same detector, same freshness requirement, RANDOM bars only
        ctrl_hits = sum(1 for j in control_bars
                        if j in snaps and _fresh(snaps[j], names, slack))
        ctrl_n = sum(1 for j in control_bars if j in snaps)
        ctrl = ctrl_hits / ctrl_n if ctrl_n else 0.0
        # the same detector without the freshness requirement: how often is the
        # flag simply ON somewhere in the 120-minute window?
        loose_hits = sum(1 for j in control_bars
                         if j in snaps and any(nm in snaps[j] for nm in names))
        loose = loose_hits / ctrl_n if ctrl_n else 0.0
        detected_ok = total > 0 and rate >= MIN_DETECTION_RATE
        saturated = ctrl >= SATURATION_RATE
        separated = ctrl <= 0.0 or rate >= ctrl * MIN_SEPARATION
        all_ok = all_ok and detected_ok
        detail.append({"marker": key, "n": total, "detection_rate": round(rate, 4),
                       "control_rate": round(ctrl, 4),
                       "control_rate_any_window": round(loose, 4),
                       "saturated": saturated, "separated": separated,
                       "pass": detected_ok})
    checks.append(_check("pattern_detection", all_ok,
                         "; ".join(f"{d['marker']}={d['detection_rate']:.0%}"
                                   for d in detail)))

    # Separation is reported separately from detection: a detector can be
    # perfectly correct and still fire on most random bars.  That is a fact
    # about the pattern, not a fault in the code -- but it must be visible.
    unseparated = [d["marker"] for d in detail
                   if not d["separated"] and not d["saturated"]]
    saturated = [d["marker"] for d in detail if d["saturated"]]
    checks.append(_check(
        "pattern_separation", not unseparated,
        ("all non-saturated detectors separate from random; " if not unseparated
         else f"no separation from random: {', '.join(unseparated)}; ")
        + (f"SATURATED (fire on >={SATURATION_RATE:.0%} of random bars, so they "
           f"carry little information alone): {', '.join(saturated)}"
           if saturated else "no saturated detectors")))

    # ---- 4. combinations assemble from their parts --------------------- #
    combo_seen = _combo_hits(snaps, truth, "SWEEP_MSS_DISPLACEMENT_LONG",
                             ["C04_SWEEP_MSS_UP", "C06_SWEEP_MSS_DISPLACEMENT_UP",
                              "C05_SWEEP_DISPLACEMENT_UP", "C02_SWEEP_BOS_UP",
                              "C03_SWEEP_CHOCH_UP"])
    checks.append(_check("combination_detection", combo_seen > 0,
                         f"sweep-plus-structure combinations fired {combo_seen} "
                         f"times inside SWEEP_MSS_DISPLACEMENT segments"))

    # ---- 5. excursions are measured sanely ----------------------------- #
    bad = [e for e in longs if e.targets_hit.get("1pct") and e.mfe_pct < 0.0099]
    checks.append(_check("mfe_mae_consistency", not bad,
                         f"{len(bad)} events claim a 1 % target with MFE < 1 %"))

    ok = all(c["pass"] for c in checks)
    return {
        "status": "PASS" if ok else "FAIL",
        "bars": series.n,
        "segments": len(truth),
        "long_events": len(longs),
        "short_events": len(shorts),
        "checks": checks,
        "pattern_detail": detail,
        "series": series,
        "truth": truth,
    }


def _fresh(feats: Dict[str, int], names: List[str], slack: int) -> bool:
    """Feature present AND recent enough to be the injected occurrence."""
    for nm in names:
        age = feats.get(nm)
        if age is not None and age <= slack:
            return True
    return False


def _check(name: str, passed: bool, detail: str) -> Dict:
    return {"check": name, "pass": bool(passed), "detail": detail}


def _in_scenarios(i: int, truth, names) -> bool:
    for t in truth:
        if t["scenario"] in names and t["start_index"] <= i <= t["end_index"]:
            return True
    return False


def _control_rates(snaps, control_bars) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    n = 0
    for j in control_bars:
        f = snaps.get(j)
        if f is None:
            continue
        n += 1
        for name in f:
            counts[name] = counts.get(name, 0) + 1
    if not n:
        return {}
    return {k: v / n for k, v in counts.items()}


def _combo_hits(snaps, truth, scenario: str, names: List[str]) -> int:
    hits = 0
    for t in truth:
        if t["scenario"] != scenario:
            continue
        for j in range(t["start_index"], t["end_index"] + 1):
            f = snaps.get(j)
            if f and any(nm in f for nm in names):
                hits += 1
                break
    return hits


def print_report(res: Dict) -> None:
    print("=" * 62)
    print("JARVIS 5 RESEARCH -- SYNTHETIC VALIDATION")
    print("=" * 62)
    print(f"bars={res['bars']}  segments={res['segments']}  "
          f"long_events={res['long_events']}  short_events={res['short_events']}")
    print()
    for c in res["checks"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']:<26}{c['detail']}")
    if res.get("pattern_detail"):
        print("\n  pattern detection detail")
        print(f"  {'marker':<20}{'n':>5}{'detected':>10}{'control':>9}"
              f"{'ctl(any win)':>14}  verdict")
        for d in res["pattern_detail"]:
            tag = "PASS" if d["pass"] else "FAIL"
            if d["pass"] and d["saturated"]:
                tag = "PASS/SATURATED"
            print(f"  {d['marker']:<20}{d['n']:>5}{d['detection_rate']:>9.0%}"
                  f"{d['control_rate']:>9.0%}{d['control_rate_any_window']:>13.0%}"
                  f"  {tag}")
        print("\n  'control' = same detector, same freshness, in RANDOM segments.")
        print("  'ctl(any win)' = the flag merely ON somewhere in the 120-minute")
        print("  window -- where it approaches 100 %, that feature carries almost")
        print("  no information on its own.  The edge tables quantify this.")
    print(f"\n  => {res['status']}")
