"""Walk-forward validation (spec 74).

There is no model to train here -- the "TRAIN" window is simply the ONLY window
you are allowed to look at while choosing parameters.  VALIDATION and OOS must
stay untouched until the parameters are frozen, otherwise the OOS number is
just another in-sample number.

Splits are strictly chronological.  Random shuffling of trades or of periods is
never performed: it would destroy the sequence information that drawdown and
streak statistics depend on.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..backtest.metrics import summarize
from ..backtest.runner import period_bounds, run_backtest, slice_data
from ..core.timeutil import to_iso


def split_periods(start_ms: int, end_ms: int,
                  ratios=(0.6, 0.2, 0.2)) -> List[Dict]:
    span = end_ms - start_ms
    names = ["TRAIN", "VALIDATION", "OOS"]
    out = []
    cursor = start_ms
    for name, r in zip(names, ratios):
        stop = cursor + int(span * r)
        out.append({"name": name, "start": cursor, "end": stop})
        cursor = stop
    out[-1]["end"] = end_ms + 1
    return out


def run(cfg, data, specs=None, funding=None, ratios=(0.6, 0.2, 0.2)) -> Dict:
    lo, hi = period_bounds(data)
    if lo is None:
        return {"error": "no data"}
    segments = split_periods(lo, hi, ratios)
    results = []
    for seg in segments:
        sub = slice_data(data, seg["start"], seg["end"])
        if not sub:
            results.append({"segment": seg["name"], "trades": 0,
                            "note": "no data in segment"})
            continue
        bt = run_backtest(cfg, sub, specs, funding)
        m = summarize(bt.trades, cfg.STARTING_EQUITY, label=seg["name"])
        m["start"] = to_iso(seg["start"])
        m["end"] = to_iso(seg["end"])
        m["segment"] = seg["name"]
        m["ending_equity"] = round(bt.equity, 2)
        results.append(m)
    return {"mode": "sequential", "ratios": list(ratios), "segments": results,
            "verdict": _verdict(results)}


def rolling(cfg, data, specs=None, funding=None, folds: int = 4,
            train_ratio: float = 0.6, val_ratio: float = 0.2) -> Dict:
    """Rolling walk-forward: train -> validation -> OOS, then roll forward."""
    lo, hi = period_bounds(data)
    if lo is None:
        return {"error": "no data"}
    span = hi - lo
    window = int(span / (1 + (folds - 1) * 0.5)) if folds > 1 else span
    step = int((span - window) / (folds - 1)) if folds > 1 else 0
    out = []
    for f in range(folds):
        w_start = lo + f * step
        w_end = w_start + window
        segs = split_periods(w_start, w_end,
                             (train_ratio, val_ratio, 1 - train_ratio - val_ratio))
        fold = {"fold": f + 1, "start": to_iso(w_start), "end": to_iso(w_end),
                "segments": []}
        for seg in segs:
            sub = slice_data(data, seg["start"], seg["end"])
            if not sub:
                continue
            bt = run_backtest(cfg, sub, specs, funding)
            m = summarize(bt.trades, cfg.STARTING_EQUITY, label=seg["name"])
            m["segment"] = seg["name"]
            m["start"] = to_iso(seg["start"])
            m["end"] = to_iso(seg["end"])
            fold["segments"].append(m)
        out.append(fold)
    oos = [s for f in out for s in f["segments"] if s["segment"] == "OOS"]
    return {"mode": "rolling", "folds": out,
            "oos_summary": {
                "folds_with_trades": sum(1 for s in oos if s["trades"]),
                "positive_expectancy_folds": sum(1 for s in oos if s["expectancy_r"] > 0),
                "total_oos_trades": sum(s["trades"] for s in oos),
            }}


def _verdict(results: List[Dict]) -> Dict:
    oos = next((r for r in results if r.get("segment") == "OOS"), None)
    val = next((r for r in results if r.get("segment") == "VALIDATION"), None)
    if not oos or not oos.get("trades"):
        return {"status": "INCONCLUSIVE", "reason": "no OOS trades"}
    ok = oos["expectancy_r"] > 0 and oos["profit_factor"] > 1.0
    val_ok = bool(val and val.get("trades") and val["expectancy_r"] > 0)
    return {
        "status": "PASS" if (ok and val_ok) else "FAIL",
        "oos_expectancy_r": oos["expectancy_r"],
        "oos_profit_factor": oos["profit_factor"],
        "oos_trades": oos["trades"],
        "validation_positive": val_ok,
    }
