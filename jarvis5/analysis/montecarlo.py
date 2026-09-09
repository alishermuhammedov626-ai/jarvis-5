"""Monte-Carlo risk analysis on the closed-trade R sequence (spec 78).

This is NOT a forecast.  Resampling the realised R multiples only answers one
question: given this trade distribution, how bad can the ordering luck get?
"""
from __future__ import annotations

import random
from typing import Dict, List

from ..backtest.metrics import percentile


def run(r_multiples: List[float], risk_fraction: float,
        start_equity: float = 10_000.0, simulations: int = 5_000,
        seed: int = 20260909) -> Dict:
    n = len(r_multiples)
    if n < 5:
        return {"error": "not enough trades for Monte Carlo", "trades": n}

    rng = random.Random(seed)
    finals: List[float] = []
    max_dds: List[float] = []
    max_dd_pcts: List[float] = []
    streaks: List[int] = []
    ruin = 0

    for _ in range(simulations):
        eq = start_equity
        peak = eq
        dd = 0.0
        dd_pct = 0.0
        streak = 0
        worst_streak = 0
        for _k in range(n):
            r = r_multiples[rng.randrange(n)]
            eq *= (1.0 + r * risk_fraction)
            if eq <= 0:
                eq = 0.0
                ruin += 1
                break
            if r < 0:
                streak += 1
                worst_streak = max(worst_streak, streak)
            else:
                streak = 0
            peak = max(peak, eq)
            dd = min(dd, eq - peak)
            dd_pct = min(dd_pct, (eq - peak) / peak if peak else 0.0)
        finals.append(eq)
        max_dds.append(dd)
        max_dd_pcts.append(dd_pct)
        streaks.append(worst_streak)

    return {
        "simulations": simulations,
        "trades_per_simulation": n,
        "risk_fraction": risk_fraction,
        "start_equity": start_equity,
        "final_equity": {
            "median": round(percentile(finals, 0.50), 2),
            "p05": round(percentile(finals, 0.05), 2),
            "p25": round(percentile(finals, 0.25), 2),
            "p75": round(percentile(finals, 0.75), 2),
            "p95": round(percentile(finals, 0.95), 2),
            "min": round(min(finals), 2),
            "max": round(max(finals), 2),
        },
        "max_drawdown_pct": {
            "median": round(percentile(max_dd_pcts, 0.50) * 100, 2),
            "p95_worst": round(percentile(max_dd_pcts, 0.05) * 100, 2),
            "worst": round(min(max_dd_pcts) * 100, 2),
        },
        "losing_streak": {
            "median": round(percentile([float(s) for s in streaks], 0.50), 1),
            "p95": round(percentile([float(s) for s in streaks], 0.95), 1),
            "worst": max(streaks),
        },
        "probability_of_loss": round(
            sum(1 for f in finals if f < start_equity) / simulations, 4),
        "risk_of_ruin": round(ruin / simulations, 6),
    }
