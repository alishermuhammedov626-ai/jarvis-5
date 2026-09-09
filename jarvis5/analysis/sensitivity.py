"""Parameter sensitivity and robustness testing (spec 75-77).

A strategy that only works at one exact threshold value is not a strategy, it
is a curve fit.  Every parameter listed here is re-run at -20 %, -10 %, +10 %
and +20 % of its configured value; if the neighbourhood collapses while the
centre shines, the parameter is flagged OVERFIT_RISK.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..backtest.metrics import summarize
from ..backtest.runner import run_backtest

# spec 76
SENSITIVITY_PARAMS = [
    "SWING_K",
    "ATR_PERIOD",
    "EXTERNAL_STRUCTURE_ATR",
    "EQUAL_LIQUIDITY_ATR",
    "LIQUIDITY_CLUSTER_ATR",
    "MIN_DISPLACEMENT_ATR",
    "MIN_FVG_ATR",
    "MAX_CHASE_ATR",
    "STRUCTURAL_BUFFER_ATR",
    "MAX_SL_PERCENT",
    "MIN_TP_PERCENT",
    "TRAILING_ACTIVATION_PERCENT",
    "COOLDOWN_MINUTES",
]

STEPS = (-0.20, -0.10, 0.0, 0.10, 0.20)
INT_PARAMS = {"SWING_K", "ATR_PERIOD", "COOLDOWN_MINUTES",
              "SETUP_EXPIRY_M5_BARS", "M1_CONFIRMATION_LOOKBACK"}


def _apply(cfg, name: str, factor: float):
    base = getattr(cfg, name)
    val = base * (1.0 + factor)
    if name in INT_PARAMS:
        val = max(1, int(round(val)))
        if val == base and factor != 0.0:
            # a 10 % step cannot move a small integer -- take the next one
            val = max(1, base + (1 if factor > 0 else -1))
    else:
        val = round(val, 12)
    return cfg.copy(**{name: val}), val


def run(cfg, data, specs=None, funding=None,
        params: Optional[List[str]] = None) -> Dict:
    params = params or SENSITIVITY_PARAMS
    out: Dict[str, Dict] = {}
    for name in params:
        rows = []
        cache: Dict[float, Dict] = {}      # integer steps can collide
        for step in STEPS:
            c2, val = _apply(cfg, name, step)
            if val in cache:
                row = dict(cache[val])
                row["step_pct"] = round(step * 100, 1)
                row["duplicate_of_value"] = val
                rows.append(row)
                continue
            try:
                c2.validate()
            except AssertionError:
                rows.append({"step_pct": round(step * 100, 1), "value": val,
                             "error": "invalid config"})
                continue
            bt = run_backtest(c2, data, specs, funding)
            m = summarize(bt.trades, c2.STARTING_EQUITY, label=f"{name}{step:+.0%}")
            row = {
                "step_pct": round(step * 100, 1), "value": val,
                "trades": m["trades"], "win_rate": m["win_rate"],
                "profit_factor": m["profit_factor"],
                "expectancy_r": m["expectancy_r"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "net_pnl": m["net_pnl"],
            }
            cache[val] = row
            rows.append(row)
        out[name] = {"rows": rows, "verdict": _verdict(rows)}
    return out


def _verdict(rows: List[Dict]) -> Dict:
    good = [r for r in rows if "expectancy_r" in r]
    if not good:
        return {"status": "NO_DATA"}
    centre = next((r for r in good if r["step_pct"] == 0.0), None)
    if centre is None:
        return {"status": "NO_DATA"}
    others = [r for r in good if r["step_pct"] != 0.0]
    if not others:
        return {"status": "NO_DATA"}
    pos = sum(1 for r in others if r["expectancy_r"] > 0)
    if centre["trades"] == 0:
        return {"status": "NO_TRADES"}
    if centre["expectancy_r"] <= 0:
        # the configured value itself loses money -- stability around a losing
        # centre says nothing useful, so never report it as STABLE
        return {"status": "NEGATIVE_CENTRE",
                "reason": f"configured value expectancy = {centre['expectancy_r']}",
                "neighbours_positive": f"{pos}/{len(others)}"}
    if centre["expectancy_r"] > 0 and pos == 0:
        return {"status": "OVERFIT_RISK",
                "reason": "only the configured value has positive expectancy"}
    if centre["expectancy_r"] > 0 and pos < len(others) / 2:
        return {"status": "UNSTABLE",
                "reason": f"{pos}/{len(others)} neighbours positive"}
    return {"status": "STABLE", "neighbours_positive": f"{pos}/{len(others)}"}
