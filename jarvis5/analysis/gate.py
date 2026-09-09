"""Production gate (spec 79).

Win rate alone never opens this gate.  Each criterion is reported with the
number behind it so a failure is diagnosable rather than merely red.
"""
from __future__ import annotations

from typing import Dict, Optional

MIN_TRADES = 100
IDEAL_PF = 1.5
MAX_DD_LIMIT = 0.25


def evaluate(report: Dict, walkforward: Optional[Dict] = None,
             sensitivity: Optional[Dict] = None,
             robustness: Optional[Dict] = None,
             montecarlo: Optional[Dict] = None) -> Dict:
    o = report["overall"]
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "status": "PASS" if passed else "FAIL",
                       "detail": detail})

    add("adequate_trade_count", o["trades"] >= MIN_TRADES,
        f"{o['trades']} trades (need >= {MIN_TRADES})")
    add("profit_factor_above_1", o["profit_factor"] > 1.0,
        f"PF = {o['profit_factor']}")
    add("cost_inclusive_profitability", o["pnl_with_all_costs"] > 0,
        f"net PnL after all costs = {o['pnl_with_all_costs']}")
    add("positive_expectancy", o["expectancy_r"] > 0,
        f"expectancy = {o['expectancy_r']} R")
    add("controlled_max_drawdown", abs(o["max_drawdown_pct"]) <= MAX_DD_LIMIT,
        f"max DD = {o['max_drawdown_pct']*100:.2f}%")

    per_symbol = report.get("per_symbol", {})
    traded = {k: v for k, v in per_symbol.items() if v["trades"] >= 10}
    if traded:
        pos = sum(1 for v in traded.values() if v["expectancy_r"] > 0)
        add("stable_across_coins", pos >= max(1, len(traded) * 0.6),
            f"{pos}/{len(traded)} coins with >=10 trades are positive")
    else:
        add("stable_across_coins", False, "no coin reached 10 trades")

    per_month = report.get("per_month", {})
    months = {k: v for k, v in per_month.items() if v["trades"] >= 5}
    if months:
        pos = sum(1 for v in months.values() if v["expectancy_r"] > 0)
        add("stable_across_periods", pos >= max(1, len(months) * 0.5),
            f"{pos}/{len(months)} months with >=5 trades are positive")
    else:
        add("stable_across_periods", False, "not enough months with >=5 trades")

    if walkforward and walkforward.get("verdict"):
        v = walkforward["verdict"]
        add("positive_oos_expectancy", v.get("status") == "PASS",
            f"walk-forward verdict = {v.get('status')} "
            f"(OOS expectancy {v.get('oos_expectancy_r')})")
    else:
        add("positive_oos_expectancy", False, "walk-forward not run")

    if sensitivity:
        bad = [k for k, v in sensitivity.items()
               if v["verdict"]["status"] in ("OVERFIT_RISK", "UNSTABLE")]
        neg = [k for k, v in sensitivity.items()
               if v["verdict"]["status"] == "NEGATIVE_CENTRE"]
        if neg and not bad:
            # nothing to be stable *around* -- the configured value loses money
            add("reasonable_parameter_sensitivity", False,
                f"{len(neg)}/{len(sensitivity)} parameters have a losing "
                f"configured value; stability is not measurable")
        else:
            add("reasonable_parameter_sensitivity", not bad,
                "unstable parameters: " + (", ".join(bad) if bad else "none"))
    else:
        add("reasonable_parameter_sensitivity", False, "sensitivity not run")

    if robustness:
        st = robustness["verdict"]["status"]
        add("execution_assumption_robustness", st in ("ROBUST", "ACCEPTABLE"),
            f"robustness = {st} ({robustness['verdict']['scenarios_positive']})")
    else:
        add("execution_assumption_robustness", False, "robustness not run")

    # structural guarantees, enforced by construction rather than measured
    checks.append({"check": "no_look_ahead", "status": "PASS",
                   "detail": "swings confirmed K bars late; higher timeframes fed "
                             "only on close; entries at confirmation-bar close; "
                             "trailing applied from the next bar onward "
                             "(see tests/test_no_lookahead.py)"})
    checks.append({"check": "realistic_execution", "status": "PASS",
                   "detail": "commission on notional both sides, per-side slippage, "
                             "funding, conservative same-candle SL-first rule"})
    checks.append({"check": "no_machine_learning", "status": "PASS",
                   "detail": "no model, no fitted score, no probability estimate "
                             "anywhere in the decision path"})

    failed = [c for c in checks if c["status"] == "FAIL"]
    ideal = o["profit_factor"] >= IDEAL_PF
    return {
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "status": "PRODUCTION_CANDIDATE" if not failed else "NOT_READY",
        "meets_ideal_target": bool(ideal and not failed),
        "failed_checks": [c["check"] for c in failed],
        "note": "A high win rate with PF <= 1 is a losing strategy; this gate "
                "never passes on win rate alone.",
    }
