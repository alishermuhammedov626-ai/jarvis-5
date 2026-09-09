"""Execution-assumption robustness (spec 77).

If a +20 % commission bump or a one-candle entry delay flips the strategy from
profitable to losing, the edge lives inside the execution assumptions rather
than inside the market.  That is reported as FRAGILE.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..backtest.metrics import summarize
from ..backtest.runner import run_backtest


def scenarios(cfg) -> List[Dict]:
    return [
        {"name": "BASELINE", "cfg": cfg},
        {"name": "COMMISSION_+10%",
         "cfg": cfg.copy(COMMISSION_RATE=cfg.COMMISSION_RATE * 1.10)},
        {"name": "COMMISSION_+20%",
         "cfg": cfg.copy(COMMISSION_RATE=cfg.COMMISSION_RATE * 1.20)},
        {"name": "SLIPPAGE_+10%",
         "cfg": cfg.copy(SLIPPAGE_BPS=cfg.SLIPPAGE_BPS * 1.10)},
        {"name": "SLIPPAGE_+20%",
         "cfg": cfg.copy(SLIPPAGE_BPS=cfg.SLIPPAGE_BPS * 1.20)},
        {"name": "ENTRY_DELAY_1_CANDLE", "cfg": cfg.copy(ENTRY_DELAY_BARS=1)},
        {"name": "SAME_CANDLE_TP_FIRST", "cfg": cfg.copy(SAME_CANDLE_RULE="TP_FIRST")},
        {"name": "NO_TRAILING", "cfg": cfg.copy(TRAILING_ENABLED=False)},
        {"name": "RISK_0.25%", "cfg": cfg.copy(RISK_PER_TRADE=0.0025)},
        {"name": "RISK_0.50%", "cfg": cfg.copy(RISK_PER_TRADE=0.0050)},
    ]


def run(cfg, data, specs=None, funding=None,
        perturbed_data: Optional[Dict] = None) -> Dict:
    rows = []
    for sc in scenarios(cfg):
        bt = run_backtest(sc["cfg"], data, specs, funding)
        m = summarize(bt.trades, sc["cfg"].STARTING_EQUITY, label=sc["name"])
        rows.append({"scenario": sc["name"], "trades": m["trades"],
                     "win_rate": m["win_rate"],
                     "profit_factor": m["profit_factor"],
                     "expectancy_r": m["expectancy_r"],
                     "net_pnl": m["net_pnl"],
                     "max_drawdown_pct": m["max_drawdown_pct"]})
    if perturbed_data:
        bt = run_backtest(cfg, perturbed_data, specs, funding)
        m = summarize(bt.trades, cfg.STARTING_EQUITY, label="PRICE_PERTURBATION")
        rows.append({"scenario": "PRICE_PERTURBATION", "trades": m["trades"],
                     "win_rate": m["win_rate"],
                     "profit_factor": m["profit_factor"],
                     "expectancy_r": m["expectancy_r"],
                     "net_pnl": m["net_pnl"],
                     "max_drawdown_pct": m["max_drawdown_pct"]})

    base = rows[0]
    others = rows[1:]
    survived = sum(1 for r in others if r["expectancy_r"] > 0)
    if base["trades"] == 0:
        status = "NO_TRADES"
    elif base["expectancy_r"] <= 0:
        status = "NEGATIVE_BASELINE"
    elif survived == len(others):
        status = "ROBUST"
    elif survived >= len(others) * 0.6:
        status = "ACCEPTABLE"
    else:
        status = "FRAGILE"
    return {"rows": rows,
            "verdict": {"status": status,
                        "scenarios_positive": f"{survived}/{len(others)}"}}
