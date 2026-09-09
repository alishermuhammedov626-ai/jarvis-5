"""Backtest statistics (spec 57-73).

Win rate is never reported alone: every table carries PF, expectancy, average R
and drawdown next to it, because a 70 % win rate with a 0.8 profit factor is a
losing strategy.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median
from typing import Callable, Dict, Iterable, List, Optional


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(pos)]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def drawdown_curve(equity_points: List[float], start_equity: float):
    peak = start_equity
    max_dd = 0.0
    max_dd_pct = 0.0
    curve = []
    for eq in equity_points:
        peak = max(peak, eq)
        dd = eq - peak
        dd_pct = _safe_div(dd, peak)
        curve.append(dd)
        if dd < max_dd:
            max_dd = dd
        if dd_pct < max_dd_pct:
            max_dd_pct = dd_pct
    return curve, max_dd, max_dd_pct


def streaks(trades) -> Dict[str, int]:
    best = cur_w = worst = cur_l = 0
    for t in trades:
        if t.net_pnl > 0:
            cur_w += 1
            cur_l = 0
        elif t.net_pnl < 0:
            cur_l += 1
            cur_w = 0
        else:
            cur_w = cur_l = 0
        best = max(best, cur_w)
        worst = max(worst, cur_l)
    return {"max_winning_streak": best, "max_losing_streak": worst}


def pnl_layers(t) -> Dict[str, float]:
    """Cost decomposition (spec 73)."""
    no_cost = t.gross_pnl + t.entry_slip + t.exit_slip
    return {
        "without_cost": no_cost,
        "with_commission": no_cost - t.fees,
        "with_commission_slippage": no_cost - t.fees - t.slippage,
        "with_all_costs": t.net_pnl,
    }


def summarize(trades, start_equity: float = 0.0, label: str = "ALL") -> Dict:
    n = len(trades)
    out: Dict[str, object] = {"label": label, "trades": n}
    if n == 0:
        out.update({k: 0 for k in (
            "wins", "losses", "win_rate", "loss_rate", "gross_profit",
            "gross_loss", "net_pnl", "profit_factor", "avg_win", "avg_loss",
            "avg_r", "median_r", "expectancy_r", "max_drawdown",
            "max_drawdown_pct", "avg_holding_minutes", "median_holding_minutes",
            "commission", "slippage", "funding", "total_costs")})
        return out

    rs = [t.r_multiple for t in trades]
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = sum(t.net_pnl for t in losses)
    hold = [t.holding_ms / 60000.0 for t in trades]

    eq = start_equity
    pts = []
    for t in trades:
        eq += t.net_pnl
        pts.append(eq)
    _curve, max_dd, max_dd_pct = drawdown_curve(pts, start_equity)

    win_r = [t.r_multiple for t in wins]
    loss_r = [t.r_multiple for t in losses]
    wr = _safe_div(len(wins), n)

    layers = defaultdict(float)
    for t in trades:
        for k, v in pnl_layers(t).items():
            layers[k] += v

    exits = Counter(t.exit_reason for t in trades)

    out.update({
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": n - len(wins) - len(losses),
        "win_rate": round(wr, 6),
        "loss_rate": round(_safe_div(len(losses), n), 6),
        "gross_profit": round(gross_profit, 6),
        "gross_loss": round(gross_loss, 6),
        "net_pnl": round(sum(t.net_pnl for t in trades), 6),
        "profit_factor": round(_safe_div(gross_profit, abs(gross_loss)), 6)
                         if gross_loss else (float("inf") if gross_profit else 0.0),
        "avg_win": round(_safe_div(gross_profit, len(wins)), 6),
        "avg_loss": round(_safe_div(gross_loss, len(losses)), 6),
        "avg_win_r": round(_safe_div(sum(win_r), len(win_r)), 6),
        "avg_loss_r": round(_safe_div(sum(loss_r), len(loss_r)), 6),
        "avg_r": round(sum(rs) / n, 6),
        "median_r": round(median(rs), 6),
        "expectancy_r": round(sum(rs) / n, 6),
        "expectancy_formula_r": round(
            wr * _safe_div(sum(win_r), len(win_r))
            + (1 - wr) * _safe_div(sum(loss_r), len(loss_r)), 6),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_pct": round(max_dd_pct, 6),
        "avg_holding_minutes": round(sum(hold) / n, 3),
        "median_holding_minutes": round(median(hold), 3),
        "max_holding_minutes": round(max(hold), 3),
        "avg_mfe_r": round(sum(t.mfe_r for t in trades) / n, 4),
        "avg_mae_r": round(sum(t.mae_r for t in trades) / n, 4),
        "median_mfe_r": round(median([t.mfe_r for t in trades]), 4),
        "median_mae_r": round(median([t.mae_r for t in trades]), 4),
        "commission": round(sum(t.fees for t in trades), 6),
        "slippage": round(sum(t.slippage for t in trades), 6),
        "funding": round(sum(t.funding for t in trades), 6),
        "total_costs": round(sum(t.fees + t.slippage - t.funding for t in trades), 6),
        "pnl_without_cost": round(layers["without_cost"], 6),
        "pnl_with_commission": round(layers["with_commission"], 6),
        "pnl_with_commission_slippage": round(layers["with_commission_slippage"], 6),
        "pnl_with_all_costs": round(layers["with_all_costs"], 6),
        "tp_rate": round(_safe_div(exits.get("TP", 0), n), 6),
        "sl_rate": round(_safe_div(exits.get("SL", 0), n), 6),
        "trailing_rate": round(_safe_div(exits.get("TRAILING_STOP", 0), n), 6),
        "invalidation_rate": round(_safe_div(exits.get("INVALIDATION", 0), n), 6),
        "timeout_rate": round(_safe_div(exits.get("TIMEOUT", 0), n), 6),
        "liquidation_rate": round(_safe_div(exits.get("LIQUIDATION", 0), n), 6),
        "exit_reasons": dict(exits),
        "ambiguous_exits": sum(1 for t in trades if t.ambiguous_exit),
        "longs": sum(1 for t in trades if t.direction == "BUY"),
        "shorts": sum(1 for t in trades if t.direction == "SELL"),
        "avg_rr_planned": round(sum(
            t.context.get("plan", {}).get("rr", 0.0) for t in trades) / n, 4),
        "avg_sl_percent": round(sum(
            t.context.get("plan", {}).get("sl_percent", 0.0) for t in trades) / n, 6),
        "avg_tp_percent": round(sum(
            t.context.get("plan", {}).get("tp_percent", 0.0) for t in trades) / n, 6),
    })
    out.update(streaks(trades))
    return out


def group(trades, keyfn: Callable, start_equity: float = 0.0) -> Dict[str, Dict]:
    buckets: Dict[str, list] = defaultdict(list)
    for t in trades:
        k = keyfn(t)
        if k is None:
            k = "UNKNOWN"
        buckets[str(k)].append(t)
    return {k: summarize(v, start_equity, label=k)
            for k, v in sorted(buckets.items())}


# ---- grouping keys ---------------------------------------------------- #
def _ctx(t, *path, default=""):
    node = t.context
    for p in path:
        if not isinstance(node, dict):
            return default
        node = node.get(p, default)
    return node


def displacement_bucket(t) -> str:
    d = _ctx(t, "m5_displacement", default=0.0) or 0.0
    if d < 1.0:
        return "<1.0_ATR"
    if d < 1.5:
        return "1.0-1.5_ATR"
    return ">=1.5_ATR"


def zone_bucket(t) -> str:
    kind = _ctx(t, "zone", "kind", default="NONE")
    fresh = _ctx(t, "zone", "fresh", default=True)
    return f"{kind}_{'FRESH' if fresh else 'MITIGATED'}"


def m1_bucket(t) -> str:
    c = t.context.get("confirmation", {})
    parts = []
    parts.append("SWEEP" if c.get("m1_micro_sweep") else "-")
    parts.append(c.get("m1_micro_break") or "-")
    parts.append("DISP" if (c.get("m1_micro_displacement") or 0) > 0 else "-")
    return "+".join(parts)


def structure_bucket(t) -> str:
    return "{}_{}{}".format(_ctx(t, "m5_break_scope", default="?"),
                            _ctx(t, "m5_break", default="?"),
                            "/MSS" if _ctx(t, "m5_mss", default=False) else "")


def sweep_penetration_bucket(t) -> str:
    p = _ctx(t, "sweep", "penetration_atr", default=0.0) or 0.0
    for edge in (0.1, 0.25, 0.5, 1.0):
        if p < edge:
            return f"<{edge}_ATR"
    return ">=1.0_ATR"


def full_report(bt) -> Dict:
    """Assemble every breakdown the spec asks for."""
    tr = bt.trades
    eq0 = bt.cfg.STARTING_EQUITY
    rep: Dict[str, object] = {
        "overall": summarize(tr, eq0),
        "starting_equity": eq0,
        "ending_equity": round(bt.equity, 6),
        "return_pct": round(_safe_div(bt.equity - eq0, eq0), 6),
        "per_symbol": group(tr, lambda t: t.symbol, eq0),
        "per_setup": group(tr, lambda t: t.setup_type, eq0),
        "per_direction": group(tr, lambda t: t.direction, eq0),
        "per_session": group(tr, lambda t: t.session, eq0),
        "per_regime": group(tr, lambda t: _ctx(t, "m15_regime", default="?"), eq0),
        "per_liquidity_type": group(tr, lambda t: _ctx(t, "sweep", "type", default="?"), eq0),
        "per_sweep_quality": group(tr, lambda t: _ctx(t, "sweep", "quality", default="?"), eq0),
        "per_sweep_penetration": group(tr, sweep_penetration_bucket, eq0),
        "per_displacement": group(tr, displacement_bucket, eq0),
        "per_zone": group(tr, zone_bucket, eq0),
        "per_structure": group(tr, structure_bucket, eq0),
        "per_m1_confirmation": group(tr, m1_bucket, eq0),
        "per_exit_reason": group(tr, lambda t: t.exit_reason, eq0),
        "per_month": group(tr, lambda t: t.day[:7], eq0),
        "direction_by_regime": group(
            tr, lambda t: f"{_ctx(t, 'm15_regime', default='?')}|{t.direction}", eq0),
        "rejections": bt.journal.rejection_histogram(),
        "rejections_by_symbol": bt.journal.rejections_by_symbol(),
        "funnel": bt.journal.funnel(),
        "data_quality": bt.data_quality,
        "ambiguous_exits": bt.ambiguous_exits,
        "config": bt.cfg.to_dict(),
    }
    return rep
