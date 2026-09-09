"""Report writers: JSON, CSV trade log, equity curve and a text summary."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

from ..core.timeutil import to_iso
from .metrics import full_report

TRADE_FIELDS = [
    "trade_id", "symbol", "entry_time", "entry_time_iso", "exit_time",
    "exit_time_iso", "direction", "setup_type", "session", "day",
    "m15_regime", "m15_external_direction", "m15_internal_direction",
    "m15_external_sequence", "m15_internal_sequence", "pd_position",
    "m15_sweep", "m15_sweep_quality", "m15_sweep_present",
    "liquidity_type", "liquidity_price", "liquidity_strength",
    "sweep_side", "sweep_quality", "sweep_score", "sweep_penetration_atr",
    "sweep_wick_ratio", "sweep_return_distance", "sweep_volume_ratio",
    "sweep_rejection", "sweep_post_displacement",
    "m5_break", "m5_break_scope", "m5_mss", "m5_displacement",
    "m5_displacement_grade", "zone_kind", "zone_lower", "zone_upper",
    "zone_height_atr", "zone_fresh", "zone_touches", "correction_depth",
    "m1_micro_sweep", "m1_micro_sweep_quality", "m1_micro_break",
    "m1_micro_break_scope", "m1_micro_displacement", "chase_atr",
    "entry_price", "position_size", "notional", "leverage", "margin",
    "sl", "sl_distance", "sl_percent", "sl_source", "risk_amount",
    "liquidation_price", "tp", "tp_distance", "tp_percent", "tp_source", "rr",
    "entry_fee", "exit_fee", "slippage", "funding",
    "exit_price", "exit_reason", "gross_pnl", "fees", "net_pnl", "r_multiple",
    "mfe", "mae", "mfe_r", "mae_r", "holding_minutes", "bars_held",
    "trail_moves", "ambiguous_exit", "equity_before", "equity_after",
    "signal_id",
]


def _c(t, *path, default=""):
    node = t.context
    for p in path:
        if not isinstance(node, dict):
            return default
        node = node.get(p, default)
    return node


def trade_row(t, leverage: float) -> Dict:
    return {
        "trade_id": t.trade_id, "symbol": t.symbol,
        "entry_time": t.entry_time, "entry_time_iso": to_iso(t.entry_time),
        "exit_time": t.exit_time, "exit_time_iso": to_iso(t.exit_time),
        "direction": t.direction, "setup_type": t.setup_type,
        "session": t.session, "day": t.day,
        "m15_regime": _c(t, "m15_regime"),
        "m15_external_direction": _c(t, "m15_external_direction"),
        "m15_internal_direction": _c(t, "m15_internal_direction"),
        "m15_external_sequence": _c(t, "m15_external_sequence"),
        "m15_internal_sequence": _c(t, "m15_internal_sequence"),
        "pd_position": _c(t, "pd_position"),
        "m15_sweep": _c(t, "m15_sweep"),
        "m15_sweep_quality": _c(t, "m15_sweep_quality"),
        "m15_sweep_present": _c(t, "m15_sweep_present"),
        "liquidity_type": _c(t, "sweep", "type"),
        "liquidity_price": _c(t, "sweep", "price"),
        "liquidity_strength": _c(t, "sweep", "liquidity_strength"),
        "sweep_side": _c(t, "sweep", "side"),
        "sweep_quality": _c(t, "sweep", "quality"),
        "sweep_score": _c(t, "sweep", "score"),
        "sweep_penetration_atr": _c(t, "sweep", "penetration_atr"),
        "sweep_wick_ratio": _c(t, "sweep", "wick_ratio"),
        "sweep_return_distance": _c(t, "sweep", "return_distance"),
        "sweep_volume_ratio": _c(t, "sweep", "volume_ratio"),
        "sweep_rejection": _c(t, "sweep", "rejection"),
        "sweep_post_displacement": _c(t, "sweep", "post_displacement"),
        "m5_break": _c(t, "m5_break"), "m5_break_scope": _c(t, "m5_break_scope"),
        "m5_mss": _c(t, "m5_mss"), "m5_displacement": _c(t, "m5_displacement"),
        "m5_displacement_grade": _c(t, "m5_displacement_grade"),
        "zone_kind": _c(t, "zone", "kind"),
        "zone_lower": _c(t, "zone", "lower"), "zone_upper": _c(t, "zone", "upper"),
        "zone_height_atr": _c(t, "zone", "height_atr"),
        "zone_fresh": _c(t, "zone", "fresh"),
        "zone_touches": _c(t, "zone", "touches"),
        "correction_depth": _c(t, "correction_depth"),
        "m1_micro_sweep": _c(t, "confirmation", "m1_micro_sweep"),
        "m1_micro_sweep_quality": _c(t, "confirmation", "m1_micro_sweep_quality"),
        "m1_micro_break": _c(t, "confirmation", "m1_micro_break"),
        "m1_micro_break_scope": _c(t, "confirmation", "m1_micro_break_scope"),
        "m1_micro_displacement": _c(t, "confirmation", "m1_micro_displacement"),
        "chase_atr": _c(t, "confirmation", "chase_atr"),
        "entry_price": t.entry, "position_size": t.qty, "notional": t.notional,
        "leverage": leverage, "margin": t.margin,
        "sl": t.initial_sl, "sl_distance": abs(t.entry - t.initial_sl),
        "sl_percent": _c(t, "plan", "sl_percent"),
        "sl_source": _c(t, "plan", "sl_source"),
        "risk_amount": t.risk_amount, "liquidation_price": t.liquidation_price,
        "tp": t.tp, "tp_distance": abs(t.tp - t.entry),
        "tp_percent": _c(t, "plan", "tp_percent"),
        "tp_source": _c(t, "plan", "tp_source"), "rr": _c(t, "plan", "rr"),
        "entry_fee": t.entry_fee, "exit_fee": t.exit_fee,
        "slippage": t.slippage, "funding": t.funding,
        "exit_price": t.exit_price, "exit_reason": t.exit_reason,
        "gross_pnl": t.gross_pnl, "fees": t.fees, "net_pnl": t.net_pnl,
        "r_multiple": t.r_multiple, "mfe": t.mfe, "mae": t.mae,
        "mfe_r": t.mfe_r, "mae_r": t.mae_r,
        "holding_minutes": t.holding_ms / 60000.0, "bars_held": t.bars_held,
        "trail_moves": t.trail_moves, "ambiguous_exit": t.ambiguous_exit,
        "equity_before": t.equity_before, "equity_after": t.equity_after,
        "signal_id": _c(t, "signal_id"),
    }


def write_trades(path: str, bt) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TRADE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for t in bt.trades:
            w.writerow(trade_row(t, bt.cfg.LEVERAGE))


def write_equity(path: str, bt) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "time_iso", "trade_id", "symbol", "net_pnl", "r",
                    "equity", "drawdown"])
        for p in bt.equity_curve:
            w.writerow([p["time"], to_iso(p["time"]), p["trade_id"], p["symbol"],
                        round(p["net_pnl"], 6), round(p["r"], 6),
                        round(p["equity"], 6), round(p["drawdown"], 6)])


def write_symbol_equity(path: str, bt) -> None:
    """Per-symbol equity curves (spec 61), each starting from STARTING_EQUITY."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    running: Dict[str, float] = {}
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "time", "time_iso", "trade_id", "net_pnl", "r",
                    "symbol_equity"])
        for t in sorted(bt.trades, key=lambda x: (x.symbol, x.exit_time)):
            eq = running.get(t.symbol, bt.cfg.STARTING_EQUITY) + t.net_pnl
            running[t.symbol] = eq
            w.writerow([t.symbol, t.exit_time, to_iso(t.exit_time), t.trade_id,
                        round(t.net_pnl, 6), round(t.r_multiple, 6), round(eq, 6)])


def _fmt_pf(v) -> str:
    if v == float("inf"):
        return "inf"
    return f"{v:.2f}"


def _table(title: str, rows: Dict[str, Dict]) -> List[str]:
    out = [f"\n{title}", "-" * len(title)]
    if not rows:
        out.append("  (no trades)")
        return out
    hdr = f"{'bucket':<28}{'n':>5}{'WR':>8}{'PF':>8}{'ExpR':>9}{'AvgR':>8}{'NetPnL':>12}{'MaxDD%':>9}"
    out.append(hdr)
    for k, m in rows.items():
        if not m["trades"]:
            continue
        out.append("{:<28}{:>5}{:>8}{:>8}{:>9}{:>8}{:>12}{:>9}".format(
            k[:28], m["trades"], f"{m['win_rate']*100:.1f}%",
            _fmt_pf(m["profit_factor"]), f"{m['expectancy_r']:.3f}",
            f"{m['avg_r']:.3f}", f"{m['net_pnl']:.2f}",
            f"{m['max_drawdown_pct']*100:.2f}%"))
    return out


def text_summary(rep: Dict, bt) -> str:
    o = rep["overall"]
    L: List[str] = []
    L.append("=" * 78)
    L.append("JARVIS 5 - BINANCE FUTURES SMC SCALPING BACKTEST")
    L.append("deterministic rule-based / no machine learning")
    L.append("=" * 78)
    cfg = bt.cfg
    L.append(f"Symbols            : {', '.join(cfg.SYMBOLS)}")
    if bt.start_ms:
        L.append(f"Period             : {to_iso(bt.start_ms)} -> {to_iso(bt.end_ms)}"
                 f"  ({(bt.end_ms - bt.start_ms)/86_400_000:.1f} days)")
    L.append(f"Leverage           : {cfg.LEVERAGE:g}x")
    L.append(f"Risk per trade     : {cfg.RISK_PER_TRADE*100:.2f}%")
    L.append(f"Max daily loss     : {cfg.MAX_DAILY_LOSS*100:.2f}%")
    L.append(f"Trades/symbol/day  : {cfg.MAX_TRADES_PER_SYMBOL_PER_DAY}")
    L.append(f"Commission         : {cfg.COMMISSION_RATE*100:.4f}% per side (on notional)")
    L.append(f"Slippage           : {cfg.SLIPPAGE_BPS:g} bps per side")
    L.append(f"Same-candle rule   : {cfg.SAME_CANDLE_RULE}")

    L.append("\nRESULT")
    L.append("-" * 6)
    L.append(f"Total trades       : {o['trades']}")
    L.append(f"Wins / Losses      : {o['wins']} / {o['losses']}")
    L.append(f"Win rate           : {o['win_rate']*100:.2f}%")
    L.append(f"Profit factor      : {_fmt_pf(o['profit_factor'])}")
    L.append(f"Expectancy (R)     : {o['expectancy_r']:.4f}")
    L.append(f"Average R          : {o['avg_r']:.4f}   Median R: {o['median_r']:.4f}")
    L.append(f"Avg win / avg loss : {o['avg_win']:.2f} / {o['avg_loss']:.2f}")
    L.append(f"Gross profit/loss  : {o['gross_profit']:.2f} / {o['gross_loss']:.2f}")
    L.append(f"Net PnL            : {o['net_pnl']:.2f}")
    L.append(f"Starting equity    : {rep['starting_equity']:.2f}")
    L.append(f"Ending equity      : {rep['ending_equity']:.2f}  "
             f"({rep['return_pct']*100:+.2f}%)")
    L.append(f"Max drawdown       : {o['max_drawdown']:.2f} "
             f"({o['max_drawdown_pct']*100:.2f}%)")
    L.append(f"Max win/loss streak: {o['max_winning_streak']} / {o['max_losing_streak']}")
    L.append(f"Holding time (min) : avg {o['avg_holding_minutes']:.1f}  "
             f"median {o['median_holding_minutes']:.1f}  max {o['max_holding_minutes']:.1f}")
    L.append(f"MFE / MAE (R)      : avg {o['avg_mfe_r']:.2f} / {o['avg_mae_r']:.2f}  "
             f"median {o['median_mfe_r']:.2f} / {o['median_mae_r']:.2f}")
    L.append(f"Planned RR (avg)   : {o['avg_rr_planned']:.2f}   "
             f"SL {o['avg_sl_percent']*100:.3f}%  TP {o['avg_tp_percent']*100:.3f}%")

    L.append("\nEXIT MIX")
    L.append("-" * 8)
    L.append(f"TP {o['tp_rate']*100:.1f}%   SL {o['sl_rate']*100:.1f}%   "
             f"TRAILING {o['trailing_rate']*100:.1f}%   "
             f"INVALIDATION {o['invalidation_rate']*100:.1f}%   "
             f"TIMEOUT {o['timeout_rate']*100:.1f}%   "
             f"LIQUIDATION {o['liquidation_rate']*100:.1f}%")
    L.append(f"Exit reasons       : {o['exit_reasons']}")
    L.append(f"Ambiguous exits    : {o['ambiguous_exits']} "
             f"(SL and TP touched in the same M1 candle; resolved as "
             f"{cfg.SAME_CANDLE_RULE}. M1 intra-candle ordering is UNKNOWN.)")

    L.append("\nCOST LAYERS")
    L.append("-" * 11)
    L.append(f"{'without cost':<32}{o['pnl_without_cost']:>14.2f}")
    L.append(f"{'+ commission':<32}{o['pnl_with_commission']:>14.2f}")
    L.append(f"{'+ commission + slippage':<32}{o['pnl_with_commission_slippage']:>14.2f}")
    L.append(f"{'+ all costs (incl. funding)':<32}{o['pnl_with_all_costs']:>14.2f}")
    L.append(f"commission {o['commission']:.2f} | slippage {o['slippage']:.2f} | "
             f"funding {o['funding']:.2f} | total {o['total_costs']:.2f}")

    for title, key in [
        ("PER COIN", "per_symbol"),
        ("PER SETUP", "per_setup"),
        ("LONG / SHORT", "per_direction"),
        ("PER SESSION", "per_session"),
        ("PER MARKET REGIME", "per_regime"),
        ("DIRECTION x REGIME", "direction_by_regime"),
        ("PER LIQUIDITY TYPE", "per_liquidity_type"),
        ("PER SWEEP QUALITY", "per_sweep_quality"),
        ("PER SWEEP PENETRATION", "per_sweep_penetration"),
        ("PER DISPLACEMENT BUCKET", "per_displacement"),
        ("PER ZONE TYPE", "per_zone"),
        ("PER STRUCTURE EVENT", "per_structure"),
        ("PER M1 CONFIRMATION", "per_m1_confirmation"),
        ("PER MONTH", "per_month"),
    ]:
        L.extend(_table(title, rep[key]))

    L.append("\nCANDIDATE FUNNEL (over-filtering check)")
    L.append("-" * 38)
    total = sum(rep["rejections"].values()) or 1
    for k, v in rep["rejections"].items():
        L.append(f"  {k or 'FOUND':<28}{v:>7}  {v/total*100:>6.2f}%")

    L.append("\nCANDIDATES PER COIN (primary rejection reason)")
    L.append("-" * 46)
    for sym, hist in rep["rejections_by_symbol"].items():
        tot = sum(hist.values()) or 1
        top = "  ".join(f"{k}={v}" for k, v in list(hist.items())[:6])
        L.append(f"  {sym:<14} candidates={tot:<7} entries={hist.get('FOUND', 0)}")
        L.append(f"  {'':<14} {top}")

    L.append("\nDATA QUALITY")
    L.append("-" * 12)
    for sym, dq in rep["data_quality"].items():
        fund = "yes" if dq["funding_available"] else "NO -- FUNDING DATA UNAVAILABLE"
        L.append(f"  {sym:<14} M1={dq['m1_bars']:<8} M5={dq['m5_bars']:<7} "
                 f"M15={dq['m15_bars']:<6} gaps={dq['m1_gaps']:<5} "
                 f"bad={dq['m1_bad_candles']:<4} funding={fund}")
    L.append("")
    return "\n".join(L)


def write_all(outdir: str, bt, name: str = "backtest") -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    rep = full_report(bt)
    paths = {}
    p = os.path.join(outdir, f"{name}_report.json")
    with open(p, "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    paths["json"] = p

    p = os.path.join(outdir, f"{name}_trades.csv")
    write_trades(p, bt)
    paths["trades"] = p

    p = os.path.join(outdir, f"{name}_equity.csv")
    write_equity(p, bt)
    paths["equity"] = p

    p = os.path.join(outdir, f"{name}_symbol_equity.csv")
    write_symbol_equity(p, bt)
    paths["symbol_equity"] = p

    p = os.path.join(outdir, f"{name}_candidates.csv")
    if bt.journal.write_csv(p):
        paths["candidates"] = p
    p = os.path.join(outdir, f"{name}_states.csv")
    if bt.journal.write_states(p):
        paths["states"] = p

    txt = text_summary(rep, bt)
    p = os.path.join(outdir, f"{name}_summary.txt")
    with open(p, "w") as fh:
        fh.write(txt)
    paths["summary"] = p
    paths["_text"] = txt
    return paths
