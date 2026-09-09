"""Research reporting: CSV tables, JSON and the final Markdown answer sheet.

Every table carries its SAMPLE SIZE and its BASELINE next to the headline
number, because a win rate without those two is not a result.  Nothing is
padded to a round number of signals; a pattern with 17 occurrences is reported
with 17.
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from ..core.timeutil import to_iso
from . import combos as C
from .stats import (DISCOVERY, INSUFFICIENT, OOS, VALIDATION, classify,
                    coin_consistency, edge_stats, rank_key)


def _pf(pos: float, neg: float) -> float:
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return round(pos / abs(neg), 4)


def _rate(a: int, b: int) -> float:
    return round(a / b, 6) if b else 0.0


# ---------------------------------------------------------------------- #
def build_rows(agg, cfg, scope: str = "ALL") -> List[Dict]:
    """One row per (pattern, window, direction) within a scope."""
    primary = f"{cfg.TARGETS[0]*100:g}pct"
    sim_keys = [f"{t*100:g}pct" for t in cfg.SIM_TARGETS]
    rows: List[Dict] = []
    for (feature, direction), buckets in agg.feature_windows(scope).items():
        # a window row is emitted only where NEW data arrives; without that,
        # a pattern seen only within 5 minutes would print six identical rows
        windows = ["STATE"] if "STATE" in buckets else \
            [str(w) for w in cfg.WINDOWS if str(w) in buckets]
        for window in windows:
            cell = agg.window_cell(scope, feature, window, direction)
            if cell is None or cell.n == 0:
                continue
            rows.append(_row(agg, cfg, scope, feature, window, direction, cell,
                             primary, sim_keys))
    return rows


def _row(agg, cfg, scope, feature, window, direction, cell, primary,
         sim_keys) -> Dict:
    base_n, base_hits = agg.baseline(scope, direction, primary)
    e = edge_stats(cell.n, cell.hits.get(primary, 0), base_n, base_hits)
    row: Dict[str, object] = {
        "pattern": feature,
        "window_minutes": window,
        "direction": direction,
        "scope": scope,
        "total_occurrences": cell.n,
        "n": cell.n,
        "sample_flag": _sample_flag(cell.n, cfg),
    }
    for t in cfg.TARGETS:
        key = f"{t*100:g}pct"
        h = cell.hits.get(key, 0)
        row[f"successful_{key}_moves"] = h
        row[f"win_rate_{key}"] = _rate(h, cell.n)
    row.update({
        "conditional_rate": e["conditional_rate"],
        "baseline_rate": e["baseline_rate"],
        "edge": e["edge"],
        "relative_lift": e["relative_lift"],
        "odds_ratio": e["odds_ratio"],
        "average_MFE": round(cell.mfe_mean, 6),
        "median_MFE": round(cell.mfe_median(), 6),
        "average_MAE": round(cell.mae_mean, 6),
        "median_MAE": round(cell.mae_median(), 6),
        "average_time_to_1pct": round(cell.t1_mean, 2),
        "median_time_to_1pct": round(cell.t1_median(), 2),
    })
    for key in sim_keys:
        n = cell.sim_n.get(key, 0)
        w = cell.sim_wins.get(key, 0)
        r = cell.sim_r.get(key, 0.0)
        rg = cell.sim_r_gross.get(key, 0.0)
        row[f"sim_trades_{key}"] = n
        row[f"sim_win_rate_{key}"] = _rate(w, n)
        row[f"expectancy_r_{key}"] = round(r / n, 6) if n else 0.0
        row[f"expectancy_r_gross_{key}"] = round(rg / n, 6) if n else 0.0
        row[f"profit_factor_{key}"] = _pf(cell.sim_r_pos.get(key, 0.0),
                                          cell.sim_r_neg.get(key, 0.0))
        row[f"net_pnl_{key}"] = round(cell.sim_pnl.get(key, 0.0), 4)
    row["expectancy_r_1pct"] = row.get(f"expectancy_r_{sim_keys[0]}", 0.0)
    row["cost_drag_r"] = round(
        row.get(f"expectancy_r_gross_{sim_keys[0]}", 0.0)
        - row.get(f"expectancy_r_{sim_keys[0]}", 0.0), 6)
    row["max_r_drawdown"] = round(cell.r_maxdd, 4)
    row["trailing_expectancy_r"] = round(cell.trail_r / cell.trail_n, 6) \
        if cell.trail_n else 0.0
    row["trailing_trades"] = cell.trail_n
    return row


def _sample_flag(n: int, cfg) -> str:
    if n < cfg.MIN_SAMPLE:
        return "INSUFFICIENT SAMPLE"
    if n < cfg.EXPLORATORY_SAMPLE:
        return "EXPLORATORY"
    return "OK"


def segment_edges(agg, cfg, feature: str, window: str, direction: str) -> Dict:
    """Discovery / validation / OOS edge for one pattern."""
    primary = f"{cfg.TARGETS[0]*100:g}pct"
    out = {}
    for seg in (DISCOVERY, VALIDATION, OOS):
        scope = f"SEG:{seg}"
        cell = agg.window_cell(scope, feature, window, direction)
        if cell is None or cell.n == 0:
            out[seg] = {"n": 0, "edge": 0.0, "conditional_rate": 0.0,
                        "baseline_rate": 0.0}
            continue
        bn, bh = agg.baseline(scope, direction, primary)
        e = edge_stats(cell.n, cell.hits.get(primary, 0), bn, bh)
        out[seg] = {"n": cell.n, "edge": e["edge"],
                    "conditional_rate": e["conditional_rate"],
                    "baseline_rate": e["baseline_rate"],
                    "relative_lift": e["relative_lift"]}
    return out


def per_coin_edges(agg, cfg, feature: str, window: str, direction: str,
                   symbols: List[str]) -> Dict:
    primary = f"{cfg.TARGETS[0]*100:g}pct"
    out = {}
    for sym in symbols:
        scope = f"SYM:{sym}"
        cell = agg.window_cell(scope, feature, window, direction)
        if cell is None or cell.n == 0:
            continue
        bn, bh = agg.baseline(scope, direction, primary)
        e = edge_stats(cell.n, cell.hits.get(primary, 0), bn, bh)
        out[sym] = {"n": cell.n, "edge": e["edge"],
                    "win_rate": e["conditional_rate"],
                    "baseline": e["baseline_rate"]}
    return out


def enrich(rows: List[Dict], agg, cfg, symbols: List[str]) -> List[Dict]:
    """Attach segment verdicts and coin consistency to every row."""
    for row in rows:
        segs = segment_edges(agg, cfg, row["pattern"], str(row["window_minutes"]),
                             row["direction"])
        row["discovery_n"] = segs[DISCOVERY]["n"]
        row["discovery_edge"] = segs[DISCOVERY]["edge"]
        row["validation_n"] = segs[VALIDATION]["n"]
        row["validation_edge"] = segs[VALIDATION]["edge"]
        row["oos_n"] = segs[OOS]["n"]
        row["oos_edge"] = segs[OOS]["edge"]
        row["oos_win_rate"] = segs[OOS]["conditional_rate"]
        row["oos_baseline"] = segs[OOS]["baseline_rate"]
        row["verdict"] = classify(row["n"], row["edge"], row["relative_lift"],
                                  cfg, segs[VALIDATION], segs[OOS])
        pc = per_coin_edges(agg, cfg, row["pattern"],
                            str(row["window_minutes"]), row["direction"], symbols)
        cc = coin_consistency(pc, cfg)
        row["coin_consistency"] = cc["status"]
        row["positive_coins"] = cc["positive_coins"]
        row["voting_coins"] = cc["voting_coins"]
        row["coin_detail"] = json.dumps(cc["detail"])
    return rows


# ---------------------------------------------------------------------- #
def write_csv(path: str, rows: List[Dict], fields: Optional[List[str]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as fh:
            fh.write("")
        return
    if fields is None:
        fields = []
        for r in rows[:500]:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def is_combo(name: str) -> bool:
    base = name.rsplit("_", 1)[0]
    return base in C.COMBO_NAMES or base in C.SETUP_NAMES


class ResearchReport:
    def __init__(self, engine, cfg):
        self.engine = engine
        self.cfg = cfg
        self.agg = engine.agg
        self.symbols = list(engine.results.keys())
        self.primary = f"{cfg.TARGETS[0]*100:g}pct"
        self.rows = enrich(build_rows(self.agg, cfg, "ALL"), self.agg, cfg,
                           self.symbols)
        self.rows.sort(key=rank_key)

    # -- helpers ------------------------------------------------------- #
    def primary_rows(self) -> List[Dict]:
        w = str(self.cfg.PRIMARY_WINDOW)
        return [r for r in self.rows if str(r["window_minutes"]) in (w, "STATE")]

    def eligible(self, rows=None) -> List[Dict]:
        rows = self.rows if rows is None else rows
        return [r for r in rows if r["n"] >= self.cfg.MIN_SAMPLE]

    def top(self, n: int, rows=None, predicate=None) -> List[Dict]:
        rows = self.eligible(self.primary_rows() if rows is None else rows)
        if predicate:
            rows = [r for r in rows if predicate(r)]
        return sorted(rows, key=rank_key)[:n]

    def totals(self) -> Dict:
        long_events = sum(len(r.long_events) for r in self.engine.results.values())
        short_events = sum(len(r.short_events) for r in self.engine.results.values())
        bn_l, bh_l = self.agg.baseline("ALL", "LONG", self.primary)
        bn_s, bh_s = self.agg.baseline("ALL", "SHORT", self.primary)
        return {
            "symbols": len(self.symbols),
            "total_long_events": long_events,
            "total_short_events": short_events,
            "sampled_bars": self.agg.sampled,
            "baseline_long_rate": _rate(bh_l, bn_l),
            "baseline_short_rate": _rate(bh_s, bn_s),
            "patterns_measured": len({r["pattern"] for r in self.rows}),
            "rows": len(self.rows),
        }

    # -- writers -------------------------------------------------------- #
    def write_all(self, outdir: str) -> Dict[str, str]:
        cfg = self.cfg
        os.makedirs(outdir, exist_ok=True)
        paths: Dict[str, str] = {}

        def P(name: str) -> str:
            p = os.path.join(outdir, name)
            paths[name] = p
            return p

        t = self.totals()
        write_csv(P("overall_summary.csv"), [t])

        coin_rows = []
        for sym, r in self.engine.results.items():
            bn_l, bh_l = self.agg.baseline(f"SYM:{sym}", "LONG", self.primary)
            bn_s, bh_s = self.agg.baseline(f"SYM:{sym}", "SHORT", self.primary)
            top = r.event_patterns.most_common(6)
            coin_rows.append({
                "symbol": sym, "m1_bars": r.bars, "sampled_bars": r.sampled,
                "long_events": len(r.long_events),
                "short_events": len(r.short_events),
                "baseline_long_rate": _rate(bh_l, bn_l),
                "baseline_short_rate": _rate(bh_s, bn_s),
                "avg_long_mfe": round(_mean([e.mfe_pct for e in r.long_events]), 6),
                "avg_short_mfe": round(_mean([e.mfe_pct for e in r.short_events]), 6),
                "top_pre_event_patterns": ";".join(f"{k}={v}" for k, v in top),
                "seconds": round(r.elapsed, 2),
            })
        write_csv(P("coin_summary.csv"), coin_rows)

        write_csv(P("strategy_summary.csv"), self.rows)
        write_csv(P("combination_summary.csv"),
                  [r for r in self.rows if is_combo(r["pattern"])])

        by_coin = []
        for sym in self.symbols:
            sub = enrich(build_rows(self.agg, cfg, f"SYM:{sym}"), self.agg, cfg,
                         self.symbols)
            for r in sub:
                r["symbol"] = sym
            by_coin.extend(sub)
        write_csv(P("strategy_by_coin.csv"), by_coin)

        by_dir = []
        seen = {}
        for r in self.rows:
            key = (r["pattern"], r["window_minutes"])
            seen.setdefault(key, {})[r["direction"]] = r
        for (pat, win), d in seen.items():
            row = {"pattern": pat, "window_minutes": win}
            for side in ("LONG", "SHORT"):
                s = d.get(side)
                row[f"{side}_n"] = s["n"] if s else 0
                row[f"{side}_win_rate"] = s["conditional_rate"] if s else 0.0
                row[f"{side}_baseline"] = s["baseline_rate"] if s else 0.0
                row[f"{side}_edge"] = s["edge"] if s else 0.0
                row[f"{side}_expectancy_r"] = s["expectancy_r_1pct"] if s else 0.0
                row[f"{side}_verdict"] = s["verdict"] if s else INSUFFICIENT
            row["direction_consistency"] = (
                "BOTH" if row["LONG_edge"] > 0 and row["SHORT_edge"] > 0 else
                "LONG_ONLY" if row["LONG_edge"] > 0 else
                "SHORT_ONLY" if row["SHORT_edge"] > 0 else "NEITHER")
            by_dir.append(row)
        write_csv(P("strategy_by_direction.csv"), by_dir)

        ev_rows = []
        for sym, r in self.engine.results.items():
            for direction, pool in (("LONG", r.long_events),
                                    ("SHORT", r.short_events)):
                if not pool:
                    continue
                row = {"symbol": sym, "direction": direction, "events": len(pool)}
                for tgt in cfg.TARGETS:
                    k = f"{tgt*100:g}pct"
                    row[f"reached_{k}"] = sum(1 for e in pool
                                              if e.targets_hit.get(k))
                    row[f"reached_{k}_rate"] = _rate(row[f"reached_{k}"], len(pool))
                    times = [e.time_to[k] for e in pool if e.time_to.get(k, -1) >= 0]
                    row[f"median_minutes_to_{k}"] = round(_median(times), 1)
                row["avg_mfe"] = round(_mean([e.mfe_pct for e in pool]), 6)
                row["median_mfe"] = round(_median([e.mfe_pct for e in pool]), 6)
                row["avg_mae"] = round(_mean([e.mae_pct for e in pool]), 6)
                row["avg_duration_bars"] = round(
                    _mean([float(e.duration_bars) for e in pool]), 1)
                ev_rows.append(row)
        write_csv(P("event_summary.csv"), ev_rows)

        oos_rows = [{
            "pattern": r["pattern"], "window_minutes": r["window_minutes"],
            "direction": r["direction"], "n": r["n"],
            "discovery_n": r["discovery_n"], "discovery_edge": r["discovery_edge"],
            "validation_n": r["validation_n"], "validation_edge": r["validation_edge"],
            "oos_n": r["oos_n"], "oos_edge": r["oos_edge"],
            "oos_win_rate": r["oos_win_rate"], "oos_baseline": r["oos_baseline"],
            "verdict": r["verdict"], "coin_consistency": r["coin_consistency"],
        } for r in self.rows]
        write_csv(P("oos_summary.csv"), oos_rows)

        write_csv(P("rejection_summary.csv"),
                  [{"reason": k, "count": v,
                    "share": _rate(v, sum(self.agg.rejections.values()))}
                   for k, v in sorted(self.agg.rejections.items(),
                                      key=lambda kv: -kv[1])])

        write_csv(P("data_quality.csv"),
                  [r.data_quality for r in self.engine.results.values()])

        write_csv(P("top_setups.csv"), self.top(cfg.TOP_N))

        db = []
        for sym, r in self.engine.results.items():
            db.extend(r.event_rows)
        db.sort(key=lambda x: (x["symbol"], x["event_start"]))
        write_csv(P("full_event_database.csv"), db)

        report = {
            "config": cfg.to_dict(),
            "totals": t,
            "coins": coin_rows,
            "events": ev_rows,
            "top_setups": self.top(cfg.TOP_N),
            "rejections": dict(self.agg.rejections),
            "data_quality": [r.data_quality for r in self.engine.results.values()],
            "answers": self.answers(),
        }
        with open(P("full_report.json"), "w") as fh:
            json.dump(report, fh, indent=2, default=str)

        md = self.markdown()
        with open(P("FINAL_REPORT.md"), "w") as fh:
            fh.write(md)
        paths["_markdown"] = md
        return paths

    # -- the 30 questions (spec 28) ------------------------------------- #
    def answers(self) -> Dict[str, object]:
        cfg = self.cfg
        t = self.totals()
        A: Dict[str, object] = {}
        A["1_total_long_1pct_events"] = t["total_long_events"]
        A["2_total_short_1pct_events"] = t["total_short_events"]
        A["3_events_per_coin"] = {
            sym: {"long": len(r.long_events), "short": len(r.short_events)}
            for sym, r in self.engine.results.items()}

        pre = Counter()
        for r in self.engine.results.values():
            for k, v in r.event_patterns.items():
                pre[k] += v
        A["4_most_frequent_patterns_before_moves"] = [
            {"direction": k.split("|", 1)[0], "pattern": k.split("|", 1)[1],
             "count": v} for k, v in pre.most_common(25)]

        A["5_highest_conditional_edge"] = _brief(self.top(15))
        A["6_best_smc_setup"] = _brief(self.top(
            5, predicate=lambda r: r["pattern"].startswith(("M1_", "M5_", "M15_"))))
        A["7_best_wyckoff_setup"] = _brief(self.top(
            5, predicate=lambda r: "WY_" in r["pattern"]))
        A["8_rsi_edge"] = _brief(self.top(
            8, predicate=lambda r: r["pattern"].startswith("RSI_")
            or "_RSI_" in r["pattern"]))
        A["9_volume_edge"] = _brief(self.top(
            8, predicate=lambda r: r["pattern"].startswith("VOLUME_")
            or "VOL" in r["pattern"]))
        A["10_price_action_edge"] = _brief(self.top(
            8, predicate=lambda r: r["pattern"].startswith("PA_")))
        A["11_liquidity_sweep_strength"] = _brief(self.find_all("C01_LIQUIDITY_SWEEP"))
        A["12_sweep_plus_mss"] = _brief(self.find_all("C04_SWEEP_MSS"))
        A["13_sweep_mss_fvg"] = _brief(self.find_all("C07_SWEEP_MSS_FVG"))
        A["14_m15_m5_m1_model"] = _brief(
            self.find_all("C14_M15_SWEEP_M5_MSS_M1_CONF")
            + self.find_all("C11_M15_SWEEP_M5_MSS"))
        A["15_double_liquidity"] = _brief(
            self.find_all("C15_DOUBLE_SWEEP") + self.find_all("C16_DOUBLE_SWEEP_MSS")
            + self.find_all("SETUP_D_DOUBLE_LIQUIDITY"))

        rev = self.find_all("SETUP_A_REVERSAL") + self.find_all("SETUP_B_RANGE_REVERSAL")
        cont = self.find_all("SETUP_C_TREND_CONTINUATION")
        A["16_reversal_vs_continuation"] = {
            "reversal": _brief(rev), "continuation": _brief(cont),
            "verdict": _compare(rev, cont, "reversal", "continuation")}

        longs = [r for r in self.eligible(self.primary_rows()) if r["direction"] == "LONG"]
        shorts = [r for r in self.eligible(self.primary_rows()) if r["direction"] == "SHORT"]
        A["17_long_vs_short"] = {
            "long_mean_edge": round(_mean([r["edge"] for r in longs]), 6),
            "short_mean_edge": round(_mean([r["edge"] for r in shorts]), 6),
            "long_baseline": t["baseline_long_rate"],
            "short_baseline": t["baseline_short_rate"],
            "verdict": _compare(longs, shorts, "LONG", "SHORT")}

        coin_score = {}
        for sym in self.symbols:
            sub = self.eligible(enrich(build_rows(self.agg, cfg, f"SYM:{sym}"),
                                       self.agg, cfg, self.symbols))
            coin_score[sym] = round(_mean([r["edge"] for r in sub]), 6)
        ranked = sorted(coin_score.items(), key=lambda kv: -kv[1])
        A["18_best_coin"] = ranked[0] if ranked else None
        A["19_worst_coin"] = ranked[-1] if ranked else None
        A["coin_mean_edge"] = dict(ranked)

        cross = [r for r in self.eligible(self.primary_rows())
                 if r["coin_consistency"] == "CROSS_COIN_CONSISTENT"]
        specific = [r for r in self.eligible(self.primary_rows())
                    if r["coin_consistency"] == "COIN_SPECIFIC" and r["edge"] > 0]
        A["20_cross_coin_stable_setups"] = _brief(sorted(cross, key=rank_key)[:15])
        A["21_coin_specific_setups"] = _brief(sorted(specific, key=rank_key)[:15])

        A["22_best_to_1pct"] = _brief(self.top(10))
        A["23_best_to_larger_targets"] = {
            f"{tg*100:g}pct": _brief(sorted(
                self.eligible(self.primary_rows()),
                key=lambda r: -r.get(f"win_rate_{tg*100:g}pct", 0.0))[:8],
                extra=[f"win_rate_{tg*100:g}pct"])
            for tg in cfg.TARGETS[1:]}
        A["24_lowest_mae"] = _brief(sorted(
            self.eligible(self.primary_rows()),
            key=lambda r: r["average_MAE"])[:10], extra=["average_MAE"])
        A["25_highest_mfe"] = _brief(sorted(
            self.eligible(self.primary_rows()),
            key=lambda r: -r["average_MFE"])[:10], extra=["average_MFE"])

        sim_key = f"{cfg.SIM_TARGETS[0]*100:g}pct"
        profitable = [r for r in self.eligible(self.primary_rows())
                      if r.get(f"expectancy_r_{sim_key}", 0.0) > 0
                      and r.get(f"sim_trades_{sim_key}", 0) >= cfg.MIN_SAMPLE]
        A["26_profitable_after_costs"] = _brief(
            sorted(profitable, key=lambda r: -r[f"expectancy_r_{sim_key}"])[:15],
            extra=[f"expectancy_r_{sim_key}", f"expectancy_r_gross_{sim_key}",
                   "cost_drag_r", f"profit_factor_{sim_key}"])
        A["27_hypothetical_backtest"] = self.simulation_summary()
        A["28_survived_oos"] = _brief([r for r in self.eligible(self.primary_rows())
                                       if r["verdict"] in ("ROBUST", "PROMISING")
                                       and r["oos_n"] >= cfg.MIN_SAMPLE
                                       and r["oos_edge"] > 0][:15],
                                      extra=["oos_edge", "oos_n"])
        A["29_overfit_risk"] = _brief([r for r in self.rows
                                       if r["verdict"] in ("OVERFIT RISK",
                                                           "OVERFIT / FAILED OOS")][:15],
                                      extra=["discovery_edge", "oos_edge"])
        A["30_candidates_for_jarvis5"] = _brief(self.candidates(), extra=[
            "oos_edge", "oos_n", "coin_consistency", f"expectancy_r_{sim_key}"])
        return A

    def find_all(self, base: str) -> List[Dict]:
        w = str(self.cfg.PRIMARY_WINDOW)
        return [r for r in self.rows
                if r["pattern"].rsplit("_", 1)[0] == base
                and str(r["window_minutes"]) == w]

    def candidates(self) -> List[Dict]:
        """Patterns that survive every honesty filter (spec 36)."""
        cfg = self.cfg
        sim_key = f"{cfg.SIM_TARGETS[0]*100:g}pct"
        out = []
        for r in self.primary_rows():
            if r["n"] < cfg.EXPLORATORY_SAMPLE:
                continue
            if r["verdict"] not in ("ROBUST", "PROMISING"):
                continue
            if r["oos_n"] < cfg.MIN_SAMPLE or r["oos_edge"] <= 0:
                continue
            # SINGLE_COIN is deliberately excluded: one coin is not evidence
            if r["coin_consistency"] not in ("CROSS_COIN_CONSISTENT", "PARTIAL"):
                continue
            if r.get(f"expectancy_r_{sim_key}", 0.0) <= 0:
                continue
            out.append(r)
        return sorted(out, key=rank_key)[:20]

    def simulation_summary(self) -> Dict:
        cfg = self.cfg
        out = {"risk_per_trade": cfg.RISK_PER_TRADE, "leverage": cfg.LEVERAGE,
               "note": "research simulation on every sampled bar, not a strategy "
                       "backtest; no daily limits or setup filters are applied",
               "targets": {}}
        for t in cfg.SIM_TARGETS:
            key = f"{t*100:g}pct"
            n = wins = 0
            r_net = r_gross = pos = neg = 0.0
            for direction in ("LONG", "SHORT"):
                cell = self.agg.totals.get(("ALL", direction))
                if cell is None:
                    continue
                n += cell.sim_n.get(key, 0)
                wins += cell.sim_wins.get(key, 0)
                r_net += cell.sim_r.get(key, 0.0)
                r_gross += cell.sim_r_gross.get(key, 0.0)
                pos += cell.sim_r_pos.get(key, 0.0)
                neg += cell.sim_r_neg.get(key, 0.0)
            out["targets"][key] = {
                "trades": n, "win_rate": _rate(wins, n),
                "expectancy_r_net": round(r_net / n, 6) if n else 0.0,
                "expectancy_r_gross": round(r_gross / n, 6) if n else 0.0,
                "profit_factor": _pf(pos, neg),
            }
        tn = tr = 0.0
        for direction in ("LONG", "SHORT"):
            cell = self.agg.totals.get(("ALL", direction))
            if cell:
                tn += cell.trail_n
                tr += cell.trail_r
        out["with_trailing"] = {
            "trades": int(tn),
            "expectancy_r_net": round(tr / tn, 6) if tn else 0.0,
            "note": f"trailing activates at +{cfg.TRAILING_ACTIVATION_PERCENT*100:g}%"
        }
        funding = [r.data_quality.get("funding_available")
                   for r in self.engine.results.values()]
        out["funding"] = "AVAILABLE" if all(funding) and funding else "UNAVAILABLE"
        return out

    # -- markdown -------------------------------------------------------- #
    def markdown(self) -> str:
        cfg = self.cfg
        t = self.totals()
        A = self.answers()
        L: List[str] = []
        L.append("# JARVIS 5 — Market Move Cause Research\n")
        L.append("**Deterministic rule-based pattern research. "
                 "No ML, no AI, no probability model.**\n")
        L.append(f"- Coins: {len(self.symbols)} — {', '.join(self.symbols)}")
        L.append(f"- Timeframe: M1 (M5 / M15 derived)")
        L.append(f"- Move definition: first touch of ±{cfg.TARGETS[0]*100:g}% "
                 f"within {cfg.EVENT_HORIZON_BARS} minutes")
        L.append(f"- Sampling: every {cfg.BASELINE_STRIDE}th bar "
                 f"({t['sampled_bars']} sampled bars)")
        L.append(f"- Risk {cfg.RISK_PER_TRADE*100:g}% · leverage {cfg.LEVERAGE:g}x "
                 f"· commission {cfg.COMMISSION_RATE*100:g}%/side "
                 f"· slippage {cfg.SLIPPAGE_BPS:g}bps/side\n")

        L.append("## How to read this report\n")
        L.append("The headline number for a pattern is **not** its win rate. It is "
                 "the **edge**: `P(move | pattern) − P(move | no pattern)`, where "
                 "the control group is every other sampled bar. A pattern that "
                 "precedes 70% of moves in a market that moves 65% of the time "
                 "has an edge of 5 points, not 70.\n")
        L.append(f"Baseline rates here: LONG **{t['baseline_long_rate']*100:.2f}%**, "
                 f"SHORT **{t['baseline_short_rate']*100:.2f}%** — that is the "
                 f"probability of a ±{cfg.TARGETS[0]*100:g}% move from a randomly "
                 f"chosen bar. Every conditional rate must be compared to it.\n")
        L.append(f"Sample-size honesty: n < {cfg.MIN_SAMPLE} is reported as "
                 f"`INSUFFICIENT SAMPLE`, n < {cfg.EXPLORATORY_SAMPLE} as "
                 f"`EXPLORATORY`. Nothing is padded to a round number of signals.\n")

        L.append("## Answers\n")
        qs = [
            ("1. How many +1% long moves across the coins?", A["1_total_long_1pct_events"]),
            ("2. How many −1% short moves?", A["2_total_short_1pct_events"]),
        ]
        for q, v in qs:
            L.append(f"**{q}** {v}\n")

        L.append("**3. Per coin**\n")
        L.append("| coin | +1% events | −1% events |")
        L.append("|---|---|---|")
        for sym, d in A["3_events_per_coin"].items():
            L.append(f"| {sym} | {d['long']} | {d['short']} |")
        L.append("")

        L.append("**4. Most frequent patterns before a move** "
                 "(frequency only — see question 5 for whether they mean anything)\n")
        L.append("| move direction | pattern | count |")
        L.append("|---|---|---|")
        for d in A["4_most_frequent_patterns_before_moves"][:20]:
            L.append(f"| {d['direction']} | {d['pattern']} | {d['count']} |")
        L.append("")

        L.append("**5. Highest conditional edge**\n")
        L.extend(_md_table(A["5_highest_conditional_edge"]))

        for qnum, title, key in [
            (6, "Best SMC setup", "6_best_smc_setup"),
            (7, "Best Wyckoff setup", "7_best_wyckoff_setup"),
            (8, "Does RSI give a real edge?", "8_rsi_edge"),
            (9, "Does volume give a real edge?", "9_volume_edge"),
            (10, "Where is price action strong?", "10_price_action_edge"),
            (11, "How strong is a liquidity sweep alone?", "11_liquidity_sweep_strength"),
            (12, "How strong is Sweep + MSS?", "12_sweep_plus_mss"),
            (13, "How strong is Sweep + MSS + FVG?", "13_sweep_mss_fvg"),
            (14, "Does the M15 → M5 → M1 model work?", "14_m15_m5_m1_model"),
            (15, "Does the double-liquidity setup work?", "15_double_liquidity"),
        ]:
            L.append(f"**{qnum}. {title}**\n")
            L.extend(_md_table(A[key]))

        L.append("**16. Reversal vs continuation**\n")
        L.append(f"{A['16_reversal_vs_continuation']['verdict']}\n")
        L.append("**17. Long vs short**\n")
        r17 = A["17_long_vs_short"]
        L.append(f"- mean edge LONG {r17['long_mean_edge']:+.4f} "
                 f"(baseline {r17['long_baseline']*100:.2f}%)")
        L.append(f"- mean edge SHORT {r17['short_mean_edge']:+.4f} "
                 f"(baseline {r17['short_baseline']*100:.2f}%)")
        L.append(f"- {r17['verdict']}\n")

        best, worst = A["18_best_coin"], A["19_worst_coin"]
        L.append(f"**18. Best coin** "
                 f"{best[0] if best else 'n/a'} (mean edge "
                 f"{best[1]:+.4f})\n" if best else "**18. Best coin** n/a\n")
        L.append(f"**19. Worst coin** "
                 f"{worst[0] if worst else 'n/a'} (mean edge "
                 f"{worst[1]:+.4f})\n" if worst else "**19. Worst coin** n/a\n")
        L.append("Mean edge across measured patterns, per coin:\n")
        L.append("| coin | mean edge |")
        L.append("|---|---|")
        for sym, sc in A["coin_mean_edge"].items():
            L.append(f"| {sym} | {sc:+.4f} |")
        L.append("")

        L.append("**20. Setups stable across coins**\n")
        L.extend(_md_table(A["20_cross_coin_stable_setups"]))
        L.append("**21. Setups that work on one coin only**\n")
        L.extend(_md_table(A["21_coin_specific_setups"]))
        L.append("**22. Best to the 1% target**\n")
        L.extend(_md_table(A["22_best_to_1pct"]))

        L.append("**23. Best to larger targets**\n")
        for tgt, rows in A["23_best_to_larger_targets"].items():
            L.append(f"\n*{tgt} target*\n")
            L.extend(_md_table(rows))

        L.append("**24. Lowest MAE**\n")
        L.extend(_md_table(A["24_lowest_mae"]))
        L.append("**25. Highest MFE**\n")
        L.extend(_md_table(A["25_highest_mfe"]))
        L.append("**26. Still profitable after commission and slippage**\n")
        L.extend(_md_table(A["26_profitable_after_costs"]))

        L.append("**27. Hypothetical simulation at "
                 f"{cfg.RISK_PER_TRADE*100:g}% risk and {cfg.LEVERAGE:g}x**\n")
        s27 = A["27_hypothetical_backtest"]
        L.append(f"_{s27['note']}_\n")
        L.append("| target | trades | win rate | expectancy R (net) | "
                 "expectancy R (gross) | profit factor |")
        L.append("|---|---|---|---|---|---|")
        for k, v in s27["targets"].items():
            L.append(f"| {k} | {v['trades']} | {v['win_rate']*100:.2f}% | "
                     f"{v['expectancy_r_net']:+.4f} | {v['expectancy_r_gross']:+.4f} "
                     f"| {v['profit_factor']} |")
        L.append("")
        L.append(f"With trailing (+{cfg.TRAILING_ACTIVATION_PERCENT*100:g}% "
                 f"activation): {s27['with_trailing']['trades']} trades, "
                 f"expectancy {s27['with_trailing']['expectancy_r_net']:+.4f} R\n")
        L.append(f"Funding data: **{s27['funding']}**\n")

        L.append("**28. Survived out-of-sample**\n")
        L.extend(_md_table(A["28_survived_oos"]))
        L.append("**29. Overfit risk / failed OOS**\n")
        L.extend(_md_table(A["29_overfit_risk"]))
        L.append("**30. Candidates for the next JARVIS 5 strategy**\n")
        cands = A["30_candidates_for_jarvis5"]
        if not cands:
            L.append("**None.** No pattern cleared every filter "
                     f"(n ≥ {cfg.EXPLORATORY_SAMPLE}, positive OOS edge with "
                     f"n ≥ {cfg.MIN_SAMPLE}, cross-coin consistency, and positive "
                     "cost-inclusive expectancy). That is a result, not a gap: "
                     "on this data none of the measured patterns earned promotion "
                     "to a trading rule.\n")
        else:
            L.extend(_md_table(cands))

        L.append("## Data quality\n")
        L.append("| coin | M1 bars | days | missing | malformed | funding |")
        L.append("|---|---|---|---|---|---|")
        for r in self.engine.results.values():
            d = r.data_quality
            L.append(f"| {d['symbol']} | {d['m1_bars']} | {d.get('days', 0)} | "
                     f"{d['missing_candles']} | {d['malformed_candles']} | "
                     f"{'yes' if d['funding_available'] else '**UNAVAILABLE**'} |")
        L.append("")
        L.append("## Method limits\n")
        L.append("- Sampled bars overlap in time, so a pattern's occurrences are "
                 "autocorrelated. Edge and lift are unbiased point estimates; "
                 "their confidence intervals are *wider* than an independent-sample "
                 "formula would suggest. Treat small edges with caution.")
        L.append("- Hundreds of patterns × windows × directions are measured, so "
                 "some will look good by chance. That is exactly why "
                 "discovery / validation / OOS are reported separately and why "
                 "nothing is promoted on the discovery number alone.")
        L.append("- The simulation enters on **every** sampled bar. It measures "
                 "what a pattern is worth, not what JARVIS 5 would trade: no "
                 "daily limits, no cooldown, no setup gating.")
        L.append("- Event outcomes deliberately use future bars. Pattern features "
                 "never do — see `tests/test_research.py` for the truncation "
                 "invariance and scale invariance proofs.\n")
        return "\n".join(L)


# ---------------------------------------------------------------------- #
def _brief(rows: List[Dict], extra: Optional[List[str]] = None) -> List[Dict]:
    base = ["pattern", "direction", "window_minutes", "n", "sample_flag",
            "conditional_rate", "baseline_rate", "edge", "relative_lift",
            "odds_ratio", "verdict", "coin_consistency", "expectancy_r_1pct"]
    keys = base + (extra or [])
    return [{k: r.get(k) for k in keys} for r in rows]


def _md_table(rows: List[Dict]) -> List[str]:
    if not rows:
        return ["_no pattern reached the minimum sample size._\n"]
    keys = list(rows[0].keys())
    out = ["| " + " | ".join(keys) + " |",
           "|" + "|".join("---" for _ in keys) + "|"]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v).replace("|", "/"))   # keep the table intact
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def _compare(a: List[Dict], b: List[Dict], name_a: str, name_b: str) -> str:
    ea = _mean([r["edge"] for r in a])
    eb = _mean([r["edge"] for r in b])
    na = sum(r["n"] for r in a)
    nb = sum(r["n"] for r in b)
    if not a and not b:
        return "neither side reached a usable sample"
    if not a:
        return f"only {name_b} reached a usable sample"
    if not b:
        return f"only {name_a} reached a usable sample"
    winner = name_a if ea > eb else name_b
    return (f"{winner} has the higher mean edge "
            f"({name_a} {ea:+.4f} on n={na}, {name_b} {eb:+.4f} on n={nb})")


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    m = len(xs) // 2
    return float(xs[m]) if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2.0
