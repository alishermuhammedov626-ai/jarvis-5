"""Research command line.

    python -m jarvis5.research.synthetic          # stage 1: synthetic gate
    python -m jarvis5.research --coins ... --days 365   # stage 2: real data

Stage 2 refuses to run until stage 1 passes.  It is never started
automatically: pointing the instrument at real money data is an explicit act.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

from ..backtest.costs import FundingBook
from ..core.timeutil import from_iso, to_iso
from ..data import binance as B
from . import validate
from .config import ResearchConfig
from .engine import ResearchEngine
from .report import ResearchReport

DAY_MS = 86_400_000


def load_config(args) -> ResearchConfig:
    cfg = ResearchConfig.load(args.config) if args.config else ResearchConfig()
    if getattr(args, "coins", None):
        cfg.SYMBOLS = [s.strip().upper() for s in args.coins.split(",") if s.strip()]
    if getattr(args, "days", None):
        cfg.DAYS = args.days
    for kv in (args.set or []):
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        k = k.strip()
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown research config key: {k}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            val = v.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(float(v))
        elif isinstance(cur, float):
            val = float(v)
        elif isinstance(cur, list):
            val = [float(x) if _isnum(x) else x.strip()
                   for x in v.split(",") if x.strip()]
        else:
            val = v
        setattr(cfg, k, val)
    cfg.validate()
    return cfg


def _isnum(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------- #
def cmd_synthetic(args) -> int:
    cfg = load_config(args)
    print("JARVIS 5 RESEARCH -- STAGE 1: SYNTHETIC VALIDATION")
    print("Real Binance data is NOT touched by this command.\n")
    res = validate.run(cfg, cycles=args.cycles, seed=args.seed)
    validate.print_report(res)

    outdir = os.path.join(args.out, "synthetic")
    os.makedirs(outdir, exist_ok=True)
    payload = {k: v for k, v in res.items() if k not in ("series", "truth")}
    with open(os.path.join(outdir, "synthetic_validation.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if args.full_report:
        print("\nRunning the full research pipeline on the synthetic market "
              "(report generation check)...")
        engine = ResearchEngine(cfg, funding=None, progress=True)
        engine.run({"SYNTH": res["series"]})
        rep = ResearchReport(engine, cfg)
        paths = rep.write_all(outdir)
        print(f"  wrote {len([k for k in paths if not k.startswith('_')])} files "
              f"to {outdir}/")

    print(f"\n  synthetic validation: {res['status']}")
    if res["status"] != "PASS":
        print("  Real-data research is BLOCKED until this passes.")
        return 1
    print("  Stage 2 (real data) is now unlocked, but will NOT start on its own.")
    print("  Run:  python -m jarvis5.research --coins <...> --days 365")
    return 0


def cmd_research(args) -> int:
    cfg = load_config(args)

    if not args.skip_gate:
        print("Stage 1 gate: synthetic validation ...")
        gate = validate.run(cfg, cycles=args.gate_cycles, seed=11)
        print(f"  {gate['status']}")
        if gate["status"] != "PASS":
            validate.print_report(gate)
            print("\nRefusing to run on real data: the synthetic gate failed.")
            return 1

    end = from_iso(args.end) if args.end else \
        int(time.time() * 1000) // 60_000 * 60_000
    start = from_iso(args.start) if args.start else end - cfg.DAYS * DAY_MS

    data = {}
    missing = []
    for sym in cfg.SYMBOLS:
        path = B.csv_path(args.data_dir, sym)
        s = B.load_series(path, sym, start, end)
        if len(s) == 0:
            missing.append(sym)
            continue
        data[sym] = s
    if missing:
        print(f"! no cached data for: {', '.join(missing)}")
        print(f"  run:  python -m jarvis5 download --days {cfg.DAYS} "
              f"--data-dir {args.data_dir}")
    if not data:
        print("No data at all -- nothing to research.")
        return 2

    funding = FundingBook(cfg.to_trading_config())
    for sym in cfg.SYMBOLS:
        funding.load(sym, B.load_funding(B.funding_path(args.data_dir, sym)))

    print()
    print("=" * 62)
    print("JARVIS 5 MARKET RESEARCH")
    print("=" * 62)
    print(f"Coins: {len(data)}")
    print(f"Period: {to_iso(start)} -> {to_iso(end)} "
          f"({(end-start)/DAY_MS:.0f} days requested)")
    print("Timeframe: M1")
    print("ML: DISABLED")
    print(f"Risk: {cfg.RISK_PER_TRADE*100:g}%")
    print(f"Leverage: {cfg.LEVERAGE:g}x")
    print()

    t0 = time.time()
    engine = ResearchEngine(cfg, funding=funding, progress=True)
    engine.run(data)
    print(f"\nreplay finished in {time.time()-t0:.1f}s")

    rep = ResearchReport(engine, cfg)
    outdir = args.out
    paths = rep.write_all(outdir)
    _print_summary(rep, cfg)
    print("\nfiles:")
    for k, v in sorted(paths.items()):
        if not k.startswith("_"):
            print(f"  {k:<30}{v}")
    return 0


def _print_summary(rep: ResearchReport, cfg) -> None:
    t = rep.totals()
    print()
    print("=" * 62)
    print("JARVIS 5 MARKET RESEARCH")
    print("=" * 62)
    print(f"Coins: {t['symbols']}")
    print(f"Timeframe: M1   ML: DISABLED")
    print(f"Risk: {cfg.RISK_PER_TRADE*100:g}%   Leverage: {cfg.LEVERAGE:g}x")
    print()
    print(f"Total +{cfg.TARGETS[0]*100:g}% LONG EVENTS : {t['total_long_events']}")
    print(f"Total -{cfg.TARGETS[0]*100:g}% SHORT EVENTS: {t['total_short_events']}")
    print(f"Sampled bars                : {t['sampled_bars']}")
    print(f"Baseline P(move) LONG/SHORT : "
          f"{t['baseline_long_rate']*100:.2f}% / {t['baseline_short_rate']*100:.2f}%")
    print(f"Patterns measured           : {t['patterns_measured']}")
    print()
    _table("TOP STRATEGIES (by edge over baseline)", rep.top(12))
    _table("TOP COMBINATIONS",
           rep.top(10, predicate=lambda r: r["pattern"].startswith(("C0", "C1", "C2"))))
    _table("BEST CROSS-COIN SETUPS",
           rep.top(10, predicate=lambda r: r["coin_consistency"] == "CROSS_COIN_CONSISTENT"))
    _table("OOS BEST SETUPS",
           sorted([r for r in rep.eligible(rep.primary_rows())
                   if r["oos_n"] >= cfg.MIN_SAMPLE and r["oos_edge"] > 0],
                  key=lambda r: -r["oos_edge"])[:10])
    cands = rep.candidates()
    print("\nCANDIDATES FOR THE NEXT JARVIS 5 STRATEGY")
    print("-" * 42)
    if not cands:
        print("  NONE -- no pattern cleared sample size, OOS, cross-coin")
        print("  consistency and cost-inclusive expectancy together.")
        print("  That is the result. Nothing is promoted to keep the list full.")
    else:
        for r in cands:
            print(f"  {r['pattern']:<44}{r['direction']:<6}n={r['n']:<7}"
                  f"edge={r['edge']:+.4f}  oos={r['oos_edge']:+.4f}  {r['verdict']}")


def _table(title: str, rows: List[Dict]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("  (nothing reached the minimum sample size)")
        return
    print(f"  {'pattern':<44}{'dir':<6}{'n':>7}{'rate':>8}{'base':>8}"
          f"{'edge':>9}{'lift':>7}  verdict")
    for r in rows:
        print(f"  {r['pattern'][:43]:<44}{r['direction']:<6}{r['n']:>7}"
              f"{r['conditional_rate']*100:>7.1f}%{r['baseline_rate']*100:>7.1f}%"
              f"{r['edge']:>+9.4f}{r['relative_lift']:>7.2f}  {r['verdict']}")


# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("jarvis5.research",
                                description="JARVIS 5 market move cause research")
    p.add_argument("--coins", help="comma separated symbols")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--timeframe", default="1m",
                   help="base timeframe (only 1m is supported)")
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="reports")
    p.add_argument("--config", help="research config JSON")
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.add_argument("--skip-gate", action="store_true",
                   help="skip the synthetic gate (not recommended)")
    p.add_argument("--gate-cycles", type=int, default=3)
    p.set_defaults(func=cmd_research)
    return p


def synthetic_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("jarvis5.research.synthetic",
                                description="Stage 1: synthetic validation")
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default="reports")
    p.add_argument("--config")
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.add_argument("--full-report", action="store_true",
                   help="also run the whole pipeline and write every report file")
    p.set_defaults(func=cmd_synthetic, coins=None, days=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeframe not in ("1m", "M1"):
        raise SystemExit("only the 1m base timeframe is supported "
                         "(M5 and M15 are derived from it)")
    return args.func(args)


def main_synthetic(argv: Optional[List[str]] = None) -> int:
    args = synthetic_parser().parse_args(argv)
    return args.func(args)
