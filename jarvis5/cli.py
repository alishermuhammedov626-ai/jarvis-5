"""JARVIS 5 command line.

    python -m jarvis5 download   --days 365
    python -m jarvis5 backtest   --days 365
    python -m jarvis5 full       --days 365      # backtest + WF + MC + sens + gate
    python -m jarvis5 selftest                   # offline pipeline check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

from .analysis import gate, montecarlo, robustness, sensitivity, walkforward
from .backtest import report as R
from .backtest.costs import FundingBook
from .backtest.runner import period_bounds, run_backtest
from .config import Config
from .core.timeutil import from_iso, to_iso
from .data import binance as B

DAY_MS = 86_400_000


# ---------------------------------------------------------------------- #
def load_config(args) -> Config:
    cfg = Config.load(args.config) if args.config else Config()
    if args.symbols:
        cfg.SYMBOLS = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for kv in (args.set or []):
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        k = k.strip()
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown config key: {k}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            val = v.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(float(v))
        elif isinstance(cur, float):
            val = float(v)
        elif isinstance(cur, list):
            val = [x.strip() for x in v.split(",") if x.strip()]
        else:
            val = v
        setattr(cfg, k, val)
    cfg.validate()
    return cfg


def resolve_period(args):
    end = from_iso(args.end) if args.end else int(time.time() * 1000) // 60000 * 60000
    if args.start:
        start = from_iso(args.start)
    else:
        start = end - args.days * DAY_MS
    return start, end


def load_data(cfg: Config, data_dir: str, start: int, end: int,
              perturb_bps: float = 0.0):
    data = {}
    missing = []
    for sym in cfg.SYMBOLS:
        path = B.csv_path(data_dir, sym)
        s = B.load_series(path, sym, start, end, perturb_bps, cfg.RANDOM_SEED)
        if len(s) == 0:
            missing.append(sym)
            continue
        data[sym] = s
    specs = B.load_specs(data_dir, cfg.SYMBOLS)
    funding = FundingBook(cfg)
    for sym in cfg.SYMBOLS:
        funding.load(sym, B.load_funding(B.funding_path(data_dir, sym)))
    return data, specs, funding, missing


# ---------------------------------------------------------------------- #
def cmd_download(args) -> int:
    cfg = load_config(args)
    start, end = resolve_period(args)
    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Downloading {to_iso(start)} -> {to_iso(end)} into {args.data_dir}/")

    try:
        info = B.download_exchange_info()
        with open(os.path.join(args.data_dir, "exchange_info.json"), "w") as fh:
            json.dump(info, fh)
        print("  exchange_info.json saved")
    except Exception as exc:
        print(f"  ! exchangeInfo failed ({exc}); fallback symbol filters will be used")

    for sym in cfg.SYMBOLS:
        path = B.csv_path(args.data_dir, sym)
        cursor = start
        if os.path.exists(path) and not args.refresh:
            last = B.last_cached_time(path)
            if last is not None and last + 60_000 >= end:
                print(f"  {sym:<14} cache already covers the period")
                cursor = None
            elif last is not None:
                cursor = max(start, last + 60_000)
                print(f"  {sym:<14} resuming from {to_iso(cursor)}")
        if cursor is None:
            pass
        else:
            try:
                rows = B.download_klines(sym, cursor, end)
            except Exception as exc:
                print(f"  {sym:<14} FAILED: {exc}")
                continue
            if os.path.exists(path) and not args.refresh:
                B.append_csv(path, rows)
            else:
                B.save_csv(path, rows)
            print(f"  {sym:<14} {len(rows)} M1 candles -> {path}")

        fpath = B.funding_path(args.data_dir, sym)
        if args.refresh or not os.path.exists(fpath):
            try:
                frows = B.download_funding(sym, start, end)
                B.save_funding(fpath, frows)
                print(f"  {sym:<14} {len(frows)} funding settlements -> {fpath}")
            except Exception as exc:
                print(f"  {sym:<14} funding FAILED: {exc} "
                      f"-- funding will be reported as UNAVAILABLE, not as 0")
    return 0


def cmd_backtest(args) -> int:
    cfg = load_config(args)
    start, end = resolve_period(args)
    data, specs, funding, missing = load_data(cfg, args.data_dir, start, end)
    if missing:
        print(f"! no cached data for: {', '.join(missing)} "
              f"(run `python -m jarvis5 download` first)")
    if not data:
        print("No data at all -- nothing to backtest.")
        return 2
    lo, hi = period_bounds(data)
    print(f"Backtesting {len(data)} symbols  {to_iso(lo)} -> {to_iso(hi)}")
    t0 = time.time()
    bt = run_backtest(cfg, data, specs, funding)
    print(f"  {len(bt.trades)} trades in {time.time()-t0:.1f}s")
    paths = R.write_all(args.out, bt, args.name)
    print(paths.pop("_text") if args.quiet else paths["_text"])
    for k, v in paths.items():
        if not k.startswith("_"):
            print(f"  {k:<14} {v}")
    return 0


def _sub_analyses(cfg, data, specs, funding, bt, args) -> Dict:
    out: Dict[str, object] = {}
    print("\nWalk-forward validation ...")
    out["walkforward"] = walkforward.run(cfg, data, specs, funding)
    v = out["walkforward"].get("verdict", {})
    print(f"  verdict: {v.get('status')}  {v}")

    if args.rolling_folds > 1:
        print(f"Rolling walk-forward ({args.rolling_folds} folds) ...")
        out["rolling_walkforward"] = walkforward.rolling(
            cfg, data, specs, funding, folds=args.rolling_folds)
        print(f"  {out['rolling_walkforward']['oos_summary']}")

    print("Monte Carlo ...")
    rs = [t.r_multiple for t in bt.trades]
    out["montecarlo"] = montecarlo.run(rs, cfg.RISK_PER_TRADE,
                                       cfg.STARTING_EQUITY, args.simulations,
                                       cfg.RANDOM_SEED)
    print(f"  {out['montecarlo'].get('final_equity', out['montecarlo'])}")

    print("Robustness ...")
    out["robustness"] = robustness.run(cfg, data, specs, funding)
    print(f"  verdict: {out['robustness']['verdict']}")

    if not args.skip_sensitivity:
        print(f"Parameter sensitivity ({len(sensitivity.SENSITIVITY_PARAMS)} "
              f"parameters x {len(sensitivity.STEPS)} runs) ...")
        out["sensitivity"] = sensitivity.run(cfg, data, specs, funding)
        for k, v in out["sensitivity"].items():
            print(f"  {k:<30} {v['verdict']['status']}")
    return out


def cmd_full(args) -> int:
    cfg = load_config(args)
    start, end = resolve_period(args)
    data, specs, funding, missing = load_data(cfg, args.data_dir, start, end)
    if missing:
        print(f"! no cached data for: {', '.join(missing)}")
    if not data:
        print("No data at all -- nothing to backtest.")
        return 2
    lo, hi = period_bounds(data)
    print(f"Primary backtest: {len(data)} symbols  {to_iso(lo)} -> {to_iso(hi)}  "
          f"leverage {cfg.LEVERAGE:g}x  risk {cfg.RISK_PER_TRADE*100:.2f}%")
    bt = run_backtest(cfg, data, specs, funding)
    paths = R.write_all(args.out, bt, args.name)
    print(paths["_text"])

    extra = _sub_analyses(cfg, data, specs, funding, bt, args)
    from .backtest.metrics import full_report
    rep = full_report(bt)
    extra["production_gate"] = gate.evaluate(
        rep, extra.get("walkforward"), extra.get("sensitivity"),
        extra.get("robustness"), extra.get("montecarlo"))

    p = os.path.join(args.out, f"{args.name}_analysis.json")
    with open(p, "w") as fh:
        json.dump(extra, fh, indent=2, default=str)

    g = extra["production_gate"]
    print("\nPRODUCTION GATE")
    print("-" * 15)
    for c in g["checks"]:
        print(f"  [{c['status']}] {c['check']:<34}{c['detail']}")
    print(f"  => {g['status']}  ({g['passed']}/{g['total']} checks passed)")
    print(f"\n  analysis json: {p}")
    return 0


def cmd_walkforward(args) -> int:
    cfg = load_config(args)
    start, end = resolve_period(args)
    data, specs, funding, _ = load_data(cfg, args.data_dir, start, end)
    res = walkforward.run(cfg, data, specs, funding)
    print(json.dumps(res, indent=2, default=str))
    return 0


def cmd_sensitivity(args) -> int:
    cfg = load_config(args)
    start, end = resolve_period(args)
    data, specs, funding, _ = load_data(cfg, args.data_dir, start, end)
    res = sensitivity.run(cfg, data, specs, funding)
    print(json.dumps(res, indent=2, default=str))
    return 0


def cmd_robustness(args) -> int:
    cfg = load_config(args)
    start, end = resolve_period(args)
    data, specs, funding, _ = load_data(cfg, args.data_dir, start, end)
    pert = None
    if args.perturb_bps:
        pert, _s, _f, _m = load_data(cfg, args.data_dir, start, end,
                                     perturb_bps=args.perturb_bps)
    res = robustness.run(cfg, data, specs, funding, pert)
    print(json.dumps(res, indent=2, default=str))
    return 0


def cmd_selftest(args) -> int:
    """Offline end-to-end check: synthetic data, full pipeline, no network."""
    from .data.synthetic import make_series
    cfg = load_config(args)
    cfg.SYMBOLS = cfg.SYMBOLS[:2]
    start = from_iso("2026-01-01")
    minutes = args.days * 1440
    data = {s: make_series(s, start, minutes, price=0.2 * (i + 1), seed=i + 1)
            for i, s in enumerate(cfg.SYMBOLS)}
    bt = run_backtest(cfg, data)
    paths = R.write_all(args.out, bt, "selftest")
    print(paths["_text"])
    print("\nSELFTEST OK -- pipeline ran end to end on synthetic data.")
    print("Synthetic prices are NOT a market model; no result above is meaningful.")
    return 0


def cmd_config(args) -> int:
    cfg = load_config(args)
    if args.write:
        cfg.save(args.write)
        print(f"config written to {args.write}")
    else:
        print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("jarvis5", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, period=True):
        q.add_argument("--config", help="JSON config file")
        q.add_argument("--symbols", help="comma separated override")
        q.add_argument("--data-dir", default="data")
        q.add_argument("--out", default="results")
        q.add_argument("--name", default="backtest")
        q.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="override any config field (repeatable)")
        if period:
            q.add_argument("--days", type=int, default=365)
            q.add_argument("--start", help="YYYY-MM-DD")
            q.add_argument("--end", help="YYYY-MM-DD")

    q = sub.add_parser("download", help="fetch klines + funding + filters")
    common(q)
    q.add_argument("--refresh", action="store_true", help="ignore existing cache")
    q.set_defaults(func=cmd_download)

    q = sub.add_parser("backtest", help="run the primary backtest")
    common(q)
    q.add_argument("--quiet", action="store_true")
    q.set_defaults(func=cmd_backtest)

    q = sub.add_parser("full", help="backtest + walk-forward + MC + sensitivity + gate")
    common(q)
    q.add_argument("--simulations", type=int, default=5000)
    q.add_argument("--rolling-folds", type=int, default=4)
    q.add_argument("--skip-sensitivity", action="store_true")
    q.set_defaults(func=cmd_full)

    q = sub.add_parser("walkforward", help="chronological train/validation/OOS")
    common(q)
    q.set_defaults(func=cmd_walkforward)

    q = sub.add_parser("sensitivity", help="parameter +/-10%% / +/-20%% sweep")
    common(q)
    q.set_defaults(func=cmd_sensitivity)

    q = sub.add_parser("robustness", help="cost / delay / perturbation scenarios")
    common(q)
    q.add_argument("--perturb-bps", type=float, default=0.0)
    q.set_defaults(func=cmd_robustness)

    q = sub.add_parser("selftest", help="offline pipeline check on synthetic data")
    common(q, period=False)
    q.add_argument("--days", type=int, default=20)
    q.set_defaults(func=cmd_selftest)

    q = sub.add_parser("config", help="print or write the effective config")
    common(q, period=False)
    q.add_argument("--write", help="path to write the config JSON to")
    q.set_defaults(func=cmd_config)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
