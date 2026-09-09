#!/usr/bin/env python3
"""Server preflight check — run this BEFORE a long download or research run.

Answers, in one pass and without downloading anything heavy:

  * is this Python new enough and are the imports present?
  * is there enough free disk for a year of M1 data?
  * is the Binance USDT-M API reachable from this host, and is the clock sane?
  * how much history does each symbol ACTUALLY have?  (a recent listing such
    as PUMPUSDT will not have 365 days, and that has to be known up front
    rather than discovered halfway through a report)
  * how long will the download and the research pass roughly take?

Exit code 0 = ready, 1 = something will block the run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MINUTE_MS = 60_000
DAY_MS = 86_400_000
BYTES_PER_M1_ROW = 82          # measured on the cached CSV format


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--json", default=None, help="write the result here")
    ap.add_argument("--offline", action="store_true",
                    help="skip every network check")
    args = ap.parse_args()

    report = {"checks": [], "symbols": {}, "ok": True}

    def check(name, ok, detail, fatal=True):
        report["checks"].append({"check": name, "ok": bool(ok),
                                 "detail": detail, "fatal": fatal})
        if not ok and fatal:
            report["ok"] = False
        flag = "OK  " if ok else ("FAIL" if fatal else "WARN")
        print(f"  [{flag}] {name:<26}{detail}")

    print("=" * 68)
    print(" JARVIS 5 — SERVER PREFLIGHT")
    print("=" * 68)
    print()

    # ---- 1. python ---------------------------------------------------- #
    v = sys.version_info
    check("python_version", v >= (3, 8),
          f"{v.major}.{v.minor}.{v.micro} (need >= 3.8)")

    try:
        import requests                                   # noqa: F401
        check("requests_installed", True, "present")
    except ImportError:
        check("requests_installed", False,
              "missing — run: pip install -r requirements.txt")

    try:
        from jarvis5.config import Config
        from jarvis5.research.config import ResearchConfig
        cfg = Config()
        rcfg = ResearchConfig()
        check("jarvis5_importable", True,
              f"trading config {len(cfg.to_dict())} params, "
              f"research config {len(rcfg.to_dict())} params")
    except Exception as exc:
        check("jarvis5_importable", False, f"{type(exc).__name__}: {exc}")
        _finish(report, args)
        return 1

    symbols = [s.strip().upper() for s in args.symbols.split(",")] \
        if args.symbols else list(cfg.SYMBOLS)

    # ---- 2. disk ------------------------------------------------------- #
    os.makedirs(args.data_dir, exist_ok=True)
    free = shutil.disk_usage(args.data_dir).free
    need = len(symbols) * args.days * 1440 * BYTES_PER_M1_ROW
    need_total = int(need * 1.6)        # + reports, archive, working files
    check("disk_space", free > need_total,
          f"{human(free)} free, need about {human(need_total)} "
          f"({len(symbols)} symbols x {args.days} days)")

    # ---- 3. cpu / memory hint ------------------------------------------ #
    try:
        cores = os.cpu_count() or 1
        with open("/proc/meminfo") as fh:
            total_kb = int(next(l for l in fh if l.startswith("MemTotal")).split()[1])
        mem = total_kb * 1024
        check("memory", mem >= 1.5 * 1024**3,
              f"{human(mem)} total, {cores} cores "
              f"(the research pass peaks near 2GB at 6 symbols)",
              fatal=False)
    except Exception:
        check("memory", True, "could not read /proc/meminfo", fatal=False)

    # ---- 4. network / exchange ----------------------------------------- #
    if args.offline:
        check("binance_reachable", True, "skipped (--offline)", fatal=False)
        _finish(report, args)
        return 0 if report["ok"] else 1

    from jarvis5.data import binance as B
    try:
        t0 = time.time()
        srv = B._get("/fapi/v1/time", {}, retries=2)
        latency = (time.time() - t0) * 1000
        skew = abs(int(srv["serverTime"]) - int(time.time() * 1000))
        check("binance_reachable", True,
              f"fapi.binance.com responded in {latency:.0f}ms")
        check("clock_skew", skew < 60_000,
              f"{skew/1000:.1f}s between this host and the exchange",
              fatal=False)
    except Exception as exc:
        check("binance_reachable", False,
              f"{type(exc).__name__}: {str(exc)[:160]}")
        print()
        print("  The host cannot reach the Binance futures API. Common causes:")
        print("    - the datacentre IP is geo-blocked by Binance")
        print("    - an outbound firewall blocks 443 to fapi.binance.com")
        print("  Check with:  curl -sS https://fapi.binance.com/fapi/v1/time")
        _finish(report, args)
        return 1

    # ---- 5. per-symbol availability ------------------------------------ #
    print()
    print("  symbol history actually available on the exchange:")
    now = int(time.time() * 1000)
    want_start = now - args.days * DAY_MS
    for sym in symbols:
        try:
            rows = B._get("/fapi/v1/klines", {
                "symbol": sym, "interval": "1m", "startTime": 0, "limit": 1},
                retries=2)
            if not rows:
                report["symbols"][sym] = {"available": False}
                print(f"    {sym:<14} NO DATA RETURNED")
                report["ok"] = False
                continue
            first = int(rows[0][0])
            have_days = (now - first) / DAY_MS
            short = first > want_start
            report["symbols"][sym] = {
                "available": True,
                "first_candle_ms": first,
                "days_available": round(have_days, 1),
                "covers_requested_period": not short,
            }
            note = "" if not short else \
                f"  <-- only {have_days:.0f} of {args.days} days requested"
            print(f"    {sym:<14} {have_days:>7.1f} days{note}")
            time.sleep(0.15)
        except Exception as exc:
            report["symbols"][sym] = {"available": False,
                                      "error": f"{type(exc).__name__}: {exc}"}
            print(f"    {sym:<14} ERROR {type(exc).__name__}")
            report["ok"] = False

    short = [s for s, d in report["symbols"].items()
             if d.get("available") and not d.get("covers_requested_period")]
    if short:
        print()
        print(f"  NOTE: {', '.join(short)} listed less than {args.days} days ago.")
        print("  The download takes whatever exists and data_quality.csv records")
        print("  the real span. Their sample sizes will be smaller — read them as")
        print("  such rather than comparing them to the older coins one-to-one.")

    # ---- 6. time estimates ---------------------------------------------- #
    total_minutes = sum(min(args.days, d.get("days_available", 0)) * 1440
                        for d in report["symbols"].values() if d.get("available"))
    requests_needed = total_minutes / 1500
    dl_seconds = requests_needed * 0.42          # request + pacing sleep
    bt_seconds = total_minutes / 9_000           # measured backtest throughput
    rs_seconds = total_minutes / 1_400           # measured research throughput
    report["estimates"] = {
        "m1_candles": int(total_minutes),
        "download_seconds": int(dl_seconds),
        "backtest_seconds": int(bt_seconds),
        "research_seconds": int(rs_seconds),
        "disk_bytes": int(total_minutes * BYTES_PER_M1_ROW),
    }
    print()
    print(f"  estimated work: {int(total_minutes):,} M1 candles, "
          f"{human(total_minutes * BYTES_PER_M1_ROW)} of CSV")
    print(f"    download  ~{hms(dl_seconds)}")
    print(f"    backtest  ~{hms(bt_seconds)}")
    print(f"    research  ~{hms(rs_seconds)}   (stride 5; --set BASELINE_STRIDE=15 is ~3x faster)")

    _finish(report, args)
    return 0 if report["ok"] else 1


def _finish(report, args) -> None:
    print()
    print("=" * 68)
    print(f" PREFLIGHT: {'READY' if report['ok'] else 'BLOCKED'}")
    print("=" * 68)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f" written to {args.json}")


if __name__ == "__main__":
    raise SystemExit(main())
