#!/usr/bin/env python3
"""Write deterministic synthetic M1 CSVs into a data directory.

Purpose: validate a server installation end to end (download-free) before the
real Binance history is available.  These prices are NOT a market model -- any
performance number produced from them is meaningless and must never be quoted.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvis5.config import Config
from jarvis5.core.timeutil import from_iso
from jarvis5.data import binance as B
from jarvis5.data.synthetic import make_series

PRICES = {"DOGEUSDT": 0.21, "1000PEPEUSDT": 0.0125, "1000SHIBUSDT": 0.0175,
          "PUMPUSDT": 0.0043, "1000BONKUSDT": 0.0258, "WIFUSDT": 1.35}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_synthetic")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--symbols", default=",".join(Config().SYMBOLS))
    args = ap.parse_args()

    start = from_iso(args.start)
    os.makedirs(args.data_dir, exist_ok=True)
    for i, sym in enumerate([s.strip() for s in args.symbols.split(",") if s.strip()]):
        s = make_series(sym, start, args.days * 1440,
                        price=PRICES.get(sym, 1.0), seed=i + 1)
        rows = [[s.ot[j], s.o[j], s.h[j], s.l[j], s.c[j], s.v[j], s.qv[j], 0,
                 0.0, 0.0] for j in range(len(s))]
        path = B.csv_path(args.data_dir, sym)
        B.save_csv(path, rows)
        print(f"  {sym:<14} {len(rows)} candles -> {path}")
    print("\nSynthetic data only -- results from it are NOT tradeable evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
