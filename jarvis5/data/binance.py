"""Binance USDT-M Futures historical data: klines, funding rates, filters.

Downloaded once into a local CSV cache so that a backtest is reproducible and
does not depend on the network.  Nothing here is used during the backtest loop
itself.
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple

from ..core.series import Series
from ..strategy.risk import SymbolSpec

BASE = "https://fapi.binance.com"
KLINE_LIMIT = 1500
FUNDING_LIMIT = 1000
MINUTE_MS = 60_000

CSV_HEADER = ["open_time", "open", "high", "low", "close", "volume",
              "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote"]


class DownloadError(RuntimeError):
    pass


def _get(path: str, params: dict, retries: int = 5):
    import requests
    url = BASE + path
    delay = 2.0
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):
                time.sleep(delay)
                delay *= 2
                continue
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as exc:                      # network hiccup
            last = repr(exc)
        time.sleep(delay)
        delay *= 2
    raise DownloadError(f"GET {path} {params} failed: {last}")


# ---------------------------------------------------------------------- #
# klines
# ---------------------------------------------------------------------- #
def download_klines(symbol: str, start_ms: int, end_ms: int,
                    interval: str = "1m", progress=None) -> List[list]:
    out: List[list] = []
    cursor = start_ms
    while cursor < end_ms:
        rows = _get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": KLINE_LIMIT})
        if not rows:
            break
        for r in rows:
            if r[0] >= end_ms:
                break
            out.append([int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                        float(r[4]), float(r[5]), float(r[7]), int(r[8]),
                        float(r[9]), float(r[10])])
        nxt = int(rows[-1][0]) + MINUTE_MS
        if nxt <= cursor:
            break
        cursor = nxt
        if progress:
            progress(symbol, cursor, end_ms)
        time.sleep(0.12)                     # stay well inside the weight limit
    return out


def csv_path(data_dir: str, symbol: str, interval: str = "1m") -> str:
    return os.path.join(data_dir, f"{symbol}_{interval}.csv")


def save_csv(path: str, rows: List[list]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        w.writerows(rows)


def append_csv(path: str, rows: List[list]) -> None:
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(CSV_HEADER)
        w.writerows(rows)


def last_cached_time(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            if row:
                last = int(row[0])
    return last


def load_series(path: str, symbol: str, start_ms: Optional[int] = None,
                end_ms: Optional[int] = None,
                perturb_bps: float = 0.0, seed: int = 0) -> Series:
    """Load a cached CSV into a Series, de-duplicated and time-sorted."""
    s = Series(symbol, "M1")
    if not os.path.exists(path):
        return s
    rows: List[tuple] = []
    with open(path) as fh:
        r = csv.reader(fh)
        header = next(r, None)
        for row in r:
            if not row or row[0] == "open_time":
                continue
            t = int(row[0])
            if start_ms is not None and t < start_ms:
                continue
            if end_ms is not None and t >= end_ms:
                continue
            rows.append((t, float(row[1]), float(row[2]), float(row[3]),
                         float(row[4]), float(row[5]),
                         float(row[6]) if len(row) > 6 and row[6] else 0.0))
    rows.sort(key=lambda x: x[0])
    seen = -1
    state = seed & 0xFFFFFFFF
    for t, o, h, l, c, v, qv in rows:
        if t == seen:
            continue
        seen = t
        if perturb_bps:
            # deterministic per-bar perturbation for the robustness test
            state = (1103515245 * (state ^ (t & 0xFFFFFFFF)) + 12345) & 0x7FFFFFFF
            f = 1.0 + ((state / 0x7FFFFFFF) * 2.0 - 1.0) * perturb_bps / 10_000.0
            o, h, l, c = o * f, h * f, l * f, c * f
        s.append(t, o, h, l, c, v, qv)
    return s


# ---------------------------------------------------------------------- #
# funding
# ---------------------------------------------------------------------- #
def download_funding(symbol: str, start_ms: int, end_ms: int) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    cursor = start_ms
    while cursor < end_ms:
        rows = _get("/fapi/v1/fundingRate", {
            "symbol": symbol, "startTime": cursor, "endTime": end_ms,
            "limit": FUNDING_LIMIT})
        if not rows:
            break
        for r in rows:
            out.append((int(r["fundingTime"]), float(r["fundingRate"])))
        nxt = int(rows[-1]["fundingTime"]) + 1
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.12)
    return out


def funding_path(data_dir: str, symbol: str) -> str:
    return os.path.join(data_dir, f"{symbol}_funding.csv")


def save_funding(path: str, rows: List[Tuple[int, float]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["funding_time", "funding_rate"])
        w.writerows(rows)


def load_funding(path: str) -> Optional[List[Tuple[int, float]]]:
    if not os.path.exists(path):
        return None
    out = []
    with open(path) as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            if row:
                out.append((int(row[0]), float(row[1])))
    return out or None


# ---------------------------------------------------------------------- #
# symbol filters
# ---------------------------------------------------------------------- #
DEFAULT_SPECS: Dict[str, dict] = {
    # tick / step / min_qty / min_notional -- fallbacks used when
    # exchangeInfo.json has not been downloaded
    "DOGEUSDT":     {"tick_size": 0.00001, "step_size": 1.0,  "min_qty": 1.0},
    "1000PEPEUSDT": {"tick_size": 0.0000001, "step_size": 1.0, "min_qty": 1.0},
    "1000SHIBUSDT": {"tick_size": 0.000001, "step_size": 1.0, "min_qty": 1.0},
    "PUMPUSDT":     {"tick_size": 0.0000001, "step_size": 1.0, "min_qty": 1.0},
    "1000BONKUSDT": {"tick_size": 0.0000001, "step_size": 1.0, "min_qty": 1.0},
    "WIFUSDT":      {"tick_size": 0.0001, "step_size": 0.1,  "min_qty": 0.1},
}


def download_exchange_info() -> dict:
    return _get("/fapi/v1/exchangeInfo", {})


def specs_from_exchange_info(info: dict, symbols: List[str]) -> Dict[str, SymbolSpec]:
    out: Dict[str, SymbolSpec] = {}
    for s in info.get("symbols", []):
        sym = s.get("symbol")
        if sym not in symbols:
            continue
        tick = step = 0.0
        min_qty = 0.0
        min_notional = 5.0
        max_qty = 1e12
        for f in s.get("filters", []):
            ft = f.get("filterType")
            if ft == "PRICE_FILTER":
                tick = float(f["tickSize"])
            elif ft == "LOT_SIZE":
                step = float(f["stepSize"])
                min_qty = float(f["minQty"])
                max_qty = float(f.get("maxQty", max_qty))
            elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                min_notional = float(f.get("notional", f.get("minNotional", 5.0)))
        out[sym] = SymbolSpec(sym, tick or 1e-8, step or 1.0, min_qty or 1.0,
                              min_notional, max_qty)
    return out


def load_specs(data_dir: str, symbols: List[str]) -> Dict[str, SymbolSpec]:
    path = os.path.join(data_dir, "exchange_info.json")
    if os.path.exists(path):
        with open(path) as fh:
            return specs_from_exchange_info(json.load(fh), symbols)
    out = {}
    for sym in symbols:
        d = DEFAULT_SPECS.get(sym, {})
        out[sym] = SymbolSpec(sym, d.get("tick_size", 1e-7),
                              d.get("step_size", 1.0), d.get("min_qty", 1.0),
                              d.get("min_notional", 5.0))
    return out
