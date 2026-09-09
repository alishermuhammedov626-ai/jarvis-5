"""Thin helpers for running one backtest and for slicing data by period."""
from __future__ import annotations

from typing import Dict, Optional

from ..config import Config
from ..core.series import Series
from .costs import FundingBook
from .engine import Backtester


def slice_data(data: Dict[str, Series], start_ms: Optional[int],
               end_ms: Optional[int]) -> Dict[str, Series]:
    out = {}
    for sym, s in data.items():
        sub = s.slice(start_ms, end_ms)
        if len(sub):
            out[sym] = sub
    return out


def run_backtest(cfg: Config, data: Dict[str, Series], specs=None,
                 funding: Optional[FundingBook] = None) -> Backtester:
    return Backtester(cfg, data, specs, funding).run()


def period_bounds(data: Dict[str, Series]):
    lo = None
    hi = None
    for s in data.values():
        if not len(s):
            continue
        lo = s.ot[0] if lo is None else min(lo, s.ot[0])
        hi = s.ot[-1] if hi is None else max(hi, s.ot[-1])
    return lo, hi
