"""Commission, slippage and funding (spec 36-38).

Commission is always charged on NOTIONAL, never on margin -- at 17x leverage
those differ by 17x and getting it wrong flatters the result enormously.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class CostModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.slip = cfg.SLIPPAGE_BPS / 10_000.0

    # -- slippage ------------------------------------------------------- #
    def fill_price(self, price: float, direction: str, side: str) -> float:
        """Always fill on the unfavourable side of the intended price.

        side: 'ENTRY' or 'EXIT'.
        BUY entry fills higher, SELL entry fills lower; exits mirror that.
        """
        s = self.slip
        if s <= 0:
            return price
        buying = (direction == "BUY" and side == "ENTRY") or \
                 (direction == "SELL" and side == "EXIT")
        return price * (1.0 + s) if buying else price * (1.0 - s)

    def slippage_cost(self, qty: float, ideal: float, actual: float) -> float:
        return abs(actual - ideal) * qty

    # -- commission ----------------------------------------------------- #
    def commission(self, qty: float, price: float) -> float:
        return qty * price * self.cfg.COMMISSION_RATE


class FundingBook:
    """Historical funding rates per symbol, keyed by settlement timestamp."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.rates: Dict[str, Dict[int, float]] = {}
        self.available: Dict[str, bool] = {}

    def load(self, symbol: str, rows: Optional[List[Tuple[int, float]]]) -> None:
        if not rows:
            self.available[symbol] = False
            self.rates[symbol] = {}
            return
        self.available[symbol] = True
        self.rates[symbol] = {int(t): float(r) for t, r in rows}

    def is_available(self, symbol: str) -> bool:
        return bool(self.available.get(symbol))

    def settlements(self, symbol: str, start_ms: int, end_ms: int):
        """Funding settlements strictly inside (start, end]."""
        book = self.rates.get(symbol)
        if not book:
            return []
        return sorted((t, r) for t, r in book.items() if start_ms < t <= end_ms)

    def charge(self, symbol: str, direction: str, notional: float,
               start_ms: int, end_ms: int) -> float:
        """Signed funding PnL: positive means the position RECEIVED funding.

        Longs pay a positive funding rate, shorts receive it.
        """
        total = 0.0
        for _t, rate in self.settlements(symbol, start_ms, end_ms):
            pay = notional * rate
            total += -pay if direction == "BUY" else pay
        return total
