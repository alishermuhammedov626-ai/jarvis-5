"""Structure-based SL, liquidity-based TP, risk-based sizing, liquidation
safety and the cost pre-check (spec 29-33, 36).

Position size comes from RISK, never from leverage:

    RiskAmount      = equity * RISK_PER_TRADE
    PositionNotional= RiskAmount / SL_fraction
    Quantity        = PositionNotional / entry      (rounded to step size)
    RequiredMargin  = PositionNotional / LEVERAGE

After exchange rounding the REAL risk is recomputed and re-validated -- the
rounded quantity, not the ideal one, is what a live account would carry.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from . import journal as J


class SymbolSpec:
    __slots__ = ("symbol", "tick_size", "step_size", "min_qty", "min_notional",
                 "max_qty")

    def __init__(self, symbol: str, tick_size: float = 0.0000001,
                 step_size: float = 1.0, min_qty: float = 1.0,
                 min_notional: float = 5.0, max_qty: float = 1e12):
        self.symbol = symbol
        self.tick_size = tick_size
        self.step_size = step_size
        self.min_qty = min_qty
        self.min_notional = min_notional
        self.max_qty = max_qty

    def round_price(self, p: float) -> float:
        if self.tick_size <= 0:
            return p
        return math.floor(p / self.tick_size + 1e-9) * self.tick_size

    def round_qty(self, q: float) -> float:
        if self.step_size <= 0:
            return q
        return math.floor(q / self.step_size + 1e-9) * self.step_size

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "tick_size": self.tick_size,
                "step_size": self.step_size, "min_qty": self.min_qty,
                "min_notional": self.min_notional}


class RiskPlan:
    __slots__ = ("ok", "reason", "entry", "sl", "tp", "sl_distance",
                 "sl_percent", "tp_distance", "tp_percent", "rr", "qty",
                 "notional", "margin", "risk_amount", "liquidation_price",
                 "entry_fee", "exit_fee", "expected_slippage", "cost_total",
                 "tp_source", "sl_source")

    def __init__(self, ok: bool, reason: str = ""):
        self.ok = ok
        self.reason = reason
        self.entry = 0.0
        self.sl = 0.0
        self.tp = 0.0
        self.sl_distance = 0.0
        self.sl_percent = 0.0
        self.tp_distance = 0.0
        self.tp_percent = 0.0
        self.rr = 0.0
        self.qty = 0.0
        self.notional = 0.0
        self.margin = 0.0
        self.risk_amount = 0.0
        self.liquidation_price = 0.0
        self.entry_fee = 0.0
        self.exit_fee = 0.0
        self.expected_slippage = 0.0
        self.cost_total = 0.0
        self.tp_source = ""
        self.sl_source = ""

    def as_dict(self) -> dict:
        return {
            "entry_price": self.entry, "sl": self.sl, "tp": self.tp,
            "sl_distance": self.sl_distance,
            "sl_percent": round(self.sl_percent, 6),
            "tp_distance": self.tp_distance,
            "tp_percent": round(self.tp_percent, 6),
            "rr": round(self.rr, 4), "position_size": self.qty,
            "notional": round(self.notional, 6),
            "margin": round(self.margin, 6),
            "risk_amount": round(self.risk_amount, 6),
            "liquidation_price": self.liquidation_price,
            "tp_source": self.tp_source, "sl_source": self.sl_source,
        }


def liquidation_price(entry: float, direction: str, leverage: float,
                      mmr: float) -> float:
    """Isolated-margin liquidation approximation for USDT-M futures."""
    if leverage <= 0:
        return 0.0
    if direction == "BUY":
        return entry * (1.0 - 1.0 / leverage + mmr)
    return entry * (1.0 + 1.0 / leverage - mmr)


def build_stop(entry: float, direction: str, invalidation: float, atr: float,
               cfg, fallback_invalidation: Optional[float] = None,
               source: str = "M1") -> Tuple[float, str, str]:
    """Return (sl_price, source, reason).  reason == '' means accepted."""
    buf = cfg.STRUCTURAL_BUFFER_ATR * atr

    def _mk(inv, src):
        sl = inv - buf if direction == "BUY" else inv + buf
        if direction == "BUY" and sl >= entry:
            return None, src, J.SL_IMPOSSIBLE
        if direction == "SELL" and sl <= entry:
            return None, src, J.SL_IMPOSSIBLE
        pct = abs(entry - sl) / entry
        if pct < cfg.MIN_SL_PERCENT:
            return sl, src, J.SL_TOO_TIGHT
        if pct > cfg.MAX_SL_PERCENT:
            return sl, src, J.SL_TOO_WIDE
        return sl, src, ""

    sl, src, reason = _mk(invalidation, source)
    if reason == "" or fallback_invalidation is None or not cfg.ALLOW_M5_INVALIDATION_FALLBACK:
        return sl, src, reason
    # M1 structure was too tight/wide -> try the M5 invalidation swing
    sl2, src2, reason2 = _mk(fallback_invalidation, "M5")
    if reason2 == "":
        return sl2, src2, reason2
    return sl, src, reason


def build_target(entry: float, direction: str, liq_engines, cfg):
    """Nearest opposing liquidity at or beyond MIN_TP_PERCENT (spec 32).

    Hierarchy: internal -> external -> equal -> session, nearest first inside
    each timeframe, M5 before M15.  A fixed candle target is never used.
    """
    min_dist = cfg.MIN_TP_PERCENT * entry
    max_dist = cfg.MAX_TP_SEARCH_PERCENT * entry
    best = None
    for tag, eng in liq_engines:
        for lv in eng.opposing_levels(entry, direction):
            dist = abs(lv.price - entry)
            if dist < min_dist or dist > max_dist:
                continue
            score = (dist, -lv.strength)
            if best is None or score < best[0]:
                best = (score, lv, tag)
    if best is not None:
        lv, tag = best[1], best[2]
        return lv.price, f"{tag}:{lv.ltype}", ""
    if cfg.TP_FALLBACK_TO_MIN:
        tp = entry * (1 + cfg.MIN_TP_PERCENT) if direction == "BUY" \
            else entry * (1 - cfg.MIN_TP_PERCENT)
        return tp, "MIN_TP_FALLBACK", ""
    return 0.0, "", J.TARGET_TOO_CLOSE


def size_position(equity: float, entry: float, sl: float, spec: SymbolSpec, cfg):
    """Risk-based sizing with exchange rounding, then risk re-validation."""
    sl_frac = abs(entry - sl) / entry
    if sl_frac <= 0:
        return 0.0, 0.0, 0.0, 0.0, J.SL_IMPOSSIBLE
    risk_amount = equity * cfg.RISK_PER_TRADE
    notional = risk_amount / sl_frac
    qty = spec.round_qty(notional / entry)
    if qty < spec.min_qty or qty <= 0:
        return 0.0, 0.0, 0.0, 0.0, J.INVALID_SIZE
    if qty > spec.max_qty:
        qty = spec.round_qty(spec.max_qty)
    notional = qty * entry
    if notional < spec.min_notional:
        return 0.0, 0.0, 0.0, 0.0, J.INVALID_SIZE
    real_risk = notional * sl_frac
    margin = notional / cfg.LEVERAGE
    if margin > equity:
        return 0.0, 0.0, 0.0, 0.0, J.INVALID_SIZE
    return qty, notional, margin, real_risk, ""


def plan_trade(direction, entry, invalidation, m5_invalidation, atr_m1,
               liq_engines, equity, spec, cfg) -> RiskPlan:
    sl, sl_src, reason = build_stop(entry, direction, invalidation, atr_m1, cfg,
                                    fallback_invalidation=m5_invalidation)
    if reason:
        return RiskPlan(False, reason)

    tp, tp_src, reason = build_target(entry, direction, liq_engines, cfg)
    if reason:
        return RiskPlan(False, reason)

    sl = spec.round_price(sl)
    tp = spec.round_price(tp)
    if direction == "BUY" and not (sl < entry < tp):
        return RiskPlan(False, J.SL_IMPOSSIBLE)
    if direction == "SELL" and not (tp < entry < sl):
        return RiskPlan(False, J.SL_IMPOSSIBLE)

    qty, notional, margin, real_risk, reason = size_position(equity, entry, sl, spec, cfg)
    if reason:
        return RiskPlan(False, reason)

    liq = liquidation_price(entry, direction, cfg.LEVERAGE, cfg.MAINTENANCE_MARGIN_RATE)
    safety = max(cfg.LIQUIDATION_SAFETY_ATR * atr_m1,
                 cfg.LIQUIDATION_SAFETY_PERCENT * entry)
    if direction == "BUY" and liq > sl - safety:
        return RiskPlan(False, J.LIQUIDATION_RISK)
    if direction == "SELL" and liq < sl + safety:
        return RiskPlan(False, J.LIQUIDATION_RISK)

    p = RiskPlan(True)
    p.entry, p.sl, p.tp = entry, sl, tp
    p.sl_distance = abs(entry - sl)
    p.sl_percent = p.sl_distance / entry
    p.tp_distance = abs(tp - entry)
    p.tp_percent = p.tp_distance / entry
    p.rr = p.tp_distance / p.sl_distance if p.sl_distance > 0 else 0.0
    p.qty, p.notional, p.margin = qty, notional, margin
    p.risk_amount = real_risk
    p.liquidation_price = liq
    p.sl_source, p.tp_source = sl_src, tp_src

    if cfg.RR_HARD_FILTER and p.rr < cfg.MIN_RR:
        return RiskPlan(False, J.RR_TOO_LOW)

    # ---- cost pre-check (spec 36) ---------------------------------------- #
    p.entry_fee = notional * cfg.COMMISSION_RATE
    p.exit_fee = qty * tp * cfg.COMMISSION_RATE
    p.expected_slippage = 2.0 * notional * (cfg.SLIPPAGE_BPS / 10_000.0)
    p.cost_total = p.entry_fee + p.exit_fee + p.expected_slippage
    gross_target = qty * p.tp_distance
    if gross_target <= 0 or p.cost_total / gross_target > cfg.MAX_COST_TO_TARGET_RATIO:
        return RiskPlan(False, J.COST_TOO_HIGH)
    return p
