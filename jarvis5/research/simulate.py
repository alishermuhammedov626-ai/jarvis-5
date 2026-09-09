"""Hypothetical entry simulation (spec 19-22).

RESEARCH ONLY.  This does not place, plan or influence a single trade in the
JARVIS 5 trading engine -- it exists so that "this pattern precedes a 1 % move
72 % of the time" can be turned into "and here is what that would actually have
paid after a structural stop, real fees and real slippage".

One simulation is run per sampled bar per direction, and its outcome is then
attributed to every pattern active at that bar.  So competing patterns are
compared on identical trades rather than on separately tuned ones.
"""
from __future__ import annotations

from typing import Dict, List, Optional

SL_TOO_TIGHT = "SL_TOO_TIGHT"
SL_TOO_WIDE = "SL_TOO_WIDE"
SL_IMPOSSIBLE = "SL_IMPOSSIBLE"
NO_STRUCTURE = "NO_STRUCTURE"
OK = "OK"


class SimResult:
    __slots__ = ("ok", "reason", "direction", "entry", "sl", "sl_pct", "qty",
                 "notional", "risk_amount", "outcomes", "trailing", "mfe_r",
                 "mae_r", "fees", "slippage")

    def __init__(self, ok: bool, reason: str = ""):
        self.ok = ok
        self.reason = reason
        self.direction = ""
        self.entry = 0.0
        self.sl = 0.0
        self.sl_pct = 0.0
        self.qty = 0.0
        self.notional = 0.0
        self.risk_amount = 0.0
        self.mfe_r = 0.0
        self.mae_r = 0.0
        self.fees = 0.0
        self.slippage = 0.0
        # target key -> {"exit", "r_net", "r_gross", "bars"}
        self.outcomes: Dict[str, dict] = {}
        self.trailing: Dict[str, dict] = {}


def structural_stop(replay, i: int, direction: str, cfg) -> Optional[float]:
    """Most recent CONFIRMED M1 swing on the invalidation side."""
    sw_engine = replay.m1.state.swings
    price = replay.m1.s.c[i]
    if direction == "LONG":
        cands = [s.price for s in sw_engine.lows[-8:] if s.price < price]
        return max(cands) if cands else None
    cands = [s.price for s in sw_engine.highs[-8:] if s.price > price]
    return min(cands) if cands else None


def simulate(replay, i: int, direction: str, cfg,
             funding=None, symbol: str = "") -> SimResult:
    s = replay.m1.s
    atr = replay.m1.state.atr_now()
    if atr <= 0:
        return SimResult(False, NO_STRUCTURE)

    inval = structural_stop(replay, i, direction, cfg)
    if inval is None:
        return SimResult(False, NO_STRUCTURE)

    ideal = s.c[i]
    slip = cfg.SLIPPAGE_BPS / 10_000.0
    entry = ideal * (1.0 + slip) if direction == "LONG" else ideal * (1.0 - slip)
    buf = cfg.STRUCTURAL_BUFFER_ATR * atr
    sl = inval - buf if direction == "LONG" else inval + buf
    if direction == "LONG" and sl >= entry:
        return SimResult(False, SL_IMPOSSIBLE)
    if direction == "SHORT" and sl <= entry:
        return SimResult(False, SL_IMPOSSIBLE)

    sl_pct = abs(entry - sl) / entry
    if sl_pct < cfg.MIN_SL_PERCENT:
        return SimResult(False, SL_TOO_TIGHT)
    if sl_pct > cfg.MAX_SL_PERCENT:
        return SimResult(False, SL_TOO_WIDE)

    risk_amount = cfg.STARTING_EQUITY * cfg.RISK_PER_TRADE
    notional = risk_amount / sl_pct
    qty = notional / entry

    res = SimResult(True, OK)
    res.direction = direction
    res.entry = entry
    res.sl = sl
    res.sl_pct = sl_pct
    res.qty = qty
    res.notional = notional
    res.risk_amount = risk_amount

    risk_per_unit = abs(entry - sl)
    horizon = min(s.n - 1, i + cfg.SIM_HORIZON_BARS)

    # ---- fixed-target outcomes (one scan resolves every target) --------- #
    targets = {}
    for t in cfg.SIM_TARGETS:
        key = f"{t*100:g}pct"
        tp = entry * (1 + t) if direction == "LONG" else entry * (1 - t)
        targets[key] = {"tp": tp, "exit": "", "bars": 0, "price": 0.0}

    best = entry
    worst = entry
    for j in range(i + 1, horizon + 1):
        hi, lo, op = s.h[j], s.l[j], s.o[j]
        if direction == "LONG":
            best = max(best, hi)
            worst = min(worst, lo)
            hit_sl = lo <= sl
            gap_sl = op <= sl
        else:
            best = min(best, lo)
            worst = max(worst, hi)
            hit_sl = hi >= sl
            gap_sl = op >= sl

        for key, tg in targets.items():
            if tg["exit"]:
                continue
            tp = tg["tp"]
            if direction == "LONG":
                hit_tp = hi >= tp
                gap_tp = op >= tp
            else:
                hit_tp = lo <= tp
                gap_tp = op <= tp
            if gap_sl:
                tg.update(exit="SL", bars=j - i, price=op)
            elif gap_tp:
                tg.update(exit="TP", bars=j - i, price=op)
            elif hit_sl and hit_tp:
                # intra-candle ordering is unknowable -> conservative branch
                if cfg.SAME_CANDLE_RULE == "TP_FIRST":
                    tg.update(exit="TP", bars=j - i, price=tp)
                else:
                    tg.update(exit="SL", bars=j - i, price=sl)
            elif hit_sl:
                tg.update(exit="SL", bars=j - i, price=sl)
            elif hit_tp:
                tg.update(exit="TP", bars=j - i, price=tp)
        if all(t["exit"] for t in targets.values()):
            break

    for key, tg in targets.items():
        if not tg["exit"]:
            tg.update(exit="TIMEOUT", bars=horizon - i, price=s.c[horizon])
        res.outcomes[key] = _score(res, tg["price"], tg["exit"], tg["bars"],
                                   risk_per_unit, cfg, funding, symbol,
                                   s.ot[i], s.ot[min(i + tg["bars"], s.n - 1)])

    if direction == "LONG":
        res.mfe_r = (best - entry) / risk_per_unit
        res.mae_r = (entry - worst) / risk_per_unit
    else:
        res.mfe_r = (entry - best) / risk_per_unit
        res.mae_r = (worst - entry) / risk_per_unit

    # ---- trailing variant (spec 22: measure it, do not assume it) ------- #
    res.trailing = {"ALL": _simulate_trailing(replay, i, direction, entry, sl,
                                              qty, risk_per_unit, cfg, horizon,
                                              funding, symbol)}
    return res


def _score(res: SimResult, exit_price: float, exit_reason: str, bars: int,
           risk_per_unit: float, cfg, funding, symbol: str,
           entry_ms: int, exit_ms: int) -> dict:
    slip = cfg.SLIPPAGE_BPS / 10_000.0
    filled = exit_price * (1.0 - slip) if res.direction == "LONG" \
        else exit_price * (1.0 + slip)
    if res.direction == "LONG":
        gross = (exit_price - res.entry) * res.qty
        net_px = (filled - res.entry) * res.qty
    else:
        gross = (res.entry - exit_price) * res.qty
        net_px = (res.entry - filled) * res.qty
    fees = (res.notional + res.qty * filled) * cfg.COMMISSION_RATE
    fund = 0.0
    if cfg.USE_FUNDING and funding is not None and funding.is_available(symbol):
        fund = funding.charge(symbol, "BUY" if res.direction == "LONG" else "SELL",
                              res.notional, entry_ms, exit_ms)
    net = net_px - fees + fund
    denom = risk_per_unit * res.qty
    return {
        "exit": exit_reason,
        "bars": bars,
        "r_gross": round(gross / denom, 6) if denom else 0.0,
        "r_net": round(net / denom, 6) if denom else 0.0,
        "pnl_gross": round(gross, 6),
        "pnl_net": round(net, 6),
        "fees": round(fees, 6),
        "funding": round(fund, 6),
    }


def _simulate_trailing(replay, i: int, direction: str, entry: float, sl: float,
                       qty: float, risk_per_unit: float, cfg, horizon: int,
                       funding, symbol: str) -> dict:
    """Same entry and stop, but the stop follows confirmed M1 structure."""
    s = replay.m1.s
    swings = replay.m1.state.swings
    atr = replay.m1.state.atr_now()
    buf = cfg.TRAILING_BUFFER_ATR * atr
    cur_sl = sl
    active = False
    exit_price = 0.0
    exit_reason = ""
    bars = 0

    # confirmed swings usable while the trade is open, with their publish delay
    lows = [(sw.index, sw.price) for sw in swings.lows[-40:]]
    highs = [(sw.index, sw.price) for sw in swings.highs[-40:]]

    for j in range(i + 1, horizon + 1):
        hi, lo, op, cl = s.h[j], s.l[j], s.o[j], s.c[j]
        if direction == "LONG":
            if op <= cur_sl:
                exit_price, exit_reason, bars = op, "SL", j - i
                break
            if lo <= cur_sl:
                exit_price, exit_reason, bars = cur_sl, "SL", j - i
                break
        else:
            if op >= cur_sl:
                exit_price, exit_reason, bars = op, "SL", j - i
                break
            if hi >= cur_sl:
                exit_price, exit_reason, bars = cur_sl, "SL", j - i
                break

        move = (cl - entry) / entry if direction == "LONG" else (entry - cl) / entry
        if not active and move >= cfg.TRAILING_ACTIVATION_PERCENT:
            active = True
        if active:
            # only swings that had already been CONFIRMED by bar j may be used
            if direction == "LONG":
                usable = [p for (idx, p) in lows
                          if idx + cfg.SWING_K <= j and p - buf > cur_sl and p < cl]
                if usable:
                    cur_sl = max(usable) - buf
            else:
                usable = [p for (idx, p) in highs
                          if idx + cfg.SWING_K <= j and p + buf < cur_sl and p > cl]
                if usable:
                    cur_sl = min(usable) + buf
    if not exit_reason:
        exit_price, exit_reason, bars = s.c[horizon], "TIMEOUT", horizon - i

    ref = SimResult(True)
    ref.direction = direction
    ref.entry = entry
    ref.qty = qty
    ref.notional = qty * entry
    return _score(ref, exit_price, exit_reason, bars, risk_per_unit, cfg,
                  funding, symbol, s.ot[i], s.ot[min(i + bars, s.n - 1)])
