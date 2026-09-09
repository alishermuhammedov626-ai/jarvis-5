"""JARVIS 5 portfolio backtest engine.

One merged M1 timeline drives every symbol, so equity, the daily loss limit and
the concurrent-position cap are shared exactly as they would be on a live
account.

Per-minute execution order (spec 56) -- and NOTHING in it may read a bar that
has not closed:

     1 M15 state      (only for M15 bars whose close_time <= now)
     2 M5 state       (same rule)
     3 M1 state       (swings / structure / liquidity / sweeps / displacement)
     4 M5 setup formation
     5 zone retest + M1 confirmation
     6 hard filters -> sizing -> cost -> liquidation -> entry
     7 open positions: trailing, exits
     8 journal
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

from ..config import Config
from ..core.series import Series, aggregate
from ..core.timeutil import session_of, trading_day
from ..engine.structure import BULLISH
from ..engine.timeframe import TimeframeState
from ..strategy import confirmation as conf
from ..strategy import journal as J
from ..strategy import setups as S
from ..strategy.journal import Journal
from ..strategy.risk import SymbolSpec, plan_trade
from .costs import CostModel, FundingBook
from .position import (DATA_ERROR, END_OF_DATA, INVALIDATION, LIQUIDATION,
                       Position, TIMEOUT)

MINUTE_MS = 60_000


class SymbolContext:
    """Everything the engine holds for one symbol."""

    def __init__(self, symbol: str, m1: Series, cfg: Config, journal: Journal,
                 spec: SymbolSpec):
        self.symbol = symbol
        self.cfg = cfg
        self.spec = spec
        self.m1_series = m1
        self.m5_series = aggregate(m1, "M5")
        self.m15_series = aggregate(m1, "M15")
        self.m1 = TimeframeState(self.m1_series, cfg)
        self.m5 = TimeframeState(self.m5_series, cfg)
        self.m15 = TimeframeState(self.m15_series, cfg)
        self.factory = S.SetupFactory(symbol, cfg, self.m15, self.m5, journal)
        self.i1 = -1
        self.p5 = 0
        self.p15 = 0
        self.position: Optional[Position] = None
        self.last_exit_time = 0
        self.deferred: List[dict] = []
        self.gaps = 0
        self.last_ot = -1


class Backtester:
    def __init__(self, cfg: Config, data: Dict[str, Series],
                 specs: Optional[Dict[str, SymbolSpec]] = None,
                 funding: Optional[FundingBook] = None):
        cfg.validate()
        self.cfg = cfg
        self.journal = Journal(cfg.JOURNAL_CANDIDATES)
        self.cost = CostModel(cfg)
        self.funding = funding or FundingBook(cfg)
        specs = specs or {}
        self.ctx: Dict[str, SymbolContext] = {}
        for sym, series in data.items():
            if len(series) == 0:
                continue
            self.ctx[sym] = SymbolContext(
                sym, series, cfg, self.journal,
                specs.get(sym) or SymbolSpec(sym))

        self.equity = cfg.STARTING_EQUITY
        self.peak_equity = self.equity
        self.trades: List[Position] = []
        self.equity_curve: List[dict] = []
        self.daily_pnl: Dict[str, float] = defaultdict(float)
        self.daily_start_equity: Dict[str, float] = {}
        self.daily_trades: Dict[tuple, int] = defaultdict(int)
        self.traded_signals: set = set()
        self.blocked_days: set = set()
        self.open_positions: List[Position] = []
        self.data_quality: Dict[str, dict] = {}
        self.ambiguous_exits = 0
        self.start_ms = 0
        self.end_ms = 0

    # ------------------------------------------------------------------ #
    def _timeline(self) -> List[int]:
        stamps = set()
        for c in self.ctx.values():
            stamps.update(c.m1_series.ot)
        return sorted(stamps)

    def run(self) -> "Backtester":
        for sym, c in self.ctx.items():
            self.data_quality[sym] = {
                "m1_bars": len(c.m1_series),
                "m5_bars": len(c.m5_series),
                "m15_bars": len(c.m15_series),
                "m1_gaps": len(c.m1_series.check_continuity()),
                "m1_bad_candles": len(c.m1_series.check_sanity()),
                "funding_available": self.funding.is_available(sym),
                "first": c.m1_series.ot[0] if len(c.m1_series) else 0,
                "last": c.m1_series.ot[-1] if len(c.m1_series) else 0,
            }

        timeline = self._timeline()
        if not timeline:
            return self
        self.start_ms, self.end_ms = timeline[0], timeline[-1] + MINUTE_MS
        self.daily_start_equity[trading_day(timeline[0], self.cfg.DAILY_RESET_HOUR_UTC)] = self.equity

        ptr = {sym: 0 for sym in self.ctx}
        for t in timeline:
            for sym, c in self.ctx.items():
                p = ptr[sym]
                s = c.m1_series
                if p >= s.n or s.ot[p] != t:
                    continue
                ptr[sym] = p + 1
                self._step(c, p, t)
        self._close_open_positions()
        return self

    # ------------------------------------------------------------------ #
    def _step(self, c: SymbolContext, i: int, t: int) -> None:
        cfg = self.cfg
        now_close = t + MINUTE_MS
        day = trading_day(t, cfg.DAILY_RESET_HOUR_UTC)
        if day not in self.daily_start_equity:
            self.daily_start_equity[day] = self.equity

        # ---- 1/2 higher timeframes, only fully CLOSED bars -------------- #
        while c.p15 < c.m15_series.n and c.m15_series.close_time(c.p15) <= now_close:
            c.m15.on_bar_closed(c.p15)
            c.m15.regime()
            c.p15 += 1
        m5_closed = False
        while c.p5 < c.m5_series.n and c.m5_series.close_time(c.p5) <= now_close:
            c.m5.on_bar_closed(c.p5)
            m5_closed = True
            last5 = c.p5
            c.p5 += 1

        # ---- 3 M1 ------------------------------------------------------- #
        gap = (c.last_ot >= 0 and t - c.last_ot != MINUTE_MS)
        if gap:
            c.gaps += 1
        c.last_ot = t
        c.i1 = i
        c.m1.on_bar_closed(i)

        # ---- 7 manage open position (bars opened strictly earlier) ------ #
        if c.position is not None and c.position.entry_index < i:
            if self._manage(c, i, t):
                pass

        # ---- 4 M5 setup formation --------------------------------------- #
        if m5_closed and c.m5.ready and c.m15.ready:
            c.factory.on_m5_closed(last5)

        # ---- deferred entries (robustness ENTRY_DELAY_BARS) ------------- #
        if c.deferred:
            still = []
            for d in c.deferred:
                if i >= d["target_index"]:
                    self._try_entry(c, d["setup"], i, t, day, d["conf"])
                else:
                    still.append(d)
            c.deferred = still

        # ---- 5/6 retest -> M1 confirmation -> entry --------------------- #
        if not c.m1.ready or not c.factory.pending:
            return
        for ps in list(c.factory.pending):
            if ps.dead:
                continue
            if not conf.check_retest(ps, c.m1, i, cfg):
                continue
            if ps.state != S.S_ZONE_RETEST:
                ps.state = S.S_ZONE_RETEST
                ps.candidate.transition(S.S_ZONE_RETEST, t,
                                        f"depth={ps.correction_depth:.2f}")
                ps.candidate.data["correction_depth"] = round(ps.correction_depth, 4)
            result = conf.check_m1(ps, c.m1, i, cfg)
            if not result.ok:
                continue
            ps.candidate.transition(S.S_M1_CONFIRMATION, t, result.reason or "ok")
            ps.candidate.data.update(result.as_dict())
            if cfg.ENTRY_DELAY_BARS > 0:
                c.deferred.append({"setup": ps, "conf": result,
                                   "target_index": i + cfg.ENTRY_DELAY_BARS})
                ps.dead = True
                continue
            self._try_entry(c, ps, i, t, day, result)

    # ------------------------------------------------------------------ #
    def _try_entry(self, c: SymbolContext, ps, i: int, t: int, day: str,
                   result) -> None:
        cfg = self.cfg
        cand = ps.candidate
        direction = ps.trade_direction
        cand.transition(S.S_RISK_CHECK, t)

        # ---- hard filters (spec 28) ------------------------------------- #
        reason = self._portfolio_filters(c, ps, i, t, day)
        if reason:
            cand.reject(reason)
            self.journal.add(cand)
            ps.dead = True
            self._drop(c, ps)
            return

        ideal = c.m1.s.c[i]
        entry = self.cost.fill_price(ideal, direction, "ENTRY")
        entry = c.spec.round_price(entry)
        if entry <= 0:
            cand.reject(J.INVALID_DATA)
            self.journal.add(cand)
            ps.dead = True
            self._drop(c, ps)
            return

        m5_inv = self._m5_invalidation(c, ps)
        plan = plan_trade(direction, entry, result.invalidation_price, m5_inv,
                          c.m1.atr_now(),
                          [("M5", c.m5.liquidity), ("M15", c.m15.liquidity)],
                          self.equity, c.spec, cfg)
        cand.data.update(plan.as_dict() if plan.ok else {})
        if not plan.ok:
            cand.reject(plan.reason)
            self.journal.add(cand)
            ps.dead = True
            self._drop(c, ps)
            return
        cand.transition(S.S_COST_CHECK, t, f"rr={plan.rr:.2f}")

        # ---- entry ------------------------------------------------------ #
        entry_fee = self.cost.commission(plan.qty, entry)
        entry_slip = self.cost.slippage_cost(plan.qty, ideal, entry)
        pos = Position(c.symbol, direction, plan, t + MINUTE_MS, i,
                       ps.setup_type, self._context_snapshot(c, ps, result, plan),
                       entry_fee, entry_slip)
        pos.equity_before = self.equity
        pos.session = session_of(t, cfg.SESSIONS, cfg.TIMEZONE_OFFSET_HOURS)
        pos.day = day
        c.position = pos
        self.open_positions.append(pos)
        self.daily_trades[(c.symbol, day)] += 1
        self.traded_signals.add(ps.signal_id)

        cand.transition(S.S_ENTRY, t, f"{direction} @{entry:.10g}")
        cand.accept()
        cand.data.update({"entry_time": pos.entry_time, "session": pos.session,
                          "trade_id": pos.trade_id})
        self.journal.add(cand)
        ps.dead = True
        self._drop(c, ps)

    def _drop(self, c: SymbolContext, ps) -> None:
        try:
            c.factory.pending.remove(ps)
        except ValueError:
            pass

    def _portfolio_filters(self, c, ps, i, t, day) -> str:
        cfg = self.cfg
        if i > 0 and c.m1_series.ot[i] - c.m1_series.ot[i - 1] != MINUTE_MS:
            return J.INVALID_DATA
        if day in self.blocked_days:
            return J.DAILY_RISK_LIMIT
        start_eq = self.daily_start_equity.get(day, self.equity)
        if self.daily_pnl[day] <= -cfg.MAX_DAILY_LOSS * start_eq:
            self.blocked_days.add(day)
            return J.DAILY_RISK_LIMIT
        if self.daily_trades[(c.symbol, day)] >= cfg.MAX_TRADES_PER_SYMBOL_PER_DAY:
            return J.DAILY_TRADE_LIMIT
        if cfg.ONE_POSITION_PER_SYMBOL and c.position is not None:
            return J.ACTIVE_POSITION
        if len(self.open_positions) >= cfg.MAX_CONCURRENT_POSITIONS:
            return J.MAX_POSITIONS
        if ps.signal_id in self.traded_signals:
            return J.DUPLICATE
        if c.last_exit_time and (t - c.last_exit_time) < cfg.COOLDOWN_MINUTES * MINUTE_MS:
            return J.COOLDOWN
        if cfg.SESSION_FILTER:
            sess = session_of(t, cfg.SESSIONS, cfg.TIMEZONE_OFFSET_HOURS)
            if sess not in cfg.SESSION_FILTER:
                return J.SESSION_FILTER
        return ""

    def _m5_invalidation(self, c, ps) -> Optional[float]:
        if ps.direction == BULLISH:
            sw = c.m5.swings.last_low()
            base = min(x for x in (sw.price if sw else None, ps.sweep.extreme)
                       if x is not None)
            return base
        sw = c.m5.swings.last_high()
        return max(x for x in (sw.price if sw else None, ps.sweep.extreme)
                   if x is not None)

    def _context_snapshot(self, c, ps, result, plan) -> dict:
        ctx = dict(ps.context)
        ctx.pop("_m15_sweep", None)
        ctx.update({
            "setup_type": ps.setup_type,
            "zone": ps.zone.as_dict(),
            "sweep": ps.sweep.as_dict(),
            "m5_break": ps.event.kind if ps.event else "",
            "m5_break_scope": ps.event.scope if ps.event else "",
            "m5_mss": bool(ps.event.is_mss) if ps.event else False,
            "m5_displacement": round(ps.displacement.ratio, 4),
            "m5_displacement_grade": ps.displacement.grade,
            "correction_depth": round(ps.correction_depth, 4),
            "m15_sweep_present": bool(ps.m15_sweep),
            "confirmation": result.as_dict(),
            "plan": plan.as_dict(),
            "signal_id": ps.signal_id,
        })
        return ctx

    # ------------------------------------------------------------------ #
    def _manage(self, c: SymbolContext, i: int, t: int) -> bool:
        cfg = self.cfg
        pos = c.position
        m1 = c.m1
        pos.update_excursion(m1.s.h[i], m1.s.l[i])
        pos.bars_held += 1

        if pos.liquidation_hit(m1, i):
            self._close(c, pos, pos.liquidation_price, LIQUIDATION, t)
            return True

        exit_now = pos.scan_exit(m1, i, cfg)
        if exit_now is not None:
            self._close(c, pos, exit_now[0], exit_now[1], t)
            return True

        # the stop is moved only AFTER bar i has been scanned for exits, so a
        # trail computed from bar i's close can first be hit on bar i+1
        pos.update_trailing(m1, i, cfg)

        if cfg.EXIT_ON_INVALIDATION and pos.structure_invalidated(m1, i):
            self._close(c, pos, m1.s.c[i], INVALIDATION, t)
            return True

        held_h = (t + MINUTE_MS - pos.entry_time) / 3_600_000.0
        if held_h >= cfg.MAX_HOLD_HOURS:
            self._close(c, pos, m1.s.c[i], TIMEOUT, t)
            return True
        return False

    def _close(self, c: SymbolContext, pos: Position, ideal_exit: float,
               reason: str, t: int) -> None:
        cfg = self.cfg
        exit_price = self.cost.fill_price(ideal_exit, pos.direction, "EXIT")
        exit_price = c.spec.round_price(exit_price)
        pos.exit_price = exit_price
        pos.exit_time = t + MINUTE_MS
        pos.exit_reason = reason
        pos.holding_ms = pos.exit_time - pos.entry_time

        pos.gross_pnl = pos.unrealised(exit_price)
        pos.exit_fee = self.cost.commission(pos.qty, exit_price)
        pos.exit_slip = self.cost.slippage_cost(pos.qty, ideal_exit, exit_price)
        pos.fees = pos.entry_fee + pos.exit_fee
        pos.slippage = pos.entry_slip + pos.exit_slip
        if cfg.USE_FUNDING:
            pos.funding = self.funding.charge(pos.symbol, pos.direction,
                                              pos.notional, pos.entry_time,
                                              pos.exit_time)
        pos.net_pnl = pos.gross_pnl - pos.fees + pos.funding
        rpu = pos.risk_per_unit
        pos.r_multiple = (pos.net_pnl / (rpu * pos.qty)) if rpu > 0 and pos.qty > 0 else 0.0

        self.equity += pos.net_pnl
        pos.equity_after = self.equity
        self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_pnl[pos.day] += pos.net_pnl
        if pos.ambiguous_exit:
            self.ambiguous_exits += 1

        self.trades.append(pos)
        self.equity_curve.append({
            "time": pos.exit_time, "equity": self.equity,
            "trade_id": pos.trade_id, "symbol": pos.symbol,
            "net_pnl": pos.net_pnl, "r": pos.r_multiple,
            "drawdown": self.equity - self.peak_equity,
        })
        c.position = None
        c.last_exit_time = pos.exit_time
        try:
            self.open_positions.remove(pos)
        except ValueError:
            pass

        start_eq = self.daily_start_equity.get(pos.day, cfg.STARTING_EQUITY)
        if self.daily_pnl[pos.day] <= -cfg.MAX_DAILY_LOSS * start_eq:
            self.blocked_days.add(pos.day)

    def _close_open_positions(self) -> None:
        for sym, c in self.ctx.items():
            if c.position is None:
                continue
            i = c.i1
            self._close(c, c.position, c.m1.s.c[i], END_OF_DATA,
                        c.m1.s.ot[i])
