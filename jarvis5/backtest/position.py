"""Open position lifecycle: trailing, exits, MFE/MAE (spec 34, 35, 37, 54).

Exit ambiguity rule: when one M1 candle touches BOTH the stop and the target we
cannot know which came first without tick data.  We take the conservative
branch (SL first) and flag the trade -- never the flattering one.
"""
from __future__ import annotations

from typing import Optional

TP = "TP"
SL = "SL"
TRAILING_STOP = "TRAILING_STOP"
INVALIDATION = "INVALIDATION"
TIMEOUT = "TIMEOUT"
LIQUIDATION = "LIQUIDATION"
DATA_ERROR = "DATA_ERROR"
END_OF_DATA = "END_OF_DATA"


class Position:
    __slots__ = ("trade_id", "symbol", "direction", "entry", "entry_time",
                 "entry_index", "qty", "notional", "margin", "sl", "tp",
                 "initial_sl", "risk_amount", "liquidation_price", "setup_type",
                 "context", "mfe", "mae", "mfe_r", "mae_r", "trailing_active",
                 "exit_price", "exit_time", "exit_reason", "gross_pnl",
                 "fees", "funding", "slippage", "net_pnl", "r_multiple",
                 "holding_ms", "ambiguous_exit", "bars_held", "trail_moves",
                 "last_trail_swing_index", "entry_fee", "exit_fee",
                 "entry_slip", "exit_slip", "equity_before", "equity_after",
                 "session", "day")

    _seq = 0

    def __init__(self, symbol, direction, plan, entry_time, entry_index,
                 setup_type, context, entry_fee, entry_slip):
        Position._seq += 1
        self.trade_id = Position._seq
        self.symbol = symbol
        self.direction = direction
        self.entry = plan.entry
        self.entry_time = entry_time
        self.entry_index = entry_index
        self.qty = plan.qty
        self.notional = plan.notional
        self.margin = plan.margin
        self.sl = plan.sl
        self.initial_sl = plan.sl
        self.tp = plan.tp
        self.risk_amount = plan.risk_amount
        self.liquidation_price = plan.liquidation_price
        self.setup_type = setup_type
        self.context = context
        self.mfe = 0.0
        self.mae = 0.0
        self.mfe_r = 0.0
        self.mae_r = 0.0
        self.trailing_active = False
        self.trail_moves = 0
        self.last_trail_swing_index = -1
        self.exit_price = 0.0
        self.exit_time = 0
        self.exit_reason = ""
        self.gross_pnl = 0.0
        self.entry_fee = entry_fee
        self.exit_fee = 0.0
        self.entry_slip = entry_slip
        self.exit_slip = 0.0
        self.fees = entry_fee
        self.funding = 0.0
        self.slippage = entry_slip
        self.net_pnl = 0.0
        self.r_multiple = 0.0
        self.holding_ms = 0
        self.bars_held = 0
        self.ambiguous_exit = False
        self.equity_before = 0.0
        self.equity_after = 0.0
        self.session = ""
        self.day = ""

    # -- helpers -------------------------------------------------------- #
    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.initial_sl)

    def unrealised(self, price: float) -> float:
        if self.direction == "BUY":
            return (price - self.entry) * self.qty
        return (self.entry - price) * self.qty

    def update_excursion(self, high: float, low: float) -> None:
        rpu = self.risk_per_unit
        if self.direction == "BUY":
            fav = high - self.entry
            adv = self.entry - low
        else:
            fav = self.entry - low
            adv = high - self.entry
        if fav > self.mfe:
            self.mfe = fav
            self.mfe_r = fav / rpu if rpu > 0 else 0.0
        if adv > self.mae:
            self.mae = adv
            self.mae_r = adv / rpu if rpu > 0 else 0.0

    # -- trailing (spec 34) --------------------------------------------- #
    # NOTE: a stop moved at the close of bar i can only ever be *hit* from bar
    # i+1 onwards -- applying it to bar i's own high/low would be look-ahead.
    def update_trailing(self, m1, i: int, cfg) -> None:
        if not cfg.TRAILING_ENABLED:
            return
        price = m1.s.c[i]
        move = (price - self.entry) / self.entry if self.direction == "BUY" \
            else (self.entry - price) / self.entry
        if not self.trailing_active:
            if move < cfg.TRAILING_ACTIVATION_PERCENT:
                return
            self.trailing_active = True

        atr = m1.atr_now()
        buf = cfg.TRAILING_STRUCTURE_BUFFER_ATR * atr

        if cfg.PROFIT_LOCK_ENABLED and move >= cfg.PROFIT_LOCK_TRIGGER_PERCENT:
            lock = self.entry * (1 + cfg.PROFIT_LOCK_PERCENT) if self.direction == "BUY" \
                else self.entry * (1 - cfg.PROFIT_LOCK_PERCENT)
            self._tighten(lock)

        # structure trail: a NEW confirmed M1 higher-low / lower-high
        if self.direction == "BUY":
            for sw in reversed(m1.swings.lows):
                if sw.index <= self.entry_index or sw.index <= self.last_trail_swing_index:
                    break
                if sw.price > self.entry - self.risk_per_unit:
                    self.last_trail_swing_index = sw.index
                    self._tighten(sw.price - buf)
                break
        else:
            for sw in reversed(m1.swings.highs):
                if sw.index <= self.entry_index or sw.index <= self.last_trail_swing_index:
                    break
                if sw.price < self.entry + self.risk_per_unit:
                    self.last_trail_swing_index = sw.index
                    self._tighten(sw.price + buf)
                break

    def _tighten(self, new_sl: float) -> None:
        """A stop may only ever move in the trade's favour."""
        if self.direction == "BUY":
            if new_sl > self.sl and new_sl < self.entry + self.risk_per_unit * 50:
                self.sl = new_sl
                self.trail_moves += 1
        else:
            if new_sl < self.sl and new_sl > self.entry - self.risk_per_unit * 50:
                self.sl = new_sl
                self.trail_moves += 1

    # -- exit scan (spec 37) -------------------------------------------- #
    def scan_exit(self, m1, i: int, cfg):
        """Return (exit_price, reason) or None, using only bar i's OHLC."""
        o, h, l = m1.s.o[i], m1.s.h[i], m1.s.l[i]
        if self.direction == "BUY":
            # gap through the stop -> filled at the open, not at the stop
            if o <= self.sl:
                return o, (TRAILING_STOP if self.trail_moves else SL)
            if o >= self.tp:
                return o, TP
            hit_sl = l <= self.sl
            hit_tp = h >= self.tp
        else:
            if o >= self.sl:
                return o, (TRAILING_STOP if self.trail_moves else SL)
            if o <= self.tp:
                return o, TP
            hit_sl = h >= self.sl
            hit_tp = l <= self.tp

        if hit_sl and hit_tp:
            self.ambiguous_exit = True
            if cfg.SAME_CANDLE_RULE == "TP_FIRST":
                return self.tp, TP
            if cfg.SAME_CANDLE_RULE == "SKIP":
                return None
            return self.sl, (TRAILING_STOP if self.trail_moves else SL)
        if hit_sl:
            return self.sl, (TRAILING_STOP if self.trail_moves else SL)
        if hit_tp:
            return self.tp, TP
        return None

    def liquidation_hit(self, m1, i: int) -> bool:
        if self.liquidation_price <= 0:
            return False
        if self.direction == "BUY":
            return m1.s.l[i] <= self.liquidation_price
        return m1.s.h[i] >= self.liquidation_price

    def structure_invalidated(self, m1, i: int) -> bool:
        """Opposite structural break on M1 while in the trade."""
        want = "BEARISH" if self.direction == "BUY" else "BULLISH"
        for ev in m1.last_events:
            if ev.direction == want and ev.scope == "EXTERNAL":
                return True
        return False
