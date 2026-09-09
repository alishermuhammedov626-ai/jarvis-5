"""The no-look-ahead contract (spec 4), tested rather than asserted.

The headline test is TRUNCATION INVARIANCE: if the engine peeked at even one
future bar, cutting the data feed at time T would change decisions made before
T.  It must not.
"""
import unittest

from jarvis5.backtest.runner import run_backtest, slice_data
from jarvis5.config import Config
from jarvis5.core.series import Series, aggregate
from jarvis5.core.timeutil import from_iso
from jarvis5.data.synthetic import make_series
from jarvis5.engine.atr import ATR
from jarvis5.engine.swings import SwingEngine

START = from_iso("2026-01-01")
MIN = 60_000


def _cfg(**kw):
    c = Config()
    c.SYMBOLS = ["DOGEUSDT", "WIFUSDT"]
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _key(t):
    return (t.symbol, t.direction, t.entry_time, round(t.entry, 12),
            t.exit_time, round(t.exit_price, 12), round(t.net_pnl, 8),
            t.exit_reason)


class TestTruncationInvariance(unittest.TestCase):
    def test_cutting_the_future_changes_nothing_in_the_past(self):
        cfg = _cfg()
        data = {s: make_series(s, START, 45 * 1440, price=0.2 * (i + 1), seed=i + 1)
                for i, s in enumerate(cfg.SYMBOLS)}
        full = run_backtest(cfg, data)
        self.assertGreaterEqual(len(full.trades), 6,
                                "synthetic fixture produced too few trades to test")

        # cut right after the median trade so both halves are populated
        exits = sorted(t.exit_time for t in full.trades)
        cut = exits[len(exits) // 2] + 5 * MIN
        truncated = run_backtest(cfg, slice_data(data, None, cut))

        a = [_key(t) for t in full.trades if t.exit_time <= cut]
        b = [_key(t) for t in truncated.trades if t.exit_time <= cut]
        self.assertGreater(len(a), 0, "test needs at least one closed trade")
        self.assertEqual(a, b,
                         "trades closed before the cut differ once future data "
                         "is removed -> look-ahead leak")


class TestDeterminism(unittest.TestCase):
    def test_same_inputs_same_output(self):
        cfg = _cfg()
        data = {s: make_series(s, START, 10 * 1440, price=0.2 * (i + 1), seed=i + 1)
                for i, s in enumerate(cfg.SYMBOLS)}
        one = run_backtest(cfg, data)
        two = run_backtest(cfg, data)
        self.assertEqual([_key(t) for t in one.trades],
                         [_key(t) for t in two.trades])
        self.assertAlmostEqual(one.equity, two.equity, places=10)


class TestScaleInvariance(unittest.TestCase):
    """Every threshold is ATR-normalised, so the same shape at a different
    price level must produce the same trades (tick rounding aside)."""

    def test_price_level_does_not_change_decisions(self):
        from jarvis5.strategy.risk import SymbolSpec
        cfg = _cfg()
        results = []
        for mult in (1.0, 50.0):
            data = {s: make_series(s, START, 30 * 1440,
                                   price=0.2 * mult * (i + 1), seed=i + 1)
                    for i, s in enumerate(cfg.SYMBOLS)}
            specs = {s: SymbolSpec(s, 1e-12, 1e-9, 1e-9, 0.0) for s in cfg.SYMBOLS}
            bt = run_backtest(cfg, data, specs)
            results.append([(t.symbol, t.entry_time, t.exit_time, t.exit_reason,
                             round(t.r_multiple, 6)) for t in bt.trades])
        self.assertGreater(len(results[0]), 0)
        self.assertEqual(results[0], results[1])


class TestSwingConfirmationDelay(unittest.TestCase):
    def test_swing_is_published_exactly_k_bars_late(self):
        s = Series("T", "M1")
        atr = ATR(3)
        eng = SwingEngine(s, 2, atr)
        bars = [(10, 12, 9, 11), (11, 13, 10, 12), (12, 20, 11, 19),
                (19, 15, 13, 14), (14, 14, 10, 11), (11, 13, 10, 12)]
        confirmed_at = {}
        for i, (o, h, l, c) in enumerate(bars):
            s.append(START + i * MIN, o, h, l, c, 1.0)
            atr.update(h, l, c)
            for sw in eng.on_bar_closed(i):
                confirmed_at[sw.index] = i
        self.assertIn(2, confirmed_at, "swing high at bar 2 was never confirmed")
        self.assertEqual(confirmed_at[2], 4,
                         "a K=2 swing must appear exactly 2 bars after it printed")


class TestAggregation(unittest.TestCase):
    def test_m5_and_m15_match_manual_aggregation(self):
        s = make_series("T", START, 60, price=1.0, seed=3)
        m5 = aggregate(s, "M5")
        self.assertEqual(len(m5), 12)
        self.assertEqual(m5.ot[0], START)
        self.assertEqual(m5.o[0], s.o[0])
        self.assertEqual(m5.c[0], s.c[4])
        self.assertEqual(m5.h[0], max(s.h[i] for i in range(5)))
        self.assertEqual(m5.l[0], min(s.l[i] for i in range(5)))
        self.assertAlmostEqual(m5.v[0], sum(s.v[i] for i in range(5)), places=9)
        m15 = aggregate(s, "M15")
        self.assertEqual(len(m15), 4)
        self.assertEqual(m15.c[0], s.c[14])

    def test_partial_bucket_is_never_published(self):
        s = make_series("T", START, 7, price=1.0, seed=3)   # 5 + 2 leftovers
        self.assertEqual(len(aggregate(s, "M5")), 1)
        self.assertEqual(len(aggregate(s, "M15")), 0)

    def test_buckets_align_to_the_exchange_clock(self):
        s = make_series("T", START + 3 * MIN, 30, price=1.0, seed=3)
        m5 = aggregate(s, "M5")
        for i in range(len(m5)):
            self.assertEqual(m5.ot[i] % (5 * MIN), 0)


class TestClosedBarsOnly(unittest.TestCase):
    def test_higher_timeframes_are_fed_only_after_their_close(self):
        from jarvis5.backtest.engine import Backtester
        cfg = _cfg()
        cfg.SYMBOLS = ["DOGEUSDT"]
        data = {"DOGEUSDT": make_series("DOGEUSDT", START, 200, seed=1)}
        bt = Backtester(cfg, data)
        c = bt.ctx["DOGEUSDT"]
        for i in range(200):
            bt._step(c, i, data["DOGEUSDT"].ot[i])
            now_close = data["DOGEUSDT"].ot[i] + MIN
            # every consumed M5/M15 bar must already have closed
            if c.p5:
                self.assertLessEqual(c.m5_series.close_time(c.p5 - 1), now_close)
            if c.p15:
                self.assertLessEqual(c.m15_series.close_time(c.p15 - 1), now_close)


if __name__ == "__main__":
    unittest.main()
