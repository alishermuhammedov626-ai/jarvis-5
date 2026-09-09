"""Rule-level conformance tests: sizing, costs, stops, breaks, sweeps."""
import os
import re
import unittest

from jarvis5.config import Config
from jarvis5.core.series import Series
from jarvis5.core.timeutil import from_iso
from jarvis5.engine.atr import ATR
from jarvis5.engine.liquidity import BUY_SIDE, SELL_SIDE, LiquidityEngine
from jarvis5.engine.timeframe import TimeframeState
from jarvis5.strategy import journal as J
from jarvis5.strategy.risk import (SymbolSpec, build_stop, build_target,
                                   liquidation_price, size_position)

START = from_iso("2026-01-01")
MIN = 60_000


class TestPositionSizing(unittest.TestCase):
    def test_matches_the_specification_worked_example(self):
        cfg = Config()                       # risk 0.35 %, leverage 17x
        spec = SymbolSpec("X", 1e-8, 1e-8, 1e-8, 0.0)
        entry = 100.0
        sl = entry * (1 - 0.0040)            # 0.40 % stop
        qty, notional, margin, risk, err = size_position(10_000.0, entry, sl, spec, cfg)
        self.assertEqual(err, "")
        self.assertAlmostEqual(notional, 8_750.0, places=2)
        self.assertAlmostEqual(risk, 35.0, places=4)
        self.assertAlmostEqual(margin, 8_750.0 / 17.0, places=4)

    def test_risk_is_recomputed_after_exchange_rounding(self):
        cfg = Config()
        spec = SymbolSpec("X", 0.01, 1.0, 1.0, 5.0)   # whole-unit steps
        entry, sl = 100.0, 99.6
        qty, notional, margin, risk, err = size_position(10_000.0, entry, sl, spec, cfg)
        self.assertEqual(err, "")
        self.assertEqual(qty, float(int(qty)), "quantity must respect step size")
        self.assertAlmostEqual(risk, notional * (0.4 / 100.0), places=9)
        self.assertLessEqual(risk, 35.0 + 1e-9,
                             "rounded risk must never exceed the intended risk")


class TestStops(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_buffer_is_applied_below_the_invalidation_low(self):
        atr = 0.5
        sl, src, reason = build_stop(100.0, "BUY", 99.65, atr, self.cfg)
        self.assertEqual(reason, "")
        self.assertAlmostEqual(sl, 99.65 - 0.10 * atr, places=9)

    def test_too_tight_and_too_wide_are_rejected(self):
        _sl, _s, reason = build_stop(100.0, "BUY", 99.99, 0.001, self.cfg)
        self.assertEqual(reason, J.SL_TOO_TIGHT)
        _sl, _s, reason = build_stop(100.0, "BUY", 98.0, 0.001, self.cfg)
        self.assertEqual(reason, J.SL_TOO_WIDE)

    def test_falls_back_to_the_m5_invalidation_swing(self):
        sl, src, reason = build_stop(100.0, "BUY", 99.99, 0.001, self.cfg,
                                     fallback_invalidation=99.6)
        self.assertEqual(reason, "")
        self.assertEqual(src, "M5")

    def test_a_stop_beyond_entry_is_impossible(self):
        _sl, _s, reason = build_stop(100.0, "BUY", 101.0, 0.1, self.cfg)
        self.assertEqual(reason, J.SL_IMPOSSIBLE)


class TestTargets(unittest.TestCase):
    def _engine(self, prices, side):
        cfg = Config()
        s = Series("X", "M5")
        eng = LiquidityEngine(s, cfg)
        for p in prices:
            eng._add(p, "INTERNAL_HIGH" if side == BUY_SIDE else "INTERNAL_LOW",
                     side, START, 0, 1.0)
        return eng

    def test_target_must_clear_the_one_percent_minimum(self):
        cfg = Config()
        eng = self._engine([100.5], BUY_SIDE)          # only 0.5 % away
        _tp, _src, reason = build_target(100.0, "BUY", [("M5", eng)], cfg)
        self.assertEqual(reason, J.TARGET_TOO_CLOSE)

    def test_nearest_qualifying_liquidity_is_chosen(self):
        cfg = Config()
        eng = self._engine([100.5, 101.4, 103.0], BUY_SIDE)
        tp, src, reason = build_target(100.0, "BUY", [("M5", eng)], cfg)
        self.assertEqual(reason, "")
        self.assertAlmostEqual(tp, 101.4, places=9)
        self.assertIn("INTERNAL_HIGH", src)


class TestLiquidation(unittest.TestCase):
    def test_stop_too_close_to_liquidation_is_rejected(self):
        from jarvis5.strategy.risk import plan_trade
        cfg = Config().copy(LEVERAGE=200.0)           # liq sits ~0.5 % away
        eng = LiquidityEngine(Series("X", "M5"), cfg)
        eng._add(102.0, "INTERNAL_HIGH", BUY_SIDE, START, 0, 1.0)
        plan = plan_trade("BUY", 100.0, 99.65, None, 0.05, [("M5", eng)],
                          10_000.0, SymbolSpec("X", 1e-8, 1e-8, 1e-8, 0.0), cfg)
        self.assertFalse(plan.ok)
        self.assertEqual(plan.reason, J.LIQUIDATION_RISK)

    def test_long_liquidation_sits_below_entry(self):
        self.assertLess(liquidation_price(100.0, "BUY", 17, 0.005), 100.0)
        self.assertGreater(liquidation_price(100.0, "SELL", 17, 0.005), 100.0)


class TestStructureBreaks(unittest.TestCase):
    """A wick through a level is never a break (spec 9)."""

    def _feed(self, bars):
        cfg = Config().copy(ATR_PERIOD=3, SWING_K=1, EXTERNAL_STRUCTURE_ATR=0.0,
                            MIN_BREAK_ATR=0.0)
        s = Series("X", "M5")
        st = TimeframeState(s, cfg)
        events = []
        for i, (o, h, l, c) in enumerate(bars):
            s.append(START + i * MIN, o, h, l, c, 100.0)
            ev, _sw = st.on_bar_closed(i)
            events.extend(ev)
        return events

    def test_wick_through_a_swing_high_is_not_a_bos(self):
        bars = [(10, 11, 9, 10), (10, 14, 10, 13), (13, 13, 11, 12),
                (12, 12, 10, 11), (11, 15, 11, 12)]   # wick to 15, close 12 < 14
        ups = [e for e in self._feed(bars) if e.direction == "BULLISH"]
        self.assertEqual(ups, [], f"wick-only break was recorded: {ups}")

    def test_close_through_a_swing_high_is_a_bos(self):
        bars = [(10, 11, 9, 10), (10, 14, 10, 13), (13, 13, 11, 12),
                (12, 12, 10, 11), (11, 15, 11, 15)]   # closes above 14
        ups = [e for e in self._feed(bars) if e.direction == "BULLISH"]
        self.assertTrue(ups, "a genuine close-through break was missed")


class TestSweeps(unittest.TestCase):
    def _engine(self):
        cfg = Config()
        s = Series("X", "M5")
        eng = LiquidityEngine(s, cfg)
        return cfg, s, eng

    def test_sweep_requires_closing_back_above_the_level(self):
        cfg, s, eng = self._engine()
        eng._add(100.0, "INTERNAL_LOW", SELL_SIDE, START, 0, 1.0)
        # bar 1 wicks to 99.7 and closes back at 100.4 -> sweep
        s.append(START, 100.5, 100.6, 100.2, 100.4, 1000.0)
        s.append(START + MIN, 100.4, 100.5, 99.7, 100.4, 1000.0)
        got = eng.on_bar_closed(1, 1.0)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].side, SELL_SIDE)
        self.assertGreater(got[0].penetration_atr, 0)

    def test_closing_through_the_level_is_a_break_not_a_sweep(self):
        cfg, s, eng = self._engine()
        eng._add(100.0, "INTERNAL_LOW", SELL_SIDE, START, 0, 1.0)
        s.append(START, 100.5, 100.6, 100.2, 100.4, 1000.0)
        s.append(START + MIN, 100.4, 100.5, 99.5, 99.6, 1000.0)
        self.assertEqual(eng.on_bar_closed(1, 1.0), [])

    def test_an_excessively_deep_penetration_is_not_a_sweep(self):
        cfg, s, eng = self._engine()
        eng._add(100.0, "INTERNAL_LOW", SELL_SIDE, START, 0, 1.0)
        s.append(START, 100.5, 100.6, 100.2, 100.4, 1000.0)
        s.append(START + MIN, 100.4, 100.5, 90.0, 100.4, 1000.0)   # 10 ATR deep
        self.assertEqual(eng.on_bar_closed(1, 1.0), [])


class TestExecution(unittest.TestCase):
    def _pos(self, cfg=None):
        from jarvis5.backtest.position import Position
        from jarvis5.strategy.risk import RiskPlan
        plan = RiskPlan(True)
        plan.entry, plan.sl, plan.tp = 100.0, 99.6, 101.0
        plan.qty, plan.notional, plan.margin = 10.0, 1000.0, 58.8
        plan.risk_amount, plan.liquidation_price = 4.0, 94.0
        return Position("X", "BUY", plan, START, 0, "A", {}, 0.4, 0.0)

    def test_same_candle_sl_and_tp_resolves_to_sl(self):
        cfg = Config()
        s = Series("X", "M1")
        s.append(START + MIN, 100.0, 101.5, 99.0, 100.5, 1.0)
        st = TimeframeState(s, cfg)
        pos = self._pos()
        price, reason = pos.scan_exit(st, 0, cfg)
        self.assertEqual(reason, "SL")
        self.assertTrue(pos.ambiguous_exit, "ambiguity must be flagged, not hidden")

    def test_tp_first_rule_is_opt_in_only(self):
        cfg = Config().copy(SAME_CANDLE_RULE="TP_FIRST")
        s = Series("X", "M1")
        s.append(START + MIN, 100.0, 101.5, 99.0, 100.5, 1.0)
        st = TimeframeState(s, cfg)
        _price, reason = self._pos().scan_exit(st, 0, cfg)
        self.assertEqual(reason, "TP")

    def test_gap_through_the_stop_fills_at_the_open(self):
        cfg = Config()
        s = Series("X", "M1")
        s.append(START + MIN, 98.0, 99.0, 97.0, 98.5, 1.0)
        st = TimeframeState(s, cfg)
        price, reason = self._pos().scan_exit(st, 0, cfg)
        self.assertEqual(reason, "SL")
        self.assertEqual(price, 98.0, "a gap must fill at the open, not at the stop")

    def test_a_stop_never_moves_against_the_trade(self):
        pos = self._pos()
        pos._tighten(99.9)
        self.assertAlmostEqual(pos.sl, 99.9)
        pos._tighten(99.2)                       # would widen the risk
        self.assertAlmostEqual(pos.sl, 99.9)

    def test_commission_is_charged_on_notional_not_margin(self):
        from jarvis5.backtest.costs import CostModel
        cfg = Config()
        cm = CostModel(cfg)
        fee = cm.commission(10.0, 100.0)
        self.assertAlmostEqual(fee, 1000.0 * cfg.COMMISSION_RATE, places=12)
        margin_fee = (1000.0 / cfg.LEVERAGE) * cfg.COMMISSION_RATE
        self.assertNotAlmostEqual(fee, margin_fee, places=6)

    def test_slippage_always_fills_on_the_unfavourable_side(self):
        from jarvis5.backtest.costs import CostModel
        cm = CostModel(Config())
        self.assertGreater(cm.fill_price(100.0, "BUY", "ENTRY"), 100.0)
        self.assertLess(cm.fill_price(100.0, "BUY", "EXIT"), 100.0)
        self.assertLess(cm.fill_price(100.0, "SELL", "ENTRY"), 100.0)
        self.assertGreater(cm.fill_price(100.0, "SELL", "EXIT"), 100.0)


class TestFunding(unittest.TestCase):
    def test_missing_funding_is_reported_not_silently_zeroed(self):
        from jarvis5.backtest.costs import FundingBook
        fb = FundingBook(Config())
        fb.load("X", None)
        self.assertFalse(fb.is_available("X"))

    def test_longs_pay_and_shorts_receive_a_positive_rate(self):
        from jarvis5.backtest.costs import FundingBook
        fb = FundingBook(Config())
        t = START + 8 * 3_600_000
        fb.load("X", [(t, 0.0001)])
        self.assertLess(fb.charge("X", "BUY", 10_000.0, START, t + 1), 0)
        self.assertGreater(fb.charge("X", "SELL", 10_000.0, START, t + 1), 0)


class TestNoMachineLearning(unittest.TestCase):
    """Spec: this build must contain no ML anywhere in the decision path."""

    BANNED = re.compile(
        r"\b(sklearn|xgboost|lightgbm|catboost|tensorflow|torch|keras|"
        r"RandomForest|LogisticRegression|GradientBoosting|MLPClassifier|"
        r"predict_proba|neural_network)\b")

    def test_no_ml_imports_or_models_in_the_package(self):
        root = os.path.join(os.path.dirname(__file__), "..", "jarvis5")
        hits = []
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                with open(path) as fh:
                    for lineno, line in enumerate(fh, 1):
                        if self.BANNED.search(line):
                            hits.append(f"{path}:{lineno}: {line.strip()}")
        self.assertEqual(hits, [], "machine-learning constructs found:\n" +
                         "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
