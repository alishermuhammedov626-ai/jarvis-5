"""Research module tests (spec 31-33).

Covers each detector, the event machinery, the honesty rules, and the two
structural guarantees: truncation invariance (no look-ahead) and scale
invariance (ATR/percentage normalisation).
"""
import os
import re
import tempfile
import unittest

from jarvis5.core.series import Series, aggregate
from jarvis5.core.timeutil import from_iso
from jarvis5.engine.atr import ATR
from jarvis5.research import combos, events, validate, wyckoff
from jarvis5.research.config import ResearchConfig
from jarvis5.research.engine import ResearchEngine, window_extreme_forward
from jarvis5.research.features import SymbolReplay
from jarvis5.research.indicators import (RSI, VolumeTracker,
                                         price_action_signals, volume_signals)
from jarvis5.research.report import ResearchReport
from jarvis5.research.stats import (INSUFFICIENT, Aggregator, Label,
                                    classify, coin_consistency, edge_stats)
from jarvis5.research.synthetic import make_market

START = from_iso("2026-01-01")
MIN = 60_000


def _flat(n, price=1.0, symbol="T"):
    s = Series(symbol, "M1")
    for i in range(n):
        s.append(START + i * MIN, price, price * 1.0002, price * 0.9998, price, 1000.0)
    return s


def _snapshots(series, cfg, bars):
    """Feature snapshots at the requested bars, taken during one forward replay."""
    m5 = aggregate(series, "M5")
    m15 = aggregate(series, "M15")
    replay = SymbolReplay(series.symbol, series, m5, m15, cfg)
    want = set(bars)
    out = {}
    for i in range(series.n):
        replay.advance(i)
        if i in want and replay.ready:
            f = replay.snapshot(i)
            f.update(combos.build(f))
            out[i] = f
    return out


class TestEventDetection(unittest.TestCase):
    def test_event_detection(self):
        s = Series("T", "M1")
        prices = [100] * 5 + [100.6, 101.3, 101.2, 101.0, 100.0]
        for i, p in enumerate(prices):
            s.append(START + i * MIN, p, p + 0.01, p - 0.01, p, 100.0)
        tt = events.time_to_targets(s, [0.01, 0.015], events.LONG, 240)
        # every flat bar at 100 reaches 101 at bar 6
        self.assertEqual(list(tt[0])[:5], [6, 5, 4, 3, 2])
        # 1.5 % (101.5) is never reached
        self.assertEqual(set(tt[1]), {-1})

    def test_target_detection(self):
        s = Series("T", "M1")
        s.append(START, 100, 100, 100, 100, 1.0)
        for k, p in enumerate([101.0, 101.6, 102.1, 103.2, 105.5], start=1):
            s.append(START + k * MIN, p, p, p, p, 1.0)
        cfg = ResearchConfig()
        tt = events.time_to_targets(s, cfg.TARGETS, events.LONG, 240)
        got = [tt[k][0] for k in range(len(cfg.TARGETS))]
        self.assertEqual(got, [1, 2, 3, 4, 5],
                         "each target must resolve at its FIRST touch")

    def test_origin_cannot_be_resolved_by_its_own_candle(self):
        s = Series("T", "M1")
        s.append(START, 100, 105, 100, 100, 1.0)      # own high already +5 %
        s.append(START + MIN, 100, 100, 100, 100, 1.0)
        tt = events.time_to_targets(s, [0.01], events.LONG, 240)
        self.assertEqual(tt[0][0], -1)

    def test_short_events_mirror_long_events(self):
        s = Series("T", "M1")
        for i, p in enumerate([100, 100, 99.5, 98.9, 98.0]):
            s.append(START + i * MIN, p, p + 0.01, p - 0.01, p, 100.0)
        tt = events.time_to_targets(s, [0.01], events.SHORT, 240)
        self.assertEqual(tt[0][0], 3)


class TestEventClustering(unittest.TestCase):
    def test_event_clustering(self):
        """One impulse must produce one event, not one per origin bar."""
        cfg = ResearchConfig()
        s, truth = make_market("T", START, cycles=1, price=1.0)
        longs, shorts, tt, _ = events.detect(s, "T", cfg)
        raw_origins = sum(1 for v in tt[0] if v >= 0)
        self.assertGreater(raw_origins, 0)
        self.assertLess(len(longs), raw_origins / 5,
                        "clustering barely reduced the origin count")
        for ev in longs:
            self.assertGreaterEqual(ev.origins, 1)

    def test_clusters_do_not_overlap_their_starts(self):
        cfg = ResearchConfig()
        s, _ = make_market("T", START, cycles=1, price=1.0)
        longs, _s, _t, _u = events.detect(s, "T", cfg)
        starts = [e.event_start for e in longs]
        self.assertEqual(starts, sorted(starts))
        for a, b in zip(longs, longs[1:]):
            self.assertGreaterEqual(b.event_start - a.event_start,
                                    cfg.MIN_EVENT_SEPARATION_BARS)

    def test_impulse_anchor_is_the_extreme_before_the_move(self):
        cfg = ResearchConfig()
        s, _ = make_market("T", START, cycles=1, price=1.0)
        longs, _a, _b, _c = events.detect(s, "T", cfg)
        for ev in longs[:20]:
            lo = min(s.l[j] for j in range(ev.event_start, ev.event_end + 1))
            self.assertAlmostEqual(s.l[ev.impulse_start], lo, places=12)


class TestMFEMAE(unittest.TestCase):
    def test_mfe_mae(self):
        cfg = ResearchConfig()
        s, _ = make_market("T", START, cycles=1, price=1.0)
        longs, shorts, _a, _b = events.detect(s, "T", cfg)
        for ev in longs:
            self.assertGreaterEqual(ev.mfe_pct, 0.0)
            self.assertGreaterEqual(ev.mae_pct, 0.0)
            if ev.targets_hit.get("1pct"):
                self.assertGreaterEqual(
                    ev.mfe_pct, 0.0099,
                    "a 1 % target was recorded with an MFE below 1 %")

    def test_window_extreme_matches_brute_force(self):
        arr = [1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0]
        n, H = len(arr), 3
        mx = window_extreme_forward(arr, n, H, True)
        mn = window_extreme_forward(arr, n, H, False)
        for i in range(n - 1):
            lo, hi = i + 1, min(n - 1, i + H)
            self.assertEqual(mx[i], max(arr[lo:hi + 1]))
            self.assertEqual(mn[i], min(arr[lo:hi + 1]))


class TestNoLookahead(unittest.TestCase):
    def test_no_lookahead(self):
        """Truncation invariance: removing the future must change nothing."""
        cfg = ResearchConfig()
        s, _ = make_market("T", START, cycles=2, price=1.0)
        cut = int(s.n * 0.6)
        bars = list(range(cfg.WARMUP_BARS, cut - 5, 37))
        full = _snapshots(s, cfg, bars)
        short = _snapshots(s.slice(None, s.ot[cut]), cfg, bars)
        self.assertGreater(len(full), 5)
        for i in bars:
            if i in full or i in short:
                self.assertEqual(full.get(i), short.get(i),
                                 f"features at bar {i} changed once the future "
                                 f"was removed -> look-ahead leak")

    def test_scale_invariance(self):
        """Spec 33: price x10 / x50 / x100 must not change detection."""
        cfg = ResearchConfig()
        base = None
        for mult in (1.0, 10.0, 50.0, 100.0):
            s, _ = make_market("T", START, cycles=1, price=1.0 * mult)
            bars = list(range(cfg.WARMUP_BARS, s.n - cfg.EVENT_HORIZON_BARS, 53))
            snaps = _snapshots(s, cfg, bars)
            names = {i: sorted(f.keys()) for i, f in snaps.items()}
            if base is None:
                base = names
                self.assertGreater(len(base), 5)
            else:
                self.assertEqual(base, names,
                                 f"pattern set changed at price x{mult}")

    def test_event_outcomes_are_scale_invariant(self):
        cfg = ResearchConfig()
        out = []
        for mult in (1.0, 100.0):
            s, _ = make_market("T", START, cycles=1, price=1.0 * mult)
            longs, shorts, _a, _b = events.detect(s, "T", cfg)
            out.append([(e.event_start, e.impulse_start, round(e.mfe_pct, 9))
                        for e in longs])
        self.assertEqual(out[0], out[1])


class TestLiquidityDetection(unittest.TestCase):
    def test_liquidity_detection(self):
        cfg = ResearchConfig()
        s, truth = make_market("T", START, cycles=1, price=1.0)
        seg = next(t for t in truth
                   if t["scenario"] == "SWEEP_MSS_DISPLACEMENT_LONG")
        idx = seg["markers"]["sweep_low"]
        snaps = _snapshots(s, cfg, range(idx, idx + 9))
        found = any("M1_SWEEP_SELLSIDE" in f or "M5_SWEEP_SELLSIDE" in f
                    for f in snaps.values())
        self.assertTrue(found, "an injected sell-side sweep was not detected")
        self.assertTrue(any("M1_LIQUIDITY_REJECTION" in f for f in snaps.values()))

    def test_liquidity_levels_of_every_required_type_exist(self):
        cfg = ResearchConfig()
        s, _ = make_market("T", START, cycles=1, price=1.0)
        m5, m15 = aggregate(s, "M5"), aggregate(s, "M15")
        replay = SymbolReplay("T", s, m5, m15, cfg)
        for i in range(min(s.n, 1500)):
            replay.advance(i)
        types = {lv.ltype for lv in replay.m1.state.liquidity.highs}
        types |= {lv.ltype for lv in replay.m1.state.liquidity.lows}
        for want in ("PREVIOUS_HIGH", "PREVIOUS_LOW", "INTERNAL_HIGH",
                     "INTERNAL_LOW", "SESSION_HIGH", "SESSION_LOW"):
            self.assertIn(want, types)


class TestSMCDetection(unittest.TestCase):
    def test_smc_detection(self):
        cfg = ResearchConfig()
        s, truth = make_market("T", START, cycles=2, price=1.0)
        seg = next(t for t in truth
                   if t["scenario"] == "SWEEP_MSS_DISPLACEMENT_LONG")
        idx = seg["markers"]["displacement_up"]
        snaps = _snapshots(s, cfg, range(idx, idx + 9))
        merged = set()
        for f in snaps.values():
            merged |= set(f)
        self.assertTrue(any(n.endswith("_DISPLACEMENT_UP") for n in merged))
        self.assertTrue(any("_BOS_UP" in n or "_CHOCH_UP" in n for n in merged),
                        "no structural break detected after an injected impulse")

    def test_premium_discount_are_mutually_exclusive(self):
        cfg = ResearchConfig()
        s, _ = make_market("T", START, cycles=1, price=1.0)
        snaps = _snapshots(s, cfg, range(cfg.WARMUP_BARS, s.n - 200, 97))
        for f in snaps.values():
            for tf in ("M1", "M5", "M15"):
                flags = [f"{tf}_PREMIUM" in f, f"{tf}_DISCOUNT" in f,
                         f"{tf}_EQUILIBRIUM" in f]
                self.assertLessEqual(sum(flags), 1)

    def test_fvg_detection(self):
        cfg = ResearchConfig()
        s, truth = make_market("T", START, cycles=1, price=1.0)
        seg = next(t for t in truth if t["scenario"] == "SWEEP_FVG_LONG")
        idx = seg["markers"]["fvg_up"]
        snaps = _snapshots(s, cfg, range(idx, idx + 9))
        self.assertTrue(any("M1_FVG_BULL" in f or "M5_FVG_BULL" in f
                            for f in snaps.values()))


class TestWyckoffDetection(unittest.TestCase):
    def test_wyckoff_detection(self):
        cfg = ResearchConfig()
        s, truth = make_market("T", START, cycles=2, price=1.0)
        springs = [t["markers"]["spring"] for t in truth
                   if t["scenario"] == "WYCKOFF_SPRING_LONG"]
        self.assertTrue(springs)
        hits = 0
        for idx in springs:
            snaps = _snapshots(s, cfg, range(idx, idx + 11))
            if any("WY_SPRING" in f for f in snaps.values()):
                hits += 1
        self.assertEqual(hits, len(springs),
                         "an injected Wyckoff spring was not detected")

    def test_range_body_excludes_the_recent_window(self):
        """Without this, a spring's own low defines the range low."""
        cfg = ResearchConfig()
        s, _ = make_market("T", START, cycles=1, price=1.0)
        atr = ATR(cfg.ATR_PERIOD)
        vals = [atr.update(s.h[i], s.l[i], s.c[i]) for i in range(s.n)]
        i = 900
        r = wyckoff.find_range(s, i, vals[i], cfg)
        if r is not None:
            self.assertLessEqual(r.end, i - cfg.WY_RECENT_BARS)


class TestRSIDetection(unittest.TestCase):
    def test_rsi_detection(self):
        r = RSI(14)
        for k in range(40):
            r.update(100 + k)
        self.assertAlmostEqual(r.value, 100.0, places=6)
        r2 = RSI(14)
        for k in range(40):
            r2.update(100 - k)
        self.assertAlmostEqual(r2.value, 0.0, places=6)

    def test_oversold_flag_before_an_injected_reversal(self):
        cfg = ResearchConfig()
        s, truth = make_market("T", START, cycles=1, price=1.0)
        seg = next(t for t in truth if t["scenario"] == "RSI_REVERSAL_LONG")
        idx = seg["markers"]["rsi_oversold"]
        snaps = _snapshots(s, cfg, range(idx, idx + 5))
        self.assertTrue(any("RSI_LT_30" in f or "RSI_LT_20" in f
                            for f in snaps.values()))


class TestVolumeDetection(unittest.TestCase):
    def test_volume_detection(self):
        cfg = ResearchConfig()
        s = _flat(120)
        vol = VolumeTracker(cfg.VOLUME_AVG_PERIOD)
        for i in range(s.n):
            vol.update(s.v[i])
        s.append(START + 120 * MIN, 1.0, 1.001, 0.999, 1.0005, 3500.0)
        vol.update(3500.0)
        got = volume_signals(s, vol, s.n - 1, 0.0005, cfg)
        self.assertIn("VOLUME_SPIKE", got)
        self.assertIn("VOLUME_GT_3X", got)

    def test_volume_breakout_is_detected_where_injected(self):
        cfg = ResearchConfig()
        s, truth = make_market("T", START, cycles=1, price=1.0)
        seg = next(t for t in truth if t["scenario"] == "VOLUME_BREAKOUT_LONG")
        idx = seg["markers"]["volume_breakout"]
        snaps = _snapshots(s, cfg, range(idx, idx + 5))
        self.assertTrue(any("VOLUME_GT_2X" in f or "VOLUME_SPIKE" in f
                            for f in snaps.values()))


class TestPriceActionDetection(unittest.TestCase):
    def _series(self, bars):
        s = Series("T", "M1")
        for i, (o, h, l, c) in enumerate(bars):
            s.append(START + i * MIN, o, h, l, c, 1000.0)
        return s

    def test_price_action_detection(self):
        cfg = ResearchConfig()
        s = self._series([(100, 100.5, 99.5, 99.6),
                          (99.6, 100.2, 99.4, 99.5),
                          (99.4, 100.6, 99.3, 100.5)])   # engulfs bar 1
        got = price_action_signals(s, 2, 0.3, cfg)
        self.assertIn("PA_BULLISH_ENGULFING", got)

    def test_hammer_and_inside_bar(self):
        cfg = ResearchConfig()
        s = self._series([(100, 101, 99, 100.5),
                          (100.4, 100.6, 100.1, 100.5),        # inside bar
                          (100.5, 100.6, 99.0, 100.45)])       # long lower wick
        self.assertIn("PA_INSIDE_BAR", price_action_signals(s, 1, 0.3, cfg))
        got = price_action_signals(s, 2, 0.3, cfg)
        self.assertIn("PA_PIN_BAR", got)
        self.assertIn("PA_HAMMER", got)


class TestCombinationDetection(unittest.TestCase):
    def test_combination_detection(self):
        f = {"M5_SWEEP_SELLSIDE": 20, "M5_MSS_UP": 10,
             "M5_DISPLACEMENT_UP": 8, "M5_FVG_BULL": 6}
        got = combos.build(f)
        self.assertEqual(got["C04_SWEEP_MSS_UP"], 10)
        self.assertEqual(got["C07_SWEEP_MSS_FVG_UP"], 6)
        self.assertIn("C06_SWEEP_MSS_DISPLACEMENT_UP", got)

    def test_order_matters(self):
        """A break that PREDATES the sweep is not sweep-plus-break."""
        f = {"M5_SWEEP_SELLSIDE": 10, "M5_MSS_UP": 30}
        self.assertNotIn("C04_SWEEP_MSS_UP", combos.build(f))

    def test_missing_element_breaks_the_chain(self):
        f = {"M5_SWEEP_SELLSIDE": 20, "M5_DISPLACEMENT_UP": 8}
        got = combos.build(f)
        self.assertIn("C05_SWEEP_DISPLACEMENT_UP", got)
        self.assertNotIn("C06_SWEEP_MSS_DISPLACEMENT_UP", got)

    def test_setup_d_requires_both_sweeps_and_confirmation(self):
        f = {"M15_SWEEP_SELLSIDE": 40, "M5_SWEEP_SELLSIDE": 25,
             "M5_MSS_UP": 15, "M1_SWEEP_SELLSIDE": 10,
             "M1_CHOCH_UP": 6, "M1_DISPLACEMENT_UP": 3}
        self.assertIn("SETUP_D_DOUBLE_LIQUIDITY_UP", combos.build(f))
        del f["M15_SWEEP_SELLSIDE"]
        self.assertNotIn("SETUP_D_DOUBLE_LIQUIDITY_UP", combos.build(f))


class TestCostCalculation(unittest.TestCase):
    def test_cost_calculation(self):
        from jarvis5.research import simulate
        cfg = ResearchConfig()
        s, _ = make_market("T", START, cycles=1, price=1.0)
        m5, m15 = aggregate(s, "M5"), aggregate(s, "M15")
        replay = SymbolReplay("T", s, m5, m15, cfg)
        found = None
        for i in range(s.n - cfg.SIM_HORIZON_BARS):
            replay.advance(i)
            if i < cfg.WARMUP_BARS or not replay.ready:
                continue
            r = simulate.simulate(replay, i, "LONG", cfg)
            if r.ok:
                found = r
                break
        self.assertIsNotNone(found, "no simulatable bar found")
        o = found.outcomes[f"{cfg.SIM_TARGETS[0]*100:g}pct"]
        self.assertLess(o["r_net"], o["r_gross"] + 1e-12,
                        "net result must be worse than gross once costs apply")
        self.assertGreater(o["fees"], 0.0)
        # commission is charged on NOTIONAL, not on margin
        self.assertGreater(o["fees"], found.notional * cfg.COMMISSION_RATE * 0.9)

    def test_funding_is_reported_not_assumed_zero(self):
        from jarvis5.backtest.costs import FundingBook
        cfg = ResearchConfig()
        fb = FundingBook(cfg.to_trading_config())
        fb.load("T", None)
        self.assertFalse(fb.is_available("T"))


class TestRiskCalculation(unittest.TestCase):
    def test_risk_calculation(self):
        from jarvis5.research import simulate
        cfg = ResearchConfig()
        s, _ = make_market("T", START, cycles=1, price=1.0)
        m5, m15 = aggregate(s, "M5"), aggregate(s, "M15")
        replay = SymbolReplay("T", s, m5, m15, cfg)
        checked = 0
        for i in range(s.n - cfg.SIM_HORIZON_BARS):
            replay.advance(i)
            if i < cfg.WARMUP_BARS or not replay.ready:
                continue
            r = simulate.simulate(replay, i, "LONG", cfg)
            if not r.ok:
                continue
            self.assertGreaterEqual(r.sl_pct, cfg.MIN_SL_PERCENT - 1e-12)
            self.assertLessEqual(r.sl_pct, cfg.MAX_SL_PERCENT + 1e-12)
            # risk = notional x SL fraction = equity x RISK_PER_TRADE
            self.assertAlmostEqual(
                r.notional * r.sl_pct,
                cfg.STARTING_EQUITY * cfg.RISK_PER_TRADE, places=6)
            checked += 1
            if checked >= 25:
                break
        self.assertGreater(checked, 0)

    def test_leverage_does_not_change_risk(self):
        from jarvis5.research import simulate
        base = ResearchConfig()
        lev = base.copy(LEVERAGE=50.0)
        s, _ = make_market("T", START, cycles=1, price=1.0)
        m5, m15 = aggregate(s, "M5"), aggregate(s, "M15")
        outs = []
        for cfg in (base, lev):
            replay = SymbolReplay("T", s, m5, m15, cfg)
            got = None
            for i in range(s.n - cfg.SIM_HORIZON_BARS):
                replay.advance(i)
                if i < cfg.WARMUP_BARS or not replay.ready:
                    continue
                r = simulate.simulate(replay, i, "LONG", cfg)
                if r.ok:
                    got = (round(r.notional, 9), round(r.sl_pct, 12))
                    break
            outs.append(got)
        self.assertEqual(outs[0], outs[1])


class TestOOSSplit(unittest.TestCase):
    def test_oos_split(self):
        cfg = ResearchConfig()
        eng = ResearchEngine(cfg, progress=False)
        segs = eng.segments(1000)
        self.assertEqual([s[0] for s in segs],
                         ["DISCOVERY", "VALIDATION", "OOS"])
        self.assertEqual(segs[0][1], 0)
        self.assertEqual(segs[-1][2], 1000)
        for a, b in zip(segs, segs[1:]):
            self.assertEqual(a[2], b[1], "segments must be contiguous")
        self.assertEqual(segs[0][2], 700)
        self.assertEqual(segs[1][2], 850)

    def test_splits_are_chronological_not_shuffled(self):
        cfg = ResearchConfig()
        eng = ResearchEngine(cfg, progress=False)
        segs = eng.segments(1000)
        for name, lo, hi in segs:
            for i in range(lo, hi, 37):
                self.assertEqual(eng._segment_of(i, segs), name)


class TestStatisticsHonesty(unittest.TestCase):
    def test_edge_is_measured_against_a_control_group(self):
        # 70 % conditional against a 65 % baseline is a 5 point edge, not 70
        e = edge_stats(1000, 700, 10000, 6500)
        self.assertAlmostEqual(e["conditional_rate"], 0.70, places=6)
        self.assertLess(e["edge"], 0.06)
        self.assertLess(e["relative_lift"], 1.15)

    def test_small_samples_are_never_rescued(self):
        cfg = ResearchConfig()
        self.assertEqual(classify(12, 0.90, 5.0, cfg), INSUFFICIENT)

    def test_failed_oos_is_reported_as_failed(self):
        cfg = ResearchConfig()
        v = classify(5000, 0.10, 2.0, cfg, {"n": 500, "edge": 0.08},
                     {"n": 500, "edge": -0.02})
        self.assertEqual(v, "OVERFIT / FAILED OOS")

    def test_one_coin_is_not_cross_coin_evidence(self):
        cfg = ResearchConfig()
        cc = coin_consistency({"DOGEUSDT": {"n": 500, "edge": 0.2}}, cfg)
        self.assertEqual(cc["status"], "SINGLE_COIN")

    def test_age_bins_sum_to_the_cumulative_window(self):
        """Disjoint bins must reconstruct nested windows exactly."""
        cfg = ResearchConfig()
        agg = Aggregator(cfg)
        keys = [f"{t*100:g}pct" for t in cfg.TARGETS]
        ages = [3, 3, 7, 18, 25, 45, 90, 200]
        for age in ages:
            label = Label("LONG", {k: True for k in keys}, 0.01, 0.005, 5)
            agg.observe_bar(["ALL"], {"F": age}, {"LONG": label}, {})
        for w in cfg.WINDOWS:
            cell = agg.window_cell("ALL", "F", str(w), "LONG")
            expected = sum(1 for a in ages if a <= w)
            self.assertEqual(cell.n if cell else 0, expected,
                             f"window {w} should contain {expected} observations")


class TestReportGeneration(unittest.TestCase):
    def test_report_generation(self):
        cfg = ResearchConfig().copy(BASELINE_STRIDE=15)
        s, _ = make_market("T", START, cycles=2, price=1.0)
        eng = ResearchEngine(cfg, progress=False)
        eng.run({"T": s})
        rep = ResearchReport(eng, cfg)
        with tempfile.TemporaryDirectory() as d:
            paths = rep.write_all(d)
            for name in ("overall_summary.csv", "coin_summary.csv",
                         "strategy_summary.csv", "strategy_by_coin.csv",
                         "strategy_by_direction.csv", "combination_summary.csv",
                         "event_summary.csv", "oos_summary.csv",
                         "rejection_summary.csv", "data_quality.csv",
                         "top_setups.csv", "full_event_database.csv",
                         "full_report.json", "FINAL_REPORT.md"):
                self.assertIn(name, paths, f"{name} was not produced")
                self.assertTrue(os.path.exists(paths[name]))
            md = paths["_markdown"]
            for q in range(1, 31):
                self.assertRegex(md, rf"\*\*{q}[\./]",
                                 f"question {q} is not answered in FINAL_REPORT.md")

    def test_no_signal_count_is_padded(self):
        """Spec 16: real counts only, never topped up to a round number."""
        cfg = ResearchConfig().copy(BASELINE_STRIDE=20)
        s, _ = make_market("T", START, cycles=1, price=1.0)
        eng = ResearchEngine(cfg, progress=False)
        eng.run({"T": s})
        rep = ResearchReport(eng, cfg)
        counts = [r["n"] for r in rep.rows]
        self.assertTrue(any(c < cfg.MIN_SAMPLE for c in counts),
                        "expected some genuinely small samples to be reported")
        for r in rep.rows:
            if r["n"] < cfg.MIN_SAMPLE:
                self.assertEqual(r["sample_flag"], "INSUFFICIENT SAMPLE")


class TestSyntheticGate(unittest.TestCase):
    def test_synthetic_validation_passes(self):
        res = validate.run(ResearchConfig(), cycles=2)
        failed = [c["check"] for c in res["checks"] if not c["pass"]]
        self.assertEqual(res["status"], "PASS", f"failed checks: {failed}")


class TestNoMachineLearning(unittest.TestCase):
    BANNED = re.compile(
        r"\b(sklearn|xgboost|lightgbm|catboost|tensorflow|torch|keras|"
        r"RandomForest|LogisticRegression|GradientBoosting|MLPClassifier|"
        r"predict_proba|neural_network)\b")

    def test_research_module_contains_no_ml(self):
        root = os.path.join(os.path.dirname(__file__), "..", "jarvis5", "research")
        hits = []
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                with open(path) as fh:
                    for lineno, line in enumerate(fh, 1):
                        if self.BANNED.search(line):
                            hits.append(f"{path}:{lineno}")
        self.assertEqual(hits, [])


class TestTradingEngineUntouched(unittest.TestCase):
    """Spec 29: the research module must not change the trading engine."""

    def test_research_does_not_mutate_trading_config_defaults(self):
        from jarvis5.config import Config
        before = Config().to_dict()
        cfg = ResearchConfig()
        cfg.to_trading_config()
        s, _ = make_market("T", START, cycles=1, price=1.0)
        eng = ResearchEngine(cfg.copy(BASELINE_STRIDE=40), progress=False)
        eng.run({"T": s})
        self.assertEqual(before, Config().to_dict())

    def test_trading_engine_has_no_import_of_research(self):
        root = os.path.join(os.path.dirname(__file__), "..", "jarvis5")
        offenders = []
        for sub in ("engine", "strategy", "backtest", "analysis", "core", "data"):
            d = os.path.join(root, sub)
            for f in os.listdir(d):
                if not f.endswith(".py"):
                    continue
                with open(os.path.join(d, f)) as fh:
                    if re.search(r"^\s*from\s+\.{1,2}research|^\s*import\s+.*research",
                                 fh.read(), re.M):
                        offenders.append(f"{sub}/{f}")
        self.assertEqual(offenders, [],
                         "the trading engine must not depend on the research module")


if __name__ == "__main__":
    unittest.main()
