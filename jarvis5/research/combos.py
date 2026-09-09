"""Pattern COMBINATIONS and the JARVIS 5 setup families (spec 12, 13).

A combination is not "both flags happened to be on".  Order matters: a break
that precedes the sweep it is supposed to react to is not a sweep-plus-break,
so every multi-step combination requires each later element to be at least as
RECENT as the earlier one (smaller age in minutes).

Derived purely from the base feature dictionary -- no new market data is read.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

UP = "UP"
DOWN = "DOWN"


def _any(feats: Dict[str, int], names: Sequence[str]) -> Optional[int]:
    ages = [feats[n] for n in names if n in feats]
    return min(ages) if ages else None


def _chain(*steps: Optional[int]) -> Optional[int]:
    """All steps present and in chronological order (oldest first).

    steps are ages in minutes; step k must be no older than step k-1.
    Returns the age of the FINAL step, i.e. how fresh the completed pattern is.
    """
    prev = None
    for age in steps:
        if age is None:
            return None
        if prev is not None and age > prev:
            return None                    # this element predates the last one
        prev = age
    return prev


def _m1_confirmation(f: Dict[str, int], d: str) -> Optional[int]:
    """The JARVIS M1 stack: micro sweep -> micro CHOCH/BOS -> micro displacement."""
    side = "SELLSIDE" if d == UP else "BUYSIDE"
    sweep = _any(f, [f"M1_SWEEP_{side}"])
    brk = _any(f, [f"M1_CHOCH_{d}", f"M1_BOS_{d}"])
    disp = _any(f, [f"M1_DISPLACEMENT_{d}"])
    return _chain(sweep, brk, disp)


def _retest(f: Dict[str, int], d: str) -> Optional[int]:
    tag = "BULL" if d == UP else "BEAR"
    return _any(f, [f"M5_IN_OB_{tag}", f"M5_IN_FVG_{tag}",
                    f"M1_IN_OB_{tag}", f"M1_IN_FVG_{tag}",
                    f"PA_BREAKOUT_RETEST_{d}"])


def build(feats: Dict[str, int]) -> Dict[str, int]:
    """Return every combination / setup that is complete, with its age."""
    out: Dict[str, int] = {}
    for d in (UP, DOWN):
        side = "SELLSIDE" if d == UP else "BUYSIDE"
        tag = "BULL" if d == UP else "BEAR"

        m5_sweep = _any(feats, [f"M5_SWEEP_{side}"])
        m15_sweep = _any(feats, [f"M15_SWEEP_{side}"])
        m1_sweep = _any(feats, [f"M1_SWEEP_{side}"])
        any_sweep = _any(feats, [f"M5_SWEEP_{side}", f"M1_SWEEP_{side}",
                                 f"M15_SWEEP_{side}"])
        bos = _any(feats, [f"M5_BOS_{d}"])
        choch = _any(feats, [f"M5_CHOCH_{d}"])
        mss = _any(feats, [f"M5_MSS_{d}"])
        disp = _any(feats, [f"M5_DISPLACEMENT_{d}"])
        fvg = _any(feats, [f"M5_FVG_{tag}"])
        ob = _any(feats, [f"M5_OB_{tag}"])
        m5_choch_or_bos = _any(feats, [f"M5_CHOCH_{d}", f"M5_BOS_{d}"])
        m1c = _m1_confirmation(feats, d)
        retest = _retest(feats, d)

        combos: List[tuple] = [
            ("C01_LIQUIDITY_SWEEP", _chain(any_sweep)),
            ("C02_SWEEP_BOS", _chain(m5_sweep, bos)),
            ("C03_SWEEP_CHOCH", _chain(m5_sweep, choch)),
            ("C04_SWEEP_MSS", _chain(m5_sweep, mss)),
            ("C05_SWEEP_DISPLACEMENT", _chain(m5_sweep, disp)),
            ("C06_SWEEP_MSS_DISPLACEMENT", _chain(m5_sweep, mss, disp)),
            ("C07_SWEEP_MSS_FVG", _chain(m5_sweep, mss, fvg)),
            ("C08_SWEEP_MSS_OB", _chain(m5_sweep, mss, ob)),
            ("C09_SWEEP_MSS_FVG_OB", _chain(m5_sweep, mss,
                                            _both(fvg, ob))),
            ("C10_SWEEP_MSS_M1_CONFIRMATION", _chain(m5_sweep, mss, m1c)),
            ("C11_M15_SWEEP_M5_MSS", _chain(m15_sweep, mss)),
            ("C12_M15_SWEEP_M5_CHOCH", _chain(m15_sweep, choch)),
            ("C13_M15_SWEEP_M5_DISPLACEMENT", _chain(m15_sweep, disp)),
            ("C14_M15_SWEEP_M5_MSS_M1_CONF", _chain(m15_sweep, mss, m1c)),
            ("C15_DOUBLE_SWEEP", _chain(m15_sweep, m5_sweep)),
            ("C16_DOUBLE_SWEEP_MSS", _chain(m15_sweep, m5_sweep, mss)),
            ("C17_DOUBLE_SWEEP_MSS_FVG", _chain(m15_sweep, m5_sweep, mss, fvg)),
            ("C18_SWEEP_BOS_RETEST", _chain(m5_sweep, bos, retest)),
            ("C19_SWEEP_CHOCH_RETEST", _chain(m5_sweep, choch, retest)),
            ("C20_SWEEP_MSS_RETEST", _chain(m5_sweep, mss, retest)),
        ]

        # ---- JARVIS 5 setup families (spec 13) ------------------------- #
        m15_reaction = _any(feats, [f"M15_BOS_{d}", f"M15_CHOCH_{d}",
                                    f"M15_MSS_{d}"])
        zone = _any(feats, [f"M5_OB_{tag}", f"M5_FVG_{tag}"])
        setup_a = _chain(m15_sweep, m15_reaction, m5_sweep, m5_choch_or_bos,
                         disp, zone, retest, m1c)
        in_range = f"M15_STRUCT_RANGE" in feats
        setup_b = _chain(m5_sweep, choch, zone, m1c) if in_range else None
        trending = (f"M15_UPTREND" in feats) if d == UP else (f"M15_DOWNTREND" in feats)
        pullback = _any(feats, ["M5_PULLBACK", "M5_DEEP_PULLBACK",
                                "M15_PULLBACK", "M15_DEEP_PULLBACK"])
        setup_c = _chain(pullback, m5_sweep, m5_choch_or_bos, zone, m1c) \
            if trending else None
        setup_d = _chain(m15_sweep, m5_sweep, mss, m1c)

        combos.extend([
            ("SETUP_A_REVERSAL", setup_a),
            ("SETUP_B_RANGE_REVERSAL", setup_b),
            ("SETUP_C_TREND_CONTINUATION", setup_c),
            ("SETUP_D_DOUBLE_LIQUIDITY", setup_d),
        ])

        for name, age in combos:
            if age is not None:
                out[f"{name}_{d}"] = age
    return out


def _both(a: Optional[int], b: Optional[int]) -> Optional[int]:
    """Both present -> the age of the more recent one."""
    if a is None or b is None:
        return None
    return min(a, b)


COMBO_NAMES = [
    "C01_LIQUIDITY_SWEEP", "C02_SWEEP_BOS", "C03_SWEEP_CHOCH", "C04_SWEEP_MSS",
    "C05_SWEEP_DISPLACEMENT", "C06_SWEEP_MSS_DISPLACEMENT", "C07_SWEEP_MSS_FVG",
    "C08_SWEEP_MSS_OB", "C09_SWEEP_MSS_FVG_OB", "C10_SWEEP_MSS_M1_CONFIRMATION",
    "C11_M15_SWEEP_M5_MSS", "C12_M15_SWEEP_M5_CHOCH",
    "C13_M15_SWEEP_M5_DISPLACEMENT", "C14_M15_SWEEP_M5_MSS_M1_CONF",
    "C15_DOUBLE_SWEEP", "C16_DOUBLE_SWEEP_MSS", "C17_DOUBLE_SWEEP_MSS_FVG",
    "C18_SWEEP_BOS_RETEST", "C19_SWEEP_CHOCH_RETEST", "C20_SWEEP_MSS_RETEST",
]

SETUP_NAMES = ["SETUP_A_REVERSAL", "SETUP_B_RANGE_REVERSAL",
               "SETUP_C_TREND_CONTINUATION", "SETUP_D_DOUBLE_LIQUIDITY"]
