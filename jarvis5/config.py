"""JARVIS 5 configuration.

Every strategy / execution parameter lives here.  Nothing in the engine is
allowed to hard-code a threshold: a run is fully described by (data, Config).

    Same data + same Config  ==  same trades, always.

NO machine learning, no probabilistic model, no fitted score is used anywhere
in this project.  All decisions are deterministic rule evaluations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # 1. UNIVERSE
    # ------------------------------------------------------------------ #
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "DOGEUSDT",
        "1000PEPEUSDT",
        "1000SHIBUSDT",
        "PUMPUSDT",
        "1000BONKUSDT",
        "WIFUSDT",
    ])
    # M1 is the only raw timeframe. M5 / M15 are deterministically aggregated.
    BASE_TIMEFRAME: str = "1m"
    USE_TIMEFRAMES: List[str] = field(default_factory=lambda: ["M1", "M5", "M15"])

    # ------------------------------------------------------------------ #
    # 2. ACCOUNT / RISK
    # ------------------------------------------------------------------ #
    STARTING_EQUITY: float = 10_000.0
    LEVERAGE: float = 17.0                # primary benchmark
    RISK_PER_TRADE: float = 0.0035        # 0.35 %  (allowed: 0.0025 / 0.0035 / 0.0050)
    MAX_DAILY_LOSS: float = 0.015         # -1.5 % realised net -> stop new entries
    MAX_TRADES_PER_SYMBOL_PER_DAY: int = 3
    MAX_CONCURRENT_POSITIONS: int = 6     # cross-symbol
    ONE_POSITION_PER_SYMBOL: bool = True
    DAILY_RESET_HOUR_UTC: int = 0

    # ------------------------------------------------------------------ #
    # 3. ATR / SWINGS / STRUCTURE
    # ------------------------------------------------------------------ #
    ATR_PERIOD: int = 14
    SWING_K: int = 2                      # fractal half-width (K bars each side)
    EXTERNAL_STRUCTURE_ATR: float = 1.5   # min separation for an external swing
    MIN_BREAK_ATR: float = 0.05           # min close-through distance for BOS/CHOCH
    MSS_SWEEP_LOOKBACK: int = 10          # bars: break must follow a sweep this recently
    RANGE_MODE_M15_BARS: int = 20         # no external BOS for N M15 bars -> RANGE

    # ------------------------------------------------------------------ #
    # 4. LIQUIDITY
    # ------------------------------------------------------------------ #
    EQUAL_LIQUIDITY_ATR: float = 0.10     # |p1-p2| <= 0.10 ATR -> equal high/low
    LIQUIDITY_CLUSTER_ATR: float = 0.25   # cluster radius
    LIQUIDITY_MAX_LEVELS: int = 120       # per timeframe, per side (memory guard)
    MAX_SWEEP_PENETRATION_ATR: float = 1.50  # deeper than this = real break, not a sweep
    MIN_SWEEP_PENETRATION_ATR: float = 0.01
    SWEEP_STRONG_SCORE: int = 6
    SWEEP_MEDIUM_SCORE: int = 4
    ALLOW_WEAK_SWEEP: bool = True         # weak sweeps only survive with other strength

    # ------------------------------------------------------------------ #
    # 5. DISPLACEMENT
    # ------------------------------------------------------------------ #
    MIN_DISPLACEMENT_ATR: float = 1.0
    STRONG_DISPLACEMENT_ATR: float = 1.5
    DISPLACEMENT_MAX_LEG_BARS: int = 3    # 1..3 bar impulse leg
    M1_MIN_DISPLACEMENT_ATR: float = 0.75

    # ------------------------------------------------------------------ #
    # 6. ZONES (OB / FVG)
    # ------------------------------------------------------------------ #
    OB_LOOKBACK: int = 12
    OB_USE_WICKS: bool = True
    MIN_FVG_ATR: float = 0.05
    MAX_ZONE_ATR: float = 3.0             # absurdly wide zones are unusable for SL
    ZONE_CONFLUENCE_ATR: float = 0.10     # OB & FVG within this -> confluence zone
    REQUIRE_ZONE: bool = True
    ZONE_RETEST_TOLERANCE_ATR: float = 0.05

    # ------------------------------------------------------------------ #
    # 7. PREMIUM / DISCOUNT
    # ------------------------------------------------------------------ #
    PD_ENABLED: bool = True
    PD_HARD_FILTER: bool = False          # soft by default (spec 18)
    PD_EQ: float = 0.50
    PD_TOLERANCE: float = 0.05            # allow slight overshoot past EQ

    # ------------------------------------------------------------------ #
    # 8. ENTRY / M1 CONFIRMATION
    # ------------------------------------------------------------------ #
    MAX_CHASE_ATR: float = 0.50           # distance from zone we refuse to chase
    M1_REQUIRE_SWEEP: bool = True
    M1_REQUIRE_STRUCTURE: bool = True     # CHOCH or BOS on M1
    M1_REQUIRE_DISPLACEMENT: bool = True
    M1_CONFIRMATION_LOOKBACK: int = 12    # M1 bars in which micro events must occur
    SETUP_EXPIRY_M5_BARS: int = 12        # pending setup lifetime
    DOUBLE_LIQUIDITY_M15_WINDOW: int = 12 # M15 bars for a matching M15 sweep
    SETUP_SWEEP_LOOKBACK_M5: int = 10     # sweep -> break window on M5
    COOLDOWN_MINUTES: int = 15

    # ------------------------------------------------------------------ #
    # 9. STOP LOSS / TAKE PROFIT
    # ------------------------------------------------------------------ #
    STRUCTURAL_BUFFER_ATR: float = 0.10
    MIN_SL_PERCENT: float = 0.0030        # 0.30 %
    MAX_SL_PERCENT: float = 0.0050        # 0.50 %
    ALLOW_M5_INVALIDATION_FALLBACK: bool = True
    MIN_TP_PERCENT: float = 0.010         # 1.00 %
    MAX_TP_SEARCH_PERCENT: float = 0.05   # look this far for opposing liquidity
    TP_FALLBACK_TO_MIN: bool = False      # if True, use MIN_TP when no liquidity found
    RR_HARD_FILTER: bool = False          # spec 33: RR is logged, not enforced
    MIN_RR: float = 0.0

    # ------------------------------------------------------------------ #
    # 10. TRAILING / EXIT
    # ------------------------------------------------------------------ #
    TRAILING_ENABLED: bool = True
    TRAILING_ACTIVATION_PERCENT: float = 0.0035   # +0.35 %
    TRAILING_STRUCTURE_BUFFER_ATR: float = 0.10
    PROFIT_LOCK_ENABLED: bool = False
    PROFIT_LOCK_TRIGGER_PERCENT: float = 0.0070   # +0.70 %
    PROFIT_LOCK_PERCENT: float = 0.0010
    EXIT_ON_INVALIDATION: bool = False    # logged either way, off by default
    MAX_HOLD_HOURS: float = 24.0

    # ------------------------------------------------------------------ #
    # 11. LIQUIDATION SAFETY
    # ------------------------------------------------------------------ #
    MAINTENANCE_MARGIN_RATE: float = 0.005
    LIQUIDATION_SAFETY_ATR: float = 1.0   # SL must sit this far in front of liq price
    LIQUIDATION_SAFETY_PERCENT: float = 0.010

    # ------------------------------------------------------------------ #
    # 12. COSTS
    # ------------------------------------------------------------------ #
    COMMISSION_RATE: float = 0.0004       # Binance USDT-M taker
    SLIPPAGE_BPS: float = 2.0             # per side, in basis points of price
    USE_FUNDING: bool = True
    FUNDING_INTERVAL_HOURS: int = 8
    MAX_COST_TO_TARGET_RATIO: float = 0.35  # costs vs expected profit -> COST_TOO_HIGH

    # ------------------------------------------------------------------ #
    # 13. EXECUTION SIMULATION
    # ------------------------------------------------------------------ #
    SAME_CANDLE_RULE: str = "SL_FIRST"    # conservative: worst-case ordering
    ENTRY_DELAY_BARS: int = 0             # robustness knob (0 = confirmation close)
    PRICE_PERTURBATION_BPS: float = 0.0   # robustness knob

    # ------------------------------------------------------------------ #
    # 14. SESSIONS (Tashkent, UTC+5)
    # ------------------------------------------------------------------ #
    TIMEZONE_OFFSET_HOURS: int = 5
    SESSIONS: Dict[str, List[int]] = field(default_factory=lambda: {
        "ASIA":   [0, 8],
        "LONDON": [10, 17],
        "NY":     [17, 23],
    })
    SESSION_FILTER: List[str] = field(default_factory=list)   # empty = no hard filter

    # ------------------------------------------------------------------ #
    # 15. JOURNAL / OUTPUT
    # ------------------------------------------------------------------ #
    JOURNAL_CANDIDATES: bool = True
    OUTPUT_DIR: str = "results"
    RANDOM_SEED: int = 20260909          # only used by Monte-Carlo resampling

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def copy(self, **overrides: Any) -> "Config":
        d = self.to_dict()
        d.update(overrides)
        return Config(**d)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @staticmethod
    def _known() -> set:
        return {f.name for f in fields(Config)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        known = cls._known()
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        return cls(**data)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def validate(self) -> None:
        assert self.SWING_K >= 1
        assert self.ATR_PERIOD >= 2
        assert 0 < self.RISK_PER_TRADE < 0.1
        assert 0 < self.MIN_SL_PERCENT < self.MAX_SL_PERCENT
        assert self.MIN_TP_PERCENT > 0
        assert self.LEVERAGE >= 1
        assert self.SAME_CANDLE_RULE in ("SL_FIRST", "TP_FIRST", "SKIP")
        assert self.SWEEP_MEDIUM_SCORE <= self.SWEEP_STRONG_SCORE


DEFAULT = Config()
