"""Research module configuration.

This module is a MEASUREMENT instrument, not a strategy.  It never feeds the
trading engine and never changes it.  No machine learning, no fitted score, no
probability model -- every pattern here is a deterministic rule evaluated on
bars that have already closed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List


@dataclass
class ResearchConfig:
    # ---- universe ---------------------------------------------------- #
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "DOGEUSDT", "1000PEPEUSDT", "1000SHIBUSDT",
        "PUMPUSDT", "1000BONKUSDT", "WIFUSDT",
    ])
    DAYS: int = 365

    # ---- move events -------------------------------------------------- #
    TARGETS: List[float] = field(default_factory=lambda: [0.010, 0.015, 0.020,
                                                          0.030, 0.050])
    EVENT_HORIZON_BARS: int = 120          # how far ahead a move may complete
    CLUSTER_GAP_BARS: int = 5              # origins this close belong to one impulse
    CLUSTER_RESET_PCT: float = 0.50        # retrace of the extension that ends a cluster
    CLUSTER_MIN_EXTENSION_PCT: float = 0.005  # reset test arms only after this much move
    MIN_EVENT_SEPARATION_BARS: int = 15    # a new event needs a real reset

    # ---- sampling ------------------------------------------------------ #
    # The baseline population is a SYSTEMATIC sample of bars, chosen without
    # any reference to the outcome -- that is what makes P(move | pattern) vs
    # P(move | no pattern) an honest comparison.
    BASELINE_STRIDE: int = 5
    WARMUP_BARS: int = 300                 # engines need history before sampling

    # ---- pattern windows (spec 5) -------------------------------------- #
    WINDOWS: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 60, 120])
    PRIMARY_WINDOW: int = 30

    # ---- indicators ---------------------------------------------------- #
    RSI_PERIOD: int = 14
    RSI_DIVERGENCE_LOOKBACK: int = 60
    VOLUME_AVG_PERIOD: int = 50
    VOLUME_SPIKE_LEVELS: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0])
    VOLUME_CONTRACTION_RATIO: float = 0.60
    LARGE_BODY_ATR: float = 1.0
    PIN_BAR_WICK_RATIO: float = 0.60
    MOMENTUM_CANDLE_ATR: float = 1.5

    # ---- wyckoff -------------------------------------------------------- #
    WY_RANGE_BARS: int = 120
    WY_MIN_RANGE_BARS: int = 40
    WY_MAX_RANGE_ATR: float = 15.0
    WY_RECENT_BARS: int = 20               # range body excludes these bars
    WY_MIN_RANGE_ATR: float = 1.5
    WY_SPRING_LOOKBACK: int = 20
    WY_BREAKOUT_LOOKBACK: int = 10
    WY_FALSE_BREAKOUT_BARS: int = 10
    WY_VOLUME_CLIMAX: float = 2.0
    WY_TREND_LEG_BARS: int = 240
    WY_TREND_LEG_PCT: float = 0.02

    # ---- proximity ------------------------------------------------------ #
    NEAR_LIQUIDITY_ATR: float = 0.50       # "price is near this level"
    ZONE_PROXIMITY_ATR: float = 0.25

    # ---- entry simulation (research only, spec 19-22) ------------------- #
    SIMULATE: bool = True
    SIM_TARGETS: List[float] = field(default_factory=lambda: [0.010, 0.015,
                                                             0.020, 0.030])
    RISK_PER_TRADE: float = 0.0035
    LEVERAGE: float = 17.0
    STARTING_EQUITY: float = 10_000.0
    MIN_SL_PERCENT: float = 0.0030
    MAX_SL_PERCENT: float = 0.0050
    STRUCTURAL_BUFFER_ATR: float = 0.10
    SIM_HORIZON_BARS: int = 240
    TRAILING_ACTIVATION_PERCENT: float = 0.0035
    TRAILING_BUFFER_ATR: float = 0.10
    COMMISSION_RATE: float = 0.0004
    SLIPPAGE_BPS: float = 2.0
    USE_FUNDING: bool = True
    SAME_CANDLE_RULE: str = "SL_FIRST"

    # ---- statistics ------------------------------------------------------ #
    MIN_SAMPLE: int = 30                   # below this: INSUFFICIENT SAMPLE
    EXPLORATORY_SAMPLE: int = 100          # below this: exploratory only
    MIN_EDGE: float = 0.02                 # absolute lift over baseline
    MIN_RELATIVE_LIFT: float = 1.15
    COIN_CONSISTENCY_MIN_COINS: int = 4    # coins that must agree
    COIN_CONSISTENCY_MIN_SAMPLE: int = 20  # per-coin sample to count as a vote

    # ---- discovery / validation / OOS (spec 25) -------------------------- #
    SPLIT_DISCOVERY: float = 0.70
    SPLIT_VALIDATION: float = 0.15
    SPLIT_OOS: float = 0.15

    # ---- engine parameters (mirrored from the trading config) ------------ #
    ATR_PERIOD: int = 14
    SWING_K: int = 2
    EXTERNAL_STRUCTURE_ATR: float = 1.5
    MIN_BREAK_ATR: float = 0.05
    EQUAL_LIQUIDITY_ATR: float = 0.10
    LIQUIDITY_CLUSTER_ATR: float = 0.25
    MIN_DISPLACEMENT_ATR: float = 1.0
    STRONG_DISPLACEMENT_ATR: float = 1.5
    MIN_FVG_ATR: float = 0.05

    # ---- output ----------------------------------------------------------- #
    OUTPUT_DIR: str = "reports"
    TOP_N: int = 40
    MAX_EVENT_ROWS: int = 100_000
    RANDOM_SEED: int = 20260909

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def copy(self, **over: Any) -> "ResearchConfig":
        d = self.to_dict()
        d.update(over)
        return ResearchConfig(**d)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown research config keys: {unknown}")
        return cls(**data)

    @classmethod
    def load(cls, path: str) -> "ResearchConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_trading_config(self):
        """A jarvis5.config.Config carrying the SAME structural definitions.

        The research module reuses the trading engine's structure / liquidity /
        zone detectors read-only, so a pattern named here means exactly what it
        means there.  The trading engine itself is never modified.
        """
        from ..config import Config
        return Config(
            ATR_PERIOD=self.ATR_PERIOD,
            SWING_K=self.SWING_K,
            EXTERNAL_STRUCTURE_ATR=self.EXTERNAL_STRUCTURE_ATR,
            MIN_BREAK_ATR=self.MIN_BREAK_ATR,
            EQUAL_LIQUIDITY_ATR=self.EQUAL_LIQUIDITY_ATR,
            LIQUIDITY_CLUSTER_ATR=self.LIQUIDITY_CLUSTER_ATR,
            MIN_DISPLACEMENT_ATR=self.MIN_DISPLACEMENT_ATR,
            STRONG_DISPLACEMENT_ATR=self.STRONG_DISPLACEMENT_ATR,
            MIN_FVG_ATR=self.MIN_FVG_ATR,
            COMMISSION_RATE=self.COMMISSION_RATE,
            SLIPPAGE_BPS=self.SLIPPAGE_BPS,
            LEVERAGE=self.LEVERAGE,
            RISK_PER_TRADE=self.RISK_PER_TRADE,
            STARTING_EQUITY=self.STARTING_EQUITY,
            MIN_SL_PERCENT=self.MIN_SL_PERCENT,
            MAX_SL_PERCENT=self.MAX_SL_PERCENT,
            STRUCTURAL_BUFFER_ATR=self.STRUCTURAL_BUFFER_ATR,
        )

    def validate(self) -> None:
        assert self.EVENT_HORIZON_BARS > 0
        assert self.BASELINE_STRIDE >= 1
        assert self.TARGETS and all(t > 0 for t in self.TARGETS)
        assert abs(self.SPLIT_DISCOVERY + self.SPLIT_VALIDATION
                   + self.SPLIT_OOS - 1.0) < 1e-9
        assert self.MIN_SAMPLE >= 1
        assert self.SAME_CANDLE_RULE in ("SL_FIRST", "TP_FIRST")


DEFAULT = ResearchConfig()
