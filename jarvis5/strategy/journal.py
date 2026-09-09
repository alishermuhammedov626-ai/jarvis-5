"""Candidate / trade journal.

EVERY candidate setup is recorded, not only the ones that became trades.  This
is the instrument that tells us whether the rule set is over-filtering: a
rejection histogram with 90 % of the mass on one reason is a design bug, not a
strategy result.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from typing import Any, Dict, List, Optional

# ---- rejection reasons (spec 52) ------------------------------------- #
NO_M15_CONTEXT = "NO_M15_CONTEXT"
NO_LIQUIDITY = "NO_LIQUIDITY"
NO_M5_SWEEP = "NO_M5_SWEEP"
NO_MSS = "NO_MSS"
NO_CHOCH = "NO_CHOCH"
NO_BOS = "NO_BOS"
NO_DISPLACEMENT = "NO_DISPLACEMENT"
NO_OB = "NO_OB"
NO_FVG = "NO_FVG"
NO_ZONE = "NO_ZONE"
NO_RETEST = "NO_RETEST"
NO_M1_CONFIRMATION = "NO_M1_CONFIRMATION"
TARGET_TOO_CLOSE = "TARGET_TOO_CLOSE"
SL_TOO_WIDE = "SL_TOO_WIDE"
SL_TOO_TIGHT = "SL_TOO_TIGHT"
SL_IMPOSSIBLE = "SL_IMPOSSIBLE"
CHASE = "CHASE"
LIQUIDATION_RISK = "LIQUIDATION_RISK"
COST_TOO_HIGH = "COST_TOO_HIGH"
DUPLICATE = "DUPLICATE"
DAILY_RISK_LIMIT = "DAILY_RISK_LIMIT"
DAILY_TRADE_LIMIT = "DAILY_TRADE_LIMIT"
ACTIVE_POSITION = "ACTIVE_POSITION"
MAX_POSITIONS = "MAX_POSITIONS"
SETUP_EXPIRED = "SETUP_EXPIRED"
INVALID_STRUCTURE = "INVALID_STRUCTURE"
INVALID_DATA = "INVALID_DATA"
DIRECTION_MISMATCH = "DIRECTION_MISMATCH"
PREMIUM_DISCOUNT = "PREMIUM_DISCOUNT"
SESSION_FILTER = "SESSION_FILTER"
COOLDOWN = "COOLDOWN"
INVALID_SIZE = "INVALID_SIZE"
RR_TOO_LOW = "RR_TOO_LOW"
SETUP_INVALIDATED = "SETUP_INVALIDATED"

FOUND = "FOUND"
REJECTED = "REJECTED"

# reasons that are pure risk-management book-keeping rather than a statement
# about the setup's structural quality
PORTFOLIO_REASONS = {DAILY_RISK_LIMIT, DAILY_TRADE_LIMIT, ACTIVE_POSITION,
                     MAX_POSITIONS, COOLDOWN, DUPLICATE}


class Candidate:
    """A setup seeded by an M5 liquidity sweep, tracked to its outcome."""

    __slots__ = ("cid", "symbol", "time", "direction", "setup_type", "status",
                 "primary_rejection_reason", "secondary_rejection_reasons",
                 "states", "data", "signal_id")

    _seq = 0

    def __init__(self, symbol: str, time: int, direction: str):
        Candidate._seq += 1
        self.cid = Candidate._seq
        self.symbol = symbol
        self.time = time
        self.direction = direction
        self.setup_type = ""
        self.status = ""
        self.primary_rejection_reason = ""
        self.secondary_rejection_reasons: List[str] = []
        self.states: List[Dict[str, Any]] = []
        self.data: Dict[str, Any] = {}
        self.signal_id = ""

    def transition(self, state: str, time: int, reason: str = "") -> None:
        prev = self.states[-1]["state"] if self.states else "WAIT"
        self.states.append({"time": time, "state": state,
                            "previous_state": prev, "reason": reason})

    def reject(self, reason: str) -> None:
        if not self.primary_rejection_reason:
            self.primary_rejection_reason = reason
            self.status = REJECTED
        elif reason not in self.secondary_rejection_reasons:
            self.secondary_rejection_reasons.append(reason)

    def accept(self) -> None:
        self.status = FOUND

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "candidate_id": self.cid,
            "symbol": self.symbol,
            "time": self.time,
            "direction": self.direction,
            "setup_type": self.setup_type,
            "status": self.status,
            "primary_rejection_reason": self.primary_rejection_reason,
            "secondary_rejection_reasons": ",".join(self.secondary_rejection_reasons),
            "signal_id": self.signal_id,
            "states": len(self.states),
            "final_state": self.states[-1]["state"] if self.states else "",
        }
        d.update(self.data)
        return d


class Journal:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.candidates: List[Candidate] = []
        self.state_log: List[Dict[str, Any]] = []
        self.counter: Counter = Counter()
        self.stage_counter: Counter = Counter()
        self.symbol_counter: Dict[str, Counter] = {}

    def add(self, cand: Candidate) -> None:
        reason = cand.primary_rejection_reason or FOUND
        self.counter[reason] += 1
        self.symbol_counter.setdefault(cand.symbol, Counter())[reason] += 1
        if cand.states:
            self.stage_counter[cand.states[-1]["state"]] += 1
        if not self.enabled:
            return
        self.candidates.append(cand)
        for st in cand.states:
            row = dict(st)
            row["candidate_id"] = cand.cid
            row["symbol"] = cand.symbol
            row["direction"] = cand.direction
            self.state_log.append(row)

    # -- reporting ------------------------------------------------------ #
    def rejection_histogram(self) -> Dict[str, int]:
        return dict(sorted(self.counter.items(), key=lambda kv: -kv[1]))

    def rejections_by_symbol(self) -> Dict[str, Dict[str, int]]:
        return {sym: dict(sorted(c.items(), key=lambda kv: -kv[1]))
                for sym, c in sorted(self.symbol_counter.items())}

    def funnel(self) -> Dict[str, int]:
        return dict(sorted(self.stage_counter.items(), key=lambda kv: -kv[1]))

    def write_csv(self, path: str) -> Optional[str]:
        if not self.candidates:
            return None
        keys: List[str] = []
        for c in self.candidates[:2000]:
            for k in c.as_dict():
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for c in self.candidates:
                w.writerow(c.as_dict())
        return path

    def write_states(self, path: str) -> Optional[str]:
        if not self.state_log:
            return None
        keys = ["candidate_id", "symbol", "direction", "time", "state",
                "previous_state", "reason"]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.state_log)
        return path
