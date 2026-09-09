"""Timestamp helpers.  Everything internal is UTC ms epoch; the only place a
local clock appears is session labelling (Tashkent, UTC+5 by default)."""
from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Optional

MS = 1000
MINUTE_MS = 60 * MS
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS

UTC = _dt.timezone.utc


def to_dt(ms: int) -> _dt.datetime:
    return _dt.datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def to_iso(ms: int) -> str:
    return to_dt(ms).strftime("%Y-%m-%d %H:%M:%S")


def from_iso(text: str) -> int:
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return int(_dt.datetime.strptime(text, fmt).replace(tzinfo=UTC).timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"Unparseable timestamp: {text!r}")


def trading_day(ms: int, reset_hour_utc: int = 0) -> str:
    """Deterministic trading-day key used by the daily counters."""
    shifted = ms - reset_hour_utc * HOUR_MS
    d = to_dt(shifted)
    return d.strftime("%Y-%m-%d")


def local_hour(ms: int, offset_hours: int) -> float:
    d = to_dt(ms + offset_hours * HOUR_MS)
    return d.hour + d.minute / 60.0


def session_of(ms: int, sessions: Dict[str, List[int]], offset_hours: int) -> str:
    """Label a timestamp with its trading session (or OFF)."""
    h = local_hour(ms, offset_hours)
    for name, (start, end) in sessions.items():
        if start <= end:
            if start <= h < end:
                return name
        else:  # wraps midnight
            if h >= start or h < end:
                return name
    return "OFF"


def session_start_ms(ms: int, sessions: Dict[str, List[int]], offset_hours: int) -> Optional[int]:
    """Start-of-current-session timestamp, used for session high/low liquidity."""
    name = session_of(ms, sessions, offset_hours)
    if name == "OFF":
        return None
    start_hour = sessions[name][0]
    local = ms + offset_hours * HOUR_MS
    day = local - (local % DAY_MS)
    start_local = day + start_hour * HOUR_MS
    if start_local > local:            # session started yesterday (wrap)
        start_local -= DAY_MS
    return start_local - offset_hours * HOUR_MS


def next_funding_ms(ms: int, interval_hours: int = 8) -> int:
    """Binance funding settles at 00:00 / 08:00 / 16:00 UTC."""
    iv = interval_hours * HOUR_MS
    return ((ms // iv) + 1) * iv
