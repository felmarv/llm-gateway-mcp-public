"""In-memory circuit breaker keyed by (provider, model).

CLOSED    -> normal
OPEN      -> N consecutive transient failures; calls rejected without hitting the provider
HALF_OPEN -> cooldown elapsed; next call is a trial that re-opens or closes the breaker

State is per-process (resets on restart), which is fine: transient outages are
typically shorter than the cooldown.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class Breaker:
    fail_threshold: int = 5
    cooldown_s: float = 60.0
    consecutive_failures: int = 0
    state: State = State.CLOSED
    opened_at: float = field(default=0.0)


_BREAKERS: dict[tuple[str, str], Breaker] = {}


def _get(provider: str, model: str) -> Breaker:
    key = (provider, model)
    if key not in _BREAKERS:
        _BREAKERS[key] = Breaker()
    return _BREAKERS[key]


def allow(provider: str, model: str) -> dict:
    b = _get(provider, model)
    now = time.monotonic()
    if b.state == State.OPEN:
        if now - b.opened_at >= b.cooldown_s:
            b.state = State.HALF_OPEN
            return {"action": "ok", "state": "half_open"}
        return {"action": "block", "state": "open",
                "reason": f"circuit breaker OPEN for {provider}/{model}, "
                          f"{b.cooldown_s - (now - b.opened_at):.0f}s cooldown left"}
    return {"action": "ok", "state": b.state.value}


def record_success(provider: str, model: str) -> None:
    b = _get(provider, model)
    b.consecutive_failures = 0
    if b.state == State.HALF_OPEN:
        b.state = State.CLOSED


def record_failure(provider: str, model: str) -> None:
    b = _get(provider, model)
    b.consecutive_failures += 1
    if b.state == State.HALF_OPEN:
        b.state = State.OPEN
        b.opened_at = time.monotonic()
    elif b.consecutive_failures >= b.fail_threshold and b.state == State.CLOSED:
        b.state = State.OPEN
        b.opened_at = time.monotonic()


def reset() -> None:
    """Clear all breakers (useful in tests)."""
    _BREAKERS.clear()


def status() -> dict[str, dict]:
    now = time.monotonic()
    return {
        f"{p}/{m}": {
            "state": b.state.value,
            "consecutive_failures": b.consecutive_failures,
            "cooldown_remaining_s": max(0.0, b.cooldown_s - (now - b.opened_at))
            if b.state == State.OPEN else 0.0,
        }
        for (p, m), b in _BREAKERS.items()
    }
