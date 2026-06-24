"""Normalize heterogeneous provider errors into stable categories.

Used to decide retry, fallback, and whether to trip the circuit breaker.
"""
from __future__ import annotations


def classify(response: dict) -> str:
    if not isinstance(response, dict):
        return "unknown"
    if response.get("status") != "error":
        return "ok"

    meta = response.get("meta") or {}
    http = meta.get("http_status")
    msg = (response.get("error") or "").lower()

    if any(s in msg for s in ("circuit breaker", "cost block", "preflight", "policy block")):
        return "policy_blocked"
    if http == 401 or "unauthor" in msg or "invalid api key" in msg or "not set" in msg:
        return "auth"
    if http == 429 or "rate limit" in msg:
        return "rate_limit"
    if any(s in msg for s in ("quota", "insufficient", "billing")):
        return "quota"
    if http in (500, 502, 503, 504) or "overloaded" in msg or "internal server error" in msg:
        return "provider_5xx"
    if http == 400 or "bad request" in msg or "invalid" in msg:
        return "invalid_request"
    if "timeout" in msg or "network" in msg:
        return "timeout"
    return "unknown"


RETRIABLE = frozenset({"rate_limit", "timeout", "provider_5xx"})
BREAKER_TRIGGERS = frozenset({"timeout", "provider_5xx", "rate_limit"})


def is_retriable(category: str) -> bool:
    return category in RETRIABLE


def should_open_breaker(category: str) -> bool:
    return category in BREAKER_TRIGGERS
