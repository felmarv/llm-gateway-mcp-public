"""Per-task_class fallback chains.

Uniform fallback is dangerous: swapping a vision/grounded model for a text model
silently breaks the task. So fallback is declared explicitly per task_class in
routing.yaml under `fallback:`. An empty/absent list means NO fallback.

    fallback:
      general_reasoning:
        - {provider: gemini, model: gemini-2.5-flash}
      deep_audit:
        - {provider: anthropic, model: claude-sonnet-4-6}
"""
from __future__ import annotations

TRANSIENT_HTTP = {429, 500, 502, 503, 504}


def get_chain(routing: dict, task_class: str) -> list[dict]:
    return list(routing.get("fallback", {}).get(task_class, []) or [])


def is_allowed(routing: dict, task_class: str) -> bool:
    return bool(get_chain(routing, task_class))


def provider_is_down(error_response: dict) -> bool:
    """Whether an error warrants trying a fallback (transient, not the caller's fault)."""
    if not isinstance(error_response, dict) or error_response.get("status") != "error":
        return False
    http = (error_response.get("meta") or {}).get("http_status")
    if http in TRANSIENT_HTTP:
        return True
    msg = (error_response.get("error") or "").lower()
    return "timeout" in msg or "network" in msg or "overloaded" in msg
