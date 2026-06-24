"""Anthropic (Claude) provider — official `anthropic` SDK.

Auth: ANTHROPIC_API_KEY from the environment (the SDK reads it automatically).

Pricing below is ILLUSTRATIVE (USD per 1M tokens) — verify at
https://platform.claude.com/docs/en/about-claude/models/overview before
relying on cost preflight in production.

Note: the Messages API takes the system prompt as a top-level `system` argument,
not as a message with role "system" — `complete()` splits it out of `messages`.
"""
from __future__ import annotations

import os
import time

MODELS: dict[str, dict[str, float]] = {
    "claude-opus-4-8":   {"in": 5.00, "out": 25.00},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5":  {"in": 1.00, "out":  5.00},
}

_client = None


def _get_client():
    """Lazy AsyncAnthropic singleton (import deferred so the package loads
    even when the SDK isn't installed until a Claude route is actually used)."""
    global _client
    if _client is not None:
        return _client
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise RuntimeError("ANTHROPIC_API_KEY not set in the environment")
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("anthropic SDK not installed — run: pip install anthropic") from exc
    _client = AsyncAnthropic()
    return _client


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Pull any role=system entries out into the top-level system string."""
    system_parts: list[str] = []
    chat: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(str(m.get("content", "")))
        else:
            chat.append({"role": m["role"], "content": m["content"]})
    return ("\n\n".join(system_parts) or None), chat


async def complete(
    messages: list[dict],
    model: str = "claude-opus-4-8",
    max_tokens: int | None = None,
    **_: object,
) -> dict:
    if model not in MODELS:
        return {"status": "error", "error": f"unknown Anthropic model: {model}",
                "meta": {"provider": "anthropic", "model": model}}

    system, chat = _split_system(messages)
    if not chat:
        chat = [{"role": "user", "content": ""}]

    t0 = time.monotonic()
    try:
        client = _get_client()
        kwargs: dict = {"model": model, "max_tokens": max_tokens or 4096, "messages": chat}
        if system:
            kwargs["system"] = system
        resp = await client.messages.create(**kwargs)
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:300],
                "meta": {"provider": "anthropic", "model": model,
                         "latency_ms": int((time.monotonic() - t0) * 1000)}}
    latency_ms = int((time.monotonic() - t0) * 1000)

    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    in_t = getattr(resp.usage, "input_tokens", 0) or 0
    out_t = getattr(resp.usage, "output_tokens", 0) or 0
    p = MODELS[model]
    cost = (in_t * p["in"] + out_t * p["out"]) / 1_000_000

    if not text:
        return {"status": "error",
                "error": f"{model}: empty response (stop_reason={resp.stop_reason}).",
                "meta": {"provider": "anthropic", "model": model, "latency_ms": latency_ms,
                         "tokens": {"input": in_t, "output": out_t, "total": in_t + out_t},
                         "cost_usd_approx": round(cost, 6)}}

    return {
        "status": "success",
        "data": {"text": text},
        "meta": {"provider": "anthropic", "model": model, "latency_ms": latency_ms,
                 "tokens": {"input": in_t, "output": out_t, "total": in_t + out_t},
                 "cost_usd_approx": round(cost, 6),
                 "stop_reason": resp.stop_reason},
    }
