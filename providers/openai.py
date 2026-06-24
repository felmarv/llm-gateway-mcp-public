"""OpenAI provider — Chat Completions over httpx.

Auth: OPENAI_API_KEY from the environment.

Pricing below is ILLUSTRATIVE (USD per 1M tokens) — verify current numbers at
https://openai.com/api/pricing before relying on cost preflight in production.
"""
from __future__ import annotations

import os
import time

import httpx

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

MODELS: dict[str, dict[str, float]] = {
    "gpt-4o":      {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out":  0.60},
    "o3":          {"in": 2.00, "out":  8.00},
    "o3-mini":     {"in": 1.10, "out":  4.40},
}

# Reasoning models bill output via max_completion_tokens, not max_tokens.
_REASONING_PREFIX = ("o1", "o3", "o4")


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in the environment")
    return key


async def complete(
    messages: list[dict],
    model: str = "gpt-4o",
    max_tokens: int | None = None,
    timeout: float = 120.0,
    **_: object,
) -> dict:
    if model not in MODELS:
        return {"status": "error", "error": f"unknown OpenAI model: {model}",
                "meta": {"provider": "openai", "model": model}}

    payload: dict = {"model": model, "messages": messages}
    if max_tokens:
        if model.startswith(_REASONING_PREFIX):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {_api_key()}",
                         "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.HTTPError as exc:
        return {"status": "error", "error": f"network: {exc}",
                "meta": {"provider": "openai", "model": model,
                         "latency_ms": int((time.monotonic() - t0) * 1000)}}
    latency_ms = int((time.monotonic() - t0) * 1000)

    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
        except Exception:
            msg = f"HTTP {resp.status_code}"
        return {"status": "error", "error": msg,
                "meta": {"provider": "openai", "model": model,
                         "latency_ms": latency_ms, "http_status": resp.status_code}}

    body = resp.json()
    content = (body["choices"][0]["message"].get("content") or "").strip()
    usage = body.get("usage", {})
    in_t = usage.get("prompt_tokens", 0)
    out_t = usage.get("completion_tokens", 0)
    p = MODELS[model]
    cost = (in_t * p["in"] + out_t * p["out"]) / 1_000_000

    if not content:
        return {"status": "error",
                "error": f"{model}: empty completion (finish_reason="
                         f"{body['choices'][0].get('finish_reason')}). Raise max_tokens.",
                "meta": {"provider": "openai", "model": model, "latency_ms": latency_ms,
                         "tokens": {"input": in_t, "output": out_t, "total": in_t + out_t},
                         "cost_usd_approx": round(cost, 6)}}

    return {
        "status": "success",
        "data": {"text": content},
        "meta": {"provider": "openai", "model": model, "latency_ms": latency_ms,
                 "tokens": {"input": in_t, "output": out_t, "total": in_t + out_t},
                 "cost_usd_approx": round(cost, 6),
                 "finish_reason": body["choices"][0].get("finish_reason")},
    }
