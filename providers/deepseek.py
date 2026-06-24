"""DeepSeek provider — OpenAI-compatible Chat Completions over httpx.

Auth: DEEPSEEK_API_KEY from the environment.

Pricing below is ILLUSTRATIVE (USD per 1M tokens) — verify at
https://api-docs.deepseek.com/quick_start/pricing before relying on cost preflight.
"""
from __future__ import annotations

import os
import time

import httpx

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

MODELS: dict[str, dict[str, float]] = {
    "deepseek-chat":     {"in": 0.27, "out": 1.10},
    "deepseek-reasoner": {"in": 0.55, "out": 2.19},
}


def _api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set in the environment")
    return key


async def complete(
    messages: list[dict],
    model: str = "deepseek-chat",
    max_tokens: int | None = None,
    timeout: float = 180.0,
    **_: object,
) -> dict:
    if model not in MODELS:
        return {"status": "error", "error": f"unknown DeepSeek model: {model}",
                "meta": {"provider": "deepseek", "model": model}}

    payload: dict = {"model": model, "messages": messages, "max_tokens": max_tokens or 4096}

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
                "meta": {"provider": "deepseek", "model": model,
                         "latency_ms": int((time.monotonic() - t0) * 1000)}}
    latency_ms = int((time.monotonic() - t0) * 1000)

    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
        except Exception:
            msg = f"HTTP {resp.status_code}"
        return {"status": "error", "error": msg,
                "meta": {"provider": "deepseek", "model": model,
                         "latency_ms": latency_ms, "http_status": resp.status_code}}

    body = resp.json()
    content = (body["choices"][0]["message"].get("content") or "").strip()
    usage = body.get("usage", {})
    in_t = usage.get("prompt_tokens", 0)
    out_t = usage.get("completion_tokens", 0)
    p = MODELS[model]
    cost = (in_t * p["in"] + out_t * p["out"]) / 1_000_000

    if not content:
        return {"status": "error", "error": f"{model}: empty completion.",
                "meta": {"provider": "deepseek", "model": model, "latency_ms": latency_ms,
                         "tokens": {"input": in_t, "output": out_t, "total": in_t + out_t},
                         "cost_usd_approx": round(cost, 6)}}

    return {
        "status": "success",
        "data": {"text": content},
        "meta": {"provider": "deepseek", "model": model, "latency_ms": latency_ms,
                 "tokens": {"input": in_t, "output": out_t, "total": in_t + out_t},
                 "cost_usd_approx": round(cost, 6)},
    }
