"""Google Gemini provider — `google-genai` SDK with an API key (NOT Vertex AI).

Auth: GEMINI_API_KEY from the environment. Get one at https://aistudio.google.com/apikey

Pricing below is ILLUSTRATIVE (USD per 1M tokens) — verify at
https://ai.google.dev/gemini-api/docs/pricing before relying on cost preflight.

The google-genai client is synchronous, so calls are run in a thread executor
to keep the async surface of the gateway non-blocking.
"""
# LLM-DIRECT-OK: this IS the gateway's Gemini provider adapter — the genai client
# lives here by design; callers reach it through llm_route/dispatch, never raw.
from __future__ import annotations

import asyncio
import os
import time

MODELS: dict[str, dict[str, float]] = {
    "gemini-2.5-pro":   {"in": 1.25, "out": 10.00},
    "gemini-2.5-flash": {"in": 0.30, "out":  2.50},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
}

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in the environment")
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("google-genai not installed — run: pip install google-genai") from exc
    _client = genai.Client(api_key=key)
    return _client


def _to_contents(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Map OpenAI-style messages to Gemini `contents` + system_instruction."""
    system_parts: list[str] = []
    contents: list[dict] = []
    for m in messages:
        role = m.get("role")
        text = str(m.get("content", ""))
        if role == "system":
            system_parts.append(text)
        else:
            contents.append({"role": "model" if role == "assistant" else "user",
                             "parts": [{"text": text}]})
    return ("\n\n".join(system_parts) or None), contents


async def complete(
    messages: list[dict],
    model: str = "gemini-2.5-pro",
    max_tokens: int | None = None,
    **_: object,
) -> dict:
    if model not in MODELS:
        return {"status": "error", "error": f"unknown Gemini model: {model}",
                "meta": {"provider": "gemini", "model": model}}

    system, contents = _to_contents(messages)
    if not contents:
        contents = [{"role": "user", "parts": [{"text": ""}]}]

    t0 = time.monotonic()
    try:
        client = _get_client()
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(model=model, contents=contents, config=config),
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:300],
                "meta": {"provider": "gemini", "model": model,
                         "latency_ms": int((time.monotonic() - t0) * 1000)}}
    latency_ms = int((time.monotonic() - t0) * 1000)

    text = (resp.text or "").strip() if hasattr(resp, "text") else ""
    usage = getattr(resp, "usage_metadata", None)
    in_t = getattr(usage, "prompt_token_count", 0) or 0
    out_t = getattr(usage, "candidates_token_count", 0) or 0
    p = MODELS[model]
    cost = (in_t * p["in"] + out_t * p["out"]) / 1_000_000

    if not text:
        return {"status": "error", "error": f"{model}: empty response.",
                "meta": {"provider": "gemini", "model": model, "latency_ms": latency_ms,
                         "tokens": {"input": in_t, "output": out_t, "total": in_t + out_t},
                         "cost_usd_approx": round(cost, 6)}}

    return {
        "status": "success",
        "data": {"text": text},
        "meta": {"provider": "gemini", "model": model, "latency_ms": latency_ms,
                 "tokens": {"input": in_t, "output": out_t, "total": in_t + out_t},
                 "cost_usd_approx": round(cost, 6)},
    }
