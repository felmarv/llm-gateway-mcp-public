"""Provider registry + dispatch.

Each provider module exposes:
    MODELS: dict[str, {"in": float, "out": float}]   # USD per 1M tokens (illustrative)
    async def complete(messages, model, max_tokens=None, **kwargs) -> dict

The common response envelope (3-state) is shared by every provider:

    {"status": "success", "data": {"text": ...},
     "meta": {"provider", "model", "latency_ms", "tokens": {...}, "cost_usd_approx"}}
    {"status": "error", "error": "...", "meta": {"provider", "model", "http_status"?}}

Add a new provider by dropping a module here and registering it in PROVIDERS.
"""
from __future__ import annotations

from typing import Any, Callable

from . import anthropic, deepseek, gemini, openai

# provider name -> coroutine(messages, model, max_tokens, **kwargs) -> dict
PROVIDERS: dict[str, Callable[..., Any]] = {
    "openai": openai.complete,
    "anthropic": anthropic.complete,
    "gemini": gemini.complete,
    "deepseek": deepseek.complete,
}

# provider name -> pricing table
PRICING: dict[str, dict[str, dict[str, float]]] = {
    "openai": openai.MODELS,
    "anthropic": anthropic.MODELS,
    "gemini": gemini.MODELS,
    "deepseek": deepseek.MODELS,
}


def model_pricing(provider: str, model: str) -> dict[str, float] | None:
    """USD per 1M tokens for (provider, model), or None if unknown."""
    return PRICING.get(provider, {}).get(model)


async def dispatch(
    provider: str,
    model: str,
    messages: list[dict],
    max_tokens: int | None = None,
    **kwargs: Any,
) -> dict:
    """Route a chat request to the named provider.

    `messages` is the OpenAI-style list of {role, content}. Each provider adapts
    it to its own wire format. Unknown providers return a structured error.
    """
    fn = PROVIDERS.get(provider)
    if fn is None:
        return {
            "status": "error",
            "error": f"unknown provider: {provider!r}. Known: {sorted(PROVIDERS)}",
            "meta": {"provider": provider, "model": model},
        }
    return await fn(messages=messages, model=model, max_tokens=max_tokens, **kwargs)
