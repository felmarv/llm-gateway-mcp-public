"""Cost preflight estimator.

Computes the MAXIMUM projected cost of a request BEFORE dispatch:

    projected_usd = (tokens_in_estimated * price_in
                  + max_tokens_out      * price_out) / 1_000_000

This catches the failure mode where a reasoning model burns its whole output
budget on hidden reasoning tokens (billed as output) without producing useful
text. Action by threshold:

    < warn_usd          -> ok, dispatch normally
    warn..block_usd     -> warn (still dispatch), surface a message in meta
    >= block_usd        -> block; caller must lower max_tokens to proceed
"""
from __future__ import annotations

from typing import Any


def estimate_tokens(total_chars: int, chars_per_token: int = 4) -> int:
    """Cheap heuristic (~4 chars/token). Real usage is reported post-call."""
    if total_chars <= 0:
        return 0
    return max(1, total_chars // chars_per_token)


def estimate_max_cost_usd(
    *, pricing: dict[str, float], tokens_in: int, max_tokens_out: int
) -> float:
    in_cost = tokens_in * pricing.get("in", 0.0) / 1_000_000
    out_cost = max_tokens_out * pricing.get("out", 0.0) / 1_000_000
    return round(in_cost + out_cost, 6)


def preflight(
    *,
    prompt: str,
    system: str | None,
    pricing: dict[str, float] | None,
    max_tokens_out: int,
    chars_per_token: int = 4,
    warn_usd: float = 0.50,
    block_usd: float = 15.00,
) -> dict[str, Any]:
    """Return {action: ok|warn|block, estimated_max_cost_usd, ...}.

    If `pricing` is unknown (None), returns action="ok" with cost None — the
    gateway never blocks a request just because a price table is missing.
    """
    if not pricing:
        return {"action": "ok", "estimated_max_cost_usd": None, "reason": "pricing unknown"}

    total_chars = (len(prompt) if prompt else 0) + (len(system) if system else 0)
    tokens_in = estimate_tokens(total_chars, chars_per_token)
    cost = estimate_max_cost_usd(pricing=pricing, tokens_in=tokens_in, max_tokens_out=max_tokens_out)

    result: dict[str, Any] = {
        "action": "ok",
        "estimated_max_cost_usd": cost,
        "tokens_in_estimated": tokens_in,
        "max_tokens_out": max_tokens_out,
    }
    if cost >= block_usd:
        result["action"] = "block"
        result["reason"] = (
            f"preflight cost ${cost:.4f} >= block_usd ${block_usd:.2f}; "
            f"lower max_tokens (in≈{tokens_in}, out={max_tokens_out})"
        )
    elif cost >= warn_usd:
        result["action"] = "warn"
        result["warning"] = f"preflight cost ${cost:.4f} >= warn_usd ${warn_usd:.2f}"
    return result
