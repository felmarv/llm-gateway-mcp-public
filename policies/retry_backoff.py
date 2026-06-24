"""Retry a coroutine that returns the gateway envelope, with exponential backoff.

Only retries TRANSIENT failures (rate limit, timeout, provider 5xx) as judged by
`error_taxonomy`. Auth / invalid-request errors are returned immediately.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from . import error_taxonomy

DEFAULT_DELAYS = (1.0, 3.0, 7.0)


async def with_retry(
    call: Callable[[], Awaitable[dict]],
    *,
    delays: tuple[float, ...] = DEFAULT_DELAYS,
) -> tuple[dict, int]:
    """Run `call`, retrying transient errors. Returns (result, retry_count)."""
    result: dict = {}
    retries = 0
    for attempt, delay in enumerate((0.0, *delays)):
        if delay:
            await asyncio.sleep(delay)
            retries += 1
        result = await call()
        if result.get("status") == "success":
            return result, retries
        if not error_taxonomy.is_retriable(error_taxonomy.classify(result)):
            return result, retries
    return result, retries
