"""Gateway policies: cost preflight, circuit breaker, retry/backoff, error
taxonomy, fallback, and an optional persistent spend ledger.

All policies are pure/generic — no provider- or domain-specific assumptions
beyond the shared response envelope used across `providers/`.
"""
