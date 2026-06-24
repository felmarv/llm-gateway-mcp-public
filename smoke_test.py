#!/usr/bin/env python3
"""Structural smoke test — validates routing, policies, and orchestration WITHOUT
calling any real provider. Every model call is replaced by a deterministic mock,
so this runs green with no API keys set.

    python smoke_test.py        # exits 0 on success, 1 on failure
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import providers  # noqa: E402
import server  # noqa: E402
from orchestration import independence  # noqa: E402
from policies import cost_estimator  # noqa: E402


async def _mock_dispatch(provider, model, messages, max_tokens=None, **_):
    """Deterministic stand-in for a provider call."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    if "Verifier" in system:
        text = "ACCEPT"
    elif "Thinker" in system:
        text = "PLAN: 1) gather facts 2) draft 3) check"
    elif "Worker" in system:
        text = f"ARTIFACT addressing: {user[:48]}"
    else:
        text = f"[{provider}:{model}] mock answer to: {user[:48]}"
    return {
        "status": "success",
        "data": {"text": text},
        "meta": {"provider": provider, "model": model, "latency_ms": 1,
                 "tokens": {"input": 10, "output": 10, "total": 20},
                 "cost_usd_approx": 0.0001},
    }


def check(label: str, cond: bool) -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


async def main() -> int:
    providers.dispatch = _mock_dispatch  # patch every model call
    ok = True

    print("routing + policy:")
    ok &= check("routing.yaml loaded with task_classes",
                bool(server.DEFAULTS) and "quick_answer" in server.DEFAULTS)
    ok &= check("independence certifier self-test", independence.self_test())
    pf = cost_estimator.preflight(prompt="x" * 4000, system=None,
                                  pricing={"in": 30.0, "out": 180.0},
                                  max_tokens_out=100000, block_usd=15.0)
    ok &= check("cost preflight blocks a runaway call", pf["action"] == "block")

    print("llm_route (single):")
    r = await server.llm_route("quick_answer", "What is 2+2?")
    ok &= check("single-model route returns text",
                r["status"] == "success" and "text" in r["data"])

    print("llm_route (blind panel):")
    p = await server.llm_route("triple_review", "Assess this claim.")
    members = p.get("data", {}).get("members", [])
    ok &= check("triple_review runs 3 members", len(members) == 3)
    ok &= check("panel certifies independence",
                (p.get("meta", {}).get("visibility") or {}).get("independence_certified") is True)

    print("llm_orchestrate (plan -> execute -> verify):")
    o = await server.llm_orchestrate("Draft a refund policy.", depth="complex")
    ok &= check("compose returns an artifact",
                o["status"] == "success" and bool(o["data"]["artifact"]))
    ok &= check("compose verdict accepted", (o["data"]["verdict"] or {}).get("accepted") is True)
    ok &= check("verifier was blind to the plan",
                o["meta"]["visibility"]["verifier_blind_to_plan"] is True)

    print()
    print("RESULT:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
