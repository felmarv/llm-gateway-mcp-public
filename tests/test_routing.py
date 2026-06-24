"""Routing + policy tests (no network)."""
import asyncio

import providers
import server
from policies import circuit_breaker, cost_estimator, error_taxonomy, fallback


async def _mock(provider, model, messages, max_tokens=None, **_):
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return {"status": "success", "data": {"text": f"[{provider}:{model}] {user[:20]}"},
            "meta": {"provider": provider, "model": model, "cost_usd_approx": 0.0001,
                     "tokens": {"input": 5, "output": 5, "total": 10}}}


def setup_function():
    circuit_breaker.reset()
    providers.dispatch = _mock


def test_routing_yaml_has_core_task_classes():
    assert "quick_answer" in server.DEFAULTS
    assert "triple_review" in server.DEFAULTS


def test_unknown_task_class_errors():
    r = asyncio.run(server.llm_route("nope", "hi"))
    assert r["status"] == "error" and "unknown task_class" in r["error"]


def test_single_route_uses_policy_model():
    r = asyncio.run(server.llm_route("quick_answer", "hi"))
    assert r["status"] == "success"
    assert r["meta"]["model"] == server.DEFAULTS["quick_answer"]["model"]


def test_override_model_is_honored():
    r = asyncio.run(server.llm_route("quick_answer", "hi", override_model="gpt-4o"))
    assert r["meta"]["model"] == "gpt-4o"


def test_panel_runs_all_members_and_certifies():
    r = asyncio.run(server.llm_route("triple_review", "claim"))
    assert len(r["data"]["members"]) == 3
    assert r["meta"]["visibility"]["independence_certified"] is True


def test_preflight_blocks_runaway():
    pf = cost_estimator.preflight(prompt="x" * 8000, system=None,
                                  pricing={"in": 30, "out": 180}, max_tokens_out=100000,
                                  block_usd=15.0)
    assert pf["action"] == "block"


def test_preflight_ok_without_pricing():
    pf = cost_estimator.preflight(prompt="hi", system=None, pricing=None, max_tokens_out=1000)
    assert pf["action"] == "ok"


def test_error_taxonomy():
    assert error_taxonomy.classify({"status": "error", "error": "rate limit"}) == "rate_limit"
    assert error_taxonomy.classify({"status": "error", "meta": {"http_status": 503}}) == "provider_5xx"
    assert error_taxonomy.classify({"status": "success"}) == "ok"


def test_fallback_chain_declared():
    assert fallback.is_allowed(server.ROUTING, "general_reasoning")
    assert not fallback.is_allowed(server.ROUTING, "triple_review")
