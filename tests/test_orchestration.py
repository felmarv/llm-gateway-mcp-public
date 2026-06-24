"""compose pipeline tests with a mock dispatch."""
import asyncio

from orchestration import compose, roles

ROUTING = {
    "orchestration": {
        "roles": {
            "planner": {"provider": "gemini", "model": "gemini-2.5-pro"},
            "worker": {"provider": "openai", "model": "gpt-4o"},
            "verifier": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        },
        "depth": {
            "trivial": {"plan": False, "verify": False},
            "standard": {"plan": False, "verify": True},
            "complex": {"plan": True, "verify": True, "replan_on_fail": True, "max_replans": 1},
        },
    }
}


def make_dispatch(verifier_says="ACCEPT", record=None):
    async def dispatch(provider, model, messages, max_tokens=None, **_):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        if record is not None:
            record.append({"model": model, "system": system, "user": user})
        if "Verifier" in system:
            text = verifier_says
        elif "Thinker" in system:
            text = "PLAN: a, b, c"
        elif "Worker" in system:
            text = "ARTIFACT"
        else:
            text = "answer"
        return {"status": "success", "data": {"text": text},
                "meta": {"provider": provider, "model": model, "cost_usd_approx": 0.0}}
    return dispatch


def test_trivial_depth_worker_only():
    out = asyncio.run(compose.compose("t", routing=ROUTING, depth="trivial",
                                      dispatch=make_dispatch()))
    assert out["status"] == "success"
    assert [s["role"] for s in out["meta"]["steps"]] == ["worker"]


def test_standard_depth_worker_then_verify():
    out = asyncio.run(compose.compose("t", routing=ROUTING, depth="standard",
                                      dispatch=make_dispatch()))
    assert [s["role"] for s in out["meta"]["steps"]] == ["worker", "verifier"]
    assert out["data"]["verdict"]["accepted"] is True


def test_complex_depth_runs_planner():
    record = []
    out = asyncio.run(compose.compose("draft a thing", routing=ROUTING, depth="complex",
                                      dispatch=make_dispatch(record=record)))
    roles_run = [s["role"] for s in out["meta"]["steps"]]
    assert roles_run[:3] == ["planner", "worker", "verifier"]
    # The verifier must NOT see the plan text.
    verifier_call = next(r for r in record if "Verifier" in r["system"])
    assert "PLAN: a, b, c" not in verifier_call["user"]


def test_verifier_blind_to_plan_in_meta():
    out = asyncio.run(compose.compose("t", routing=ROUTING, depth="complex",
                                      dispatch=make_dispatch()))
    assert out["meta"]["visibility"]["verifier_blind_to_plan"] is True


def test_revise_triggers_one_gated_replan_then_stops():
    out = asyncio.run(compose.compose("t", routing=ROUTING, depth="complex",
                                      dispatch=make_dispatch(verifier_says="REVISE: missing X")))
    # Never-accepting verifier -> caps at max_replans then returns needs_human_review.
    assert out["status"] == "needs_human_review"
    assert out["meta"]["rounds"] == 1
    planner_runs = [s for s in out["meta"]["steps"] if s["role"] == "planner"]
    assert len(planner_runs) == 2  # initial + one re-plan


def test_worker_failure_propagates():
    async def failing(provider, model, messages, max_tokens=None, **_):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "Worker" in system:
            return {"status": "error", "error": "boom", "meta": {"model": model}}
        return {"status": "success", "data": {"text": "ok"}, "meta": {"model": model}}
    out = asyncio.run(compose.compose("t", routing=ROUTING, depth="standard", dispatch=failing))
    assert out["status"] == "error" and out["meta"]["failed_stage"] == "worker"


def test_parse_verdict():
    assert roles.parse_verdict("ACCEPT")[0] is True
    assert roles.parse_verdict("REVISE: nope")[0] is False
    assert roles.parse_verdict("REVISE: nope")[1] == "nope"
