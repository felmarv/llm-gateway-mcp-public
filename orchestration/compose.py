"""compose: a plan -> execute -> verify orchestration pipeline.

Topology (sequential, for producing ONE artifact):

    [plan]    PLANNER  : decompose the task into a plan.            (sees: task)
    EXECUTE   WORKER   : produce the artifact following the plan.    (sees: task + plan)
    [verify]  VERIFIER : ACCEPT|REVISE against the ORIGINAL task.    (sees: task + artifact;
                                                                       NOT the plan)
    [gated]   RE-PLAN  : only if REVISE and depth allows; cap 1 round, then stop.

Visibility is controlled per step: the verifier is deliberately blind to the plan
so it cannot "buy into" the plan's logic — it judges the artifact against the task
itself (Conductor access_list applied for independence, not synergy).

Depth graduates the work:
    trivial  -> worker only
    standard -> worker + verify
    complex  -> plan + worker + verify (+ one gated re-plan)

`depth` is a declared parameter, NOT an LLM difficulty classifier — the caller
states how hard the task is; no extra model call guesses it.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from . import roles

# A dispatch callable: (provider, model, messages, max_tokens) -> envelope dict.
Dispatch = Callable[..., Awaitable[dict]]


async def _run_role(
    routing: dict,
    role: str,
    user_content: str,
    dispatch: Dispatch,
) -> dict:
    cfg = roles.role_config(routing, role)
    messages = [
        {"role": "system", "content": roles.role_instruction(routing, role)},
        {"role": "user", "content": user_content},
    ]
    return await dispatch(
        provider=cfg["provider"], model=cfg["model"],
        messages=messages, max_tokens=cfg.get("max_tokens"),
    )


def _text(envelope: dict) -> str:
    return (envelope.get("data") or {}).get("text", "")


async def compose(
    task: str,
    *,
    routing: dict,
    depth: str = "standard",
    dispatch: Dispatch,
) -> dict:
    """Run the orchestration pipeline for `task`.

    Returns:
        {
          "status": "success" | "error",
          "data": {"artifact": str, "plan": str|None, "verdict": {...}|None},
          "meta": {"depth", "steps":[...], "rounds", "visibility": {...}}
        }
    """
    dcfg = roles.depth_config(routing, depth)
    steps: list[dict] = []

    async def fail(stage: str, env: dict) -> dict:
        return {
            "status": "error",
            "error": f"{stage} failed: {env.get('error')}",
            "data": {"artifact": None, "plan": None, "verdict": None},
            "meta": {"depth": depth, "steps": steps, "failed_stage": stage},
        }

    plan_text: str | None = None
    rounds = 0
    max_replans = int(dcfg.get("max_replans", 1)) if dcfg.get("replan_on_fail") else 0

    while True:
        # 1) PLAN (optional) -----------------------------------------------------
        if dcfg.get("plan"):
            plan_prompt = task if plan_text is None else (
                f"{task}\n\n--- Previous attempt was REVISED for: {plan_text} ---\n"
                "Produce an improved plan addressing those reasons."
            )
            plan_env = await _run_role(routing, "planner", plan_prompt, dispatch)
            steps.append({"role": "planner", "status": plan_env.get("status"),
                          "model": (plan_env.get("meta") or {}).get("model")})
            if plan_env.get("status") != "success":
                return await fail("planner", plan_env)
            plan_text = _text(plan_env)

        # 2) EXECUTE -------------------------------------------------------------
        worker_input = task if not plan_text else (
            f"TASK:\n{task}\n\nPLAN TO FOLLOW:\n{plan_text}\n\n"
            "Produce the artifact requested by the task, following the plan."
        )
        worker_env = await _run_role(routing, "worker", worker_input, dispatch)
        steps.append({"role": "worker", "status": worker_env.get("status"),
                      "model": (worker_env.get("meta") or {}).get("model")})
        if worker_env.get("status") != "success":
            return await fail("worker", worker_env)
        artifact = _text(worker_env)

        # 3) VERIFY (optional) ---------------------------------------------------
        if not dcfg.get("verify"):
            return _result("success", artifact, plan_text, None, steps, depth, rounds)

        # The verifier sees the ORIGINAL task + the artifact, but NOT the plan.
        verify_input = (
            f"ORIGINAL TASK:\n{task}\n\nCANDIDATE ARTIFACT:\n{artifact}\n\n"
            "Does the artifact satisfy the original task on its own merits?"
        )
        verify_env = await _run_role(routing, "verifier", verify_input, dispatch)
        steps.append({"role": "verifier", "status": verify_env.get("status"),
                      "model": (verify_env.get("meta") or {}).get("model")})
        if verify_env.get("status") != "success":
            return await fail("verifier", verify_env)

        accepted, reasons = roles.parse_verdict(_text(verify_env))
        verdict = {"accepted": accepted, "reasons": reasons, "raw": _text(verify_env)}

        if accepted or rounds >= max_replans:
            status = "success" if accepted else "needs_human_review"
            return _result(status, artifact, plan_text, verdict, steps, depth, rounds)

        # Gated re-plan: feed reasons back, loop once.
        rounds += 1
        plan_text = reasons  # carried into the re-plan prompt above
        if not dcfg.get("plan"):
            # No planner in this depth: re-run worker with the reasons appended.
            task = f"{task}\n\n--- Reviewer asked to revise: {reasons} ---"
            plan_text = None


def _result(status, artifact, plan, verdict, steps, depth, rounds) -> dict:
    visibility = {
        "verifier_sees": ["original_task", "artifact"],
        "verifier_blind_to_plan": True,
        "independence_certified": True,
    }
    return {
        "status": status,
        "data": {"artifact": artifact, "plan": plan, "verdict": verdict},
        "meta": {"depth": depth, "rounds": rounds, "steps": steps, "visibility": visibility},
    }
