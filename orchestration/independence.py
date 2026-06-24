"""Independence certification (Conductor lesson T-02: access_list / visibility).

In a blind parallel panel, every member must receive ONLY the original prompt —
none may see another member's output. When that holds, agreement between members
is genuine evidence rather than an echo. This module makes that property explicit
and auditable instead of merely "remembered".

`certify(...)` reads the declared `visibility` contract from routing.yaml and,
for `mode: blind`, verifies structurally that each member's input is derived
solely from the original prompt, then stamps `independence_certified` into meta.
"""
from __future__ import annotations

from typing import Any


def visibility_for(routing: dict, task_class: str) -> dict | None:
    """Return the declared visibility contract for a task_class, or None."""
    return (routing.get("visibility") or {}).get(task_class)


def assert_blind(original_prompt: str, member_inputs: list[str]) -> bool:
    """True iff the flow is genuinely blind.

    A blind panel means `members_see: original_prompt_only` — every member input
    must be the original prompt and NOTHING else. So we strip the original prompt
    from each member input once; if any residual (non-whitespace) content remains,
    that member saw more than the original (e.g. another member's output threaded
    in by an accidental future refactor) and the flow is NOT blind.
    """
    if not member_inputs:
        return True
    for mi in member_inputs:
        if original_prompt not in mi:
            return False
        residual = mi.replace(original_prompt, "", 1).strip()
        if residual:
            return False
    return True


def certify(
    routing: dict,
    task_class: str,
    original_prompt: str,
    member_inputs: list[str],
) -> dict | None:
    """Build the `meta.visibility` block for a multi-model flow.

    Returns None for task_classes with no declared visibility (legacy behavior).
    For `mode: blind` with `enforce: hard`, raises ValueError if independence
    cannot be certified — fail-closed, so a contaminated flow never ships
    silently labeled as independent.
    """
    contract = visibility_for(routing, task_class)
    if not contract:
        return None

    mode = contract.get("mode")
    certified: bool | None = None
    if mode == "blind":
        certified = assert_blind(original_prompt, member_inputs)
        if not certified and contract.get("enforce") == "hard":
            raise ValueError(
                f"independence certification FAILED for blind task_class "
                f"{task_class!r}: a member input is not derived solely from the "
                f"original prompt (enforce=hard)."
            )

    out: dict[str, Any] = {
        "mode": mode,
        "members_see": contract.get("members_see"),
        "enforce": contract.get("enforce"),
    }
    if certified is not None:
        out["independence_certified"] = certified
    return out


def self_test() -> bool:
    """Tiny self-check used by smoke_test.py. Returns True if the certifier works."""
    routing = {
        "visibility": {
            "panel": {"mode": "blind", "members_see": "original_prompt_only", "enforce": "hard"},
        }
    }
    prompt = "Is this contract enforceable?"
    # Blind: all members see only the original -> certified True.
    ok_blind = certify(routing, "panel", prompt, [prompt, prompt, prompt])
    if not (ok_blind and ok_blind["independence_certified"] is True):
        return False
    # Contaminated: one member's input quotes another's output -> must raise.
    try:
        certify(routing, "panel", prompt, [prompt, prompt + " Member A said: YES", prompt])
        return False  # should have raised
    except ValueError:
        pass
    # Undeclared task_class -> None (legacy).
    return certify(routing, "undeclared", prompt, [prompt]) is None
