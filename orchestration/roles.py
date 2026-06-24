"""Roles: Thinker (planner) / Worker / Verifier.

The delta over "which model for which task_class" is "which ROLE each model is
best at" (Trinity T-03 / Conductor F-03). Both the role->model assignment and the
per-role instruction templates are declarative in routing.yaml under
`orchestration:`, so they can be tuned without touching code.

    orchestration:
      roles:
        planner:  {provider: gemini,    model: gemini-2.5-pro}
        worker:   {provider: openai,    model: gpt-4o}
        verifier: {provider: anthropic, model: claude-sonnet-4-6}
      instructions:
        planner:  "Decompose ..."
        worker:   "Execute only ..."
        verifier: "Reply ACCEPT or REVISE:<reasons> ..."
      depth:
        trivial:  {plan: false, verify: false}
        standard: {plan: false, verify: true}
        complex:  {plan: true,  verify: true, replan_on_fail: true, max_replans: 1}
"""
from __future__ import annotations

ROLES = ("planner", "worker", "verifier")

# Defaults used when routing.yaml omits an instruction for a role.
DEFAULT_INSTRUCTIONS = {
    "planner": (
        "You are the Thinker. Decompose the task into atomic, verifiable steps. "
        "Do NOT solve or draft the artifact. Return: assumptions, subtasks, risks, "
        "and a verification checklist."
    ),
    "worker": (
        "You are the Worker. Execute ONLY the assigned task using the provided plan "
        "and context. Do not critique the plan unless it is materially contradictory. "
        "Return only the requested product."
    ),
    "verifier": (
        "You are the Verifier. Judge whether the output satisfies the ORIGINAL task "
        "on its own merits. Reply with exactly one of: 'ACCEPT' or 'REVISE: <concrete reasons>'. "
        "No other prose."
    ),
}


def role_config(routing: dict, role: str) -> dict:
    """Return {provider, model} for a role from routing.yaml/orchestration.roles."""
    cfg = ((routing.get("orchestration") or {}).get("roles") or {}).get(role)
    if not cfg or "provider" not in cfg or "model" not in cfg:
        raise ValueError(
            f"orchestration.roles.{role} missing provider/model in routing.yaml"
        )
    return {"provider": cfg["provider"], "model": cfg["model"],
            "max_tokens": cfg.get("max_tokens")}


def role_instruction(routing: dict, role: str) -> str:
    instr = ((routing.get("orchestration") or {}).get("instructions") or {}).get(role)
    return instr or DEFAULT_INSTRUCTIONS[role]


def depth_config(routing: dict, depth: str) -> dict:
    table = (routing.get("orchestration") or {}).get("depth") or {}
    cfg = table.get(depth)
    if cfg is None:
        # Sensible built-in graduation if routing.yaml doesn't declare one.
        cfg = {
            "trivial": {"plan": False, "verify": False},
            "standard": {"plan": False, "verify": True},
            "complex": {"plan": True, "verify": True, "replan_on_fail": True, "max_replans": 1},
        }.get(depth)
    if cfg is None:
        raise ValueError(f"unknown orchestration depth: {depth!r}")
    return cfg


def parse_verdict(text: str) -> tuple[bool, str]:
    """Parse a verifier reply into (accepted, reasons)."""
    head = (text or "").strip()
    upper = head.upper()
    if upper.startswith("ACCEPT"):
        return True, ""
    if upper.startswith("REVISE"):
        reasons = head.split(":", 1)[1].strip() if ":" in head else head
        return False, reasons
    # Be lenient: an ACCEPT/REVISE anywhere in the first line still counts.
    first_line = head.splitlines()[0].upper() if head else ""
    if "ACCEPT" in first_line and "REVISE" not in first_line:
        return True, ""
    return False, head  # default to "needs revision" when ambiguous
