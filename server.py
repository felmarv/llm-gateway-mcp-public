#!/usr/bin/env python3
"""llm-gateway-mcp — a policy-routed MCP gateway over multiple LLM providers.

Tools exposed to an MCP client:
  - llm_route(task_class, prompt, system?, max_tokens?, override_model?)
  - llm_orchestrate(task, depth?)        # plan -> execute -> verify pipeline
  - llm_routing_info()                   # inspect the active policy

Routing policy is declarative in routing.yaml. Providers (OpenAI, Anthropic,
Gemini, DeepSeek) each read their own API key from the environment. See README.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Load .env if python-dotenv is available (optional convenience).
try:  # pragma: no cover
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass

from mcp.server.fastmcp import FastMCP

import providers
from orchestration import compose as _compose
from orchestration import independence
from policies import (
    circuit_breaker,
    cost_estimator,
    cost_ledger,
    error_taxonomy,
    fallback as fb,
    retry_backoff,
)

ROUTING_FILE = ROOT / "routing.yaml"
ROUTING = yaml.safe_load(ROUTING_FILE.read_text()) if ROUTING_FILE.exists() else {}
DEFAULTS = ROUTING.get("defaults", {})
PREFLIGHT = ROUTING.get("cost_preflight", {})

mcp = FastMCP("llm-gateway")


# ── Core dispatch with policy enforcement ─────────────────────────────────────
async def _call(
    provider: str,
    model: str,
    prompt: str,
    system: str | None,
    max_tokens: int | None,
    task_class: str | None = None,
    _allow_fallback: bool = True,
) -> dict:
    """Dispatch one (provider, model) call through the policy stack."""
    # 1) Cost preflight + caps + kill switches.
    pricing = providers.model_pricing(provider, model)
    if PREFLIGHT.get("enabled") and max_tokens:
        pf = cost_estimator.preflight(
            prompt=prompt, system=system, pricing=pricing, max_tokens_out=max_tokens,
            chars_per_token=PREFLIGHT.get("chars_per_token", 4),
            warn_usd=PREFLIGHT.get("warn_usd", 0.50),
            block_usd=PREFLIGHT.get("block_usd", 15.00),
        )
        if pf["action"] == "block":
            return {"status": "error", "error": f"cost block: {pf['reason']}",
                    "meta": {"provider": provider, "model": model, "preflight": pf}}
        caps = cost_ledger.check_caps(
            provider=provider, model=model,
            projected_cost_usd=pf.get("estimated_max_cost_usd") or 0.0,
        )
        if caps["action"] == "block":
            return {"status": "error", "error": f"cost block: {caps['reason']}",
                    "meta": {"provider": provider, "model": model, "caps": caps}}
    else:
        pf = None

    # 2) Circuit breaker.
    br = circuit_breaker.allow(provider, model)
    if br["action"] == "block":
        return {"status": "error", "error": br["reason"],
                "meta": {"provider": provider, "model": model, "circuit_breaker": br}}

    # 3) Dispatch with transient-retry.
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    result, retries = await retry_backoff.with_retry(
        lambda: providers.dispatch(provider, model, messages, max_tokens=max_tokens)
    )

    # 4) Record breaker state.
    category = error_taxonomy.classify(result)
    if result.get("status") == "success":
        circuit_breaker.record_success(provider, model)
    elif error_taxonomy.should_open_breaker(category):
        circuit_breaker.record_failure(provider, model)

    # 5) Fallback (per task_class) if the provider looks down.
    if (result.get("status") == "error" and task_class and _allow_fallback
            and fb.provider_is_down(result) and fb.is_allowed(ROUTING, task_class)):
        for alt in fb.get_chain(ROUTING, task_class):
            if alt["provider"] == provider and alt["model"] == model:
                continue
            alt_result = await _call(
                alt["provider"], alt["model"], prompt, system, max_tokens,
                task_class=task_class, _allow_fallback=False,
            )
            if alt_result.get("status") == "success":
                alt_result.setdefault("meta", {})["fallback_from"] = {"provider": provider, "model": model}
                result = alt_result
                break

    # 6) Annotate + record spend.
    meta = result.setdefault("meta", {})
    meta["task_class"] = task_class
    meta["retries"] = retries
    if pf is not None:
        meta["preflight"] = pf
    if category != "ok":
        meta["error_category"] = category
    if result.get("status") == "success":
        cost_ledger.record_spend(
            provider=meta.get("provider", provider), model=meta.get("model", model),
            task_class=task_class, amount_usd=meta.get("cost_usd_approx", 0.0) or 0.0,
        )
    return result


# ── Tool: llm_route ───────────────────────────────────────────────────────────
@mcp.tool()
async def llm_route(
    task_class: str,
    prompt: str,
    system: str | None = None,
    max_tokens: int | None = None,
    override_model: str | None = None,
) -> dict:
    """Route a prompt to the right model by task_class (see llm_routing_info).

    Single-model task_classes return one answer. Panel task_classes (those with
    `members:` in routing.yaml, e.g. dual_opinion / triple_review) run every
    member in parallel on the SAME original prompt and certify independence in
    meta — agreement between members is then real evidence.

    Args:
        task_class: a key under routing.yaml/defaults.
        prompt: the user content.
        system: optional system prompt.
        max_tokens: override the routed default output cap.
        override_model: force a specific model id (escape hatch; skips the policy
            model choice but keeps cost/breaker/fallback policies).
    """
    cfg = DEFAULTS.get(task_class)
    if cfg is None:
        return {"status": "error",
                "error": f"unknown task_class {task_class!r}. Options: {sorted(DEFAULTS)}"}

    mt = max_tokens if max_tokens is not None else cfg.get("max_tokens")

    # Panel flow (parallel, blind, independence-certified).
    if "members" in cfg:
        members = cfg["members"]
        results = await asyncio.gather(*[
            _call(m["provider"], m["model"], prompt, system, mt, task_class=task_class)
            for m in members
        ])
        # Independence: every member received the same original prompt.
        member_inputs = [prompt for _ in members]
        try:
            vis = independence.certify(ROUTING, task_class, prompt, member_inputs)
        except ValueError as exc:
            return {"status": "error", "error": str(exc), "meta": {"task_class": task_class}}
        total_cost = sum((r.get("meta") or {}).get("cost_usd_approx", 0.0) or 0.0 for r in results)
        cap = cfg.get("cost_cap_usd")
        return {
            "status": "success",
            "data": {"members": results},
            "meta": {
                "task_class": task_class,
                "visibility": vis,
                "total_cost_usd_approx": round(total_cost, 6),
                "cost_cap_usd": cap,
                "cap_exceeded": bool(cap and total_cost > cap),
                "synthesis_pending": "caller compares convergence + dissent across members",
            },
        }

    # Single-model flow.
    provider = cfg["provider"]
    model = override_model or cfg["model"]
    return await _call(provider, model, prompt, system, mt, task_class=task_class)


# ── Tool: llm_orchestrate (compose) ───────────────────────────────────────────
@mcp.tool()
async def llm_orchestrate(task: str, depth: str = "standard") -> dict:
    """Run a plan -> execute -> verify pipeline over multiple roles/models.

    Roles (planner/worker/verifier) map to models in routing.yaml/orchestration.
    The verifier is deliberately blind to the plan and judges the artifact against
    the original task. Depth graduates the work:
      - trivial  : worker only
      - standard : worker + verify
      - complex  : plan + worker + verify (+ one failure-gated re-plan)

    Returns {status, data:{artifact, plan, verdict}, meta:{steps, visibility, ...}}.
    """
    if depth not in ("trivial", "standard", "complex"):
        return {"status": "error", "error": f"depth must be trivial|standard|complex, got {depth!r}"}
    return await _compose.compose(
        task, routing=ROUTING, depth=depth,
        dispatch=lambda **kw: _orchestration_dispatch(**kw),
    )


async def _orchestration_dispatch(provider, model, messages, max_tokens=None, **_):
    """Adapt compose's role calls (system+user messages) onto the policy stack."""
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return await _call(provider, model, user, system, max_tokens, task_class="orchestrate")


# ── Tool: llm_routing_info ────────────────────────────────────────────────────
@mcp.tool()
async def llm_routing_info() -> dict:
    """Inspect the active routing policy (version, task_classes, providers)."""
    return {
        "status": "success",
        "data": {
            "version": ROUTING.get("version"),
            "task_classes": sorted(DEFAULTS),
            "providers": sorted(providers.PROVIDERS),
            "panels": [k for k, v in DEFAULTS.items() if "members" in v],
            "visibility": sorted(ROUTING.get("visibility", {})),
            "orchestration_depths": sorted((ROUTING.get("orchestration") or {}).get("depth", {})),
            "circuit_breakers": circuit_breaker.status(),
        },
    }


if __name__ == "__main__":
    mcp.run()
