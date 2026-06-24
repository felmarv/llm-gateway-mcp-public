# llm-gateway-mcp

A small, self-hostable **MCP server** that routes prompts to multiple LLM
providers by a **declarative policy**, with **multi-role orchestration** inspired
by two Sakana AI papers. Clone it, drop in *your own* API keys, and point any MCP
client at it.

> The caller never picks a model. It names a **`task_class`**; the gateway
> imposes the right model, applies cost/reliability policies, and (optionally)
> runs a plan → execute → verify pipeline across several models.

- **4 pluggable providers** — OpenAI, Anthropic, Google Gemini (API key, *not*
  Vertex), DeepSeek. Each reads its own key from the environment.
- **Declarative routing** (`routing.yaml`) — model × provider × max_tokens per
  task_class, with cost preflight, circuit breaker, retry/backoff, and per-task
  fallback chains.
- **Sakana-inspired orchestration that actually runs** — independence
  certification, Thinker/Worker/Verifier roles, and a compose pipeline with
  controlled per-step visibility and failure-gated re-planning.
- **No keys? Still testable.** `smoke_test.py` and the `tests/` suite mock every
  provider call, so the whole thing runs green offline.

---

## The orchestration features (the interesting part)

These implement, in a **domain-agnostic** way, ideas from:

- **"TRINITY: An Evolved LLM Coordinator"** — Sakana AI, arXiv:2512.04695, ICLR 2026
- **"Learning to Orchestrate Agents in Natural Language with the Conductor"** —
  Sakana AI, arXiv:2512.04388, ICLR 2026

A key adaptation: the papers optimize **synergy** (workers reading each other to
converge). For cross-checking we often want the **opposite — independence** — so
that agreement between models is *evidence*, not an echo. This gateway keeps the
two planes separate on purpose.

| # | Feature | Where | What it does |
|---|---------|-------|--------------|
| 1 | **Declarative routing by `task_class`** | `routing.yaml`, `server.py` | model × provider × `max_tokens`, cost preflight before dispatch. |
| 2 | **Independence certification** (Conductor *T-02 access_list / visibility*) | `orchestration/independence.py` | For blind parallel panels, proves each member saw **only** the original prompt and stamps `independence_certified` into `meta`. `enforce: hard` fails closed. |
| 3 | **Thinker / Worker / Verifier roles** (Trinity *T-03*) | `orchestration/roles.py`, `routing.yaml` | Per-role instruction templates + a configurable role → model table. |
| 4 | **compose: plan → execute → verify** | `orchestration/compose.py` | One model plans, another executes, a third verifies. The **verifier is blind to the plan** and judges the artifact against the original task. One **failure-gated** re-plan (cap 1). |
| 5 | **Depth by difficulty** | `routing.yaml/orchestration.depth` | `trivial` = 1 step, `standard` = +verify, `complex` = full pipeline. A declared parameter — **not** an LLM difficulty classifier. |

Plus the generic reliability policies: cost preflight + caps, circuit breaker,
retry with backoff, and per-task fallback.

---

## Architecture

```
llm-gateway-mcp/
├── server.py              # FastMCP entry: llm_route, llm_orchestrate, llm_routing_info
├── routing.yaml           # declarative policy (task_classes, panels, visibility, roles, depth)
├── providers/             # one adapter per provider, shared 3-state response envelope
│   ├── __init__.py        #   registry + dispatch()
│   ├── openai.py          #   OPENAI_API_KEY    (httpx, Chat Completions)
│   ├── anthropic.py       #   ANTHROPIC_API_KEY (official anthropic SDK, Messages API)
│   ├── gemini.py          #   GEMINI_API_KEY    (google-genai, API key — NOT Vertex)
│   └── deepseek.py        #   DEEPSEEK_API_KEY  (httpx, OpenAI-compatible)
├── policies/              # generic, provider-agnostic
│   ├── cost_estimator.py  #   preflight max-cost projection
│   ├── cost_ledger.py     #   optional SQLite spend ledger + caps + kill switches
│   ├── circuit_breaker.py #   per (provider, model) breaker
│   ├── retry_backoff.py   #   transient-only retry
│   ├── error_taxonomy.py  #   normalize provider errors
│   └── fallback.py        #   per-task_class fallback chains
├── orchestration/         # the Sakana-inspired layer
│   ├── independence.py    #   access_list / visibility certification
│   ├── roles.py           #   Thinker/Worker/Verifier + depth
│   └── compose.py         #   plan → execute → verify pipeline
├── smoke_test.py          # offline structural test (mocks every model call)
└── tests/                 # pytest suite (offline)
```

**Response envelope** (every provider, every tool):

```json
{ "status": "success",
  "data": { "text": "..." },
  "meta": { "provider": "...", "model": "...", "latency_ms": 0,
            "tokens": {"input": 0, "output": 0, "total": 0},
            "cost_usd_approx": 0.0, "task_class": "..." } }
```

---

## Install & run

```bash
git clone <your-fork-url> llm-gateway-mcp
cd llm-gateway-mcp

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e ".[dev]"

cp .env.example .env                   # then put YOUR keys in .env
```

Verify it works with **no keys needed** (everything is mocked):

```bash
python smoke_test.py     # 9/9 PASS
pytest -q                # 22 passed
```

Run as an MCP server (stdio):

```bash
python server.py
```

Register it with an MCP client (example `mcp.json` entry):

```json
{
  "mcpServers": {
    "llm-gateway": {
      "command": "python",
      "args": ["/absolute/path/to/llm-gateway-mcp/server.py"]
    }
  }
}
```

You only need keys for the providers your `routing.yaml` actually targets.

---

## Tools

### `llm_route(task_class, prompt, system?, max_tokens?, override_model?)`

Routes by `task_class`. Single-model classes return one answer; **panel** classes
(those with `members:` in `routing.yaml`, e.g. `dual_opinion`, `triple_review`)
run every member in parallel on the **same original prompt** and certify
independence in `meta`.

```python
llm_route(task_class="general_reasoning", prompt="Plan a migration from X to Y.")
llm_route(task_class="triple_review",     prompt="Is this argument sound? ...")
# -> data.members = [3 independent answers], meta.visibility.independence_certified = true
```

### `llm_orchestrate(task, depth?)`

Runs the plan → execute → verify pipeline. `depth` ∈ `trivial | standard | complex`.

```python
llm_orchestrate(task="Draft a concise refund policy for a SaaS product.", depth="complex")
# -> data: { artifact, plan, verdict }, meta: { steps, rounds, visibility }
```

### `llm_routing_info()`

Returns the active policy: version, task_classes, providers, panels, visibility
contracts, orchestration depths, and current circuit-breaker state.

---

## Configuration

Everything routable lives in **`routing.yaml`** — edit it freely:

- `defaults.<task_class>` → `{provider, model, max_tokens}` (or `members:` for a panel)
- `cost_preflight` → `warn_usd` / `block_usd` thresholds
- `visibility.<task_class>` → `mode: blind`, `enforce: hard|soft`
- `fallback.<task_class>` → ordered alternates (empty = no fallback)
- `orchestration` → `roles`, per-role `instructions`, and `depth` table

Optional spend controls (env, **disabled by default** — see `.env.example`):
`LLM_GATEWAY_LEDGER`, `LLM_GATEWAY_CAP_TOTAL_MONTHLY`,
`LLM_GATEWAY_CAP_<PROVIDER>_MONTHLY`, plus kill switches
`LLM_GATEWAY_DISABLED` and `LLM_GATEWAY_EXPENSIVE_DISABLED`.

> ⚠️ Pricing in each `providers/*.py` `MODELS` table is **illustrative**. Verify
> against each provider's live pricing before trusting cost preflight in production.

---

## Providers at a glance

| Provider | Env var | Transport | Example models |
|----------|---------|-----------|----------------|
| OpenAI | `OPENAI_API_KEY` | httpx (Chat Completions) | `gpt-4o`, `gpt-4o-mini`, `o3-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic` SDK (Messages) | `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
| Google Gemini | `GEMINI_API_KEY` | `google-genai` (API key) | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` |
| DeepSeek | `DEEPSEEK_API_KEY` | httpx (OpenAI-compatible) | `deepseek-chat`, `deepseek-reasoner` |

**Adding a provider:** drop a module in `providers/` exposing `MODELS` and an
`async def complete(messages, model, max_tokens, **kwargs)` returning the shared
envelope, then register it in `providers/__init__.py`.

---

## License

MIT © Felipe Márquez. See [LICENSE](LICENSE).

Paper credits: TRINITY (arXiv:2512.04695) and Conductor (arXiv:2512.04388),
Sakana AI, ICLR 2026. This project implements ideas from those papers in a
generic form; it is not affiliated with or endorsed by Sakana AI.
