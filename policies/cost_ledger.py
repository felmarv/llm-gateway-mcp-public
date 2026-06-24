"""Optional persistent spend ledger (SQLite) + monthly caps + kill switches.

DISABLED BY DEFAULT — the gateway never writes files unless you opt in. Enable
and configure via environment variables:

    LLM_GATEWAY_LEDGER=1                 # turn the ledger on
    LLM_GATEWAY_LEDGER_PATH=/path/db     # default: ./.llm_gateway/ledger.db
    LLM_GATEWAY_CAP_TOTAL_MONTHLY=500    # USD; 0/unset = no cap
    LLM_GATEWAY_CAP_<PROVIDER>_MONTHLY=200

Kill switches (work even with the ledger disabled):
    LLM_GATEWAY_DISABLED=1              # block all calls
    LLM_GATEWAY_EXPENSIVE_DISABLED=1   # block any projected call > $1
"""
from __future__ import annotations

import datetime as _dt
import os
import sqlite3
from pathlib import Path


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def enabled() -> bool:
    return _truthy("LLM_GATEWAY_LEDGER")


def _ledger_path() -> Path:
    raw = os.environ.get("LLM_GATEWAY_LEDGER_PATH", "").strip()
    return Path(raw).expanduser() if raw else Path.cwd() / ".llm_gateway" / "ledger.db"


def _conn() -> sqlite3.Connection:
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS spend ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, yearmonth TEXT, "
        "provider TEXT, model TEXT, task_class TEXT, amount_usd REAL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ym ON spend(yearmonth)")
    return conn


def record_spend(*, provider: str, model: str, task_class: str | None, amount_usd: float) -> None:
    if not enabled() or not amount_usd or amount_usd <= 0:
        return
    try:
        now = _dt.datetime.now()
        with _conn() as c:
            c.execute(
                "INSERT INTO spend (timestamp, yearmonth, provider, model, task_class, amount_usd) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (now.isoformat(timespec="seconds"), now.strftime("%Y-%m"),
                 provider, model, task_class, float(amount_usd)),
            )
    except Exception:
        pass  # the ledger is observability; never break the response path


def spend_month_usd(provider: str | None = None) -> float:
    if not enabled():
        return 0.0
    ym = _dt.date.today().strftime("%Y-%m")
    try:
        with _conn() as c:
            q = "SELECT COALESCE(SUM(amount_usd), 0) FROM spend WHERE yearmonth = ?"
            params: list = [ym]
            if provider:
                q += " AND provider = ?"
                params.append(provider)
            return round(float(c.execute(q, params).fetchone()[0] or 0), 4)
    except Exception:
        return 0.0


def _cap(name: str) -> float:
    try:
        return float(os.environ.get(name, "0") or 0)
    except ValueError:
        return 0.0


def check_caps(*, provider: str, model: str, projected_cost_usd: float) -> dict:
    """Return {action: ok|block, reason?}. Honors kill switches + monthly caps."""
    if _truthy("LLM_GATEWAY_DISABLED"):
        return {"action": "block", "reason": "LLM_GATEWAY_DISABLED=1 kill switch"}
    if _truthy("LLM_GATEWAY_EXPENSIVE_DISABLED") and projected_cost_usd > 1.00:
        return {"action": "block",
                "reason": f"LLM_GATEWAY_EXPENSIVE_DISABLED=1 — projected ${projected_cost_usd:.2f} > $1"}
    if not enabled():
        return {"action": "ok"}

    cap_provider = _cap(f"LLM_GATEWAY_CAP_{provider.upper()}_MONTHLY")
    if cap_provider > 0:
        spent = spend_month_usd(provider=provider)
        if spent + projected_cost_usd > cap_provider:
            return {"action": "block",
                    "reason": f"{provider} monthly cap: ${spent:.2f}+${projected_cost_usd:.2f} > ${cap_provider:.2f}"}

    cap_total = _cap("LLM_GATEWAY_CAP_TOTAL_MONTHLY")
    if cap_total > 0:
        spent_total = spend_month_usd()
        if spent_total + projected_cost_usd > cap_total:
            return {"action": "block",
                    "reason": f"total monthly cap: ${spent_total:.2f}+${projected_cost_usd:.2f} > ${cap_total:.2f}"}
    return {"action": "ok"}
