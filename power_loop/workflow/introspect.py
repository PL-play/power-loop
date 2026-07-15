"""Workflow introspection (D4) — read the durable journal.

``list_workflows`` / ``get_workflow`` read the per-parent-session journal written
by :mod:`power_loop.workflow.journal`. ``detail=True`` enriches each step with
live ``get_session_stats`` for its sub-agent session.
"""

from __future__ import annotations

from typing import Any

from . import journal
from .journal import _now_ms

__all__ = ["list_workflows", "get_workflow"]


async def list_workflows(loop: Any, parent_sid: str) -> list[dict[str, Any]]:
    """Light summaries of every workflow run created under ``parent_sid``."""
    store = await loop._ensure_store()
    out: list[dict[str, Any]] = []
    for run_id in await journal.list_run_ids(store, parent_sid):
        j = await journal.read(store, parent_sid, run_id)
        if j is None:
            continue
        out.append({
            "run_id": run_id,
            "workflow": j.get("workflow"),
            "status": j.get("status"),
            "created_at_ms": j.get("created_at_ms"),
            "finished_at_ms": j.get("finished_at_ms"),
            "steps": len(j.get("steps", [])),
        })
    return out


async def get_workflow(
    loop: Any, parent_sid: str, run_id: str, *, detail: bool = False
) -> dict[str, Any] | None:
    """The full journal blob for one run, or ``None`` if unknown.

    With ``detail=True``, each step is enriched with ``session_stats`` from the
    sub-agent session (sends / rounds / total_tokens), best-effort.
    """
    store = await loop._ensure_store()
    j = await journal.read(store, parent_sid, run_id)
    if j is None:
        return None
    # A running run's ``steps`` only fills as nodes SETTLE — surface the live heartbeat
    # (which leaves are executing right now, for how long) so a long single-leaf run is
    # distinguishable from a hung one without waiting for it to finish.
    if j.get("status") == "running":
        j = dict(j)
        now = _now_ms()
        created = int(j.get("created_at_ms") or now)
        j["elapsed_s"] = max(0, (now - created) // 1000)
        live = j.get("live") or {}
        j["live_nodes"] = [
            {"node_id": nid, "running_for_s": max(0, (now - int(ts or now)) // 1000)}
            for nid, ts in sorted(live.items())
        ]
        if not j.get("steps") and j["live_nodes"]:
            j["note"] = (
                "steps 只在节点完成时落账;live_nodes 是当前正在执行的叶子及其已运行时长 —— "
                "有 live_nodes 且时长在涨 = 在干活,不是卡死。"
            )
    if not detail:
        return j
    enriched = dict(j)
    steps = []
    for s in j.get("steps", []):
        s = dict(s)
        sid = s.get("session_id")
        if sid:
            stats = None
            try:
                stats = await loop.get_session_stats(sid)
            except Exception:  # noqa: BLE001 — session may be gone; skip stats
                stats = None
            if stats is not None:
                s["session_stats"] = {
                    "sends": stats.sends,
                    "rounds": stats.rounds,
                    "total_tokens": stats.total_tokens,
                }
        steps.append(s)
    enriched["steps"] = steps
    return enriched
