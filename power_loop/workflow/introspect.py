"""Workflow introspection (D4) — read the durable journal.

``list_workflows`` / ``get_workflow`` read the per-parent-session journal written
by :mod:`power_loop.workflow.journal`. ``detail=True`` enriches each step with
live ``get_session_stats`` for its sub-agent session.
"""

from __future__ import annotations

from typing import Any

from . import journal

__all__ = ["list_workflows", "get_workflow"]


def list_workflows(loop: Any, parent_sid: str) -> list[dict[str, Any]]:
    """Light summaries of every workflow run created under ``parent_sid``."""
    store = loop.store
    out: list[dict[str, Any]] = []
    for run_id in journal.list_run_ids(store, parent_sid):
        j = journal.read(store, parent_sid, run_id)
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


def get_workflow(
    loop: Any, parent_sid: str, run_id: str, *, detail: bool = False
) -> dict[str, Any] | None:
    """The full journal blob for one run, or ``None`` if unknown.

    With ``detail=True``, each step is enriched with ``session_stats`` from the
    sub-agent session (sends / rounds / total_tokens), best-effort.
    """
    store = loop.store
    j = journal.read(store, parent_sid, run_id)
    if j is None:
        return None
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
                stats = loop.get_session_stats(sid)
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
