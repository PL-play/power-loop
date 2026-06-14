"""Typed results and the shared token budget for workflow runs.

The base library returns sub-agent results as a loose ``dict`` and drops token
usage on the ``run_agent_spec`` path; deterministic control flow needs typed
results it can branch on, so the workflow layer wraps every step in
:class:`AgentResult` and aggregates a run into :class:`WorkflowResult`.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AgentResult", "WorkflowResult", "SharedBudget", "WorkflowRunHandle"]


@dataclass
class AgentResult:
    """Typed result of a single ``agent`` node.

    ``payload`` is the parsed structured output (when the node declared an
    ``output_schema``); ``text`` is always the sub-agent's final text. ``ok`` is
    a convenience for ``status == "completed"``.
    """

    node_id: str
    status: str
    text: str
    payload: dict[str, Any] | None = None
    usage: dict[str, int] = field(default_factory=dict)
    session_id: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "text": self.text,
            "payload": self.payload,
            "usage": self.usage,
            "session_id": self.session_id,
            "error": self.error,
        }


@dataclass
class WorkflowResult:
    """Aggregate result of a workflow run.

    ``results`` maps each ``agent`` node id to its :class:`AgentResult`.
    ``final`` is the result of the last leaf to complete (best-effort), handy
    for the common "give me the final answer" case.
    """

    name: str
    status: str  # "completed" | "failed" | "budget_exceeded"
    results: dict[str, AgentResult] = field(default_factory=dict)
    final: AgentResult | None = None
    usage: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def summary(self) -> str:
        """A short human/LLM-readable rollup (used by the create_workflow tool)."""
        lines = [f"workflow '{self.name}' -> {self.status}"]
        for nid, r in self.results.items():
            mark = "ok" if r.ok else r.status
            lines.append(f"  [{mark}] {nid}: {_oneline(r.text)}")
        if self.errors:
            lines.append("errors:")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.usage.get("total_tokens"):
            lines.append(f"tokens: {self.usage['total_tokens']}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "final": self.final.to_dict() if self.final else None,
            "usage": self.usage,
            "errors": self.errors,
        }


def _oneline(text: str, limit: int = 160) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


class SharedBudget:
    """A token pool shared across all sub-agents of one workflow run.

    The base library's budget is strictly per-run (per ``send``); a workflow
    fanning out N sub-agents would otherwise have an unbounded total. This pool
    is *soft*: :meth:`can_spawn` lets the engine stop launching new steps once
    the pool is near-empty, while in-flight steps finish. :meth:`commit` records
    actual usage after each step. Thread/async safe via a lock (commits may land
    from concurrent fan-out tasks).
    """

    def __init__(self, max_tokens: int, *, stop_at_remaining_pct: float = 0.0) -> None:
        self.max_tokens = int(max_tokens)
        self._floor = self.max_tokens * float(stop_at_remaining_pct) / 100.0
        self._spent = 0
        self._lock = threading.Lock()

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_tokens - self._spent)

    def can_spawn(self) -> bool:
        """True while enough budget remains to be worth launching another step."""
        with self._lock:
            return (self.max_tokens - self._spent) > self._floor

    def commit(self, usage: dict[str, int] | None) -> None:
        if not usage:
            return
        with self._lock:
            self._spent += int(usage.get("total_tokens") or 0)

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {"max_tokens": self.max_tokens, "spent": self._spent,
                    "remaining": max(0, self.max_tokens - self._spent)}


@dataclass
class WorkflowRunHandle:
    """Handle for a detached workflow run.

    ``run_id`` is the journal key suffix (``workflow:run:<run_id>``) used by
    :func:`power_loop.workflow.get_workflow` to read status/detail; ``task`` is
    the background ``asyncio.Task`` driving the run (retained by the
    :class:`Workflow` so it is not garbage-collected mid-flight).

    NOTE: the completion **wake** of the parent agent is delivered by a durable
    timer — it fires only while a ``TimerRunner`` (or an external poller of
    ``store.due_timers()``) is running on the host. Without one, the run still
    completes and the journal updates, but the parent is not auto-woken.
    """

    run_id: str
    task: asyncio.Task[None]
