"""Durable journal for workflow runs (D2).

Persists a run's status + per-step records into the existing
``session_runtime_state`` table via ``SessionStore.set/get_runtime_state`` —
**no new core tables**. Everything is keyed on the *parent* agent session (the
session that created the workflow): that session outlives the workflow's driver
session and is the one the completion wake targets, so introspection reads from
one place.

Layout (per parent session):

* ``workflow:run:<run_id>``  → one journal blob per run (status, steps, result).
* ``workflow:index``         → ``list[run_id]`` so runs can be enumerated
  (``SessionStore`` has no list-by-prefix / query-by-metadata).

Caveats baked in here:

* ``get_runtime_state`` cannot tell "absent" from a stored JSON ``null`` — so we
  never store a meaningful ``null`` (status is always a string, lists default to
  ``[]``).
* Every blob mutation (status/steps/index) is a read-modify-write. Bare
  ``get_runtime_state`` + ``set_runtime_state`` is NOT safe here: the async store yields
  the event loop between the read and the write, so concurrent parallel-branch steps (one
  run) or concurrent ``seed`` calls (the shared index) would lose updates. We therefore
  funnel every mutation through the store's atomic ``mutate_runtime_state`` (row-locked
  read-modify-write), which serializes them correctly even across coroutines/processes.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from power_loop.runtime.store.store import MUTATE_SKIP

if TYPE_CHECKING:
    from power_loop.runtime.store.store import SessionStore

    from .result import WorkflowResult

__all__ = [
    "JOURNAL_PREFIX",
    "INDEX_KEY",
    "run_key",
    "new_journal",
    "seed",
    "read",
    "update",
    "record_step",
    "finalize",
    "fail",
    "list_run_ids",
]

JOURNAL_PREFIX = "workflow:run:"
INDEX_KEY = "workflow:index"

# A run in any of these states is FINISHED: its status/result/steps are frozen. budget_exceeded is
# a settled outcome too (M-workflow-durability-2): without it the journal stayed writable, so an
# orphaned sub-agent settling late could still mutate a budget-exhausted run. Resume is unaffected —
# it flips a terminal run back to "running" via allow_terminal=True.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "budget_exceeded"})


def _is_terminal(status: Any) -> bool:
    return status in _TERMINAL_STATUSES


def _now_ms() -> int:
    return int(time.time() * 1000)


def run_key(run_id: str) -> str:
    return f"{JOURNAL_PREFIX}{run_id}"


def new_journal(
    run_id: str,
    workflow: str,
    driver_sid: str | None = None,
    spec: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    now = _now_ms()
    return {
        "run_id": run_id,
        "workflow": workflow,
        "driver_sid": driver_sid,
        # The serialized WorkflowSpec (WorkflowSpec.to_dict()) so the run can be
        # resumed after a process restart without the caller re-supplying it.
        "spec": spec,
        # The run's capability clamp (sorted list; null = unrestricted), persisted
        # so a resume applies the SAME leaf-tool intersection — without it a
        # resume would silently widen a clamped run's permissions. Absent in
        # pre-3.14 journals → treated as null (old behavior).
        "allowed_tools": allowed_tools,
        "status": "running",
        "created_at_ms": now,
        "updated_at_ms": now,
        "finished_at_ms": None,
        "woke": False,
        # Bumped each time the run is (re)started; lets introspection tell a
        # resumed run from a first run.
        "attempts": 1,
        "steps": [],
        "result": None,
        "error": None,
    }


async def seed(
    store: SessionStore,
    parent_sid: str,
    run_id: str,
    workflow: str,
    spec: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Create the run journal and add it to the parent's run index."""
    j = new_journal(run_id, workflow, spec=spec, allowed_tools=allowed_tools)
    await store.set_runtime_state(parent_sid, run_key(run_id), j)
    await _append_index(store, parent_sid, run_id)
    return j


async def _append_index(store: SessionStore, parent_sid: str, run_id: str) -> None:
    def _add(idx: Any) -> list[str]:
        idx = list(idx or [])
        if run_id not in idx:
            idx.append(run_id)
        return idx

    # Atomic RMW: concurrent seed() calls for different runs all mutate this one shared
    # index blob; a plain get+set would drop run_ids (lost-update). See module docstring.
    await store.mutate_runtime_state(parent_sid, INDEX_KEY, _add, default=[])


async def read(store: SessionStore, parent_sid: str, run_id: str) -> dict[str, Any] | None:
    return await store.get_runtime_state(parent_sid, run_key(run_id), default=None)


async def update(
    store: SessionStore,
    parent_sid: str,
    run_id: str,
    *,
    allow_terminal: bool = False,
    **fields: Any,
) -> dict[str, Any] | None:
    """Read-modify-write a run journal (single-writer; bumps ``updated_at_ms``).

    Once a run is terminal (completed/failed/cancelled) the journal is **frozen**: a
    late write — e.g. an orphaned sub-agent settling after the run already finalized
    (H1.2), or a duplicate finalize — must not revert status/result. Callers that
    legitimately mutate a finished run pass ``allow_terminal=True``: the completion
    wake's ``woke`` flag, and resume flipping a finished run back to ``running``.
    """
    def _apply(j: Any) -> Any:
        if j is None:
            return MUTATE_SKIP  # gone
        if _is_terminal(j.get("status")) and not allow_terminal:
            return MUTATE_SKIP  # frozen — ignore the late write, first finalize wins
        j.update(fields)
        j["updated_at_ms"] = _now_ms()
        return j

    # Atomic RMW; on skip mutate_runtime_state returns the unchanged current blob
    # (None when absent, the frozen journal when terminal) — matching the old contract.
    return await store.mutate_runtime_state(parent_sid, run_key(run_id), _apply, default=None)


async def record_step(
    store: SessionStore,
    parent_sid: str,
    run_id: str,
    *,
    node_id: str,
    status: str,
    session_id: str | None = None,
    usage: dict[str, int] | None = None,
    error: str | None = None,
    text: str | None = None,
    payload: dict[str, Any] | None = None,
    db_path: str | None = None,
) -> None:
    """Append (or replace) a per-step record as the engine progresses.

    ``text`` / ``payload`` capture the step's *output* so a completed step can be
    replayed (not re-run) on resume and feed downstream ``inputs_from`` /
    ``items_from`` / ``branch.on`` references. ``db_path`` records an
    out-of-process leaf's private db so it can be inspected after the fact.
    """
    step = {
        "node_id": node_id,
        "status": status,
        "session_id": session_id,
        "usage": usage or {},
        "error": error,
        "text": text or "",
        "payload": payload,
        "db_path": db_path,
    }

    def _merge(j: Any) -> Any:
        # Gone, or already finalized: a step settling this late (an orphaned sibling
        # under on_error='halt', H1.2) must not clobber the terminal status/result.
        if j is None or _is_terminal(j.get("status")):
            return MUTATE_SKIP
        # Replace-or-append this node's record (idempotent on replay), merging onto the
        # freshest blob so a concurrent finalize's status/result is preserved.
        steps = [s for s in j.get("steps", []) if s.get("node_id") != node_id]
        steps.append(step)
        j["steps"] = steps
        j["updated_at_ms"] = _now_ms()
        return j

    # Atomic, row-locked RMW: parallel branches of one run record concurrently, so a
    # plain get+set would let siblings clobber each other (lost-update). See docstring.
    await store.mutate_runtime_state(parent_sid, run_key(run_id), _merge, default=None)


async def finalize(store: SessionStore, parent_sid: str, run_id: str, result: WorkflowResult) -> None:
    await update(
        store,
        parent_sid,
        run_id,
        status=result.status,
        result=result.to_dict(),
        finished_at_ms=_now_ms(),
        error=(result.errors[0] if result.errors else None),
    )


async def fail(store: SessionStore, parent_sid: str, run_id: str, exc: BaseException) -> None:
    await update(
        store,
        parent_sid,
        run_id,
        status="failed",
        finished_at_ms=_now_ms(),
        error=f"{type(exc).__name__}: {exc}",
    )


async def list_run_ids(store: SessionStore, parent_sid: str) -> list[str]:
    return list(await store.get_runtime_state(parent_sid, INDEX_KEY, default=[]) or [])
