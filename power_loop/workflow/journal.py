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
* ``set_runtime_state`` is last-write-wins with no CAS — callers must funnel all
  writes of a given run's blob through a single coroutine (the detached task);
  the only other writer is the wake-guard, which touches a disjoint field.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from power_loop.runtime.session_store import SessionStore

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

# A run in any of these states is FINISHED: its status/result/steps are frozen.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


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
) -> dict[str, Any]:
    now = _now_ms()
    return {
        "run_id": run_id,
        "workflow": workflow,
        "driver_sid": driver_sid,
        # The serialized WorkflowSpec (WorkflowSpec.to_dict()) so the run can be
        # resumed after a process restart without the caller re-supplying it.
        "spec": spec,
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


def seed(
    store: SessionStore,
    parent_sid: str,
    run_id: str,
    workflow: str,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the run journal and add it to the parent's run index."""
    j = new_journal(run_id, workflow, spec=spec)
    store.set_runtime_state(parent_sid, run_key(run_id), j)
    _append_index(store, parent_sid, run_id)
    return j


def _append_index(store: SessionStore, parent_sid: str, run_id: str) -> None:
    idx = store.get_runtime_state(parent_sid, INDEX_KEY, default=[]) or []
    if run_id not in idx:
        idx.append(run_id)
        store.set_runtime_state(parent_sid, INDEX_KEY, idx)


def read(store: SessionStore, parent_sid: str, run_id: str) -> dict[str, Any] | None:
    return store.get_runtime_state(parent_sid, run_key(run_id), default=None)


def update(
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
    j = store.get_runtime_state(parent_sid, run_key(run_id), default=None)
    if j is None:
        return None
    if _is_terminal(j.get("status")) and not allow_terminal:
        return j  # frozen — ignore the late write, first finalize wins
    j.update(fields)
    j["updated_at_ms"] = _now_ms()
    store.set_runtime_state(parent_sid, run_key(run_id), j)
    return j


def record_step(
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
    j = store.get_runtime_state(parent_sid, run_key(run_id), default=None)
    if j is None or _is_terminal(j.get("status")):
        # Gone, or already finalized: a step settling this late (an orphaned sibling
        # under on_error='halt', H1.2) must not clobber the terminal status/result
        # via its stale full-blob write.
        return
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
    # Re-read immediately before writing so the step merges onto the freshest blob
    # (preserving a status/result a concurrent finalize just wrote) and bail if the
    # run finalized in between — shrinks the un-CAS'd read-modify-write window.
    fresh = store.get_runtime_state(parent_sid, run_key(run_id), default=None)
    if fresh is None or _is_terminal(fresh.get("status")):
        return
    steps = [s for s in fresh.get("steps", []) if s.get("node_id") != node_id]
    steps.append(step)
    fresh["steps"] = steps
    fresh["updated_at_ms"] = _now_ms()
    store.set_runtime_state(parent_sid, run_key(run_id), fresh)


def finalize(store: SessionStore, parent_sid: str, run_id: str, result: WorkflowResult) -> None:
    update(
        store,
        parent_sid,
        run_id,
        status=result.status,
        result=result.to_dict(),
        finished_at_ms=_now_ms(),
        error=(result.errors[0] if result.errors else None),
    )


def fail(store: SessionStore, parent_sid: str, run_id: str, exc: BaseException) -> None:
    update(
        store,
        parent_sid,
        run_id,
        status="failed",
        finished_at_ms=_now_ms(),
        error=f"{type(exc).__name__}: {exc}",
    )


def list_run_ids(store: SessionStore, parent_sid: str) -> list[str]:
    return list(store.get_runtime_state(parent_sid, INDEX_KEY, default=[]) or [])
