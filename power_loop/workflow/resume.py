"""Orchestration-level resume across a process restart (full tier).

A detached run journals its spec (``WorkflowSpec.to_dict()``) and, per node as it
settles, the node's status + output (``text`` / ``payload``). If the process dies
mid-run, :func:`resume_run` / :func:`resume_detached` rebuild the engine from that
journal and re-run **only the unfinished tail**:

* completed nodes are *replayed* (their cached result is returned, the sub-agent is
  not called again) so downstream ``inputs_from`` / ``items_from`` / ``branch.on``
  references still resolve;
* everything else re-runs.

Granularity & idempotency
-------------------------
Replay is keyed by node id. A ``foreach`` is atomic: it is replayed only via its
journaled *aggregate* (its body leaves share one id across iterations, so they
cannot be replayed individually) — a foreach that died mid-fan-out re-runs in
full. Re-running a node re-executes its sub-agent from scratch, which re-runs any
side-effecting tools; the engine threads a stable ``idempotency_key``
(``run_id:node_id``) into each leaf's metadata so a tool *can* dedupe, but the
library does not force it.

Caveats
-------
* Only data is journaled, so runtime objects (``executor`` / ``budget``) must be
  re-supplied on resume. Pass a *fresh* budget: it is pre-charged with the tokens
  replayed steps already spent, so a partially-spent budget would double-count.
* Runs created before resume support landed have no persisted spec and cannot be
  resumed — :func:`rehydrate` raises a clear :class:`WorkflowRunError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from power_loop.runtime.cancellation import CancellationToken

from . import journal
from .engine import WorkflowEngine, WorkflowRunError
from .result import AgentResult, WorkflowResult
from .runner import _LIVE_RUN_IDS, is_run_live, make_on_step, spawn_background
from .spec import (
    AgentNode,
    BranchNode,
    ForeachNode,
    ParallelNode,
    SequenceNode,
    WorkflowNode,
    WorkflowSpec,
)

if TYPE_CHECKING:
    from power_loop.runtime.store.store import SessionStore

    from .result import WorkflowRunHandle

__all__ = ["replayable_node_ids", "rehydrate", "resume_run", "resume_detached"]

# Module-level GC guard for detached resume tasks (first-run uses Workflow._tasks).
_RESUME_TASKS: set[Any] = set()


def replayable_node_ids(spec: WorkflowSpec) -> set[str]:
    """Ids whose journaled result may be replayed individually on resume.

    Excludes everything inside a ``foreach`` body (those ids collide across
    iterations); includes ordinary agent ids and a foreach's own aggregate id,
    as long as they are not themselves nested in another foreach body.
    """
    out: set[str] = set()
    _collect(spec.root, in_foreach_body=False, out=out)
    return out


def _collect(node: WorkflowNode, *, in_foreach_body: bool, out: set[str]) -> None:
    if isinstance(node, AgentNode):
        if not in_foreach_body:
            out.add(node.id)
        return
    if isinstance(node, SequenceNode):
        for s in node.steps:
            _collect(s, in_foreach_body=in_foreach_body, out=out)
        return
    if isinstance(node, ParallelNode):
        for b in node.branches:
            _collect(b, in_foreach_body=in_foreach_body, out=out)
        return
    if isinstance(node, ForeachNode):
        if node.id and not in_foreach_body:
            out.add(node.id)  # the aggregate is atomically replayable
        _collect(node.body, in_foreach_body=True, out=out)  # body ids are not
        return
    if isinstance(node, BranchNode):
        for c in node.cases.values():
            _collect(c, in_foreach_body=in_foreach_body, out=out)
        if node.default is not None:
            _collect(node.default, in_foreach_body=in_foreach_body, out=out)
        return


async def rehydrate(
    store: SessionStore, parent_sid: str, run_id: str
) -> tuple[WorkflowSpec, dict[str, AgentResult], dict[str, Any]]:
    """Load a run's journal and rebuild (spec, replay-cache, journal blob).

    Raises :class:`WorkflowRunError` if the run is unknown or predates spec
    persistence (and so cannot be resumed).
    """
    j = await journal.read(store, parent_sid, run_id)
    if j is None:
        raise WorkflowRunError(f"cannot resume: no journal for run {run_id!r} under {parent_sid!r}")
    spec_dict = j.get("spec")
    if not spec_dict:
        raise WorkflowRunError(
            f"cannot resume run {run_id!r}: no spec persisted (run predates resume support)"
        )
    spec = WorkflowSpec.from_json(spec_dict)
    replayable = replayable_node_ids(spec)
    replay: dict[str, AgentResult] = {}
    for step in j.get("steps", []):
        if step.get("status") != "completed":
            continue
        nid = step.get("node_id")
        if nid not in replayable:
            continue
        replay[nid] = AgentResult(
            node_id=nid,
            status="completed",
            text=step.get("text") or "",
            payload=step.get("payload"),
            usage=step.get("usage") or {},
            session_id=step.get("session_id"),
            error=step.get("error"),
            db_path=step.get("db_path"),
        )
    return spec, replay, j


def _guard_resumable(run_id: str, force: bool) -> None:
    """Refuse to resume a run whose engine is still executing IN THIS PROCESS unless ``force``
    (workflow-durability-3): a second engine for the same run would race the journal. Liveness is
    tracked by the in-process registry (set while a detached/sync run's task is alive), so a CRASHED
    run — the normal resume case — is not blocked (its task already discarded the id). Cross-process
    liveness can't be detected from here; ``force=True`` overrides if you've confirmed it's dead."""
    if not force and is_run_live(run_id):
        raise WorkflowRunError(
            f"run {run_id} is still executing in this process — resuming would start a second "
            f"concurrent engine and race the journal. Wait for it to finish (or cancel it), or pass "
            f"force=True if you've confirmed it is no longer running."
        )


async def _mark_resuming(store: SessionStore, parent_sid: str, run_id: str, j: dict[str, Any]) -> None:
    """Flip a finished/failed journal back to running and bump the attempt count."""
    await journal.update(
        store, parent_sid, run_id,
        allow_terminal=True,  # an intentional terminal→running restart, not a late clobber
        status="running",
        attempts=int(j.get("attempts", 1)) + 1,
        finished_at_ms=None,
        error=None,
        woke=False,  # allow the completion wake to fire again for this attempt
    )


async def resume_run(
    loop: Any,
    parent_sid: str,
    run_id: str,
    *,
    executor: Any | None = None,
    budget: Any | None = None,
    force: bool = False,
) -> WorkflowResult:
    """Resume a run synchronously (in-process), returning the final result.

    Pass the ``StatefulAgentLoop`` whose store holds the journal. Completed steps
    are replayed; the unfinished tail re-runs. The journal is updated live and
    finalized at the end.

    Runtime objects are not journaled (only data is), so re-supply a non-default
    ``executor`` / ``budget`` if the original run used one. A ``budget`` is
    pre-charged with the tokens already spent by replayed steps.
    """
    store = await loop._ensure_store()
    spec, replay, j = await rehydrate(store, parent_sid, run_id)
    _guard_resumable(run_id, force)
    await _mark_resuming(store, parent_sid, run_id, j)
    engine = WorkflowEngine(
        loop, executor=executor, budget=budget,
        on_step=make_on_step(store, parent_sid, run_id), replay=replay, run_id=run_id,
    )
    # Mark live so a concurrent resume of the same run is refused; discard in finally.
    _LIVE_RUN_IDS.add(run_id)
    try:
        result = await engine.run(spec)
    except Exception as exc:  # noqa: BLE001 — mirror detached: journal then re-raise
        await journal.fail(store, parent_sid, run_id, exc)
        raise
    finally:
        _LIVE_RUN_IDS.discard(run_id)
    await journal.finalize(store, parent_sid, run_id, result)
    return result


async def resume_detached(
    loop: Any,
    parent_sid: str,
    run_id: str,
    *,
    executor: Any | None = None,
    budget: Any | None = None,
    eager_wake: bool = False,
    force: bool = False,
) -> WorkflowRunHandle:
    """Resume a run on a background task; wake the parent on completion.

    Mirror of :func:`run_detached` for the resume path. Requires a live parent
    session and (for the wake to land) a running ``TimerRunner`` +
    ``register_wake_guard``. Re-supply ``executor`` / ``budget`` if the original
    run used non-default ones.
    """
    store = await loop._ensure_store()
    if await store.get_session(parent_sid) is None:
        raise WorkflowRunError("parent session not found; cannot resume detached run")
    spec, replay, j = await rehydrate(store, parent_sid, run_id)
    _guard_resumable(run_id, force)
    await _mark_resuming(store, parent_sid, run_id, j)
    cancel_token = CancellationToken()

    def _build_engine() -> WorkflowEngine:
        return WorkflowEngine(
            loop, executor=executor, budget=budget,
            on_step=make_on_step(store, parent_sid, run_id),
            replay=replay, run_id=run_id, stop_event=cancel_token,
        )

    return spawn_background(
        loop, parent_sid, run_id, store=store,
        build_engine=_build_engine, run_spec=spec,
        cancel_token=cancel_token, eager_wake=eager_wake, task_set=_RESUME_TASKS,
    )
