"""Detached execution + completion-callback wake (D3).

``run_detached`` runs a workflow as a background ``asyncio.Task`` and returns a
:class:`WorkflowRunHandle` immediately, so the parent agent can ``pass_turn`` out.
When the run finishes (success or failure) it:

1. finalizes the durable journal (:mod:`power_loop.workflow.journal`);
2. publishes a ``SYSTEM_LOG`` progress event (no new ``AgentEventType``); and
3. **wakes the parent agent** by scheduling a durable timer
   (``loop.schedule_timer(parent_sid, delay_s=0, note=...)``) — delivered by the
   host's ``TimerRunner`` as a normal ``follow_up`` into the parent session.

The timer is the durable, "one path into the conversation" wake. A non-durable
``eager_wake`` fast-path is available; the :func:`make_wake_guard` TIMER_FIRE
hook dedupes the at-least-once timer so the parent wakes exactly once.

Everything here uses an EXPLICITLY captured ``loop`` / ``parent_sid`` — never the
contextvars (which are unreliable inside a detached task).
"""

from __future__ import annotations

import asyncio
import json
import secrets
from typing import TYPE_CHECKING, Any

from power_loop.contracts.event_payloads import SystemLogPayload
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.hooks import HookDirective, HookPoint

from . import journal
from .engine import WorkflowEngine, WorkflowRunError
from .result import AgentResult, WorkflowRunHandle

if TYPE_CHECKING:
    from power_loop.contracts.hook_contexts import TimerFireCtx

    from .api import Workflow

__all__ = [
    "run_detached", "make_wake_guard", "register_wake_guard", "make_on_step", "spawn_background",
]


def make_on_step(store: Any, parent_sid: str, run_id: str):
    """Build the per-step journaling callback the engine fires as nodes settle.

    Captures the step's output (``text`` / ``payload``) too, so a completed step
    can be replayed (not re-run) when the run is resumed.
    """

    def _on_step(res: AgentResult) -> None:
        journal.record_step(
            store, parent_sid, run_id,
            node_id=res.node_id, status=res.status,
            session_id=res.session_id, usage=res.usage, error=res.error,
            text=res.text, payload=res.payload,
        )

    return _on_step


def _wake_note(run_id: str, status: str, extra: str = "") -> str:
    base = f"{journal.JOURNAL_PREFIX}{run_id} {status}"
    return f"{base} — {extra}" if extra else base


def _parse_run_id(note: str | None) -> str | None:
    if not note or not note.startswith(journal.JOURNAL_PREFIX):
        return None
    rest = note[len(journal.JOURNAL_PREFIX):].strip()
    return rest.split()[0] if rest else None


def _publish(loop: Any, parent_sid: str, run_id: str, event: str, status: str, *, level: str = "info") -> None:
    try:
        loop.event_bus.publish(
            AgentEvent(
                type=AgentEventType.SYSTEM_LOG,
                data=SystemLogPayload(
                    message=json.dumps({"workflow": run_id, "event": event, "status": status}),
                    level=level,
                ),
                session_id=parent_sid,
                source="workflow",
            )
        )
    except Exception:  # noqa: BLE001 — observability must never break a run
        pass


def _wake(loop: Any, parent_sid: str, note: str) -> None:
    try:
        loop.schedule_timer(parent_sid, delay_s=0, note=note)
    except Exception:  # noqa: BLE001 — parent may have been closed; journal is the source of truth
        pass


async def run_detached(workflow: Workflow, *, eager_wake: bool = False) -> WorkflowRunHandle:
    """Start ``workflow`` as a background task; return a handle immediately.

    Requires ``workflow`` to carry a ``parent_session_id`` (the session that
    will be woken on completion). The parent must be live, and a ``TimerRunner``
    must be running on the host for the wake to be delivered.
    """
    parent_sid = workflow._parent_sid
    if not parent_sid:
        raise WorkflowRunError("detached run requires a parent_session_id on the workflow")
    loop = workflow._loop
    store = loop.store
    if store.get_session(parent_sid) is None:
        raise WorkflowRunError("parent session not found; cannot start detached run")

    run_id = secrets.token_hex(8)
    journal.seed(store, parent_sid, run_id, workflow.spec.name, spec=workflow.spec.to_dict())

    def _build_engine() -> WorkflowEngine:
        return WorkflowEngine(
            loop, executor=workflow._executor, budget=workflow._budget,
            on_step=make_on_step(store, parent_sid, run_id),
            stop_event=workflow._cancel, run_id=run_id,
        )

    return spawn_background(
        loop, parent_sid, run_id,
        build_engine=_build_engine, run_spec=workflow.spec,
        cancel_token=workflow._cancel, eager_wake=eager_wake, task_set=workflow._tasks,
    )


def spawn_background(
    loop: Any,
    parent_sid: str,
    run_id: str,
    *,
    build_engine: Any,
    run_spec: Any,
    cancel_token: Any,
    eager_wake: bool,
    task_set: set[Any],
) -> WorkflowRunHandle:
    """Drive a workflow engine on a background task; journal + wake on finish.

    Shared by first-run (``run_detached``) and ``resume_detached``. The engine is
    built lazily inside the task so its construction happens on the run's own
    task. Returns a handle immediately.
    """
    store = loop.store

    async def _bg() -> None:
        try:
            result = await build_engine().run(run_spec)
            await asyncio.to_thread(journal.finalize, store, parent_sid, run_id, result)
            note = _wake_note(run_id, result.status)
            _publish(loop, parent_sid, run_id, "completed", result.status,
                     level=("error" if result.status == "failed" else "info"))
        except Exception as exc:  # noqa: BLE001 — capture everything; the task must not die silently
            await asyncio.to_thread(journal.fail, store, parent_sid, run_id, exc)
            note = _wake_note(run_id, "failed", repr(exc))
            _publish(loop, parent_sid, run_id, "failed", "failed", level="error")
        _wake(loop, parent_sid, note)
        if eager_wake:
            # Non-durable fast-path; the timer above is the durable backstop and
            # the wake-guard dedupes the double delivery.
            with _suppress():
                asyncio.create_task(loop.follow_up(note, parent_sid))

    task = asyncio.create_task(_bg(), name=f"workflow-{run_id}")
    task_set.add(task)
    task.add_done_callback(task_set.discard)
    return WorkflowRunHandle(run_id=run_id, task=task, cancel_token=cancel_token)


class _suppress:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return True  # swallow any error from the optional eager wake


def make_wake_guard(store: Any):
    """A ``HookPoint.TIMER_FIRE`` guard that delivers each workflow wake exactly
    once (timers are at-least-once). Ignores non-workflow timers."""

    def guard(ctx: TimerFireCtx) -> None:
        run_id = _parse_run_id(ctx.note)
        if run_id is None:
            return  # not a workflow timer → CONTINUE
        j = store.get_runtime_state(ctx.session_id, journal.run_key(run_id), default=None)
        if j is None:
            return
        if j.get("woke"):
            ctx.directive = HookDirective.SKIP  # already delivered once
            return
        j["woke"] = True
        store.set_runtime_state(ctx.session_id, journal.run_key(run_id), j)

    return guard


def register_wake_guard(loop: Any) -> None:
    """Install the workflow wake-dedupe guard on ``loop``'s hooks (call once)."""
    loop.hooks.register(HookPoint.TIMER_FIRE, make_wake_guard(loop.store))
