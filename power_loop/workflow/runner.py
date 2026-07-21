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

**Host wake seam (``on_complete``).** A host may pass ``on_complete`` to
:func:`run_detached` / :func:`resume_detached`, taking over the parent wake
entirely: the durable timer and the ``eager_wake`` fast-path are BOTH skipped,
and a :class:`WorkflowCompletion` is handed to the callback instead. This is the
in-process, timer-free wake — the host typically re-resolves the parent's current
loop and calls ``loop.follow_up``. It lives and dies with the process (no durable
backstop): a raising callback is logged and swallowed (the run is journal-final,
so a lost wake means the caller must re-request), and it never falls back to a
timer. ``on_complete`` supersedes ``eager_wake``.

Everything here uses an EXPLICITLY captured ``loop`` / ``parent_sid`` — never the
contextvars (which are unreliable inside a detached task).
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from power_loop.contracts.event_payloads import SystemLogPayload
from power_loop.contracts.events import AgentEvent, AgentEventType
from power_loop.contracts.hooks import HookDirective, HookPoint
from power_loop.runtime.store.store import MUTATE_SKIP

from . import journal
from .engine import WorkflowEngine, WorkflowRunError
from .result import AgentResult, WorkflowCompletion, WorkflowRunHandle

if TYPE_CHECKING:
    from power_loop.contracts.hook_contexts import TimerFireCtx

    from .api import Workflow

logger = logging.getLogger(__name__)

#: Host completion-wake seam (see module docstring). Receives a
#: :class:`WorkflowCompletion`; when supplied it replaces the durable-timer wake.
OnComplete = Callable[["WorkflowCompletion"], Awaitable[None]]

__all__ = [
    "run_detached", "make_wake_guard", "register_wake_guard", "make_on_step", "spawn_background",
    "claim_wake", "parse_workflow_wake",
]

#: run_ids whose engine is currently executing IN THIS PROCESS (set by spawn_background +
#: resume_run). Lets the resume entry points refuse to start a second concurrent engine for a
#: still-live run (workflow-durability-3) without blocking resume-after-crash (a crashed run's
#: task already discarded its id).
_LIVE_RUN_IDS: set[str] = set()


def is_run_live(run_id: str) -> bool:
    """True iff ``run_id``'s engine is currently executing in this process (workflow-durability-3)."""
    return run_id in _LIVE_RUN_IDS


def make_on_step(store: Any, parent_sid: str, run_id: str):
    """Build the per-step journaling callback the engine fires as nodes settle.

    Captures the step's output (``text`` / ``payload``) too, so a completed step
    can be replayed (not re-run) when the run is resumed. Async: the engine awaits
    it as each node settles, and the journal write hits the async store.
    """

    async def _on_step(res: AgentResult) -> None:
        await journal.record_step(
            store, parent_sid, run_id,
            node_id=res.node_id, status=res.status,
            session_id=res.session_id, usage=res.usage, error=res.error,
            text=res.text, payload=res.payload, db_path=res.db_path,
        )

    return _on_step


def make_on_node_start(store: Any, parent_sid: str, run_id: str):
    """Build the leaf-start callback: journals the node into ``live`` (the run's heartbeat)."""

    async def _on_node_start(node_id: str) -> None:
        await journal.node_start(store, parent_sid, run_id, node_id=node_id)

    return _on_node_start


def _wake_note(run_id: str, status: str, extra: str = "") -> str:
    base = f"{journal.JOURNAL_PREFIX}{run_id} {status}"
    return f"{base} — {extra}" if extra else base


def parse_workflow_wake(note: str | None) -> str | None:
    """Extract the run id from a workflow completion-wake timer note.

    Returns ``None`` for any non-workflow note, so a host
    :data:`~power_loop.runtime.timers.TimerDelivery` can branch: workflow wakes
    get enriched (e.g. with ``workflow_status`` detail) or routed differently
    from ordinary timers. PROVISIONAL (3.14).
    """
    if not note or not note.startswith(journal.JOURNAL_PREFIX):
        return None
    rest = note[len(journal.JOURNAL_PREFIX):].strip()
    return rest.split()[0] if rest else None


async def claim_wake(store: Any, parent_sid: str, run_id: str) -> bool:
    """Atomically claim the completion wake for ``run_id`` — True = this caller
    delivers, False = the wake was already delivered once.

    Only needed by hosts that bypass ``TimerRunner`` entirely (polling
    ``store.due_timers()`` themselves): with a running ``TimerRunner`` +
    :func:`register_wake_guard`, dedupe already happens in the TIMER_FIRE hook
    — BEFORE any custom delivery — so a host delivery seam gets it for free.

    A missing journal (or a deleted parent session) claims as True: there is
    nothing to dedupe against, and dropping the wake would strand the parent.
    PROVISIONAL (3.14).
    """
    outcome = await _claim_wake_state(store, parent_sid, run_id)
    return outcome != "already"


async def _claim_wake_state(store: Any, parent_sid: str, run_id: str) -> str:
    """Shared atomic claim: ``"claimed"`` | ``"already"`` | ``"no_journal"``.

    The claim is an atomic RMW via ``mutate_runtime_state`` (row-locked): a bare
    get→set would race concurrent journal writes on the same run key and let two
    concurrent fires both observe ``woke=False`` → double-wake.
    """
    seen = {"woke": False}

    def _claim(cur: Any) -> Any:
        if cur is None:
            return MUTATE_SKIP  # no journal → nothing to dedupe
        if cur.get("woke"):
            seen["woke"] = True
            return MUTATE_SKIP  # already delivered once
        return {**cur, "woke": True}  # first delivery — set woke, preserve every other key

    try:
        await store.mutate_runtime_state(parent_sid, journal.run_key(run_id), _claim, default=None)
    except ValueError:
        # Session/state row gone (a stale timer firing on a deleted session):
        # treat as no-journal, matching the old get_runtime_state(default=None)
        # tolerance; nothing to dedupe.
        return "no_journal"
    return "already" if seen["woke"] else "claimed"


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


async def _wake(loop: Any, parent_sid: str, note: str) -> None:
    try:
        await loop.schedule_timer(parent_sid, delay_s=0, note=note)
    except Exception:  # noqa: BLE001 — parent may have been closed; journal is the source of truth
        pass


async def _run_on_complete(
    on_complete: OnComplete, parent_sid: str, run_id: str, status: str, note: str
) -> None:
    """Invoke the host completion callback in place of the durable-timer wake.

    The host owns delivery (die-with-process, no timer fallback). A raising
    callback is logged and swallowed: the run is already journal-final, so a lost
    wake means the caller must re-request — it must NOT crash the detached task
    (which would leave the run marked live-but-dead) or fall back to a timer.
    """
    try:
        await on_complete(
            WorkflowCompletion(
                parent_session_id=parent_sid, run_id=run_id, status=status, note=note,
            )
        )
    except Exception:  # noqa: BLE001 — a lost wake is recoverable by re-request; a crash is not
        logger.warning(
            "workflow %s on_complete callback failed (wake lost; caller must re-request)",
            run_id, exc_info=True,
        )


async def run_detached(
    workflow: Workflow,
    *,
    eager_wake: bool = False,
    on_complete: OnComplete | None = None,
) -> WorkflowRunHandle:
    """Start ``workflow`` as a background task; return a handle immediately.

    Requires ``workflow`` to carry a ``parent_session_id`` (the session that
    will be woken on completion).

    Default wake (``on_complete=None``): a durable timer — the parent must be
    live and a ``TimerRunner`` must be running on the host for it to be delivered.
    ``eager_wake`` adds a non-durable instant wake that calls ``loop.follow_up``
    DIRECTLY — it does not go through the TimerRunner, so a host that installed
    a custom ``TimerDelivery`` should leave it False (the eager path would
    bypass the host's pipeline; the durable timer still honors it).

    Host wake seam (``on_complete``): supply a callback to take over the wake
    (in-process, timer-free). It receives a :class:`WorkflowCompletion`; the
    durable timer and ``eager_wake`` are both skipped. See the module docstring.
    """
    parent_sid = workflow._parent_sid
    if not parent_sid:
        raise WorkflowRunError("detached run requires a parent_session_id on the workflow")
    loop = workflow._loop
    store = await loop._ensure_store()
    if await store.get_session(parent_sid) is None:
        raise WorkflowRunError("parent session not found; cannot start detached run")

    run_id = secrets.token_hex(8)
    await journal.seed(
        store, parent_sid, run_id, workflow.spec.name, spec=workflow.spec.to_dict(),
        allowed_tools=(
            sorted(workflow._allowed_tools) if workflow._allowed_tools is not None else None
        ),
    )

    def _build_engine() -> WorkflowEngine:
        return WorkflowEngine(
            loop, executor=workflow._executor, budget=workflow._budget,
            on_step=make_on_step(store, parent_sid, run_id),
            on_node_start=make_on_node_start(store, parent_sid, run_id),
            stop_event=workflow._cancel, run_id=run_id,
            allowed_tools=workflow._allowed_tools,
        )

    return spawn_background(
        loop, parent_sid, run_id, store=store,
        build_engine=_build_engine, run_spec=workflow.spec,
        cancel_token=workflow._cancel, eager_wake=eager_wake, task_set=workflow._tasks,
        on_complete=on_complete,
    )


def spawn_background(
    loop: Any,
    parent_sid: str,
    run_id: str,
    *,
    store: Any,
    build_engine: Any,
    run_spec: Any,
    cancel_token: Any,
    eager_wake: bool,
    task_set: set[Any],
    on_complete: OnComplete | None = None,
) -> WorkflowRunHandle:
    """Drive a workflow engine on a background task; journal + wake on finish.

    Shared by first-run (``run_detached``) and ``resume_detached``. The engine is
    built lazily inside the task so its construction happens on the run's own
    task. Returns a handle immediately.

    ``store`` is the caller's already-resolved store (this fn is sync and cannot
    ``await loop._ensure_store()`` itself; reading ``loop.store`` here would capture
    ``None`` if a future caller hadn't pre-opened it).

    ``on_complete`` (host wake seam): when supplied, it OWNS the parent wake — the
    durable timer and ``eager_wake`` are both skipped and a
    :class:`WorkflowCompletion` is handed to it instead (in-process, timer-free).
    """

    async def _bg() -> None:
        status = "failed"
        try:
            result = await build_engine().run(run_spec)
            await journal.finalize(store, parent_sid, run_id, result)
            status = result.status
            note = _wake_note(run_id, status)
            _publish(loop, parent_sid, run_id, "completed", status,
                     level=("error" if status == "failed" else "info"))
        except Exception as exc:  # noqa: BLE001 — capture everything; the task must not die silently
            await journal.fail(store, parent_sid, run_id, exc)
            status = "failed"
            note = _wake_note(run_id, "failed", repr(exc))
            _publish(loop, parent_sid, run_id, "failed", "failed", level="error")
        if on_complete is not None:
            # Host owns delivery (timer-free, die-with-process). Supersedes eager_wake.
            await _run_on_complete(on_complete, parent_sid, run_id, status, note)
        else:
            await _wake(loop, parent_sid, note)
            if eager_wake:
                await _eager_wake(loop, store, parent_sid, run_id, note, task_set)

    task = asyncio.create_task(_bg(), name=f"workflow-{run_id}")
    task_set.add(task)
    task.add_done_callback(task_set.discard)
    # In-process liveness (workflow-durability-3): mark live now, discard when the task settles —
    # a crashed run is therefore NOT marked live, so resume-after-crash still works.
    _LIVE_RUN_IDS.add(run_id)
    task.add_done_callback(lambda _t: _LIVE_RUN_IDS.discard(run_id))
    return WorkflowRunHandle(run_id=run_id, task=task, cancel_token=cancel_token)


async def _eager_wake(
    loop: Any, store: Any, parent_sid: str, run_id: str, note: str, task_set: set[Any]
) -> None:
    """Non-durable fast-path layered on the durable timer (already armed above).

    The eager delivery bypasses ``TIMER_FIRE``, so :func:`make_wake_guard` can't
    dedupe it — we CLAIM the wake (journal ``woke=True``) so the later durable timer
    is skipped and the parent wakes exactly once.

    The follow_up is a real, fallible coroutine, so we must NOT fire-and-forget it:
    a bare ``create_task`` is only weakly referenced by CPython and can be GC'd
    mid-flight, and the coroutine can raise *after* we claimed the wake — and since
    the durable timer is then suppressed, the parent would be lost forever. So we
    RETAIN the task in ``task_set`` and, if it fails/cancels, re-open the claim and
    re-arm a durable timer (see :func:`_recover_eager_wake`).
    """
    j = await journal.read(store, parent_sid, run_id)
    if j is None or j.get("woke"):
        return
    await journal.update(store, parent_sid, run_id, allow_terminal=True, woke=True)
    eager = asyncio.create_task(
        loop.follow_up(note, parent_sid), name=f"workflow-eagerwake-{run_id}"
    )
    task_set.add(eager)
    eager.add_done_callback(task_set.discard)
    eager.add_done_callback(
        lambda t: _recover_eager_wake(loop, store, parent_sid, run_id, note, t, task_set)
    )


def _recover_eager_wake(
    loop: Any, store: Any, parent_sid: str, run_id: str, note: str, task: Any, task_set: set[Any]
) -> None:
    """Done-callback for the eager follow_up.

    On success, leave ``woke=True`` (the durable timer stays suppressed). On failure
    or cancellation the parent was never woken yet the durable timer is suppressed —
    so schedule an async recovery that re-opens the claim and re-arms a fresh durable
    timer. The wake-guard dedupes if the original timer is also still pending, so the
    parent still wakes exactly once. A done-callback must never raise (and cannot
    await), so the real work runs on a retained recovery task.
    """
    if not task.cancelled() and task.exception() is None:
        return  # eager delivery succeeded
    try:
        recovery = asyncio.create_task(
            _recover_eager_wake_async(loop, store, parent_sid, run_id, note),
            name=f"workflow-eagerwake-recover-{run_id}",
        )
        task_set.add(recovery)
        recovery.add_done_callback(task_set.discard)
    except Exception:  # noqa: BLE001 — recovery scheduling must not raise inside a callback
        return


async def _recover_eager_wake_async(
    loop: Any, store: Any, parent_sid: str, run_id: str, note: str
) -> None:
    """Re-open a failed eager-wake claim and re-arm the durable timer (off the
    done-callback so it can await the async store)."""
    try:
        j = await journal.read(store, parent_sid, run_id)
        if j is None or not j.get("woke"):
            return  # already re-opened, or the run is gone
        await journal.update(store, parent_sid, run_id, allow_terminal=True, woke=False)
    except Exception:  # noqa: BLE001 — recovery must not raise
        return
    await _wake(loop, parent_sid, note)


def make_wake_guard(store: Any):
    """A ``HookPoint.TIMER_FIRE`` guard that delivers each workflow wake exactly
    once (timers are at-least-once). Ignores non-workflow timers. Async because the
    store is async; ``run_typed_async`` awaits it.

    Runs BEFORE the runner's delivery step, so the dedupe also covers a host
    ``TimerDelivery`` — a custom delivery does not need to re-claim.
    """

    async def guard(ctx: TimerFireCtx) -> None:
        run_id = parse_workflow_wake(ctx.note)
        if run_id is None:
            return  # not a workflow timer → CONTINUE
        if await _claim_wake_state(store, ctx.session_id, run_id) == "already":
            ctx.directive = HookDirective.SKIP

    return guard


def register_wake_guard(loop: Any) -> None:
    """Install the workflow wake-dedupe guard on ``loop``'s hooks (call once).

    Resolves the loop's (lazily-opened) async store at fire time, so it is safe
    to register before the first session is created.
    """

    async def guard(ctx: TimerFireCtx) -> None:
        store = await loop._ensure_store()
        await make_wake_guard(store)(ctx)

    loop.hooks.register(HookPoint.TIMER_FIRE, guard)
