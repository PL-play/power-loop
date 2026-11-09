"""Asynchronous subagent sessions, durable results and serialized continuation.

The host owns idle-parent wake routing. No notification callback starts a bare parent loop.
Execution and control use persisted task references across workers.
"""
from __future__ import annotations

import asyncio
import contextlib
import weakref
from dataclasses import dataclass, field
from typing import Any

from power_loop.runtime.recall import dumps
from power_loop.runtime.spec import (
    AgentSpec,
)

RESULT_PREFIX = "[subagent_result]\n"


@dataclass
class Running:
    run_id: str
    task: asyncio.Task | None = None
    token: Any = None
    parent: str = ""
    title: str = ""
    suppress_wake: bool = False
    phase: str = "running"
    descendant_usage: dict = field(default_factory=dict)
    projection_expected: bool = True
    task_binding: Any = None
    canonical_result: dict | None = None
    canonical_attempted: bool = False
    answer_attempted: bool = False
    completion_max_chars: int | None = None
    admission: Any = field(default=None, repr=False)
    admission_lease: Any = field(default=None, repr=False)
    admission_task: Any = field(default=None, repr=False)





async def _acquire_admission(run):
    if run.admission is None:
        return False
    if run.admission_task is not None:
        if run.admission_task is not asyncio.current_task():
            from .jobs import JobError
            raise JobError("subagent_admission_owner_changed")
        return False
    from power_loop.workflow.admission import Lease
    lease = Lease((run.admission,), run.token)
    await lease.acquire()
    run.admission_lease, run.admission_task = lease, asyncio.current_task()
    return True


def _release_admission(run):
    if run.admission_lease is not None:
        run.admission_lease.release()
    run.admission_lease, run.admission_task = None, None


@contextlib.asynccontextmanager
async def readmit(run):
    acquired = await _acquire_admission(run)
    try:
        yield
    finally:
        if acquired:
            _release_admission(run)


@contextlib.asynccontextmanager
async def readmit_session(loop, sid):
    yield


_GATES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _gate(store, sid):
    return _GATES.setdefault((id(store), sid), asyncio.Lock())


async def start(loop, spec: AgentSpec, task: str) -> dict[str, Any]:
    from .child_tasks import create
    return await create(loop, spec, task)


async def run_inline(spec, task, *, parent_loop):
    from power_loop.agent.runtime_types import JobRef, plain

    from .child_tasks import tool_actor
    receipt = await start(parent_loop, spec, task)
    if getattr(parent_loop, '_host_command', None) is None:
        _, actor = await tool_actor(parent_loop)
    else:
        from power_loop.agent.runtime_types import TrustedActor
        runtime = parent_loop.task_runtime
        actor = TrustedActor(runtime.tenant_id, runtime.principal_id, None)
    worker = parent_loop._runtime_worker
    from power_loop.agent.runtime_store import now_ms
    result = await worker.runtime.client.wait(JobRef(**receipt["job_ref"]), actor=actor,
        deadline_ms=now_ms() + 1800000)
    if getattr(parent_loop, '_host_command', None) is not None:
        from .host_commands import consume_result
        await consume_result(parent_loop, result)
    return {**plain(result), "session_id": receipt["session_id"], "run_id": receipt["run_id"]}


def result_view(text: str, cap: int = 600) -> str | None:
    """Tool-context rule: only final_text may change, never envelope metadata."""
    import json
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or "session_id" not in value or not value.get("run_id"):
        return None
    if isinstance(value.get("final_text"), str) and len(value["final_text"]) > cap:
        value["final_text"] = value["final_text"][:max(0, cap)] + "…[truncated]"
    return dumps(value)


async def await_children(store, parent, stop_event=None, timeout_s=1800):
    """Wait for stored descendants without retaining their Python executions."""
    from power_loop.agent.runtime_store import now_ms
    from power_loop.agent.runtime_types import JobRef, TargetRef
    from power_loop.core.agent_context import get_current_loop

    from .child_tasks import tool_actor
    loop = get_current_loop()
    worker = loop._runtime_worker
    _, actor = await tool_actor(loop)
    page = await worker.runtime.client.list_jobs(TargetRef(store.store_id, parent), actor=actor, scope="subtree")
    for status in page.items:
        if status.job_ref != loop._runtime_worker_context.job_ref:
            await worker.runtime.client.wait(JobRef(store.store_id, status.job_ref.job_id, status.job_ref.generation),
                actor=actor, deadline_ms=now_ms() + int(timeout_s * 1000))


def completion_view(loop, result, *, max_chars=None):
    config = getattr(loop, 'config', None)
    projection = getattr(getattr(config, 'representation', None), 'projection_cfg', {}) or {}
    cap = max_chars if max_chars is not None else projection.get('final_text_cap') or 600
    return result_view(dumps(result), int(cap))
