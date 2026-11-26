"""Subagent creation and ordinary TargetRef/InputRef continuation."""
from __future__ import annotations

import asyncio
import contextvars
from typing import Any

from power_loop.runtime.recall import dumps
from power_loop.runtime.spec import AgentSpec

RESULT_PREFIX = "[subagent_result]\n"


async def start(loop, spec: AgentSpec, task: str, *, on_created=None) -> dict[str, Any]:
    from .child_tasks import create
    return await create(loop, spec, task, on_created=on_created)


async def _wait_input(
    worker, input_ref, *, actor, deadline_ms, cancel_token=None,
):
    from power_loop.contracts.errors import CancellationRequested

    cancellable = cancel_token is not None and not cancel_token.is_never
    if cancellable:
        cancel_token.raise_if_cancelled()
    if not cancellable:
        return await worker.runtime.client.wait(
            input_ref, actor=actor, deadline_ms=deadline_ms,
        )

    wait_task = asyncio.create_task(
        worker.runtime.client.wait(input_ref, actor=actor, deadline_ms=deadline_ms)
    )
    stop_task = asyncio.create_task(cancel_token.wait())
    try:
        done, _ = await asyncio.wait(
            {wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_task in done or cancel_token.is_cancelled():
            wait_task.cancel()
            stop_task.cancel()
            await asyncio.gather(wait_task, stop_task, return_exceptions=True)
            raise CancellationRequested(cancel_token.reason or "cancelled")
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)
        return wait_task.result()
    except CancellationRequested:
        raise
    except BaseException:
        stop_task.cancel()
        wait_task.cancel()
        await asyncio.gather(wait_task, stop_task, return_exceptions=True)
        raise


async def run_inline(spec, task, *, parent_loop):
    from power_loop.agent.runtime_store import now_ms
    from power_loop.agent.runtime_types import InputRef, plain
    from power_loop.contracts.event_payloads import (
        SubagentCompletedPayload,
        SubagentLimitPayload,
        SubagentTaskStartPayload,
        SubagentTextPayload,
    )
    from power_loop.contracts.events import AgentEventType
    from power_loop.core.agent_context import get_current_cancel_token
    from power_loop.runtime.spec import _publish_subagent_event

    from .child_tasks import tool_actor

    worker = parent_loop._runtime_worker
    ctx = getattr(parent_loop, "_runtime_worker_context", None)
    parent_sid = ctx.session_id if ctx is not None else None
    child_depth = (
        int(ctx.batch.depth) + 1
        if ctx is not None and getattr(ctx, "batch", None) is not None
        else 1
    )
    effective_max_rounds = int(spec.max_rounds)

    def _on_created(target, max_rounds: int) -> None:
        nonlocal effective_max_rounds
        effective_max_rounds = max_rounds
        _publish_subagent_event(
            parent_loop,
            parent_sid,
            AgentEventType.SUBAGENT_TASK_START,
            SubagentTaskStartPayload(
                task=str(task)[:500],
                preset=spec.name,
                sub_session_id=target.session_id,
                depth=child_depth,
            ),
        )

    receipt = await start(parent_loop, spec, task, on_created=_on_created)
    sub_sid = receipt["session_id"]
    _, actor = await tool_actor(parent_loop)
    if parent_sid is None:
        parent_sid = actor.session_id
    cancel_token = get_current_cancel_token() or (ctx.stop if ctx is not None else None)
    input_ref = InputRef(**receipt["input_ref"])
    async with worker.suspend_execution_slot(
        ctx, cancel_token=cancel_token, child_sids=(sub_sid,),
    ):
        if getattr(parent_loop, "_host_command", None) is not None and not worker._started:
            await asyncio.create_task(worker._run_session(sub_sid), context=contextvars.Context())
        result = await _wait_input(
            worker, input_ref, actor=actor, deadline_ms=now_ms() + 1800000,
            cancel_token=cancel_token,
        )
    rounds = int(result.rounds)
    final_text = (result.final_text or "")[:2000]
    if result.status == "completed":
        _publish_subagent_event(
            parent_loop,
            parent_sid,
            AgentEventType.SUBAGENT_TEXT,
            SubagentTextPayload(
                sub_session_id=sub_sid,
                status=result.status,
                rounds=rounds,
                final_text=final_text,
            ),
        )
    elif result.error_code == "hit_round_limit":
        _publish_subagent_event(
            parent_loop,
            parent_sid,
            AgentEventType.SUBAGENT_LIMIT,
            SubagentLimitPayload(
                sub_session_id=sub_sid,
                max_rounds=effective_max_rounds,
            ),
        )
    _publish_subagent_event(
        parent_loop,
        parent_sid,
        AgentEventType.SUBAGENT_COMPLETED,
        SubagentCompletedPayload(
            sub_session_id=sub_sid,
            status=result.status,
            rounds=rounds,
            final_text=final_text,
        ),
    )
    if getattr(parent_loop, "_host_command", None) is not None:
        from .host_commands import consume_result
        await consume_result(parent_loop)
    return {**plain(result), "session_id": sub_sid, "run_id": receipt["run_id"]}


def result_view(text: str, cap: int = 600) -> str | None:
    """Only final_text may change in the tool view, never envelope metadata."""
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
    """Wait for direct child start inputs without retaining their Python executions."""
    from power_loop.agent.runtime_store import RuntimeInputStore, now_ms
    from power_loop.core.agent_context import get_current_cancel_token, get_current_loop
    from power_loop.runtime.cancellation import CancellationToken

    from .child_tasks import tool_actor

    loop = get_current_loop()
    _, actor = await tool_actor(loop)
    worker = loop._runtime_worker
    ctx = getattr(loop, "_runtime_worker_context", None)
    cancel_token = (
        CancellationToken.from_any(stop_event)
        if stop_event is not None
        else (get_current_cancel_token() or (ctx.stop if ctx is not None else None))
    )
    port = RuntimeInputStore(store)
    children = await port.list_children(parent, actor=actor, limit=1000)
    pending_children = []
    queued_child_sids = []
    for child in children.items:
        session = await port._session(port.db, child.session_id)
        if session["start_input_message_id"] is None:
            continue
        from power_loop.agent.runtime_types import InputRef
        start_ref = InputRef(store.store_id, child.session_id, session["start_input_message_id"])
        pending_children.append(start_ref)
        if session["execution_state"] == "queued":
            queued_child_sids.append(child.session_id)
    async with worker.suspend_execution_slot(
        ctx, cancel_token=cancel_token, child_sids=tuple(queued_child_sids),
    ):
        for start_ref in pending_children:
            await _wait_input(
                worker, start_ref, actor=actor, deadline_ms=now_ms() + int(timeout_s * 1000),
                cancel_token=cancel_token,
            )


def completion_view(loop, result, *, max_chars=None):
    config = getattr(loop, "config", None)
    projection = getattr(getattr(config, "representation", None), "projection_cfg", {}) or {}
    cap = max_chars if max_chars is not None else projection.get("final_text_cap") or 600
    return result_view(dumps(result), int(cap))
