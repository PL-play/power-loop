"""Subagent creation and ordinary TargetRef/InputRef continuation."""
from __future__ import annotations

from typing import Any

from power_loop.runtime.recall import dumps
from power_loop.runtime.spec import AgentSpec

RESULT_PREFIX = "[subagent_result]\n"


async def start(loop, spec: AgentSpec, task: str) -> dict[str, Any]:
    from .child_tasks import create
    return await create(loop, spec, task)


async def run_inline(spec, task, *, parent_loop):
    from power_loop.agent.runtime_store import now_ms
    from power_loop.agent.runtime_types import InputRef, plain

    from .child_tasks import tool_actor

    receipt = await start(parent_loop, spec, task)
    worker = parent_loop._runtime_worker
    if getattr(parent_loop, "_host_command", None) is not None:
        await worker._run_session_owned(receipt["session_id"])
    _, actor = await tool_actor(parent_loop)
    result = await worker.runtime.client.wait(
        InputRef(**receipt["input_ref"]), actor=actor, deadline_ms=now_ms() + 1800000
    )
    if getattr(parent_loop, "_host_command", None) is not None:
        from .host_commands import consume_result
        await consume_result(parent_loop)
    return {**plain(result), "session_id": receipt["session_id"], "run_id": receipt["run_id"]}


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
    from power_loop.core.agent_context import get_current_loop

    from .child_tasks import tool_actor

    loop = get_current_loop()
    _, actor = await tool_actor(loop)
    port = RuntimeInputStore(store)
    children = await port.list_children(parent, actor=actor, limit=1000)
    for child in children.items:
        session = await port._session(port.db, child.session_id)
        if session["start_input_message_id"] is None:
            continue
        from power_loop.agent.runtime_types import InputRef
        start = InputRef(store.store_id, child.session_id, session["start_input_message_id"])
        await loop._runtime_worker.runtime.client.wait(
            start, actor=actor, deadline_ms=now_ms() + int(timeout_s * 1000)
        )


def completion_view(loop, result, *, max_chars=None):
    config = getattr(loop, "config", None)
    projection = getattr(getattr(config, "representation", None), "projection_cfg", {}) or {}
    cap = max_chars if max_chars is not None else projection.get("final_text_cap") or 600
    return result_view(dumps(result), int(cap))
