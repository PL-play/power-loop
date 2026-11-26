"""Factory-resolved host commands on an ordinary claimed session send."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager

from power_loop.agent.runtime_types import RuntimeRequestError, canonical
from power_loop.agent.stateful_loop import StatefulResult
from power_loop.contracts.hook_contexts import SessionEndCtx, SessionStartCtx
from power_loop.contracts.hooks import HookPoint
from power_loop.core import agent_context

from .pricing import Free
from .task_runtime import _SESSION_SEND, SessionSend


@contextmanager
def _bound(setter, value, reset):
    token = setter(value)
    try:
        yield
    finally:
        reset(token)


async def run_session(loop, ctx, command, *, effective_tools: frozenset[str] | None = None):
    name, arguments = command
    registered = loop.tool_registry.get(name) if loop.tool_registry is not None else None
    if (
        registered is None
        or (ctx.options.tools is not None and name not in ctx.options.tools)
        or (effective_tools is not None and name not in effective_tools)
    ):
        raise RuntimeRequestError("forbidden", "host command outside the frozen tool grant")
    if type(registered.definition.pricing) is not Free:
        raise RuntimeRequestError("invalid_options", "host command router must be explicitly free")
    loop.tool_registry._raise_if_invalid(name, arguments)
    store, sid = loop.store, ctx.session_id
    loop._host_command = {
        "next_child": 0,
        "message_seq": ctx.batch.message_seq,
        "send_index": ctx.send_index,
    }
    send = SessionSend(store=store, target=ctx.batch.target, batch=ctx.batch, ownership=ctx.ownership)
    token = _SESSION_SEND.set(send)
    text = ""
    error = None

    async def stop_command():
        while not ctx.stop.is_cancelled():
            state = await loop._runtime_worker.inputs.boundary(ctx.batch.target)
            if state is not None and (
                state["lifecycle"] in ("closing", "closed")
                or int(state["cancel_seq"]) > ctx.batch.cancel_seq
            ):
                ctx.stop.cancel(state["cancel_reason"] or "cancelled")
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(loop.config.stop_policy.finish_tool_s)
        ctx.task.cancel("host command cancellation grace elapsed")

    try:
        async with store.send_read_scope(sid, ownership=ctx.ownership):
            with ExitStack() as scopes:
                for setter, value, reset in (
                    (agent_context.set_current_loop, loop, agent_context.reset_current_loop),
                    (agent_context.set_session_id, sid, agent_context.reset_session_id),
                    (agent_context.set_effective_tools,
                     effective_tools,
                     agent_context.reset_effective_tools),
                    (agent_context.set_current_cancel_token, ctx.stop,
                     agent_context.reset_current_cancel_token),
                ):
                    scopes.enter_context(_bound(setter, value, reset))
                await loop.hooks.run_typed_async(
                    HookPoint.SESSION_START,
                    SessionStartCtx(scope="host_command", stop_event=ctx.stop),
                )
                stopper = asyncio.create_task(stop_command())
                try:
                    ctx.stop.raise_if_cancelled()
                    output = await loop.tool_registry.invoke_async(name, arguments)
                    text = output if isinstance(output, str) else canonical(output)
                    seq = await store.append_message(
                        sid,
                        role="assistant",
                        content=text,
                        send_index=ctx.send_index,
                        round_index=0,
                        meta={"host_command_result": name},
                    )
                    return StatefulResult(
                        sid, "completed", text, usage={}, result_seq=seq,
                    )
                finally:
                    stopper.cancel()
                    await asyncio.gather(stopper, return_exceptions=True)
    except BaseException as exc:
        error = exc
        raise
    finally:
        _SESSION_SEND.reset(token)
        await loop.hooks.run_typed_async(
            HookPoint.SESSION_END,
            SessionEndCtx(
                scope="host_command",
                reason="failed" if error else "completed",
                final_text=text,
            ),
        )
        loop._host_command = None


async def consume_result(loop):
    """Consume routed child results as ordinary input in the current host send."""
    return await loop._runtime_worker.inputs.claim_follow_up(loop._runtime_worker_context.batch)
