"""Background tools execute once in their own session and route ordinary results."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace

from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import (
    InputRef,
    RuntimeRequestError,
    TargetRef,
    canonical,
    plain,
    sha256_canonical,
)
from power_loop.contracts.tools import ToolResult
from power_loop.core import agent_context

from .child_tasks import tool_actor
from .tool_contracts import background_allowed, validate_registration


async def create(loop, name, arguments, *, command=False, automatic=False):
    from power_loop.tools.registry import async_capable_actions, async_capable_for

    _, actor = await tool_actor(loop)
    registry = loop.tool_registry
    registered = registry.get(name)
    allowed = agent_context.get_effective_tools()
    if registered is None or allowed is not None and name not in allowed:
        raise RuntimeRequestError("forbidden", "background target is outside the tool grant")
    if not background_allowed(registered) or (
        not command and not automatic and not async_capable_for(registered.definition, arguments)
    ):
        choices = []
        for definition in registry.definitions():
            candidate = registry.get(definition.name)
            if definition.name == "background_run" or not background_allowed(candidate):
                continue
            if allowed is not None and definition.name not in allowed:
                continue
            actions = async_capable_actions(definition)
            if actions:
                choices.append(definition.name + "(" + "/".join(sorted(actions)) + ")")
            elif async_capable_for(definition, {}):
                choices.append(definition.name)
        raise RuntimeRequestError(
            "invalid_options",
            f"tool does not allow background execution: {name}. Available: " + ", ".join(sorted(choices)),
        )
    registry._raise_if_invalid(name, arguments)
    validate_registration(registry, name, registered)
    parent_descriptor = loop._runtime_worker_context.descriptor
    frozen = {
        "tool": name,
        "arguments": arguments,
        "definition_hash": sha256_canonical(registered.definition.to_openai_tool()),
        "call_ref": actor.call_ref.to_dict(),
        "call_id": actor.tool_call_id,
    }
    descriptor = replace(
        parent_descriptor,
        kind="background",
        config={**plain(parent_descriptor.config), "background": frozen},
    )
    parent = TargetRef(loop.store.store_id, actor.session_id)
    target, start, receipt = await loop._runtime_worker.inputs.create_child(
        descriptor,
        parent=parent,
        actor=actor,
        call_ref=actor.call_ref,
        tool_call_id=actor.tool_call_id,
        body=frozen,
        result_kind="background_result",
    )
    await loop._runtime_worker.runtime.client._notify(
        target, start.message_id, start, ["work", "activation"]
    )
    return {
        "target_ref": plain(target),
        "input_ref": plain(start),
        "task_id": target.session_id,
        "session_id": target.session_id,
        "run_id": start.message_id,
        "status": "accepted",
        "newly_accepted": receipt.newly_accepted,
        "mode": "create",
    }


def automatic(registered):
    return registered.definition.interrupt == "background" and background_allowed(registered)


async def invoke_and_wait(pipeline, registered, arguments):
    loop = agent_context.get_current_loop()
    receipt = await create(loop, registered.definition.name, arguments, automatic=True)
    _, actor = await tool_actor(loop)
    result = await loop._runtime_worker.runtime.client.wait(
        InputRef(**receipt["input_ref"]), actor=actor, deadline_ms=None,
    )
    return result.final_text


async def _start_for_task(port, actor, task_id):
    target = TargetRef(port.store.store_id, task_id)
    session = await port.authorize(port.db, target, actor)
    if session["session_kind"] != "background" or session["start_input_message_id"] is None:
        raise RuntimeRequestError("invalid_options", "not a background session")
    return target, InputRef(store_id=target.store_id, session_id=task_id,
                            message_id=session["start_input_message_id"])


async def action(loop, action, *, command=None, task_id=None, tool=None, args=None,
                 task_ids=None, timeout_s=None):
    from .background import encode_results

    _, actor = await tool_actor(loop)
    worker, port = loop._runtime_worker, loop._runtime_worker.inputs
    client = worker.runtime.client
    if action in {"tool", "run"}:
        receipt = await create(
            loop,
            "bash" if action == "run" else tool,
            {"command": command, "description": "Run the requested background command"}
            if action == "run" else dict(args or {}),
            command=action == "run",
        )
        return canonical({
            **receipt,
            "message": "The result will be delivered automatically. Do not poll unless it appears stuck.",
        })
    if action not in {"list", "check", "wait", "cancel"}:
        raise RuntimeRequestError("invalid_options", "unknown background action")
    ids = list(task_ids or ([task_id] if task_id else []))
    if not ids:
        page = await client.list_children(
            TargetRef(loop.store.store_id, actor.session_id), actor=actor, limit=1000,
        )
        for child in page.items:
            session = await port._session(port.db, child.session_id)
            if session["session_kind"] == "background":
                ids.append(child.session_id)
    if action in {"check", "wait", "cancel"} and not ids:
        raise RuntimeRequestError("invalid_options", "task_id is required")
    deadline = now_ms() + int(min(max(timeout_s or 600, 0.01), 3600) * 1000)
    records = []
    for current_id in ids:
        target, start = await _start_for_task(port, actor, current_id)
        if action == "cancel":
            return canonical(plain(await client.cancel(
                target, actor=actor, scope="subtree", reason="background cancelled",
            )))
        if action == "wait":
            from power_loop.agent.runtime_types import WaitTimeout
            try:
                await client.wait(start, actor=actor, deadline_ms=deadline)
            except WaitTimeout:
                pass
        status = await client.inspect(start, actor=actor)
        result = await port.get_result(start, actor=actor)
        records.append({
            "task_id": current_id,
            "target_ref": plain(target),
            "input_ref": plain(start),
            "status": result.status if result else status.execution_state,
            "result": result.final_text if result else None,
            "result_ref": plain(start),
        })
    return encode_results(records)


@dataclass
class SessionToolResult:
    status: str
    final_text: str = ""
    usage: dict | None = None
    error: str | None = None
    error_code: str | None = None
    structured_output: dict | None = None


async def run_session(worker, ctx, env, config):
    """Invoke the frozen original registration once; recovery never calls this again."""
    frozen = ctx.descriptor.config["background"]
    name, arguments = frozen["tool"], frozen["arguments"]
    registered = env.tool_registry.get(name) if env.tool_registry is not None else None
    if registered is None or sha256_canonical(
        registered.definition.to_openai_tool()
    ) != frozen["definition_hash"]:
        return SessionToolResult(
            "failed", error="definition_unavailable: background tool changed",
            error_code="definition_unavailable",
        )
    loop = SimpleNamespace(
        config=config,
        tool_registry=env.tool_registry,
        _runtime_worker=worker,
        _runtime_worker_context=ctx,
        store=worker.runtime.store,
    )
    @contextmanager
    def bound(setter, value, reset):
        token = setter(value)
        try:
            yield
        finally:
            reset(token)

    async def stop_original():
        while not ctx.stop.is_cancelled():
            state = await worker.inputs.boundary(ctx.batch.target)
            if state is not None and (
                state["lifecycle"] in ("closing", "closed")
                or int(state["cancel_seq"]) > ctx.batch.cancel_seq
            ):
                ctx.stop.cancel(state["cancel_reason"] or "cancelled")
                break
            await asyncio.sleep(0.05)
        policy = config.stop_policy
        grace = (
            policy.abort_tool_s if registered.definition.interrupt == "abort"
            else policy.subtask_s if registered.definition.interrupt == "background"
            else policy.finish_tool_s
        )
        await asyncio.sleep(grace)
        ctx.task.cancel("background cancellation grace elapsed")

    with ExitStack() as stack:
        for setter, value, reset in (
            (agent_context.set_current_loop, loop, agent_context.reset_current_loop),
            (agent_context.set_session_id, ctx.session_id, agent_context.reset_session_id),
            (agent_context.set_current_cancel_token, ctx.stop, agent_context.reset_current_cancel_token),
        ):
            stack.enter_context(bound(setter, value, reset))
        stopper = asyncio.create_task(stop_original())
        try:
            output = await env.tool_registry.invoke_async(name, arguments)
            if isinstance(output, ToolResult):
                output = output.content
            text = output if isinstance(output, str) else canonical(output)
            if text.startswith("Error:"):
                return SessionToolResult("failed", text, error=text, error_code="tool_failed")
            return SessionToolResult("completed", text, usage={})
        except asyncio.CancelledError as exc:
            if ctx.stop.is_cancelled():
                reason = ctx.stop.reason or "cancelled"
                return SessionToolResult("cancelled", error=reason, error_code="cancelled")
            return SessionToolResult("interrupted", error=str(exc), error_code="interrupted")
        except Exception as exc:
            return SessionToolResult(
                "failed", error=f"{type(exc).__name__}: {exc}", error_code="tool_failed",
            )
        finally:
            stopper.cancel()
            await asyncio.gather(stopper, return_exceptions=True)
