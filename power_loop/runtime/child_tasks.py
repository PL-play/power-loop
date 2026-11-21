"""Deterministic child sessions admitted by the original assistant tool call."""
from __future__ import annotations

from dataclasses import asdict, replace

from power_loop.agent.runtime_types import (
    RequestOptions,
    RuntimeRequestError,
    TargetRef,
    TrustedActor,
    plain,
    sha256_canonical,
)

from .invocation import current_invocation


async def tool_actor(loop):
    from .task_runtime import current_session_send

    binding = current_session_send(loop.store, loop._runtime_worker_context.session_id)
    if binding is None:
        raise RuntimeRequestError("invalid_options", "session tool call required")
    sid = binding.session_id
    host = getattr(loop, "_host_command", None)
    if host is not None:
        seq = host["message_seq"]
        call_id = f"inline:{loop._runtime_worker_context.input_ref.message_id}:{max(1, host['next_child'])}"
    else:
        seq, call_id = current_invocation(sid)
    if seq is None or not call_id:
        raise RuntimeRequestError("invalid_options", "persisted tool call required")
    ref = await loop.store.message_ref(sid, seq)
    actor = TrustedActor(
        loop._runtime_worker_context.actor.tenant_id,
        loop._runtime_worker_context.actor.principal_id,
        sid,
        ref,
        call_id,
    )
    return binding, actor


async def create(loop, spec, task):
    from power_loop.core.agent_context import get_effective_tools

    from .execution_roles import require_orchestration, role_tools
    from .spec import _child_config

    await require_orchestration(loop, "subagent")
    host = getattr(loop, "_host_command", None)
    if host is not None:
        if host["next_child"] >= 1000:
            from power_loop.agent.runtime_types import CapacityError
            raise CapacityError()
        host["next_child"] += 1
    binding, actor = await tool_actor(loop)
    parent = TargetRef(loop.store.store_id, actor.session_id)
    parent_descriptor = loop._runtime_worker_context.descriptor
    names = list(loop.tool_registry.names()) if loop.tool_registry else []
    effective = get_effective_tools()
    allowed = [name for name in names if effective is None or name in effective]
    if spec.tools is not None:
        missing = set(spec.tools) - set(allowed)
        if missing:
            raise RuntimeRequestError("forbidden", "child tools exceed the current tool grant")
        allowed = list(spec.tools)
    allowed = role_tools(allowed, "subagent", registry=loop.tool_registry)
    config = _child_config(spec, loop)
    options = RequestOptions(tools=allowed, **{name: getattr(config, name) for name in (
        "model", "temperature", "max_tokens", "max_rounds", "max_tokens_per_run",
        "system_prompt", "response_format")})
    descriptor = replace(parent_descriptor, config={
        **plain(parent_descriptor.config),
        "child_spec": asdict(spec),
        "child_options": plain(options),
    })
    target, start, receipt = await loop._runtime_worker.inputs.create_child(
        descriptor,
        parent=parent,
        actor=actor,
        call_ref=actor.call_ref,
        tool_call_id=actor.tool_call_id,
        body=task,
        cancel_propagation="linked" if spec.lifecycle == "linked" else "independent",
        require_assistant_call=host is None,
    )
    await loop._runtime_worker.runtime.client._notify(
        target, start.message_id, start, ["work", "activation"]
    )
    return {
        "target_ref": plain(target),
        "input_ref": plain(start),
        "session_id": target.session_id,
        "run_id": start.message_id,
        "status": "accepted",
        "newly_accepted": receipt.newly_accepted,
        "mode": "create",
    }


async def send_input(loop, sid, message, *, urgent=False):
    from .execution_roles import require_orchestration

    await require_orchestration(loop, "subagent")
    _, actor = await tool_actor(loop)
    target = TargetRef(loop.store.store_id, sid)
    message_id = "inp_" + sha256_canonical([
        "child_send", actor.session_id, plain(actor.call_ref), actor.tool_call_id,
    ])[:56]
    return plain(await loop._runtime_worker.runtime.client.send(
        target,
        message,
        message_id=message_id,
        actor=actor,
        priority="urgent" if urgent else "normal",
    ))
