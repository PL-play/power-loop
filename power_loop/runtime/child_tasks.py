"""Independent child admission using the original tool call and frozen input."""
from __future__ import annotations

from dataclasses import asdict, replace

from power_loop.agent.runtime_types import (
    Command,
    InputRef,
    JobRef,
    RequestOptions,
    RuntimeRequestError,
    TargetRef,
    TrustedActor,
    plain,
)

from .invocation import current_invocation
from .jobs import JobError, _hash


async def tool_actor(loop):
    from .task_runtime import _SEND
    binding = _SEND.get()
    if binding is None or binding.runtime is not loop.task_runtime:
        raise JobError("task_send_required")
    sid = binding.actor.session_id
    seq, call_id = current_invocation(sid)
    if seq is None or not call_id:
        raise JobError("persisted_tool_call_required")
    ref = await loop.store.message_ref(sid, seq)
    return binding, TrustedActor(binding.actor.tenant_id, binding.actor.principal_id, sid, ref, call_id)


async def create(loop, spec, task):
    from power_loop.core.agent_context import get_effective_tools

    from .execution_roles import require_orchestration, role_tools
    from .spec import _child_config

    await require_orchestration(loop, "subagent")
    host = getattr(loop, '_host_command', None)
    system_key = None
    if host is None:
        binding, actor = await tool_actor(loop)
    else:
        from .task_runtime import _SEND
        binding = _SEND.get()
        actor = TrustedActor(binding.actor.tenant_id, binding.actor.principal_id, binding.actor.session_id)
        if host['next_child'] >= 1000:
            from power_loop.agent.runtime_types import CapacityError
            raise CapacityError('host command child limit reached')
        host['next_child'] += 1
        system_key = f"host-child:{loop._runtime_worker_context.input_ref.message_id}:{host['next_child']}"
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
    descriptor = replace(parent_descriptor, config={**plain(parent_descriptor.config),
        "child_spec": asdict(spec), "child_options": plain(options)})
    from power_loop.runtime.store.types import SubagentLifecycle
    return await admit(loop, binding, actor, kind="subagent", title=spec.name, descriptor=descriptor,
                       options=options, body=task, role="subagent", lifecycle=SubagentLifecycle(spec.lifecycle),
                       system_key=system_key)


async def admit(loop, binding, actor, *, kind, title, descriptor, options, body, role, lifecycle, system_key=None):
    worker = loop._runtime_worker
    port, manager = worker.inputs, loop.task_runtime.manager
    # Stable IDs make retries refer to the same original admission, even after
    # the first child has started. The job creation event owns deduplication.
    key = manager._origin_key(binding.actor, "create", actor.call_ref, actor.tool_call_id, system_key)
    sid = "child_" + key[:32]
    input_ref = InputRef(port.store.store_id, sid, "start_" + key[:32])
    async with manager.joined_write() as tx:
        await port.lock_owned(tx, TargetRef(port.store.store_id, actor.session_id), loop._runtime_worker_context)
        await manager._fence(tx, binding.actor, binding.lease)
        if system_key is None:
            await manager._validate_call(tx, binding.actor, actor.call_ref, actor.tool_call_id)
        existing = await tx.fetchone(f"SELECT job_id FROM {manager.events} WHERE operation_key=?", (key,))
        if existing is None:
            await port.store._create_session_tx(tx, sid, manager.now(),
                system_prompt=options.system_prompt, model=options.model,
                lifecycle=lifecycle,
                parent_session_id=actor.session_id, spawn_tool_call_id=actor.tool_call_id,
                metadata={"tenant_id": actor.tenant_id, "principal_id": actor.principal_id,
                    "execution_role": role, "standalone_subagent": kind == "subagent",
                    "runtime_descriptor": plain(descriptor)})
        registration = await manager.register_job(binding.actor, kind=kind, title=title,
            session_id=sid, parent=binding.lease, independent=loop._runtime_worker_context.descriptor.kind != "workflow_node",
            call_ref=actor.call_ref, tool_call_id=actor.tool_call_id,
            system_key=system_key, request={"input_ref": plain(input_ref)})
        command = Command(input_ref.message_id, TargetRef(port.store.store_id, sid),
            "send", {"body": body}, options=plain(options))
        submitter = actor if system_key is None else TrustedActor(actor.tenant_id, actor.principal_id, None)
        receipt = await port.accept_input(tx, command, submitter, descriptor=descriptor, child_start=True)
        if receipt.newly_accepted:
            await tx.execute(f"UPDATE {port.q} SET delivered_job_id=?,delivered_job_generation=0 "
                "WHERE session_id=? AND item_id=?", (registration.job.job_id, sid, input_ref.message_id))
    ref = JobRef(port.store.store_id, registration.job.job_id, 0)
    await worker.runtime.client._notify(TargetRef(ref.store_id, sid), input_ref.message_id,
                                       input_ref, ["work", "activation"])
    return {"job_ref": plain(ref), "input_ref": plain(input_ref), "job_id": ref.job_id,
        "session_id": sid, "run_id": registration.operation.operation_id[3:],
        "operation_id": registration.operation.operation_id, "status": "accepted", "mode": "create"}


async def send_input(loop, sid, message, *, urgent=False):
    binding, actor = await tool_actor(loop)
    port = loop._runtime_worker.inputs
    await port.authorize(port.db, TargetRef(port.store.store_id, sid), actor)
    row = await port.db.fetchone(f"SELECT job_id,generation,outcome FROM {port.jobs} "
        "WHERE session_id=? AND kind='subagent'", (sid,))
    if row is None:
        raise RuntimeRequestError("not_found")
    target = TargetRef(port.store.store_id, sid, row["job_id"])
    mid = "nudge_" + _hash([plain(actor.call_ref), actor.tool_call_id])[:32]
    if row['outcome'] is not None:
        receipt = await loop._runtime_worker.runtime.client.resume(
            JobRef(port.store.store_id, row['job_id'], row['generation']),
            message_id=mid, actor=actor, body=message)
        return plain(receipt)
    manager = loop.task_runtime.manager
    async with manager.joined_write() as tx:
        await manager._fence(tx, binding.actor, binding.lease)
        receipt = await port.accept_input(tx, Command(mid, target, "send", {"body": message}, urgent,
            expected_job_generation=row["generation"]), actor)
    await loop._runtime_worker.runtime.client._notify(target, mid, receipt.input_ref,
        ["control", "work", "activation"] if urgent else ["work", "activation"])
    return plain(receipt)
