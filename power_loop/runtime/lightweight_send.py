"""An ordinary unbilled send keeps its control state in memory.

The turn still has an execution identity, so model and tool attempts can record
what they did. Nothing is written to the task tables unless the turn later
needs a real task. The caller's result is remembered on the input store.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from power_loop.agent.runtime_types import JobRef, Status

from .execution_state import ExecutionIdentity, ExecutionState
from .jobs import JobLease, _hash
from .task_runtime import Execution


class _MemoryJournal:
    @asynccontextmanager
    async def transaction(self):
        yield None

    async def create(self, identity, summary, *, transaction=None):
        return None

    async def append(self, identity, number, records, summary, *, transaction=None):
        return None


def lightweight_job_id(store, session_id, message_id):
    return "job_" + _hash([store.store_id, session_id, "lightweight", message_id])[:32]


def open_lightweight(runtime, actor, session_id, message_id):
    lease = JobLease(lightweight_job_id(runtime.manager.store, session_id, message_id),
                     runtime.executor_id, runtime.boot_id, 1)
    binding = Execution(runtime, actor, lease)
    identity = ExecutionIdentity(runtime.manager.store.store_id, lease.job_id, lease.generation,
                                 lease.executor_id, lease.boot_id, actor.tenant_id,
                                 actor.principal_id, session_id)
    state = ExecutionState(identity, _MemoryJournal(), now_ms=lambda: runtime.manager.now())
    binding.resource_evidence.update(
        lightweight=True,
        lightweight_message_id=message_id,
        _control_state=state,
        _control_tool_plan_required=True,
        _control_source={
            "job_id": lease.job_id, "generation": lease.generation, "kind": "send",
            "execution_role": "main", "session_id": session_id, "tenant_id": actor.tenant_id,
            "principal_id": actor.principal_id, "root_job_id": lease.job_id, "parent_job_id": None,
            "executor_id": lease.executor_id, "boot_id": lease.boot_id,
            "created_by_session_id": session_id,
        },
    )
    from .root_context import _authority
    binding.resource_evidence["_control_authority"] = _authority(runtime)
    return binding


async def upgrade(binding, tx):
    """Turn this ordinary send into a real task before it controls another one.

    The in-memory lease already has generation 1. The inserted row uses that
    same identity, so later checks see the holder that has been running.
    """
    if not binding.resource_evidence.get("lightweight") or binding.resource_evidence.get("upgraded"):
        return
    manager, actor, lease = binding.runtime.manager, binding.actor, binding.lease
    message_id = binding.resource_evidence["lightweight_message_id"]
    from .root_context import for_send
    request = {"input_ref": {"message_id": message_id}, "send_index": binding.resource_evidence.get("_runtime_send_index")}
    provenance = for_send(binding.runtime, actor.session_id)
    if provenance is not None:
        request["host_context"] = provenance
    from .jobs import _hash
    key = manager._origin_key(actor, "create", None, None, "input:" + message_id)
    now = manager.now()
    # Check if runtime requires executor registration
    executor_key = None
    if hasattr(binding.runtime, '_lifecycle') and binding.runtime._lifecycle is not None:
        from .executor_registry import ExecutorIdentity, _key
        identity = ExecutorIdentity(actor.tenant_id, actor.principal_id, lease.executor_id, lease.boot_id)
        executor_key = _key(manager, identity)
    await tx.insert_job(lease.job_id, lease.job_id,
        f"INSERT INTO {manager.jobs} (job_id,parent_job_id,root_job_id,tenant_id,principal_id,"
        "session_id,created_by_session_id,kind,title,completion_scope_job_id,blocks_completion,"
        "status,version,generation,intake,resource_state,supervision_status,result_status,created_at,updated_at,"
        "execution_role,control_owner_job_id,executor_id,boot_id,executor_key) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (lease.job_id, None, lease.job_id, actor.tenant_id, actor.principal_id, actor.session_id,
         actor.session_id, "send", "", lease.job_id, 0, "running", 1, lease.generation, "open",
         "running", "open", "none", now, now, "main", None, lease.executor_id, lease.boot_id, executor_key))
    await manager._insert_operation(tx, job_id=lease.job_id, version=1, actor=actor, action="create",
        key=key, fingerprint=_hash(["lightweight", message_id]), payload={"request": request})
    await tx.execute(
        f"UPDATE {manager.store.t.follow_up_queue} SET delivered_job_id=?,delivered_job_generation=? "
        "WHERE session_id=? AND item_id=?",
        (lease.job_id, lease.generation, actor.session_id, message_id))
    # Create execution_scope event for membership verification
    from dataclasses import asdict
    from .execution_membership import scope_id, _insert as _insert_scope, _remember_after_commit
    from .jobs import _json
    scope_data = {
        "version": 1,
        "lease": asdict(lease),
        "owner": None,
        "tenant_id": actor.tenant_id,
        "principal_id": actor.principal_id,
        "kind": "send",
        "execution_role": "main",
        "parent_job_id": None,
        "created_by_session_id": actor.session_id,
        "session_id": actor.session_id,
    }
    # Add executor binding if runtime requires registration
    if executor_key is not None:
        from .executor_registry import ExecutorIdentity, registration_binding
        scope_data["version"] = 2
        # Create a minimal row dict for registration_binding
        row = {
            "job_id": lease.job_id,
            "generation": lease.generation,
            "tenant_id": actor.tenant_id,
            "principal_id": actor.principal_id,
            "executor_id": lease.executor_id,
            "boot_id": lease.boot_id,
            "executor_key": executor_key,
        }
        scope_data["executor_binding"] = await registration_binding(manager, tx, row)
    await _insert_scope(manager, tx, scope_id(manager, lease.job_id, lease.generation),
                        lease.job_id, "execution_scope", 1, scope_data)
    _remember_after_commit(manager, tx, "_membership_scopes", (lease.job_id, lease.generation), _json(scope_data))
    binding.resource_evidence["upgraded"] = True


async def ensure(binding):
    if binding is None or not binding.resource_evidence.get("lightweight") or binding.resource_evidence.get("upgraded"):
        return
    manager = binding.runtime.manager
    current = manager._joined_write.get()
    if current is not None:
        await upgrade(binding, current)
        return
    async with manager.joined_write() as tx:
        await upgrade(binding, tx)


def remember_result(loop, ref, result, evidence=None):
    worker = getattr(loop, "_runtime_worker", None)
    if worker is None:
        return
    store = worker.runtime.store
    results = getattr(store, "lightweight_results", None)
    if results is None:
        results = {}
        store.lightweight_results = results
    key = (ref.session_id, ref.message_id)
    getattr(store, "lightweight_accepted", {}).pop(key, None)
    results[key] = (
        Status(ref, result.job_ref, result.status, result.error, True,
               result.status == "cancelled", "stopped"),
        result, JobRef(ref.store_id, result.job_ref.job_id, result.job_ref.generation))
    if evidence is not None:
        remembered = getattr(store, "lightweight_evidence", None)
        if remembered is None:
            remembered = {}
            store.lightweight_evidence = remembered
        remembered[key] = dict(evidence)
