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
    await tx.insert_job(lease.job_id, lease.job_id,
        f"INSERT INTO {manager.jobs} (job_id,parent_job_id,root_job_id,tenant_id,principal_id,"
        "session_id,created_by_session_id,kind,title,completion_scope_job_id,blocks_completion,"
        "status,version,generation,intake,resource_state,supervision_status,result_status,created_at,updated_at,"
        "execution_role,control_owner_job_id,executor_id,boot_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (lease.job_id, None, lease.job_id, actor.tenant_id, actor.principal_id, actor.session_id,
         actor.session_id, "send", "", lease.job_id, 0, "running", 1, lease.generation, "open",
         "running", "open", "none", now, now, "main", None, lease.executor_id, lease.boot_id))
    await manager._insert_operation(tx, job_id=lease.job_id, version=1, actor=actor, action="create",
        key=key, fingerprint=_hash(["lightweight", message_id]), payload={"request": request})
    await tx.execute(
        f"UPDATE {manager.store.t.follow_up_queue} SET delivered_job_id=?,delivered_job_generation=? "
        "WHERE session_id=? AND item_id=?",
        (lease.job_id, lease.generation, actor.session_id, message_id))
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


def remember_result(loop, ref, result):
    worker = getattr(loop, "_runtime_worker", None)
    if worker is None:
        return
    results = getattr(worker.runtime.store, "lightweight_results", None)
    if results is None:
        results = {}
        worker.runtime.store.lightweight_results = results
    getattr(worker.runtime.store, "lightweight_accepted", {}).pop((ref.session_id, ref.message_id), None)
    results[(ref.session_id, ref.message_id)] = (
        Status(ref, result.job_ref, result.status, result.error, True,
               result.status == "cancelled", "stopped"),
        result, JobRef(ref.store_id, result.job_ref.job_id, result.job_ref.generation))
