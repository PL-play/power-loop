"""Original-context stops are durable admission barriers, never exit proofs."""
from __future__ import annotations

import asyncio

import pytest

from power_loop.runtime import root_context, root_control
from power_loop.runtime.jobs import JobActor, JobError, JobManager

from .test_job_registry_192 import registry_store
from .test_root_context_192 import NAMESPACE, root_case

__all__ = ["registry_store", "root_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
PAYLOAD = {"origin_run_id": 101}


async def stop(c, *, key="original-cancel", payload=None, manager=None):
    return await root_control.cancel(manager or c.manager, c.actor, namespace=NAMESPACE,
        payload=PAYLOAD if payload is None else payload, system_key=key, reason="user stop")


async def send(c, payload=None):
    result = await c.send("original work", source={"namespace":NAMESPACE, "payload":PAYLOAD if payload is None else payload})
    return result


async def test_unstarted_context_is_blocked_without_creating_a_root_or_usage(root_case):
    c = root_case
    receipt = await stop(c)
    assert receipt["state"] == "accepted" and receipt["roots"] == []
    assert all(receipt[k] is False for k in ("execution_authority", "resource_stop_confirmed", "billing_settled"))
    rejected = await send(c)
    assert rejected.status == 'cancelled' and 'root_context_cancelled' in rejected.error, rejected
    assert not c.model.calls and not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs}")
    await send(c, {"origin_run_id": 202})
    assert len(c.model.calls) == 1


async def test_stop_recall_survives_fresh_manager_and_preserves_closed_result(root_case):
    c = root_case
    await send(c)
    original = await c.manager.get_job(c.actor, c.model.calls[0])
    receipt = await stop(c)
    [effect] = receipt["roots"]
    assert effect["source"]["root_job_id"] == original.job_id
    assert effect["operation"]["result"]["cancel_requested"] is True
    saved = await c.manager.get_job(c.actor, original.job_id)
    assert saved.status == original.status and saved.result == original.result
    assert saved.cancel_requested_at is not None
    other = JobManager(await c.connect())
    assert await stop(c, manager=other) == receipt
    rejected = await send(c)
    assert rejected.status == 'cancelled' and 'root_context_cancelled' in rejected.error, rejected
    assert len(c.model.calls) == 1


async def test_same_key_with_changed_body_is_rejected_and_different_context_is_untouched(root_case):
    c = root_case
    await send(c)
    await send(c, {"origin_run_id": 202})
    receipt = await stop(c)
    assert len(receipt["roots"]) == 1
    assert (await c.manager.get_job(c.actor, c.model.calls[1])).cancel_requested_at is None
    with pytest.raises(JobError, match="operation_conflict"):
        await stop(c, payload={"origin_run_id": 202})
    await send(c, {"origin_run_id": 202})
    assert len(c.model.calls) == 3


async def test_control_is_mechanical_after_root_session_archival(root_case):
    c = root_case
    await send(c)
    await c.store.archive_session(c.sid)
    assert len((await stop(c))["roots"]) == 1
    assert (await c.manager.get_job(c.actor, c.model.calls[0])).cancel_requested_at is not None


async def test_foreign_actor_cannot_stop_an_existing_owner(root_case):
    c = root_case
    await send(c)
    foreign = JobActor(c.actor.tenant_id, "foreign", c.sid)
    receipt = await root_control.cancel(c.manager, foreign, namespace=NAMESPACE,
        payload=PAYLOAD, system_key="foreign", reason="not owner")
    assert receipt["roots"] == []
    assert (await c.manager.get_job(c.actor, c.model.calls[0])).cancel_requested_at is None
    await send(c)
    assert len(c.model.calls) == 2


async def test_running_owner_blocks_descendant_execution_without_claiming_exit(root_case):
    c = root_case
    await send(c)
    # A separate unclosed original execution models a remote owner still running.
    document = root_context._document(NAMESPACE, PAYLOAD)
    registered = await c.manager.register_job(c.actor, kind="send", session_id=c.sid,
        system_key="input:999", request={"host_context": document, "send_index": 999, "input_ref":{"store_id":c.store.store_id,"session_id":c.sid,"message_id":"999"}})
    lease = await c.manager.claim_job(c.actor, registered.job.job_id,
        executor_id="remote", boot_id="remote-boot", expected_version=1)
    child = await c.manager.register_job(c.actor, kind="subagent", session_id=await c.store.create_session(parent_session_id=c.sid), parent=lease, system_key="child")
    receipt = await stop(c)
    assert len(receipt["roots"]) == 2 and receipt["resource_stop_confirmed"] is False
    owner = await c.manager.get_job(c.actor, registered.job.job_id)
    assert owner.status == "cancelling" and owner.resource_state == "running"
    with pytest.raises(JobError, match="cancel"):
        await c.manager.claim_job(c.actor, child.job.job_id,
            executor_id="child", boot_id="boot", expected_version=1)
    closed = await c.manager.close_model_execution(c.actor, lease, evidence={"usage": {"total_tokens": 19}})
    assert closed.resource_state == "stopped"  # This model ended; descendant cancellation was checked separately.
    assert (await c.manager.execution_usage(c.actor, registered.job.job_id))['total_tokens'] == 19
    assert (await stop(c)) == receipt  # Historical intent does not rewrite itself as exit evidence.


async def test_register_and_cancel_share_a_cross_connection_serialization_barrier(root_case):
    c = root_case
    other = JobManager(await c.connect())
    document = root_context._document(NAMESPACE, PAYLOAD)
    async def register():
        try:
            return await other.register_job(c.actor, kind="send", session_id=c.sid,
                system_key="input:1", request={"host_context": document, "send_index": 1, "input_ref":{"store_id":c.store.store_id,"session_id":c.sid,"message_id":"1"}})
        except JobError as exc:
            assert "root_context_cancelled" in str(exc)
    root, receipt = await asyncio.gather(register(), stop(c))
    if root is not None:
        assert (await c.manager.get_job(c.actor, root.job.job_id)).cancel_requested_at is not None
        assert [r["source"]["root_job_id"] for r in receipt["roots"]] == [root.job.job_id]
        replay = await other.register_job(c.actor, kind="send", session_id=c.sid,
            system_key="input:1", request={"host_context": document, "send_index": 1, "input_ref":{"store_id":c.store.store_id,"session_id":c.sid,"message_id":"1"}})
        assert replay.replayed and replay.job.cancel_requested_at is not None
    with pytest.raises(JobError, match="root_context_cancelled"):
        await other.register_job(c.actor, kind="send", session_id=c.sid,
            system_key="new-root", request={"host_context": document})
    assert not c.model.calls
