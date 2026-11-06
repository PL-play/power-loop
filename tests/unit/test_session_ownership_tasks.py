"""Independent units never inherit the parent's execution or cancellation."""
import asyncio
from types import SimpleNamespace

import pytest

from power_loop.core.agent_context import _current_cancel_token
from power_loop.core.execution_context import get_execution_context
from power_loop.runtime.invocation import _INVOCATION
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.session_ownership import (
    MemorySessionOwnershipCoordinator,
    hold_session_ownership,
    recover_acquired_session,
)
from power_loop.runtime.task_runtime import _SEND, TaskRuntime

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def holder(store, coordinator, name='node', ttl=90):
    runtime = TaskRuntime(JobManager(store), tenant_id='tenant', principal_id='host',
                          executor_id=name, boot_id=name + '-boot')
    return SimpleNamespace(store=store, session_coordinator=coordinator, task_runtime=runtime,
        config=SimpleNamespace(session_lease_ttl_s=ttl), _owner_id=name,
        _cache_invalidate=lambda sid: None)


async def test_child_has_own_grant_and_context_and_survives_parent_release(registry_store):
    store, _ = registry_store
    parent_sid, child_sid = await store.create_session(), await store.create_session()
    coordinator = MemorySessionOwnershipCoordinator()
    parent, child = holder(store, coordinator, 'parent'), holder(store, coordinator, 'child')
    entered, finish = asyncio.Event(), asyncio.Event()
    async def run_child():
        async with hold_session_ownership(child, child_sid) as lease:
            assert _SEND.get() is None and _INVOCATION.get() is None
            assert _current_cancel_token.get() is None
            assert get_execution_context().session_id == child_sid
            assert lease is not parent_lease
            entered.set()
            await finish.wait()
            lease.require_owned()
            await store.append_message(child_sid, role='assistant', content='child finished')
    async with hold_session_ownership(parent, parent_sid) as parent_lease:
        tokens = [(v, v.set(value)) for v, value in ((_SEND, object()),
            (_INVOCATION, (parent_sid, 1, 'parent-tool')), (_current_cancel_token, object()))]
        task = asyncio.create_task(run_child())
        try:
            await asyncio.wait_for(entered.wait(), 2)
        finally:
            for var, token in reversed(tokens):
                var.reset(token)
    assert parent_lease.released
    finish.set()
    await task
    assert not coordinator._leases


async def test_parent_recovery_only_interrupts_its_session(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    child_sid = await store.create_session(parent_session_id=sid)
    loop = holder(store, MemorySessionOwnershipCoordinator())
    manager, actor = loop.task_runtime.manager, loop.task_runtime.actor(sid)
    parent = await manager.register_job(actor, kind='send', session_id=sid, system_key='root')
    proof = await manager.claim_job(actor, parent.job.job_id, executor_id='old', boot_id='old', expected_version=1)
    child = await manager.register_job(actor, kind='subagent', session_id=child_sid, parent=proof, system_key='child')
    child_proof = await manager.claim_job(actor, child.job.job_id, executor_id='remote', boot_id='remote', expected_version=1)
    async with hold_session_ownership(loop, sid):
        await recover_acquired_session(loop, sid, cache_valid=False)
        assert (await manager.get_job(actor, parent.job.job_id)).outcome == 'interrupted'
        live = await manager.get_job(actor, child.job.job_id)
        assert live.status == 'running' and live.generation == child_proof.generation
        await manager.check_execution(actor, child_proof)


async def test_rebuilding_under_same_grant_keeps_generation(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    coordinator = MemorySessionOwnershipCoordinator()
    loop, rebuilt = holder(store, coordinator), holder(store, coordinator)
    async with hold_session_ownership(loop, sid) as original:
        await recover_acquired_session(loop, sid, cache_valid=False)
        manager, actor = loop.task_runtime.manager, loop.task_runtime.actor(sid)
        job = await manager.register_job(actor, kind='send', session_id=sid, system_key='same')
        proof = await manager.claim_job(actor, job.job.job_id, executor_id='node', boot_id='node-boot', expected_version=1)
        async with hold_session_ownership(rebuilt, sid) as reused:
            await recover_acquired_session(rebuilt, sid, cache_valid=False)
            assert reused is original
            current = await manager.get_job(actor, proof.job_id)
            assert current.generation == 1 and current.status == 'running'


async def test_old_generation_cannot_publish_over_a_new_execution(registry_store):
    from power_loop.runtime.jobs import JobError
    store, connect = registry_store
    loop = holder(store, MemorySessionOwnershipCoordinator())
    manager = loop.task_runtime.manager
    sid = await store.create_session()
    actor = loop.task_runtime.actor(sid)
    registered = await manager.register_job(actor, kind='send', session_id=sid, system_key='generation')
    old = await manager.claim_job(actor, registered.job.job_id, executor_id='old', boot_id='old', expected_version=1)
    newer = JobManager(await connect())
    # The generic transition used by resume writers locks the original tree and
    # conditions the update on the observed version and generation.
    async with newer.tree_write(old.job_id) as tx:
        changed = await tx.write_job(old.job_id, f"UPDATE {manager.jobs} SET generation=2,version=version+1,"
            "executor_id='new',boot_id='new' WHERE job_id=? AND generation=1 AND version=2", (old.job_id,))
        assert changed == 1
        async def late_result():
            with pytest.raises(JobError, match='stale_execution'):
                await manager.record_execution_result(actor, old, outcome='interrupted',
                    result={'late': True}, resource_state='unknown')
        pending = asyncio.create_task(late_result())
        await asyncio.sleep(.05)
        assert not pending.done()
    before = await manager.get_job(actor, old.job_id)
    await pending
    assert await manager.get_job(actor, old.job_id) == before
