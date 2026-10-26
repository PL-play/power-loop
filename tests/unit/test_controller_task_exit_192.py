"""Only the original retained Task finishing creates a durable exit checkpoint."""
import asyncio

import pytest

from power_loop.runtime import delegation
from power_loop.runtime.controller_task_exit import read
from power_loop.runtime.jobs import JobError

from .test_job_registry_192 import registry_store, runtime
from .test_original_owner_review_192 import original_owner_case, wait_owned

__all__ = ['registry_store', 'runtime']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('kind', ['subagent', 'workflow_node'])
async def test_actual_task_exit_is_saved_without_releasing_result_or_billing_obligations(registry_store, kind):
    c = await original_owner_case(registry_store, kind, allowed_calls=1)
    initial = set(delegation._TASKS)
    try:
        await c.loop.send('Launch original work once.', c.state.root)
        await wait_owned(c, kind, initial)
        await c.runtime.flush_control_checkpoints()
        owner = c.state.owner_binding
        async with c.manager._write() as tx:
            row = await c.manager._authorize(tx, owner.actor, owner.lease.job_id)
            data = await read(c.manager, tx, row, owner.lease)
        assert data['task_exit']['state'] in ('returned', 'raised', 'cancelled')
        assert not data['execution_authority'] and not data['descendant_stop_confirmed'] and not data['billing_settled']
        job = await c.manager.get_job(owner.actor, owner.lease.job_id)
        assert job.resource_state == 'unknown' and job.result_status == 'none' and job.supervision_status == 'open'
        assert c.state.counts['BUILD'] == 1 and not c.state.effects
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.runtime.flush_control_checkpoints()
        await c.loop.aclose(drain_timeout_s=.1)


async def test_live_original_task_cannot_be_replaced_by_a_missing_handle_or_expired_lease(registry_store):
    c = await original_owner_case(registry_store, 'subagent', allowed_calls=2)
    initial = set(delegation._TASKS)
    try:
        await c.loop.send('Launch original work once.', c.state.root)
        await asyncio.wait_for(c.state.entered.wait(), 5)
        owner = c.state.owner_binding
        async with c.manager._write() as tx:
            row = await c.manager._authorize(tx, owner.actor, owner.lease.job_id)
            with pytest.raises(JobError):
                await read(c.manager, tx, row, owner.lease)
        assert not c.runtime._controller_task_exits.slots
        c.state.release.set()
        await wait_owned(c, 'subagent', initial)
        await c.runtime.flush_control_checkpoints()
        async with c.manager._write() as tx:
            row = await c.manager._authorize(tx, owner.actor, owner.lease.job_id)
            assert await read(c.manager, tx, row, owner.lease)
    finally:
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.runtime.flush_control_checkpoints()
        await c.loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('fault', ['before_commit', 'lost_ack', 'cancelled'])
async def test_failed_actual_exit_write_keeps_original_witness_for_mechanical_retry(registry_store, monkeypatch, fault):
    from power_loop.runtime import controller_task_exit

    c = await original_owner_case(registry_store, 'subagent', allowed_calls=2)
    initial = set(delegation._TASKS)
    original = controller_task_exit.persist
    observed = asyncio.Event()
    enabled = True

    async def fail(runtime, slot):
        if enabled and slot.lease.job_id == c.state.owner_binding.lease.job_id:
            if fault == 'lost_ack':
                await original(runtime, slot)
            observed.set()
            if fault == 'cancelled':
                raise asyncio.CancelledError('exit checkpoint interrupted')
            raise OSError('exit checkpoint unavailable')
        return await original(runtime, slot)

    try:
        await c.loop.send('Launch original work once.', c.state.root)
        await asyncio.wait_for(c.state.entered.wait(), 5)
        monkeypatch.setattr(controller_task_exit, 'persist', fail)
        c.state.release.set()
        await wait_owned(c, 'subagent', initial)
        await asyncio.wait_for(observed.wait(), 5)
        # Observe the bounded writer itself before inspecting retained state.
        writer = c.runtime._controller_task_exits._writer
        if writer is not None:
            await asyncio.gather(writer, return_exceptions=True)
        owner = c.state.owner_binding
        before = await c.manager.execution_usage(owner.actor, owner.lease.job_id, generation=1)
        assert c.runtime._controller_task_exits.failed_count() > 0
        with pytest.raises(JobError, match='controller_task_exit_requires_reconciliation'):
            c.runtime.check_local_admission()
        status = await c.runtime.executor_status()
        assert not status['store_safe_to_close'] and status['controller_task_exit_pending'] > 0
        enabled = False
        await asyncio.gather(*(c.runtime.flush_control_checkpoints() for _ in range(4)))
        assert not c.runtime._controller_task_exits.slots
        events = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='controller_task_exit'", (owner.lease.job_id,))
        assert len(events) == 1
        assert await c.manager.execution_usage(owner.actor, owner.lease.job_id, generation=1) == before
        assert not c.state.effects
    finally:
        enabled = False
        monkeypatch.setattr(controller_task_exit, 'persist', original)
        c.state.release.set()
        await c.runtime.shutdown(timeout_s=3)
        await c.runtime.flush_control_checkpoints()
        await c.loop.aclose(drain_timeout_s=.1)


async def primitive_controller(fixture, runtime, key, release):
    store, manager, actor, root, lease, connect, now = fixture
    reg = await manager.register_job(actor, kind='subagent', parent=lease, system_key=key)
    binding = await runtime.claim(actor, reg)
    async def body():
        async with runtime.execution(binding, owner=True):
            await release.wait()
    task = asyncio.create_task(body())
    runtime.track_task(task, job_id=binding.lease.job_id)
    return task, binding


async def test_new_exit_during_writer_snapshot_is_saved_without_another_model_or_host_call(runtime, monkeypatch):
    from power_loop.runtime import controller_task_exit
    from power_loop.runtime.task_runtime import TaskRuntime

    fixture = runtime
    store, manager, actor, root, lease, connect, now = fixture
    actual = TaskRuntime(manager, tenant_id=actor.tenant_id, principal_id=actor.principal_id,
                         executor_id='executor', boot_id='boot', lease_ms=300000)
    first_release, second_release = asyncio.Event(), asyncio.Event()
    first, first_binding = await primitive_controller(fixture, actual, 'first-exit', first_release)
    second, second_binding = await primitive_controller(fixture, actual, 'second-exit', second_release)
    writing, unblock, saved = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = controller_task_exit.persist
    async def gated(rt, slot):
        if slot.lease == first_binding.lease:
            writing.set()
            await unblock.wait()
        await original(rt, slot)
        if slot.lease == second_binding.lease:
            saved.set()
    monkeypatch.setattr(controller_task_exit, 'persist', gated)
    try:
        first_release.set()
        await first
        await asyncio.wait_for(writing.wait(), 5)
        second_release.set()
        await second
        unblock.set()
        await asyncio.wait_for(saved.wait(), 5)
        writer = actual._controller_task_exits._writer
        await asyncio.gather(writer, return_exceptions=False)
        assert not actual._controller_task_exits.slots
    finally:
        unblock.set()
        first_release.set()
        second_release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        await actual.flush_control_checkpoints()


async def test_allocating_writer_after_previous_writer_finished_keeps_witness_and_closes_coroutine(runtime, monkeypatch):
    from power_loop.runtime import controller_task_exit
    from power_loop.runtime.task_runtime import TaskRuntime

    store, manager, actor, root, lease, connect, now = runtime
    actual = TaskRuntime(manager, tenant_id=actor.tenant_id, principal_id=actor.principal_id,
                         executor_id='executor', boot_id='boot', lease_ms=300000)
    release = asyncio.Event()
    first, binding = await primitive_controller(runtime, actual, 'successful-exit', release)
    release.set()
    await first
    await actual.flush_control_checkpoints()
    writer = actual._controller_task_exits._writer
    if writer is not None:
        await asyncio.gather(writer, return_exceptions=False)
    original = asyncio.create_task
    refused = []
    def refuse(coroutine, *args, **kwargs):
        if getattr(coroutine, 'cr_code', None) is controller_task_exit.TaskExitCache._write_ready.__code__:
            refused.append(coroutine)
            raise RuntimeError('cannot allocate metadata writer')
        return original(coroutine, *args, **kwargs)
    monkeypatch.setattr(asyncio, 'create_task', refuse)
    release2 = asyncio.Event()
    second, binding2 = await primitive_controller(runtime, actual, 'allocation-failed-exit', release2)
    release2.set()
    await second
    await asyncio.sleep(0)  # Run the Task's registered done callbacks.
    assert refused and all(coroutine.cr_frame is None for coroutine in refused)
    assert actual._controller_task_exits.failed_count() == 1
    await actual.flush_control_checkpoints()
    async with manager._write() as tx:
        row = await manager._authorize(tx, actor, binding2.lease.job_id)
        assert await read(manager, tx, row, binding2.lease)


async def test_new_controller_event_identifiers_fit_every_declared_task_store_dialect():
    from types import SimpleNamespace

    from power_loop.runtime.controller_handoff import identity as handoff_id
    from power_loop.runtime.controller_recovery import identity as stop_id
    from power_loop.runtime.controller_recovery import notice_id
    from power_loop.runtime.controller_retirement import identity as retirement_id
    from power_loop.runtime.controller_task_exit import identity as exit_id
    from power_loop.runtime.jobs import JobLease

    manager = SimpleNamespace(store=SimpleNamespace(store_id='original-store'))
    lease = JobLease('job_' + 'a'*32, 'original-executor', 'original-boot', 1)
    source = {'job_id': lease.job_id, 'generation': lease.generation}
    for identity in (handoff_id(manager, source), retirement_id(manager, lease), exit_id(manager, lease),
                     stop_id(manager, source), notice_id(manager, source)):
        assert len(identity) <= 64
    assert len(handoff_id(manager, source, legacy=True)) > 64
    assert len(retirement_id(manager, lease, legacy=True)) > 64
