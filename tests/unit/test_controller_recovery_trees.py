"""Controller recovery proves foreign handling before its own-tree writer."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop.runtime import controller_recovery
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_controller_recovery_192 import cleanup, ready
from .test_executor_drain_trees import stop_worker
from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture(params=['subagent', 'workflow_node'])
async def recovery_case(registry_store, request):
    case = await ready(registry_store, request.param)
    try:
        yield case
    finally:
        await cleanup(case)


@asynccontextmanager
async def observe(manager, monkeypatch):
    rows = await manager.db.fetchall(f"SELECT job_id,root_job_id FROM {manager.jobs}")
    roots = {row['job_id']: row['root_job_id'] for row in rows}
    write, one, many = manager._write, ControlTransaction.fetchone, ControlTransaction.fetchall
    footprints = []

    def note(tx, rows):
        tx.__dict__.setdefault('_recovery_reads', set()).update(
            row['job_id'] for row in rows if row is not None and 'job_id' in row)

    async def fetchone(tx, sql, params=()):
        row = await one(tx, sql, params)
        note(tx, [row])
        return row

    async def fetchall(tx, sql, params=()):
        rows = await many(tx, sql, params)
        note(tx, rows)
        return rows

    @asynccontextmanager
    async def scoped():
        async with write() as tx:
            yield tx
        jobs = tx.touched_jobs | tx.__dict__.get('_recovery_reads', set())
        footprints.append({roots[job] for job in jobs})

    with monkeypatch.context() as patch:
        patch.setattr(manager, '_write', scoped)
        patch.setattr(ControlTransaction, 'fetchone', fetchone)
        patch.setattr(ControlTransaction, 'fetchall', fetchall)
        yield footprints


async def test_original_controller_recovery_only_publishes_its_own_tree(recovery_case, monkeypatch):
    c = recovery_case
    runtime, actor = c.recovery_runtime, c.actor
    job_id = c.state.owner_binding.lease.job_id
    before = await runtime.manager.execution_usage(actor, job_id, generation=1)
    append, routed = ControlTransaction.append_event, []

    async def original_recipient(tx, columns, values, **kwargs):
        changed = await append(tx, columns, values, **kwargs)
        event = dict(zip(columns, values, strict=True))
        if event['event_type'] == 'controller_stopped':
            assert event['caller_session_id'] == actor.session_id
            assert actor.session_id in tx.sessions
            routed.append(event['event_id'])
        return changed

    monkeypatch.setattr(ControlTransaction, 'append_event', original_recipient)
    async with observe(runtime.manager, monkeypatch) as footprints:
        receipt = await controller_recovery.reconcile(runtime, actor, job_id)
    saved = await runtime.manager.get_job(actor, job_id)
    assert saved.outcome == 'interrupted' and saved.resource_state == 'stopped'
    assert saved.result['usage'] == before and saved.result_status == 'pending_delivery'
    assert receipt['notification_id'] and not receipt['execution_authority']
    assert c.state.counts['BUILD'] == 1 and c.state.counts['AFTER'] == 0 and not c.state.effects
    assert len(routed) == 1
    assert footprints and all(len(scope) <= 1 for scope in footprints), footprints


async def test_completed_recovery_replays_from_originals_without_another_writer(recovery_case, monkeypatch):
    c = recovery_case
    runtime, actor = c.recovery_runtime, c.actor
    manager, job_id = runtime.manager, c.state.owner_binding.lease.job_id
    receipt = await controller_recovery.reconcile(runtime, actor, job_id)
    before = await manager.db.fetchall(f'SELECT * FROM {manager.events} ORDER BY event_id')
    jobs = await manager.db.fetchall(f'SELECT * FROM {manager.jobs} ORDER BY job_id')
    async with observe(manager, monkeypatch) as footprints:
        assert await controller_recovery.reconcile(runtime, actor, job_id) == receipt
    assert not footprints
    assert await manager.db.fetchall(f'SELECT * FROM {manager.events} ORDER BY event_id') == before
    assert await manager.db.fetchall(f'SELECT * FROM {manager.jobs} ORDER BY job_id') == jobs


async def test_paused_recovery_proof_does_not_hold_the_control_writer(recovery_case, registry_store, monkeypatch):
    c = recovery_case
    runtime, actor = c.recovery_runtime, c.actor
    manager, job_id = runtime.manager, c.state.owner_binding.lease.job_id
    other = JobManager(await registry_store[1](), now_ms=manager.now)
    third = await other.register_job(actor, kind='send', session_id=actor.session_id, system_key='third-recovery-root')
    facts = controller_recovery._facts
    entered, release = asyncio.Event(), asyncio.Event()

    async def held(current, reader, *args, **kwargs):
        assert current._control_intents.get() is None
        result = await facts(current, reader, *args, **kwargs)
        if current is manager:
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(controller_recovery, '_facts', held)
    action = asyncio.create_task(controller_recovery.reconcile(runtime, actor, job_id))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.wait_for(other.request_cancel(actor, third.job.job_id,
            system_key='stop-third-recovery-root', reason='test'), 3)
    finally:
        release.set()
        await asyncio.wait_for(action, 10)
    assert (await other.get_job(actor, third.job.job_id)).cancel_requested_at is not None
    assert c.state.counts['BUILD'] == 1 and not c.state.effects


async def test_cancellation_after_preparation_replans_before_publishing_the_original_result(recovery_case, registry_store, monkeypatch):
    c = recovery_case
    runtime, actor = c.recovery_runtime, c.actor
    manager, job_id = runtime.manager, c.state.owner_binding.lease.job_id
    other = JobManager(await registry_store[1](), now_ms=manager.now)
    publish, attempts = controller_recovery._publish_recovery, []

    async def stopped(current, actual_actor, prepared):
        attempts.append(prepared)
        if len(attempts) == 1:
            await other.request_cancel(actor, job_id, system_key='stop-between-recovery-phases', reason='test')
        return await publish(current, actual_actor, prepared)

    monkeypatch.setattr(controller_recovery, '_publish_recovery', stopped)
    await controller_recovery.reconcile(runtime, actor, job_id)
    saved = await manager.get_job(actor, job_id)
    assert len(attempts) == 2 and saved.outcome == 'cancelled'
    assert saved.result['status'] == 'cancelled' and saved.result['execution_status'] == 'interrupted'
    assert saved.resource_state == 'stopped' and c.state.counts['BUILD'] == 1 and not c.state.effects
    assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='result'", (job_id,))) == 1


async def test_two_actual_preparations_publish_one_result_and_one_delivery(recovery_case, monkeypatch):
    c = recovery_case
    runtime, actor = c.recovery_runtime, c.actor
    manager, job_id = runtime.manager, c.state.owner_binding.lease.job_id
    publish, prepared = controller_recovery._publish_recovery, []
    entered = asyncio.Event()

    async def together(current, actual_actor, plan):
        prepared.append(plan)
        if len(prepared) == 2:
            entered.set()
        await asyncio.wait_for(entered.wait(), 5)
        return await publish(current, actual_actor, plan)

    monkeypatch.setattr(controller_recovery, '_publish_recovery', together)
    a, b = await asyncio.gather(*(controller_recovery.reconcile(runtime, actor, job_id) for _ in range(2)))
    assert a == b and len(prepared) == 2
    assert prepared[0].version_sum == prepared[1].version_sum
    events = await manager.db.fetchall(f"SELECT event_type FROM {manager.events} WHERE job_id=? "
        "AND event_type IN ('result','controller_stopped','controller_result_delivery')", (job_id,))
    assert sorted(row['event_type'] for row in events) == ['controller_result_delivery', 'controller_stopped', 'result']
    assert c.state.counts['BUILD'] == 1 and not c.state.effects


async def test_missing_derived_recovery_inbox_rebuilds_without_rewriting_originals(recovery_case, monkeypatch):
    c = recovery_case
    runtime, actor = c.recovery_runtime, c.actor
    manager, job_id = runtime.manager, c.state.owner_binding.lease.job_id
    receipt = await controller_recovery.reconcile(runtime, actor, job_id)
    originals = await manager.db.fetchall(f'SELECT * FROM {manager.events} WHERE job_id=? ORDER BY event_id', (job_id,))
    await manager.db.execute(f'DELETE FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND item_id=?',
        (actor.session_id, receipt['notification_id']))
    async with observe(manager, monkeypatch) as footprints:
        assert await controller_recovery.reconcile(runtime, actor, job_id) == receipt
    assert await manager.db.fetchall(f'SELECT * FROM {manager.events} WHERE job_id=? ORDER BY event_id', (job_id,)) == originals
    queue = await manager.db.fetchall(f'SELECT * FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND item_id=?',
        (actor.session_id, receipt['notification_id']))
    assert len(queue) == 1 and queue[0]['status'] == 'pending'
    assert footprints and all(len(scope) <= 1 for scope in footprints)
    assert c.state.counts['BUILD'] == 1 and not c.state.effects


async def test_original_workflow_handling_is_not_reread_during_commit(registry_store, monkeypatch):
    from power_loop.runtime import processing

    c = await ready(registry_store, 'workflow_node')
    runtime, actor = c.recovery_runtime, c.actor
    manager = runtime.manager
    try:
        node = await manager.get_job(actor, c.state.owner_binding.lease.job_id)
        workflow = await manager.get_job(actor, node.parent_job_id)
        await controller_recovery.reconcile(runtime, actor, node.job_id)
        await c.recovery_loop.send('Handle the original interrupted node without executing work.', actor.session_id, tools=[])
        write, handling = manager._write, processing.require_event_handled
        active, observed = False, []

        @asynccontextmanager
        async def held():
            nonlocal active
            active = True
            try:
                async with write() as tx:
                    yield tx
            finally:
                active = False

        async def outside(current, reader, result, **kwargs):
            if current is manager and result['job_id'] == workflow.job_id:
                observed.append(active)
                assert not active
            return await handling(current, reader, result, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(manager, '_write', held)
            patch.setattr(processing, 'require_event_handled', outside)
            await controller_recovery.reconcile(runtime, actor, workflow.job_id)
        final = await manager.get_job(actor, workflow.job_id)
        assert observed and not any(observed)
        assert final.result == workflow.result and final.outcome == workflow.outcome
        assert final.resource_state == 'stopped' and final.supervision_status == 'closed'
        assert c.state.counts['BUILD'] == 1 and c.state.counts['AFTER'] == 0 and not c.state.effects
    finally:
        await cleanup(c)


@pytest.mark.parametrize('fault', ['rollback', 'lost_ack'])
async def test_native_publication_failure_replays_the_same_result_and_exact_usage(recovery_case, monkeypatch, fault):
    c = recovery_case
    runtime, actor = c.recovery_runtime, c.actor
    manager, job_id = runtime.manager, c.state.owner_binding.lease.job_id
    before = await manager.execution_usage(actor, job_id, generation=1)
    publish, write = controller_recovery._publish_recovery, manager._write
    reached = []

    @asynccontextmanager
    async def failed():
        async with write() as tx:
            yield tx
            assert await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='result'", (job_id,))
            reached.append(job_id)
            if fault == 'rollback':
                raise OSError('original controller publication rolled back')
        if fault == 'lost_ack':
            raise OSError('original controller publication acknowledgement lost')

    async def failing(current, actual_actor, prepared):
        with monkeypatch.context() as patch:
            patch.setattr(current, '_write', failed)
            return await publish(current, actual_actor, prepared)

    with monkeypatch.context() as patch:
        patch.setattr(controller_recovery, '_publish_recovery', failing)
        with pytest.raises(OSError):
            await controller_recovery.reconcile(runtime, actor, job_id)
    assert reached == [job_id]
    originals = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? "
        "AND event_type IN ('result','controller_stopped','controller_result_delivery')", (job_id,))
    assert len(originals) == (3 if fault == 'lost_ack' else 0)
    a, b = await asyncio.gather(*(controller_recovery.reconcile(runtime, actor, job_id) for _ in range(2)))
    assert a == b
    if fault == 'lost_ack':
        assert await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? "
            "AND event_type IN ('result','controller_stopped','controller_result_delivery')", (job_id,)) == originals
    assert (await manager.get_job(actor, job_id)).result['usage'] == before
    assert c.state.counts['BUILD'] == 1 and not c.state.effects


@pytest.mark.parametrize('stage', ['prepared', 'result', 'stopped', 'inbox', 'before_commit', 'after_commit'])
async def test_real_recovery_death_keeps_result_notice_and_inbox_atomic(recovery_case, tmp_path, stage):
    c = recovery_case
    runtime, actor = c.recovery_runtime, c.actor
    manager, job_id = runtime.manager, c.state.owner_binding.lease.job_id
    before = await manager.execution_usage(actor, job_id, generation=1)
    if manager.db.dialect.name == 'sqlite':
        rows = await manager.db.fetchall('PRAGMA database_list')
        dsn = next(row['file'] for row in rows if row['name'] == 'main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN' if manager.db.dialect.name == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    marker = tmp_path/'recovery.ready'
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._controller_recovery_tree_worker',
        dsn, manager.prefix, json.dumps({'actor': asdict(actor), 'job_id': job_id, 'now': manager.now()}),
        stage, str(marker), cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 92, output.decode()
        assert marker.read_text() == stage
        actual = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? "
            "AND event_type IN ('result','controller_stopped','controller_result_delivery')", (job_id,))
        queue = await manager.db.fetchall(f'SELECT * FROM {manager.store.t.follow_up_queue} WHERE session_id=? AND item_id=?',
            (actor.session_id, controller_recovery.notice_id(manager, {'job_id': job_id, 'generation': 1})))
        committed = stage == 'after_commit'
        assert len(actual) == (3 if committed else 0) and len(queue) == int(committed)
        if not committed:
            pending = await manager.get_job(actor, job_id)
            assert pending.result_status == 'none' and pending.resource_state == 'unknown'
        receipt = await controller_recovery.reconcile(runtime, actor, job_id)
        saved = await manager.get_job(actor, job_id)
        assert saved.result['usage'] == before and saved.result_status == 'pending_delivery'
        if committed:
            assert await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE job_id=? "
                "AND event_type IN ('result','controller_stopped','controller_result_delivery')", (job_id,)) == actual
        assert receipt == await controller_recovery.reconcile(runtime, actor, job_id)
        await c.recovery_loop.send('Handle this original interrupted controller result without repeating work.', actor.session_id, tools=[])
        assert (await manager.get_job(actor, job_id)).supervision_status == 'closed'
        assert await manager.execution_usage(actor, job_id, generation=1) == before
        assert c.state.counts['BUILD'] == 1 and c.state.counts['AFTER'] == 0 and not c.state.effects
    finally:
        await stop_worker(process)
