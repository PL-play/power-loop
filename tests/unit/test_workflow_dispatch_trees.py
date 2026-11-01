"""Whole-workflow dispatch must not hold its caller's and target's writers."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from power_loop.runtime import control_requests, resume_dispatch, session_obligations, workflow_resume
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_job_registry_192 import registry_store
from .test_workflow_restore_runtime_192 import native_restore
from .test_workflow_resume_admission_192 import admitting
from .test_workflow_task_runtime_192 import managed_workflow

__all__ = ['registry_store', 'managed_workflow', 'admitting', 'native_restore']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def pending_dispatch(admitting, monkeypatch):
    c, args, held = admitting, {}, []
    claim = workflow_resume.claim_dispatch
    entered = asyncio.Event()
    c.runtime.lease_ms = 30000

    async def pause(manager, actor, operation_id, **kwargs):
        args.update(manager=manager, actor=actor, operation_id=operation_id, **kwargs)
        held.append(asyncio.current_task())
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(workflow_resume, 'claim_dispatch', pause)
    source = asyncio.create_task(c.admit(claim=True))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        yield SimpleNamespace(c=c, args=args, source=source, claim=claim)
    finally:
        source.cancel()
        await asyncio.gather(source, return_exceptions=True)
        for task in held:
            task.cancel()
        await asyncio.gather(*held, return_exceptions=True)


async def permissions(c, manager=None):
    manager = manager or c.manager
    rows = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='task_control_request'")
    return [row for row in rows if (lambda data:data['phase']=='accepted' and data['action']=='dispatch'
        and data['request']['mode']=='workflow_continue')(json.loads(row['payload_json']))], rows


async def test_original_workflow_dispatch_uses_one_tree_per_writer(admitting, monkeypatch):
    c = admitting
    claim = workflow_resume.claim_dispatch
    one, many = ControlTransaction.fetchone, ControlTransaction.fetchall
    footprints = []

    def note(tx, rows):
        tx.__dict__.setdefault('_workflow_dispatch_reads', set()).update(
            row['job_id'] for row in rows if row is not None and 'job_id' in row)

    async def fetchone(tx, sql, params=()):
        row = await one(tx, sql, params)
        note(tx, [row])
        return row

    async def fetchall(tx, sql, params=()):
        rows = await many(tx, sql, params)
        note(tx, rows)
        return rows

    async def observed(manager, *args, **kwargs):
        roots = {row['job_id']:row['root_job_id'] for row in await manager.db.fetchall(
            f"SELECT job_id,root_job_id FROM {manager.jobs}")}
        write = manager._write

        @asynccontextmanager
        async def scoped():
            async with write() as tx:
                yield tx
                jobs = tx.touched_jobs | tx.__dict__.get('_workflow_dispatch_reads', set())
                footprints.append({roots[job] for job in jobs})

        with monkeypatch.context() as patch:
            patch.setattr(manager, '_write', scoped)
            patch.setattr(ControlTransaction, 'fetchone', fetchone)
            patch.setattr(ControlTransaction, 'fetchall', fetchall)
            return await claim(manager, *args, **kwargs)

    monkeypatch.setattr(workflow_resume, 'claim_dispatch', observed)
    await c.admit(claim=True)
    assert c.claimed
    assert footprints and all(len(roots)<=1 for roots in footprints), footprints


async def test_closed_source_cannot_grant_a_fresh_workflow_dispatch(pending_dispatch):
    p, c = pending_dispatch, pending_dispatch.c
    p.source.cancel()
    await asyncio.gather(p.source, return_exceptions=True)
    with pytest.raises(JobError, match='intake_closed|not_running|cancel_requested'):
        await p.claim(**p.args)
    assert not (await permissions(c))[0]
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_dispatch'")
    assert c.model.counts['BUILD']==1


async def test_retained_dispatch_permission_can_finish_after_its_source_exits(pending_dispatch, monkeypatch):
    p, c = pending_dispatch, pending_dispatch.c
    publish = resume_dispatch._publish

    async def exit_source(manager, actor, accepted, outcome):
        assert manager._control_intents.get() is None
        p.source.cancel()
        await asyncio.gather(p.source, return_exceptions=True)
        [pending] = await manager.db.fetchall(f"SELECT source_id FROM {manager.prefix}session_obligations WHERE kind='control_request'")
        assert pending['source_id']==accepted.event['event_id']
        assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='workflow_dispatch'")
        return await publish(manager, actor, accepted, outcome)

    with monkeypatch.context() as patch:
        patch.setattr(resume_dispatch, '_publish', exit_source)
        assert await p.claim(**p.args)
    cold = JobManager(await c.connect())
    [grant], _ = await permissions(c)
    assert await resume_dispatch.reconcile_request(cold, c.actor, grant['event_id'])
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    assert len(await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='workflow_dispatch'"))==1
    assert c.model.counts['BUILD']==1
    assert not await c.store.get_session(c.result[0].result['request']['driver_sid'])
    await session_obligations.verify(cold, c.actor)


async def test_stop_after_dispatch_permission_refuses_without_constructing_a_workflow(pending_dispatch, monkeypatch):
    p, c = pending_dispatch, pending_dispatch.c
    publish, observed = resume_dispatch._publish, []

    async def stopped(manager, actor, accepted, outcome):
        await manager.request_cancel(actor, c.handle.job_id, system_key='stop-before-dispatch', reason='test')
        observed.append(await manager.get_job(actor, c.handle.job_id))
        return await publish(manager, actor, accepted, outcome)

    with monkeypatch.context() as patch:
        patch.setattr(resume_dispatch, '_publish', stopped)
        with pytest.raises(JobError, match='cancel_requested'):
            await p.claim(**p.args)
    assert await c.manager.get_job(c.actor, c.handle.job_id)==observed[0]
    [grant], _ = await permissions(c)
    cold = JobManager(await c.connect())
    with pytest.raises(JobError, match='cancel_requested'):
        await resume_dispatch.reconcile_request(cold, c.actor, grant['event_id'])
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='workflow_dispatch'")
    assert c.model.counts['BUILD']==1


async def test_cold_dispatch_rejects_a_changed_source_permission_without_repair(pending_dispatch, monkeypatch):
    p, c = pending_dispatch, pending_dispatch.c

    async def unavailable(*args, **kwargs):
        raise OSError('target unavailable after source permission')

    with monkeypatch.context() as patch:
        patch.setattr(resume_dispatch, '_publish', unavailable)
        with pytest.raises(OSError, match='target unavailable'):
            await p.claim(**p.args)
    [grant], _ = await permissions(c)
    value = json.loads(grant['payload_json'])
    value['request']['driver_sid'] = 'another-driver'
    damaged = json.dumps(value)
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (damaged, grant['event_id']))
    try:
        cold = JobManager(await c.connect())
        with pytest.raises(JobError, match='resume_dispatch_request_corrupt'):
            await resume_dispatch.reconcile_request(cold, c.actor, grant['event_id'])
        assert (await cold.db.fetchone(f"SELECT payload_json FROM {cold.events} WHERE event_id=?", (grant['event_id'],)))['payload_json']==damaged
        assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='workflow_dispatch'")
        assert c.model.counts['BUILD']==1
    finally:
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (grant['payload_json'], grant['event_id']))


async def test_resource_admission_is_checked_before_any_tree_writer(pending_dispatch, monkeypatch):
    from power_loop.runtime import workflow_restore_resources

    p, c = pending_dispatch, pending_dispatch.c
    admission, inspected = workflow_restore_resources.admission, []

    def outside(resources, manager, actor, profile, **kwargs):
        assert manager._control_intents.get() is None
        inspected.append(profile)
        return admission(resources, manager, actor, profile, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workflow_restore_resources, 'admission', outside)
        assert await p.claim(**p.args)
    assert len(inspected)==1 and c.model.counts['BUILD']==1


async def test_paused_original_workflow_proof_allows_an_unrelated_tree_to_stop(pending_dispatch, monkeypatch):
    p, c = pending_dispatch, pending_dispatch.c
    peer = JobManager(await c.connect())
    third = await peer.register_job(c.actor, kind='send', session_id=c.parent, system_key='third-root')
    entered, release = asyncio.Event(), asyncio.Event()
    execution, paused = workflow_resume.execution, []

    async def pause(manager, reader, row, *args, **kwargs):
        if not paused and row['job_id']==c.handle.job_id:
            assert manager._control_intents.get() is None
            paused.append(row['job_id'])
            entered.set()
            await release.wait()
        return await execution(manager, reader, row, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workflow_resume, 'execution', pause)
        action = asyncio.create_task(p.claim(**p.args))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await asyncio.wait_for(peer.request_cancel(c.actor, third.job.job_id, system_key='third-stop', reason='test'), 3)
            assert (await peer.get_job(c.actor, third.job.job_id)).cancel_requested_at is not None
        finally:
            release.set()
            assert await asyncio.wait_for(action, 5)
    assert c.model.counts['BUILD']==1


@pytest.mark.parametrize('phase', ['accepted', 'target', 'completed'])
@pytest.mark.parametrize('committed', [False, True])
async def test_real_dispatch_owner_death_is_repaired_without_a_workflow_factory(admitting, tmp_path, phase, committed):
    c = admitting
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine=='sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    context = {'actor':asdict(c.actor), 'job_id':c.handle.job_id, 'now':c.manager.now(),
        'executor_id':c.runtime.executor_id, 'boot_id':c.runtime.boot_id}
    actual, marker, output = tmp_path/'source.json', tmp_path/'owner.ready', tmp_path/'owner.json'
    mode = phase+':'+str(int(committed))
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._workflow_dispatch_worker',
        dsn, c.manager.prefix, json.dumps(context), mode, str(actual), str(marker), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode==80, stdout.decode()
        assert marker.read_text()==mode and not output.exists()
        source = json.loads(actual.read_text())
        accepted, rows = await permissions(c)
        permitted = phase!='accepted' or committed
        applied = phase=='completed' or phase=='target' and committed
        completed = phase=='completed' and committed
        assert len(accepted)==int(permitted)
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_dispatch'"))==int(applied)
        cold = JobManager(await c.connect())
        if not permitted:
            with pytest.raises(JobError, match='resume_dispatch_request_original_missing'):
                await resume_dispatch.reconcile_request(cold, c.actor, source['request_id'])
            assert c.model.counts['BUILD']==1
            return
        [grant] = accepted
        assert sum(row['event_id']==control_requests.completion_id(grant['event_id']) for row in rows)==int(completed)
        assert await resume_dispatch.reconcile_request(cold, c.actor, grant['event_id'])
        events = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await resume_dispatch.reconcile_request(cold, c.actor, grant['event_id'])
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")==events
        assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
        [dispatch] = await cold.db.fetchall(f"SELECT * FROM {cold.events} WHERE event_type='workflow_dispatch'")
        assert json.loads(dispatch['payload_json'])['token']=='workflow-owner'
        assert (await cold.get_job(c.actor, c.handle.job_id)).generation==2
        assert not await c.store.get_session(source['driver_sid'])
        assert c.model.counts['BUILD']==1
        await session_obligations.verify(cold, c.actor)
    finally:
        await stop_worker(process)


@pytest.mark.parametrize('phase', ['accepted', 'target', 'completed'])
@pytest.mark.parametrize('committed', [False, True])
async def test_native_workflow_owner_recovers_dispatch_ack_without_duplicate_requests(native_restore, monkeypatch, phase, committed):
    from power_loop.runtime.managed_effects import _ref
    from power_loop.runtime.tool_call_plans import identity, start_id
    from power_loop.workflow.resume import resume_detached

    c, hits, handles = native_restore, [], []
    transaction = c.manager.db.transaction

    @asynccontextmanager
    async def boundary():
        async with transaction() as tx:
            yield tx
            rows = await tx.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='task_control_request'")
            accepted = [row for row in rows if (lambda data:data['phase']=='accepted' and data['action']=='dispatch'
                and data['request']['mode']=='workflow_continue')(json.loads(row['payload_json']))]
            current = None
            if accepted:
                grant = accepted[0]
                if any(row['event_id']==control_requests.completion_id(grant['event_id']) for row in rows):
                    current = 'completed'
                elif await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_dispatch'"):
                    current = 'target'
                else:
                    current = 'accepted'
            hit = not hits and current==phase
            if hit:
                request = json.loads(grant['payload_json'])['request']
                source = await c.manager._row(tx, grant['job_id'])
                started = start_id(identity(c.manager, source, _ref(request['call_ref'])), request['tool_call_id'])
                assert await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_id=? AND event_type='tool_call_dispatch_started'", (started,))
                hits.append(phase)
                if not committed:
                    raise OSError('dispatch '+phase+' rolled back')
        if hit:
            raise OSError('dispatch '+phase+' committed; ACK lost')

    async def restore():
        handle = await resume_detached(c.loop, c.parent, c.initial.run_id, continuation=('BUILD', 'NUDGED-43', 3))
        handles.append(handle)
        c.state['handles'].append(handle)
        return {'operation_id':handle.operation_id}

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, 'transaction', boundary)
        await c.issue(handler=restore)
    assert hits==[phase]
    if phase=='accepted' and not committed:
        assert not handles and not (await permissions(c))[0]
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_dispatch'")
        assert c.model.counts['BUILD']==c.model.counts['REVIEW']==1
        return
    [handle] = handles
    await asyncio.wait_for(handle.task, 10)
    assert c.model.counts['BUILD']==c.model.counts['REVIEW']==2
    assert (await c.manager.get_job(c.actor, c.initial.job_id)).outcome=='succeeded'
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_dispatch'"))==1
    assert not await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
