"""Whole-workflow admission must prepare foreign proof outside its writer."""
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

from power_loop.runtime import control_requests, session_obligations, workflow_admission, workflow_resume
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
async def pending_admission(admitting, monkeypatch):
    c, args, held = admitting, {}, []
    admit = workflow_resume.admit
    c.runtime.lease_ms = 30000
    entered = asyncio.Event()

    async def pause(manager, actor, job_id, **kwargs):
        args.update(manager=manager, actor=actor, job_id=job_id, **kwargs)
        held.append(asyncio.current_task())
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(workflow_resume, 'admit', pause)
    source = asyncio.create_task(c.admit())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        yield SimpleNamespace(c=c, args=args, source=source, admit=admit)
    finally:
        source.cancel()
        await asyncio.gather(source, return_exceptions=True)
        for task in held:
            task.cancel()
        await asyncio.gather(*held, return_exceptions=True)


async def permissions(c, manager=None):
    manager = manager or c.manager
    rows = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='task_control_request'")
    return [row for row in rows if (lambda data:data['phase']=='accepted' and data['action']==workflow_admission.ACTION)(
        json.loads(row['payload_json']))], rows


async def test_whole_workflow_admission_uses_one_tree_per_writer(admitting, monkeypatch):
    c = admitting
    admit = workflow_resume.admit
    one, many = ControlTransaction.fetchone, ControlTransaction.fetchall
    footprints = []

    def note(tx, rows):
        tx.__dict__.setdefault('_workflow_admission_reads', set()).update(
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
        roots = {row['job_id']:row['root_job_id'] for row in await manager.db.fetchall(f"SELECT job_id,root_job_id FROM {manager.jobs}")}
        write = manager._write

        @asynccontextmanager
        async def scoped():
            async with write() as tx:
                yield tx
                jobs = tx.touched_jobs | tx.__dict__.get('_workflow_admission_reads', set())
                footprints.append({roots[job] for job in jobs})

        with monkeypatch.context() as patch:
            patch.setattr(manager, '_write', scoped)
            patch.setattr(ControlTransaction, 'fetchone', fetchone)
            patch.setattr(ControlTransaction, 'fetchall', fetchall)
            return await admit(manager, *args, **kwargs)

    monkeypatch.setattr(workflow_resume, 'admit', observed)
    await c.admit()
    assert c.result and not c.result[1]
    assert footprints and all(len(roots)<=1 for roots in footprints), footprints


async def test_closed_source_cannot_create_a_fresh_workflow_admission(pending_admission):
    p, c = pending_admission, pending_admission.c
    p.source.cancel()
    await asyncio.gather(p.source, return_exceptions=True)
    with pytest.raises(JobError, match='intake_closed|not_running|cancel_requested'):
        await p.admit(**p.args)
    assert not (await permissions(c))[0]
    assert (await c.manager.get_job(c.actor, c.handle.job_id)).generation==1
    assert c.model.counts['BUILD']==1


async def test_all_original_workflow_and_resource_proof_is_prepared_outside_writers(pending_admission, monkeypatch):
    from power_loop.runtime import workflow_restore_resources

    p, c = pending_admission, pending_admission.c
    originals, resource = workflow_admission._originals, workflow_restore_resources.admission
    inspected, authorized = [], []

    async def outside(manager, reader, actor, client, executor_required):
        assert manager._control_intents.get() is None
        inspected.append(client['target']['job_id'])
        return await originals(manager, reader, actor, client, executor_required)

    def pure(prepared, manager, actor, profile, **kwargs):
        assert manager._control_intents.get() is None
        authorized.append(profile)
        return resource(prepared, manager, actor, profile, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workflow_admission, '_originals', outside)
        patch.setattr(workflow_restore_resources, 'admission', pure)
        operation, repeated = await p.admit(**p.args)
    assert inspected==[c.handle.job_id] and len(authorized)==1
    assert not repeated and operation.result['generation']==2 and c.model.counts['BUILD']==1


async def test_source_exit_after_permission_keeps_pending_until_workflow_admission_is_acknowledged(pending_admission, monkeypatch):
    p, c = pending_admission, pending_admission.c
    publish = workflow_admission._publish

    async def close_source(manager, actor, accepted, prepared):
        assert manager._control_intents.get() is None
        p.source.cancel()
        await asyncio.gather(p.source, return_exceptions=True)
        [pending] = await manager.db.fetchall(f"SELECT source_id FROM {manager.prefix}session_obligations WHERE kind='control_request'")
        assert pending['source_id']==accepted.event['event_id']
        assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='execution' AND job_id=?", (c.handle.job_id,))
        return await publish(manager, actor, accepted, prepared)

    with monkeypatch.context() as patch:
        patch.setattr(workflow_admission, '_publish', close_source)
        operation, repeated = await p.admit(**p.args)
    assert not repeated and operation.result['generation']==2
    cold = JobManager(await c.connect())
    [grant], _ = await permissions(c)
    assert await workflow_admission.reconcile_request(cold, c.actor, grant['event_id'])==operation
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='workflow_dispatch'")
    assert not await c.store.get_session(operation.result['request']['driver_sid'])
    assert c.model.counts['BUILD']==1
    await session_obligations.verify(cold, c.actor)


@pytest.mark.parametrize('change', ['stop', 'inbox'])
async def test_target_changes_after_permission_refuse_without_replacing_the_old_generation(pending_admission, monkeypatch, change):
    p, c = pending_admission, pending_admission.c
    publish, observed = workflow_admission._publish, []
    sid = c.plan['nodes']['BUILD']['binding']['session_id']

    async def changed(manager, actor, accepted, prepared):
        if change=='stop':
            await manager.request_cancel(actor, c.handle.job_id, system_key='stop-before-admission', reason='test')
        else:
            await c.store.inbox_put(sid, [{'item_id':'earlier-input', 'content':'preserve this actual input'}])
        observed.append(await manager.get_job(actor, c.handle.job_id))
        return await publish(manager, actor, accepted, prepared)

    code = 'cancel_requested' if change=='stop' else 'resume_input_pending'
    with monkeypatch.context() as patch:
        patch.setattr(workflow_admission, '_publish', changed)
        with pytest.raises(JobError, match=code):
            await p.admit(**p.args)
    assert await c.manager.get_job(c.actor, c.handle.job_id)==observed[0]
    assert observed[0].generation==1 and c.model.counts['BUILD']==1
    [grant], _ = await permissions(c)
    cold = JobManager(await c.connect())
    with pytest.raises(JobError, match=code):
        await workflow_admission.reconcile_request(cold, c.actor, grant['event_id'])
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='workflow_node_reservation'")
    if change=='inbox':
        [item] = await c.store.inbox_items(sid)
        assert item['item_id']=='earlier-input' and item['status']=='pending'


async def test_paused_workflow_proof_allows_an_unrelated_root_to_stop(pending_admission, monkeypatch):
    p, c = pending_admission, pending_admission.c
    peer = JobManager(await c.connect())
    third = await peer.register_job(c.actor, kind='send', session_id=c.parent, system_key='third-root')
    entered, release = asyncio.Event(), asyncio.Event()
    originals = workflow_admission._originals

    async def pause(manager, reader, actor, client, executor_required):
        assert manager._control_intents.get() is None
        entered.set()
        await release.wait()
        return await originals(manager, reader, actor, client, executor_required)

    with monkeypatch.context() as patch:
        patch.setattr(workflow_admission, '_originals', pause)
        action = asyncio.create_task(p.admit(**p.args))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await asyncio.wait_for(peer.request_cancel(c.actor, third.job.job_id, system_key='third-stop', reason='test'), 3)
            assert (await peer.get_job(c.actor, third.job.job_id)).cancel_requested_at is not None
        finally:
            release.set()
            await asyncio.wait_for(action, 5)
    assert c.model.counts['BUILD']==1


@pytest.mark.parametrize('phase', ['accepted', 'target', 'completed'])
@pytest.mark.parametrize('committed', [False, True])
async def test_real_admission_owner_death_never_reconstructs_a_workflow_factory(admitting, tmp_path, phase, committed):
    c = admitting
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine=='sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    context = {'actor':asdict(c.actor), 'job_id':c.handle.job_id, 'now':c.manager.now(),
        'executor_id':c.runtime.executor_id, 'boot_id':c.runtime.boot_id}
    actual, marker, output = tmp_path/'source.json', tmp_path/'owner.ready', tmp_path/'owner.json'
    mode = phase+':'+str(int(committed))
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._workflow_admission_worker',
        dsn, c.manager.prefix, json.dumps(context), mode, str(actual), str(marker), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode==81, stdout.decode()
        assert marker.read_text()==mode and not output.exists()
        source = json.loads(actual.read_text())
        accepted, rows = await permissions(c)
        permitted = phase!='accepted' or committed
        applied = phase=='completed' or phase=='target' and committed
        completed = phase=='completed' and committed
        assert len(accepted)==int(permitted)
        assert (await c.manager.get_job(c.actor, c.handle.job_id)).generation==(2 if applied else 1)
        cold = JobManager(await c.connect())
        if not permitted:
            with pytest.raises(JobError, match='restore_request_original_missing'):
                await workflow_admission.reconcile_request(cold, c.actor, source['request_id'])
            assert c.model.counts['BUILD']==1
            return
        [grant] = accepted
        assert sum(row['event_id']==control_requests.completion_id(grant['event_id']) for row in rows)==int(completed)
        operation = await workflow_admission.reconcile_request(cold, c.actor, grant['event_id'])
        assert operation.result['generation']==2 and operation.result['request']['caller']==source['caller']
        after = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await workflow_admission.reconcile_request(cold, c.actor, grant['event_id'])==operation
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")==after
        assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
        assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='workflow_dispatch'")
        assert not await c.store.get_session(operation.result['request']['driver_sid'])
        assert c.model.counts['BUILD']==1
        assert len(await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='workflow_node_reservation'"))==2
        await session_obligations.verify(cold, c.actor)
    finally:
        await stop_worker(process)


@pytest.mark.parametrize('phase', ['accepted', 'target', 'completed'])
@pytest.mark.parametrize('committed', [False, True])
async def test_original_native_owner_recovers_admission_ack_and_runs_its_workflow_only_once(native_restore, monkeypatch, phase, committed):
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
            accepted = [row for row in rows if (lambda data:data['phase']=='accepted' and data['action']==workflow_admission.ACTION)(json.loads(row['payload_json']))]
            current = None
            if accepted:
                grant = accepted[0]
                if any(row['event_id']==control_requests.completion_id(grant['event_id']) for row in rows):
                    current = 'completed'
                elif await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_type='execution' AND job_id=?", (c.initial.job_id,)):
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
                    raise OSError('admission '+phase+' rolled back')
        if hit:
            raise OSError('admission '+phase+' committed; ACK lost')

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
        assert (await c.manager.get_job(c.actor, c.initial.job_id)).generation==1
        assert c.model.counts['BUILD']==c.model.counts['REVIEW']==1
        return
    [handle] = handles
    await asyncio.wait_for(handle.task, 10)
    assert c.model.counts['BUILD']==c.model.counts['REVIEW']==2
    assert (await c.manager.get_job(c.actor, c.initial.job_id)).outcome=='succeeded'
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_dispatch'"))==1
    assert not await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")


async def test_committed_input_cannot_hide_behind_an_earlier_workflow_admission_snapshot(pending_admission, monkeypatch):
    p, c = pending_admission, pending_admission.c
    peer = await c.connect()
    sid = c.plan['nodes']['BUILD']['binding']['session_id']
    entered, release, started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    lookup, paused = ControlTransaction.fetchall, []

    async def after_lookup(tx, sql, params=()):
        rows = await lookup(tx, sql, params)
        if not paused and sql.startswith('SELECT * FROM '+c.manager.events+' WHERE operation_key=? OR event_id=?'):
            paused.append(True)
            entered.set()
            await release.wait()
        return rows

    async def put_input():
        started.set()
        return await peer.inbox_put(sid, [{'item_id':'concurrent-input', 'content':'preserve this original input'}])

    with monkeypatch.context() as patch:
        patch.setattr(ControlTransaction, 'fetchall', after_lookup)
        action = asyncio.create_task(p.admit(**p.args))
        put = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            put = asyncio.create_task(put_input())
            await asyncio.wait_for(started.wait(), 3)
            try:
                await asyncio.wait_for(asyncio.shield(put), .2)
                before_admission = True
            except TimeoutError:
                before_admission = False
            release.set()
            if before_admission:
                with pytest.raises(JobError, match='resume_input_pending'):
                    await asyncio.wait_for(action, 5)
                assert (await c.manager.get_job(c.actor, c.handle.job_id)).generation==1
            else:
                operation, _ = await asyncio.wait_for(action, 5)
                assert operation.result['generation']==2
            await asyncio.wait_for(put, 5)
            [item] = await c.store.inbox_items(sid)
            assert item['item_id']=='concurrent-input' and item['status']=='pending'
            assert c.model.counts['BUILD']==c.model.counts['REVIEW']==1
        finally:
            release.set()
            await asyncio.gather(action, return_exceptions=True)
            if put is not None:
                await asyncio.wait_for(put, 5)


async def test_cold_admission_rejects_a_changed_source_permission_without_repair(pending_admission, monkeypatch):
    p, c = pending_admission, pending_admission.c

    async def unavailable(*args, **kwargs):
        raise OSError('target unavailable after permission')

    with monkeypatch.context() as patch:
        patch.setattr(workflow_admission, '_publish', unavailable)
        with pytest.raises(OSError, match='target unavailable'):
            await p.admit(**p.args)
    [grant], _ = await permissions(c)
    value = json.loads(grant['payload_json'])
    value['request']['input']['previous_generation'] += 1
    damaged = json.dumps(value)
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (damaged, grant['event_id']))
    try:
        cold = JobManager(await c.connect())
        with pytest.raises(JobError, match='restore_request_corrupt'):
            await workflow_admission.reconcile_request(cold, c.actor, grant['event_id'])
        assert (await cold.db.fetchone(f"SELECT payload_json FROM {cold.events} WHERE event_id=?", (grant['event_id'],)))['payload_json']==damaged
        assert (await cold.get_job(c.actor, c.handle.job_id)).generation==1
        assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type IN ('execution','workflow_dispatch')")
        assert c.model.counts['BUILD']==c.model.counts['REVIEW']==1
    finally:
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (grant['payload_json'], grant['event_id']))


async def test_cold_admission_keeps_source_pending_when_the_target_execution_original_is_missing(pending_admission, monkeypatch):
    from power_loop.runtime.result_revisions import execution_id

    p, c = pending_admission, pending_admission.c

    async def unavailable(*args, **kwargs):
        raise OSError('source completion unavailable')

    with monkeypatch.context() as patch:
        patch.setattr(control_requests, 'complete', unavailable)
        with pytest.raises(OSError, match='source completion unavailable'):
            await p.admit(**p.args)
    [grant], _ = await permissions(c)
    key = execution_id(c.handle.job_id, 2)
    original = await c.manager.db.fetchone(f"SELECT * FROM {c.manager.events} WHERE event_id=?", (key,))
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (key,))
    cold = JobManager(await c.connect())
    try:
        with pytest.raises(JobError, match='workflow_restore_execution_mismatch'):
            await workflow_admission.reconcile_request(cold, c.actor, grant['event_id'])
        assert not await cold.db.fetchone(f"SELECT event_id FROM {cold.events} WHERE event_id=?", (key,))
        assert not await cold.db.fetchone(f"SELECT event_id FROM {cold.events} WHERE event_id=?", (control_requests.completion_id(grant['event_id']),))
        [pending] = await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
        assert pending['source_id']==grant['event_id']
        assert c.model.counts['BUILD']==c.model.counts['REVIEW']==1
    finally:
        await c.manager.db.execute(f"INSERT INTO {c.manager.events} ("+','.join(original)+') VALUES ('+','.join('?' for _ in original)+')', tuple(original.values()))
    operation = await workflow_admission.reconcile_request(cold, c.actor, grant['event_id'])
    assert operation.result['generation']==2
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")


async def test_pending_admission_without_a_local_handle_is_recovered_before_any_resource_factory(native_restore, monkeypatch):
    from power_loop.runtime import workflow_restore_resources
    from power_loop.workflow import runner
    from power_loop.workflow.resume import resume_detached

    c, failed, recovered = native_restore, [], []
    publish = workflow_admission._publish

    async def unavailable(manager, actor, accepted, prepared):
        if len(failed)<2:
            failed.append(accepted.event['event_id'])
            raise OSError('target publication unavailable')
        return await publish(manager, actor, accepted, prepared)

    async def forbidden(*args, **kwargs):
        pytest.fail('a receipt replay must not reconstruct workflow resources')

    async def retry():
        with pytest.raises(OSError, match='target publication unavailable'):
            await resume_detached(c.loop, c.parent, c.initial.run_id, continuation=('BUILD', 'NUDGED-43', 3))
        assert (await c.manager.get_job(c.actor, c.initial.job_id)).generation==1
        assert runner.get_run_handle(c.initial.run_id) is None
        with monkeypatch.context() as patch:
            patch.setattr(workflow_restore_resources, 'prepare', forbidden)
            with pytest.raises(JobError, match='workflow_admission_requires_reconciliation'):
                await resume_detached(c.loop, c.parent, c.initial.run_id, continuation=('BUILD', 'NUDGED-43', 3))
        [operation] = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")
        recovered.append(operation['event_id'])
        return {'operation_id':operation['event_id'], 'status':'requires_reconciliation'}

    monkeypatch.setattr(workflow_admission, '_publish', unavailable)
    await c.issue(handler=retry)
    assert len(failed)==2 and failed[0]==failed[1] and len(recovered)==1
    assert (await c.manager.get_job(c.actor, c.initial.job_id)).generation==2
    assert c.model.counts['BUILD']==c.model.counts['REVIEW']==1
    assert recovered[0] not in c.runtime._workflow_handles
    assert runner.get_run_handle(c.initial.run_id) is None
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='workflow_dispatch'")
    assert not await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
