"""An original continuation grant can finish after exit without new dispatch."""
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

from power_loop.runtime import control_requests, session_obligations, subagent_admission
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_job_registry_192 import registry_store
from .test_nudge_runtime_192 import issue_nudge
from .test_subagent_resume_192 import nudge_receipt, ready
from .test_task_runtime_192 import finish, managed

__all__ = ["registry_store", "managed", "ready"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def pending(ready, monkeypatch):
    c, args = ready, {}
    entered = asyncio.Event()
    admit = c.manager.resume_subagent

    async def hold(actor, job_id, **kwargs):
        args.update(actor=actor, job_id=job_id, **kwargs)
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(c.manager, 'resume_subagent', hold)
    source = asyncio.create_task(issue_nudge(c.fixture, monkeypatch))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        yield SimpleNamespace(c=c, args=args, source=source, admit=admit)
    finally:
        source.cancel()
        await asyncio.gather(source, return_exceptions=True)
        await finish(c.fixture)


async def records(c, manager=None):
    manager = manager or c.manager
    rows = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='task_control_request'")
    accepted = [row for row in rows if (lambda data:data['phase']=='accepted' and data['action']==subagent_admission.ACTION)(
        json.loads(row['payload_json']))]
    return accepted, rows


async def test_closed_caller_cannot_create_a_new_continuation_permission(pending):
    p, c = pending, pending.c
    p.source.cancel()
    await asyncio.gather(p.source, return_exceptions=True)
    with pytest.raises(JobError, match='intake_closed|not_running|cancel_requested'):
        await p.admit(**p.args)
    assert not (await records(c))[0]
    assert (await c.manager.get_job(p.args['actor'], c.old.job_id)).generation == 1
    assert c.model.child_calls == 1


async def test_original_route_and_contract_are_verified_only_before_the_tree_writer(pending, monkeypatch):
    p, c = pending, pending.c
    immutable, inspected = subagent_admission._immutable, []

    async def outside(owner, reader, actor, request):
        assert owner._control_intents.get() is None
        inspected.append(request['job_id'])
        return await immutable(owner, reader, actor, request)

    with monkeypatch.context() as patch:
        patch.setattr(subagent_admission, '_immutable', outside)
        operation, repeated = await p.admit(**p.args)
    assert inspected==[c.old.job_id] and not repeated and operation.result['generation']==2
    assert c.model.child_calls==1


async def test_source_exit_after_continuation_grant_keeps_the_original_pending_until_target_ack(pending, monkeypatch):
    p, c = pending, pending.c
    publish = subagent_admission._publish

    async def exit_source(owner, actor, accepted, prepared):
        assert owner._control_intents.get() is None
        p.source.cancel()
        await asyncio.gather(p.source, return_exceptions=True)
        [entry] = await owner.db.fetchall(f"SELECT source_id FROM {owner.prefix}session_obligations WHERE kind='control_request'")
        assert entry['source_id'] == accepted.event['event_id']
        assert not await owner.db.fetchall(f"SELECT event_id FROM {owner.events} WHERE action='nudge'")
        return await publish(owner, actor, accepted, prepared)

    with monkeypatch.context() as patch:
        patch.setattr(subagent_admission, '_publish', exit_source)
        operation, repeated = await p.admit(**p.args)
    assert not repeated and operation.result['generation'] == 2
    assert not await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    [accepted], _ = await records(c)
    assert await subagent_admission.reconcile_request(cold, p.args['actor'], accepted['event_id']) == operation
    assert c.model.child_calls == 1
    assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='dispatch'")
    await session_obligations.verify(cold, p.args['actor'])


@pytest.mark.parametrize('change', ['stop', 'inbox'])
async def test_target_change_after_permission_refuses_without_starting_a_new_generation(pending, monkeypatch, change):
    p, c = pending, pending.c
    publish, stopped = subagent_admission._publish, []

    async def changed(owner, actor, accepted, prepared):
        if change == 'stop':
            await owner.request_cancel(actor, c.old.job_id, system_key='stop-before-resume', reason='test')
        else:
            await c.store.inbox_put(c.sid, [{'item_id':'earlier-input', 'content':'do not skip this original input'}])
        stopped.append(await owner.get_job(actor, c.old.job_id))
        return await publish(owner, actor, accepted, prepared)

    code = 'cancel_requested' if change == 'stop' else 'resume_input_pending'
    with monkeypatch.context() as patch:
        patch.setattr(subagent_admission, '_publish', changed)
        with pytest.raises(JobError, match=code):
            await p.admit(**p.args)
    assert await c.manager.get_job(p.args['actor'], c.old.job_id) == stopped[0]
    assert stopped[0].generation == 1 and c.model.child_calls == 1
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'")
    [accepted], _ = await records(c)
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    with pytest.raises(JobError, match=code):
        await subagent_admission.reconcile_request(cold, p.args['actor'], accepted['event_id'])
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    if change == 'inbox':
        [item] = await c.store.inbox_items(c.sid)
        assert item['item_id'] == 'earlier-input' and item['status'] == 'pending'


async def test_paused_old_result_proof_holds_no_writer_and_allows_a_third_tree_to_stop(pending, monkeypatch):
    p, c = pending, pending.c
    peer = JobManager(await c.connect(), now_ms=c.manager.now)
    third = await peer.register_job(p.args['actor'], kind='send', session_id=c.parent, system_key='third-tree')
    entered, release = asyncio.Event(), asyncio.Event()
    prepare = subagent_admission._prepare

    async def pause(owner, reader, actor, request):
        assert owner._control_intents.get() is None
        entered.set()
        await release.wait()
        return await prepare(owner, reader, actor, request)

    with monkeypatch.context() as patch:
        patch.setattr(subagent_admission, '_prepare', pause)
        action = asyncio.create_task(p.admit(**p.args))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await asyncio.wait_for(peer.request_cancel(p.args['actor'], third.job.job_id,
                system_key='third-stop', reason='test'), 3)
            assert (await peer.get_job(p.args['actor'], third.job.job_id)).cancel_requested_at is not None
        finally:
            release.set()
            await asyncio.wait_for(action, 5)
    assert c.model.child_calls == 1


async def test_a_committed_inbox_input_cannot_hide_behind_an_earlier_admission_snapshot(pending, monkeypatch):
    p, c = pending, pending.c
    peer = await c.connect()
    entered, release, put_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    lookup = ControlTransaction.fetchall
    paused = []

    async def after_lookup(tx, sql, params=()):
        rows = await lookup(tx, sql, params)
        if not paused and sql.startswith('SELECT * FROM '+c.manager.events+' WHERE operation_key=? OR event_id=?'):
            paused.append(True)
            entered.set()
            await release.wait()
        return rows

    async def put_input():
        put_started.set()
        return await peer.inbox_put(c.sid, [{'item_id':'concurrent-input', 'content':'must not be skipped'}])

    with monkeypatch.context() as patch:
        patch.setattr(ControlTransaction, 'fetchall', after_lookup)
        action = asyncio.create_task(p.admit(**p.args))
        put = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            put = asyncio.create_task(put_input())
            await asyncio.wait_for(put_started.wait(), 3)
            try:
                await asyncio.wait_for(asyncio.shield(put), .2)
                before_admission = True
            except TimeoutError:
                before_admission = False
            release.set()
            if before_admission:
                with pytest.raises(JobError, match='resume_input_pending'):
                    await asyncio.wait_for(action, 5)
                assert (await c.manager.get_job(p.args['actor'], c.old.job_id)).generation==1
            else:
                operation, _ = await asyncio.wait_for(action, 5)
                assert operation.result['generation']==2
            await asyncio.wait_for(put, 5)
            items = await c.store.inbox_items(c.sid)
            assert any(item['item_id']=='concurrent-input' and item['status']=='pending' for item in items)
            assert len(items)==(1 if before_admission else 2)
            assert c.model.child_calls==1
        finally:
            release.set()
            await asyncio.gather(action, return_exceptions=True)
            if put is not None:
                await asyncio.wait_for(put, 5)


@pytest.mark.parametrize('phase', ['accepted', 'target', 'completed'])
@pytest.mark.parametrize('committed', [False, True])
async def test_original_native_owner_recovers_a_real_lost_admission_ack_and_dispatches_only_once(ready, monkeypatch, phase, committed):
    from power_loop.runtime.managed_effects import _ref
    from power_loop.runtime.tool_call_plans import identity, start_id

    c = ready
    transaction, hits = c.manager.db.transaction, []

    @asynccontextmanager
    async def lost_ack():
        async with transaction() as tx:
            yield tx
            rows = await tx.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='task_control_request'")
            accepted = [row for row in rows if (lambda data:data['phase']=='accepted' and data['action']==subagent_admission.ACTION)(
                json.loads(row['payload_json']))]
            current = None
            if accepted:
                grant = accepted[0]
                if any(row['event_id']==control_requests.completion_id(grant['event_id']) for row in rows):
                    current = 'completed'
                elif await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE action='nudge'"):
                    current = 'target'
                else:
                    current = 'accepted'
            hit = not hits and current==phase
            if hit:
                request = json.loads(grant['payload_json'])['request']
                source = await c.manager._row(tx, grant['job_id'])
                started = start_id(identity(c.manager, source, _ref(request['call_ref'])), request['tool_call_id'])
                assert await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_id=? "
                    "AND event_type='tool_call_dispatch_started'", (started,))
                hits.append(phase)
                if not committed:
                    raise OSError('admission '+phase+' transaction rolled back')
        if hit:
            raise OSError('admission '+phase+' committed; ACK lost')

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db, 'transaction', lost_ack)
        _, sent, _ = await issue_nudge(c.fixture, monkeypatch)
    assert hits == [phase] and sent.status == 'completed'
    if phase=='accepted' and not committed:
        messages = await c.store.load_all_messages(c.parent)
        assert 'transaction rolled back' in next(row.content for row in reversed(messages) if row.role=='tool')
        assert (await c.manager.get_job(c.runtime.actor(c.parent), c.old.job_id)).generation==1
        assert not (await records(c))[0]
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'")
        assert c.model.child_calls==1
        return
    result = await nudge_receipt(c)
    assert result['generation']==2 and result['status']=='accepted'
    await finish(c.fixture)
    assert c.model.child_calls==2
    assert not await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
    [dispatch] = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'")
    assert dispatch['event_id']=='dispatch_'+result['operation_id']


async def test_pending_continuation_without_a_local_handle_is_recovered_before_constructing_a_child_loop(ready, monkeypatch):
    from power_loop.runtime import delegation, spec

    c = ready
    publish, nudge = subagent_admission._publish, delegation.nudge
    failed, recovered = [], []

    async def unavailable(owner, actor, accepted, prepared):
        if len(failed)<2:
            failed.append(accepted.event['event_id'])
            raise OSError('target publication unavailable')
        return await publish(owner, actor, accepted, prepared)

    async def forbidden(*args, **kwargs):
        pytest.fail('a receipt replay must not reconstruct the original child loop')

    async def retry(*args, **kwargs):
        with pytest.raises(OSError, match='target publication unavailable'):
            await nudge(*args, **kwargs)
        assert (id(c.store), c.sid) not in delegation._RUNNING
        assert (await c.manager.get_job(c.runtime.actor(c.parent), c.old.job_id)).generation==1
        with monkeypatch.context() as patch:
            patch.setattr(spec, 'workflow_continuation_loop', forbidden)
            result = await nudge(*args, **kwargs)
        recovered.append(result)
        return result

    monkeypatch.setattr(subagent_admission, '_publish', unavailable)
    monkeypatch.setattr(delegation, 'nudge', retry)
    _, sent, _ = await issue_nudge(c.fixture, monkeypatch)
    assert sent.status=='completed' and len(failed)==2 and failed[0]==failed[1]
    assert len(recovered)==1 and recovered[0]['replayed'] and recovered[0]['generation']==2
    assert c.model.child_calls==1 and (id(c.store), c.sid) not in delegation._RUNNING
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='dispatch'")
    assert not await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")


@pytest.mark.parametrize('phase', ['accepted', 'target', 'completed'])
@pytest.mark.parametrize('committed', [False, True])
async def test_real_continuation_owner_death_keeps_original_generation_and_never_recreates_a_callback(ready, tmp_path, phase, committed):
    c = ready
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine == 'sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    context = {'actor':asdict(c.runtime.actor(c.parent)), 'job_id':c.old.job_id, 'sid':c.sid,
        'generation':c.old.generation, 'now':c.manager.now(), 'executor_id':'continuation-worker', 'boot_id':'worker-boot'}
    actual, marker, output = tmp_path/'source.json', tmp_path/'owner.ready', tmp_path/'owner.json'
    mode = phase+':'+str(int(committed))
    process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._subagent_admission_worker',
        dsn, c.manager.prefix, json.dumps(context), mode, str(actual), str(marker), str(output),
        cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        assert process.returncode == 79, stdout.decode()
        assert marker.read_text() == mode and not output.exists()
        source = json.loads(actual.read_text())
        accepted, rows = await records(c)
        permitted = phase != 'accepted' or committed
        applied = phase == 'completed' or phase == 'target' and committed
        completed = phase == 'completed' and committed
        assert len(accepted) == int(permitted)
        assert (await c.manager.get_job(c.runtime.actor(c.parent), c.old.job_id)).generation == (2 if applied else 1)
        cold = JobManager(await c.connect(), now_ms=c.manager.now)
        if not permitted:
            with pytest.raises(JobError, match='resume_request_original_missing'):
                await subagent_admission.reconcile_request(cold, c.runtime.actor(c.parent), source['request_id'])
            assert c.model.child_calls == 1
            return
        [accepted] = accepted
        assert sum(row['event_id']==control_requests.completion_id(accepted['event_id']) for row in rows) == int(completed)
        operation = await subagent_admission.reconcile_request(cold, c.runtime.actor(c.parent), accepted['event_id'])
        assert operation.result['generation'] == 2 and operation.result['request']['caller'] == source['caller']
        after = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await subagent_admission.reconcile_request(cold, c.runtime.actor(c.parent), accepted['event_id']) == operation
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id") == after
        assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
        assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='dispatch'")
        [item] = await c.store.inbox_items(c.sid)
        assert item['item_id']=='nudge:'+operation.operation_id and item['status']=='pending'
        assert c.model.child_calls == 1
        await session_obligations.verify(cold, c.runtime.actor(c.parent))
    finally:
        await stop_worker(process)
