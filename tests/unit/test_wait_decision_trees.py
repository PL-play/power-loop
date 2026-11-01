"""A main-model decision must not share a writer with an older wait tree."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from power_loop.runtime import (
    control_requests,
    wait_decision_admission,
    wait_decisions,
    wait_effects,
    wait_notifications,
)
from power_loop.runtime.execution_state import CheckpointPolicy
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_wait_effects_192 import decide
from .test_wait_handoff_192 import handoff, registry_store, runtime, waiting

__all__ = ['handoff', 'registry_store', 'runtime', 'waiting']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_wait_decision_uses_one_tree_per_writer(handoff, monkeypatch):
    h = handoff
    record = wait_decisions.record_decision
    one, many = ControlTransaction.fetchone, ControlTransaction.fetchall
    footprints = []

    def note(tx, rows):
        tx.__dict__.setdefault('_wait_decision_reads', set()).update(
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
                jobs = tx.touched_jobs | tx.__dict__.get('_wait_decision_reads', set())
                footprints.append({roots[job] for job in jobs})

        with monkeypatch.context() as patch:
            patch.setattr(manager, '_write', scoped)
            patch.setattr(ControlTransaction, 'fetchone', fetchone)
            patch.setattr(ControlTransaction, 'fetchall', fetchall)
            return await record(manager, *args, **kwargs)

    monkeypatch.setattr(wait_decisions, 'record_decision', observed)
    await decide(h)
    assert footprints and all(len(roots)<=1 for roots in footprints), footprints


async def grants(c, manager=None):
    owner = manager or c.manager
    rows = await owner.db.fetchall(f"SELECT * FROM {owner.events} WHERE event_type='task_control_request'")
    return [row for row in rows if (lambda data:data['phase']=='accepted' and data['action']==wait_decision_admission.ACTION)(json.loads(row['payload_json']))]


@pytest.fixture
async def pending_decision(handoff, monkeypatch):
    from power_loop.runtime.control_runtime import flush_binding
    from power_loop.runtime.task_runtime import _SEND

    h, entered, args = handoff, asyncio.Event(), {}
    record = wait_decisions.record_decision

    async def pause(manager, actor, **kwargs):
        args.update(manager=manager, actor=actor, **kwargs)
        await flush_binding(_SEND.get())
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(wait_decisions, 'record_decision', pause)
    source = asyncio.create_task(decide(h))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        yield SimpleNamespace(h=h,c=h.c,args=args,source=source,record=record)
    finally:
        source.cancel()
        await asyncio.gather(source, return_exceptions=True)


async def test_an_exited_main_cannot_grant_a_fresh_wait_decision(pending_decision):
    p, c = pending_decision, pending_decision.c
    p.source.cancel()
    await asyncio.gather(p.source, return_exceptions=True)
    with pytest.raises(JobError, match='stop_caller_closed|intake_closed|not_running|cancel_requested'):
        await p.record(**p.args)
    assert not await grants(c)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'")
    assert p.h.model.calls==1


async def test_retained_decision_permission_survives_main_exit_without_executing_an_effect(pending_decision, monkeypatch):
    p, c = pending_decision, pending_decision.c

    async def unavailable(*args, **kwargs):
        raise OSError('target unavailable after source permission')

    with monkeypatch.context() as patch:
        patch.setattr(wait_decision_admission, '_publish', unavailable)
        with pytest.raises(OSError, match='target unavailable'):
            await p.record(**p.args)
    [grant] = await grants(c)
    [pending] = await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
    assert pending['source_id']==grant['event_id']
    p.source.cancel()
    await asyncio.gather(p.source, return_exceptions=True)
    cold = JobManager(await c.connect(), now_ms=c.manager.now)
    operation = await wait_decision_admission.reconcile_request(cold, c.main, grant['event_id'])
    rows = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
    assert await wait_decision_admission.reconcile_request(cold, c.main, grant['event_id'])==operation
    assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")==rows
    assert operation.result['response']['status']=='accepted' and operation.receipt_ref is None
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    with pytest.raises(JobError, match='wait_decision_receipt_required'):
        await wait_effects.apply(cold, c.main, p.h.event_id)
    assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='wait_effect'")
    assert p.h.model.calls==1


async def test_paused_wait_originals_allow_an_unrelated_root_to_stop(pending_decision, monkeypatch):
    p, c = pending_decision, pending_decision.c
    other = JobManager(await c.connect(), now_ms=c.manager.now)
    third = await other.register_job(c.main,kind='send',session_id=c.main.session_id,system_key='third-root')
    prepare = wait_decision_admission._prepare
    entered, release = asyncio.Event(), asyncio.Event()

    async def pause(manager, reader, actor, request, **kwargs):
        assert manager._control_intents.get() is None
        entered.set()
        await release.wait()
        return await prepare(manager,reader,actor,request,**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wait_decision_admission,'_prepare',pause)
        action = asyncio.create_task(p.record(**p.args))
        try:
            await asyncio.wait_for(entered.wait(),3)
            await asyncio.wait_for(other.request_cancel(c.main,third.job.job_id,system_key='third-stop',reason='test'),3)
            assert (await other.get_job(c.main,third.job.job_id)).cancel_requested_at is not None
        finally:
            release.set()
            await asyncio.wait_for(action,5)
    assert p.h.model.calls==1


@pytest.mark.parametrize('change',['cancel','deadline'])
async def test_target_change_after_model_permission_refuses_without_consuming_the_due(pending_decision,monkeypatch,change):
    p,c = pending_decision,pending_decision.c
    publish,observed = wait_decision_admission._publish,[]

    async def changed(manager,actor,accepted,prepared):
        if change=='cancel':
            await manager.request_cancel(c.main,p.h.registration.job.job_id,system_key='cancel-before-decision',reason='test')
        else:
            c.now[0] = c.policy.deadline_at_ms
        observed.append(await manager.get_job(c.main,p.h.registration.job.job_id))
        return await publish(manager,actor,accepted,prepared)

    code = 'cancel_requested|wait_state_changed' if change=='cancel' else 'wait_deadline_exceeded'
    with monkeypatch.context() as patch:
        patch.setattr(wait_decision_admission,'_publish',changed)
        with pytest.raises(JobError,match=code):
            await p.record(**p.args)
    assert await c.manager.get_job(c.main,p.h.registration.job.job_id)==observed[0]
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'")
    [due] = await c.manager.db.fetchall(f"SELECT delivery_status FROM {c.manager.events} WHERE event_id=?",(p.h.event_id,))
    assert due['delivery_status']=='delivered'
    [grant] = await grants(c)
    cold = JobManager(await c.connect(),now_ms=c.manager.now)
    with pytest.raises(JobError,match=code):
        await wait_decision_admission.reconcile_request(cold,c.main,grant['event_id'])
    assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='wait_effect'")
    assert p.h.model.calls==1


async def test_cold_wait_decision_refuses_a_changed_permission_without_repair(pending_decision,monkeypatch):
    p,c = pending_decision,pending_decision.c

    async def unavailable(*args,**kwargs):
        raise OSError('target unavailable after permission')

    with monkeypatch.context() as patch:
        patch.setattr(wait_decision_admission,'_publish',unavailable)
        with pytest.raises(OSError,match='target unavailable'):
            await p.record(**p.args)
    [grant] = await grants(c)
    value = json.loads(grant['payload_json'])
    value['request']['proof']['continuation_hash'] = '0'*64
    damaged = json.dumps(value)
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",(damaged,grant['event_id']))
    try:
        cold = JobManager(await c.connect(),now_ms=c.manager.now)
        with pytest.raises(JobError,match='decide_request_corrupt'):
            await wait_decision_admission.reconcile_request(cold,c.main,grant['event_id'])
        assert (await cold.db.fetchone(f"SELECT payload_json FROM {cold.events} WHERE event_id=?",(grant['event_id'],)))['payload_json']==damaged
        assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='wait_decision'")
        assert p.h.model.calls==1
    finally:
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",(grant['payload_json'],grant['event_id']))


async def test_two_managers_publish_one_original_decision_from_the_same_model_permission(pending_decision):
    p,c = pending_decision,pending_decision.c
    other = JobManager(await c.connect(),now_ms=c.manager.now)
    args = {k:v for k,v in p.args.items() if k!='manager'}
    first,second = await asyncio.gather(p.record(manager=c.manager,**args),p.record(manager=other,**args))
    assert first[0]==second[0] and sorted([first[1],second[1]])==[False,True]
    assert len(await grants(c))==1
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'"))==1
    assert not await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
    assert p.h.model.calls==1


@pytest.mark.parametrize('phase',['accepted','target','completed'])
@pytest.mark.parametrize('committed',[False,True])
async def test_real_process_death_between_decision_trees_never_repeats_a_model_or_external_action(handoff,tmp_path,phase,committed):
    h, c = handoff, handoff.c
    await wait_notifications.enqueue_notification(h.loop,c.main,h.event_id)
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine=='sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    config = {'now':c.now[0],'tenant':c.main.tenant_id,'principal':c.main.principal_id,'sid':c.main.session_id,'event_id':h.event_id}
    evidence, marker = tmp_path/'decision.json', tmp_path/'owner.ready'
    mode = phase+':'+str(int(committed))
    process = await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._wait_decision_tree_worker',
        dsn,c.manager.prefix,json.dumps(config),mode,str(evidence),str(marker),
        cwd=Path(__file__).resolve().parents[2],stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        output,_ = await asyncio.wait_for(process.communicate(),30)
        assert process.returncode==82,output.decode()
        assert marker.read_text()==mode
        saved = json.loads(evidence.read_text())
        permitted = phase!='accepted' or committed
        applied = phase=='completed' or phase=='target' and committed
        completed = phase=='completed' and committed
        accepted = await grants(c)
        assert len(accepted)==int(permitted)
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'"))==int(applied)
        phases = await c.manager.db.fetchall(f"SELECT event_type FROM {c.manager.events} WHERE job_id=? AND event_type IN ('model_attempt_prepared','model_attempt_dispatched','model_attempt_finished')",(saved['caller']['job_id'],))
        assert sorted(row['event_type'] for row in phases)==(['model_attempt_dispatched','model_attempt_finished','model_attempt_prepared'] if permitted else [])
        if not permitted:
            assert not await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
            assert saved['model_calls']==1 and h.model.calls==0
            return
        [grant] = accepted
        pending = await c.manager.db.fetchall(f"SELECT source_id FROM {c.manager.prefix}session_obligations WHERE kind='control_request'")
        assert len(pending)==int(not completed)
        cold = JobManager(await c.connect(),now_ms=c.manager.now)
        operation = await wait_decision_admission.reconcile_request(cold,c.main,grant['event_id'])
        rows = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await wait_decision_admission.reconcile_request(cold,c.main,grant['event_id'])==operation
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")==rows
        assert operation.receipt_ref is None
        assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
        with pytest.raises(JobError,match='wait_decision_receipt_required'):
            await wait_effects.apply(cold,c.main,h.event_id)
        assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='wait_effect'")
        assert saved['model_calls']==1 and h.model.calls==0
    finally:
        await stop_worker(process)


@pytest.mark.parametrize('phase',['accepted','target','completed'])
@pytest.mark.parametrize('committed',[False,True])
async def test_native_decision_commit_failure_keeps_original_model_and_receipt_barrier(handoff,monkeypatch,phase,committed):
    from power_loop.runtime.managed_effects import _ref
    from power_loop.runtime.tool_call_plans import identity, start_id

    h,c,hits = handoff,handoff.c,[]
    h.loop.task_runtime.checkpoint_policy = CheckpointPolicy(interval_s=300)
    transaction = c.manager.db.transaction

    @asynccontextmanager
    async def boundary():
        async with transaction() as tx:
            yield tx
            rows = await tx.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='task_control_request'")
            accepted = [row for row in rows if (lambda data:data['phase']=='accepted' and data['action']==wait_decision_admission.ACTION)(json.loads(row['payload_json']))]
            current = None
            if accepted:
                if any(row['event_id']==control_requests.completion_id(accepted[0]['event_id']) for row in rows):
                    current = 'completed'
                elif await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_decision'"):
                    current = 'target'
                else:
                    current = 'accepted'
            hit = not hits and current==phase
            if hit:
                request = json.loads(accepted[0]['payload_json'])['request']
                source = await c.manager._row(tx,accepted[0]['job_id'])
                started = start_id(identity(c.manager,source,_ref(request['call_ref'])),request['tool_call_id'])
                assert await tx.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_id=? AND event_type='tool_call_dispatch_started'",(started,))
                hits.append(phase)
                if not committed:
                    raise OSError('wait '+phase+' rolled back')
        if hit:
            raise OSError('wait '+phase+' committed; ACK lost')

    with monkeypatch.context() as patch:
        patch.setattr(c.manager.db,'transaction',boundary)
        await decide(h)
    assert hits==[phase] and h.model.calls==2
    [tool] = [row for row in await c.store.load_all_messages(c.main.session_id) if row.role=='tool' and row.name=='wait_handoff']
    response = json.loads(tool.content)
    permitted = phase!='accepted' or committed
    applied = phase=='completed' or phase=='target' and committed
    assert (response.get('status')=='accepted')==applied
    accepted = await grants(c)
    assert len(accepted)==int(permitted)
    if permitted:
        [grant] = accepted
        cold = JobManager(await c.connect(),now_ms=c.manager.now)
        operation = await wait_decision_admission.reconcile_request(cold,c.main,grant['event_id'])
        if applied:
            assert operation.receipt_ref['id']==str(tool.id)
        else:
            assert operation.receipt_ref is None
            with pytest.raises(JobError,match='wait_decision_receipt_required'):
                await wait_effects.apply(cold,c.main,h.event_id)
        assert not await cold.db.fetchall(f"SELECT source_id FROM {cold.prefix}session_obligations WHERE kind='control_request'")
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_effect'")
    assert h.model.calls==2
