"""Required-wait settlement does not inspect another root inside its writer."""
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

from power_loop.runtime import wait_decision_admission, wait_effects
from power_loop.runtime.jobs import JobLease, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_owner_wait_recovery_192 import paused
from .test_owner_waits_192 import original
from .test_wait_effects_192 import child_loop, decide
from .test_wait_handoff_192 import handoff, registry_store, runtime, waiting

__all__ = ['handoff','registry_store','runtime','waiting','original','paused']
pytestmark = [pytest.mark.unit,pytest.mark.asyncio]


@pytest.fixture
async def consumed(handoff):
    h,c = handoff,handoff.c
    await decide(h)
    await wait_effects.apply(c.manager,c.main,h.event_id)
    loop,model,binding = await child_loop(h)
    async with loop.task_runtime.execution(binding,owner=True):
        answer = await loop.flush_follow_ups(c.sid)
    return SimpleNamespace(h=h,c=c,model=model,answer=answer)


@pytest.mark.parametrize('port',['settle','owner_result'])
async def test_consumed_wait_settlement_uses_one_tree_per_writer(consumed,monkeypatch,port):
    c,h = consumed.c,consumed.h
    rows = await c.manager.db.fetchall(f"SELECT job_id,root_job_id FROM {c.manager.jobs}")
    roots = {row['job_id']:row['root_job_id'] for row in rows}
    write,one,many = c.manager._write,ControlTransaction.fetchone,ControlTransaction.fetchall
    footprints = []

    def note(tx,rows):
        tx.__dict__.setdefault('_settlement_reads',set()).update(
            row['job_id'] for row in rows if row is not None and 'job_id' in row)

    async def fetchone(tx,sql,params=()):
        row = await one(tx,sql,params)
        note(tx,[row])
        return row

    async def fetchall(tx,sql,params=()):
        rows = await many(tx,sql,params)
        note(tx,rows)
        return rows

    @asynccontextmanager
    async def scoped():
        async with write() as tx:
            yield tx
        jobs = tx.touched_jobs|tx.__dict__.get('_settlement_reads',set())
        footprints.append({roots[job] for job in jobs})

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',scoped)
        patch.setattr(ControlTransaction,'fetchone',fetchone)
        patch.setattr(ControlTransaction,'fetchall',fetchall)
        if port=='settle':
            await wait_effects.settle_owner(c.manager,c.actor,c.owner)
        else:
            await c.manager.record_execution_result(c.actor,c.owner,outcome='succeeded',
                result={'final_text':consumed.answer.final_text},resource_state='stopped')
    assert (await c.manager.get_job(c.main,h.registration.job.job_id)).outcome=='succeeded'
    assert consumed.model.calls==1 and h.model.calls==2
    assert footprints and all(len(scope)<=1 for scope in footprints),footprints


async def test_completed_public_settlement_replay_is_readonly(consumed,monkeypatch):
    c = consumed.c
    await wait_effects.settle_owner(c.manager,c.actor,c.owner)

    @asynccontextmanager
    async def forbidden():
        pytest.fail('completed required-wait originals must not reacquire a writer')
        yield

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',forbidden)
        await wait_effects.settle_owner(c.manager,c.actor,c.owner)
    assert consumed.model.calls==1 and consumed.h.model.calls==2


@pytest.mark.parametrize('cancel_owner',[False,True])
async def test_paused_settlement_source_proof_allows_a_third_root_and_keeps_current_stop(consumed,monkeypatch,cancel_owner):
    c,h = consumed.c,consumed.h
    other = JobManager(await c.connect(),now_ms=c.manager.now)
    third = await other.register_job(c.main,kind='send',session_id=c.main.session_id,system_key='third-settlement-root')
    stored = wait_decision_admission._stored_request
    entered,release = asyncio.Event(),asyncio.Event()

    async def pause(manager,reader,*args,**kwargs):
        assert manager._control_intents.get() is None
        result = await stored(manager,reader,*args,**kwargs)
        entered.set()
        await release.wait()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(wait_decision_admission,'_stored_request',pause)
        action = asyncio.create_task(c.manager.record_execution_result(c.actor,c.owner,outcome='succeeded',
            result={'final_text':consumed.answer.final_text},resource_state='stopped'))
        try:
            await asyncio.wait_for(entered.wait(),3)
            await asyncio.wait_for(other.request_cancel(c.main,third.job.job_id,system_key='stop-third-settlement-root',reason='test'),3)
            if cancel_owner:
                await asyncio.wait_for(other.request_cancel(c.main,c.owner.job_id,system_key='stop-owner-after-source-proof',reason='test'),3)
        finally:
            release.set()
            await asyncio.wait_for(action,5)
    assert (await other.get_job(c.main,third.job.job_id)).cancel_requested_at is not None
    assert (await other.get_job(c.main,c.owner.job_id)).outcome==('cancelled' if cancel_owner else 'succeeded')
    assert (await other.get_job(c.main,h.registration.job.job_id)).outcome==(None if cancel_owner else 'succeeded')
    assert consumed.model.calls==1 and h.model.calls==2


async def test_two_owner_result_publishers_resolve_one_original_wait(consumed):
    c = consumed.c
    other = JobManager(await c.connect(),now_ms=c.manager.now)
    results = await asyncio.gather(*(manager.record_execution_result(c.actor,c.owner,outcome='succeeded',
        result={'final_text':consumed.answer.final_text},resource_state='stopped') for manager in (c.manager,other)))
    assert results[0]==results[1]
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_resolved'"))==1
    assert consumed.model.calls==1 and consumed.h.model.calls==2


@pytest.mark.parametrize('cancel_scope',['owner','root'])
async def test_cancelled_waiting_owner_keeps_actual_exit_without_consuming_the_input(paused,cancel_scope):
    p,c = paused,paused.c
    actor = c.runtime.actor(c.state.parent)
    row = await c.manager._row(c.manager.db,p.owner.job_id)
    target = row['job_id'] if cancel_scope=='owner' else row['root_job_id']
    await c.manager.request_cancel(actor,target,system_key='cancel-before-exit-evidence',reason='test')
    usage = await c.manager.execution_usage(actor,p.owner.job_id,generation=1)
    await c.manager.record_execution_result(actor,JobLease(**p.saved['owner']),outcome='succeeded',
        result={**p.saved['raw'],'usage':usage},resource_state='stopped')
    owner = await c.manager.get_job(actor,p.owner.job_id)
    assert owner.outcome=='cancelled' and owner.resource_state=='stopped'
    assert owner.result['usage']==usage and usage['total_tokens']==20
    assert owner.result['execution_status']=='completed'
    assert (await c.manager.get_job(actor,p.registration.job.job_id)).outcome is None
    assert c.model.calls['child']==2


@pytest.mark.parametrize('stage',['prepared','resolution','timer_result','owner_result','before_commit','after_commit'])
async def test_real_owner_result_death_never_leaves_a_partial_wait_resolution(consumed,tmp_path,stage):
    c,h = consumed.c,consumed.h
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine=='sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    config = {'now':c.manager.now(),'actor':asdict(c.actor),'owner':asdict(c.owner),
        'wait_id':h.registration.job.job_id,'result':{'final_text':consumed.answer.final_text}}
    marker = tmp_path/'settlement.ready'
    messages = await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} ORDER BY id")
    process = await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._wait_settlement_tree_worker',
        dsn,c.manager.prefix,json.dumps(config),stage,str(marker),
        cwd=Path(__file__).resolve().parents[2],stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        output,_ = await asyncio.wait_for(process.communicate(),30)
        assert process.returncode==89,output.decode()
        assert marker.read_text()==stage
        expected = 'succeeded' if stage=='after_commit' else None
        assert (await c.manager.get_job(c.main,c.owner.job_id)).outcome==expected
        assert (await c.manager.get_job(c.main,h.registration.job.job_id)).outcome==expected
        cold = JobManager(await c.connect(),now_ms=c.manager.now)
        first = await cold.record_execution_result(c.actor,c.owner,outcome='succeeded',result=config['result'],resource_state='stopped')
        events = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await cold.record_execution_result(c.actor,c.owner,outcome='succeeded',result=config['result'],resource_state='stopped')==first
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")==events
        assert await cold.db.fetchall(f"SELECT * FROM {c.store.t.messages} ORDER BY id")==messages
        assert (await cold.get_job(c.main,c.owner.job_id)).outcome=='succeeded'
        assert (await cold.get_job(c.main,h.registration.job.job_id)).outcome=='succeeded'
        assert consumed.model.calls==1 and h.model.calls==2
    finally:
        await stop_worker(process)
