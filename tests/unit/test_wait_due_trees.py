"""Historical reviews must not pull their main model tree into a timer writer."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop.runtime import timer_waits, wait_decision_admission, wait_effects
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_wait_effects_192 import decide
from .test_wait_handoff_192 import handoff, registry_store, runtime, waiting

__all__ = ['handoff','registry_store','runtime','waiting']
pytestmark = [pytest.mark.unit,pytest.mark.asyncio]


@pytest.mark.parametrize('stage',['armed','new_due','existing_due'])
async def test_due_preparation_uses_only_its_own_tree_writer(handoff,monkeypatch,stage):
    h,c = handoff,handoff.c
    await decide(h,'defer')
    await wait_effects.apply(c.manager,c.main,h.event_id)
    if stage!='armed':
        c.now[0] += 1001
    if stage=='existing_due':
        await timer_waits.prepare_due(c.manager,c.main,h.registration.job.job_id)
    write,one,many = c.manager._write,ControlTransaction.fetchone,ControlTransaction.fetchall
    footprints = []

    def note(tx,rows):
        tx.__dict__.setdefault('_wait_due_reads',set()).update(
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
            roots = {row['job_id']:row['root_job_id'] for row in await tx._tx.fetchall(f"SELECT job_id,root_job_id FROM {c.manager.jobs}")}
            jobs = tx.touched_jobs|tx.__dict__.get('_wait_due_reads',set())
            footprints.append({roots[job] for job in jobs})

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',scoped)
        patch.setattr(ControlTransaction,'fetchone',fetchone)
        patch.setattr(ControlTransaction,'fetchall',fetchall)
        result = await timer_waits.prepare_due(c.manager,c.main,h.registration.job.job_id)
    assert result['status']==('armed' if stage=='armed' else 'decision_pending')
    assert h.model.calls==2
    assert all(len(roots)<=1 for roots in footprints),footprints
    if stage!='new_due':
        assert footprints==[]


async def ready(h):
    await decide(h,'defer')
    await wait_effects.apply(h.c.manager,h.c.main,h.event_id)
    h.c.now[0] += 1001


async def test_previous_review_model_proof_is_read_outside_the_due_writer(handoff,monkeypatch):
    h,c = handoff,handoff.c
    await ready(h)
    stored,seen = wait_decision_admission._stored_request,[]

    async def outside(manager,*args,**kwargs):
        assert manager._control_intents.get() is None
        seen.append(args[2]['event_id'])
        return await stored(manager,*args,**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wait_decision_admission,'_stored_request',outside)
        due = await timer_waits.prepare_due(c.manager,c.main,h.registration.job.job_id)
    assert due['status']=='decision_pending' and len(seen)==1 and h.model.calls==2


async def test_due_original_preparation_does_not_block_an_unrelated_root(handoff,monkeypatch):
    h,c = handoff,handoff.c
    await ready(h)
    other = JobManager(await c.connect(),now_ms=c.manager.now)
    third = await other.register_job(c.main,kind='send',session_id=c.main.session_id,system_key='third-root')
    entered,release = asyncio.Event(),asyncio.Event()
    stored = wait_decision_admission._stored_request

    async def pause(manager,*args,**kwargs):
        assert manager._control_intents.get() is None
        entered.set()
        await release.wait()
        return await stored(manager,*args,**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wait_decision_admission,'_stored_request',pause)
        action = asyncio.create_task(timer_waits.prepare_due(c.manager,c.main,h.registration.job.job_id))
        try:
            await asyncio.wait_for(entered.wait(),3)
            await asyncio.wait_for(other.request_cancel(c.main,third.job.job_id,system_key='third-stop',reason='test'),3)
            assert (await other.get_job(c.main,third.job.job_id)).cancel_requested_at is not None
        finally:
            release.set()
            await asyncio.wait_for(action,5)
    assert h.model.calls==2


async def test_owner_stop_between_due_preparation_and_publication_keeps_only_prior_due(handoff,monkeypatch):
    h,c = handoff,handoff.c
    await ready(h)
    write,stopped = c.manager._write,[]

    @asynccontextmanager
    async def changed():
        if not stopped:
            stopped.append(True)
            other = JobManager(await c.connect(),now_ms=c.manager.now)
            await other.request_cancel(c.main,c.owner.job_id,system_key='stop-before-due',reason='test')
        async with write() as tx:
            yield tx

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',changed)
        with pytest.raises(JobError,match='cancel_requested|intake_closed'):
            await timer_waits.prepare_due(c.manager,c.main,h.registration.job.job_id)
    assert stopped==[True]
    [row] = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_due'")
    assert row['event_id']==h.event_id and h.model.calls==2


async def test_two_due_publishers_use_one_original_and_both_return_the_same_event(handoff):
    h,c = handoff,handoff.c
    await ready(h)
    other = JobManager(await c.connect(),now_ms=c.manager.now)
    left,right = await asyncio.gather(timer_waits.prepare_due(c.manager,c.main,h.registration.job.job_id),
        timer_waits.prepare_due(other,c.main,h.registration.job.job_id))
    assert left==right and left['event_id']!=h.event_id
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_due'"))==2
    assert h.model.calls==2


@pytest.mark.parametrize('stage',['prepared','first_write','before_commit','after_commit'])
async def test_real_due_publisher_death_never_loses_review_or_repeats_its_model(handoff,tmp_path,stage):
    h,c = handoff,handoff.c
    await ready(h)
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine=='sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    config = {'now':c.now[0],'actor':asdict(c.main),'job_id':h.registration.job.job_id}
    marker = tmp_path/'due.ready'
    process = await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._wait_due_tree_worker',
        dsn,c.manager.prefix,json.dumps(config),stage,str(marker),
        cwd=Path(__file__).resolve().parents[2],stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        output,_ = await asyncio.wait_for(process.communicate(),30)
        assert process.returncode==84,output.decode()
        assert marker.read_text()==stage
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_due'"))==(2 if stage=='after_commit' else 1)
        cold = JobManager(await c.connect(),now_ms=c.manager.now)
        result = await timer_waits.prepare_due(cold,c.main,h.registration.job.job_id)
        assert result['status']=='decision_pending' and result['event_id']!=h.event_id
        after = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await timer_waits.prepare_due(cold,c.main,h.registration.job.job_id)==result
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")==after
        assert len(await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='wait_review'"))==1
        assert len(await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='wait_effect'"))==1
        assert not await c.store.inbox_items(c.sid) and h.model.calls==2
    finally:
        await stop_worker(process)
