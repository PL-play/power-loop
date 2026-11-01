"""Publishing an original wait effect must not lock its model-review source tree."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop.runtime import timer_waits, wait_decision_admission, wait_decisions, wait_effects
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_wait_effects_192 import decide
from .test_wait_handoff_192 import handoff, registry_store, runtime, waiting

__all__ = ['handoff', 'registry_store', 'runtime', 'waiting']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('choice',['continue','defer','cancel'])
async def test_wait_effect_uses_one_tree_per_writer(handoff,monkeypatch,choice):
    h,c = handoff,handoff.c
    await decide(h,choice)
    write,one,many = c.manager._write,ControlTransaction.fetchone,ControlTransaction.fetchall
    footprints = []

    def note(tx,rows):
        tx.__dict__.setdefault('_wait_effect_reads',set()).update(
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
            jobs = tx.touched_jobs|tx.__dict__.get('_wait_effect_reads',set())
            footprints.append({roots[job] for job in jobs})

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',scoped)
        patch.setattr(ControlTransaction,'fetchone',fetchone)
        patch.setattr(ControlTransaction,'fetchall',fetchall)
        reply = await wait_effects.apply(c.manager,c.main,h.event_id)
    assert reply['decision']==choice
    assert footprints and all(len(roots)<=1 for roots in footprints),footprints


async def test_main_model_and_receipt_originals_are_verified_before_the_wait_writer(handoff,monkeypatch):
    h,c = handoff,handoff.c
    await decide(h)
    stored,validate = wait_decision_admission._stored_request,c.manager._validate_ref
    grants,receipts = [],[]

    async def original(manager,*args,**kwargs):
        assert manager._control_intents.get() is None
        grants.append(args[2]['event_id'])
        return await stored(manager,*args,**kwargs)

    async def raw(reader,actor,ref,**kwargs):
        if actor.session_id==c.main.session_id:
            assert c.manager._control_intents.get() is None
            receipts.append(ref.role)
        return await validate(reader,actor,ref,**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wait_decision_admission,'_stored_request',original)
        patch.setattr(c.manager,'_validate_ref',raw)
        await wait_effects.apply(c.manager,c.main,h.event_id)
    assert len(grants)==1 and 'tool' in receipts and h.model.calls==2


async def test_a_slow_source_proof_does_not_hold_up_an_unrelated_tree(handoff,monkeypatch):
    h,c = handoff,handoff.c
    await decide(h)
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
        action = asyncio.create_task(wait_effects.apply(c.manager,c.main,h.event_id))
        try:
            await asyncio.wait_for(entered.wait(),3)
            await asyncio.wait_for(other.request_cancel(c.main,third.job.job_id,system_key='third-stop',reason='test'),3)
            assert (await other.get_job(c.main,third.job.job_id)).cancel_requested_at is not None
        finally:
            release.set()
            await asyncio.wait_for(action,5)
    assert h.model.calls==2


async def test_a_new_review_source_is_prepared_again_outside_the_writer(handoff,monkeypatch):
    h,c = handoff,handoff.c
    await decide(h,'defer')
    original_event = h.event_id
    peer = JobManager(await c.connect(),now_ms=c.manager.now)
    entered,release = asyncio.Event(),asyncio.Event()
    apply,attempts = wait_effects._apply,[]

    async def pause(manager,actor,event_id):
        if manager is c.manager:
            attempts.append(wait_decisions._SOURCE_PROOFS.get())
            if len(attempts)==1:
                entered.set()
                await release.wait()
        return await apply(manager,actor,event_id)

    with monkeypatch.context() as patch:
        patch.setattr(wait_effects,'_apply',pause)
        action = asyncio.create_task(wait_effects.apply(c.manager,c.main,original_event))
        try:
            await asyncio.wait_for(entered.wait(),3)
            expected = await wait_effects.apply(peer,c.main,original_event)
            c.now[0] += 1001
            due = await timer_waits.prepare_due(peer,c.main,h.registration.job.job_id)
            h.event_id = due['event_id']
            h.model.calls = 0
            await decide(h,'defer')
            await wait_effects.apply(peer,c.main,h.event_id)
            assert len(h.model.requests)==4
            release.set()
            assert await asyncio.wait_for(action,5)==expected
        finally:
            release.set()
            await asyncio.gather(action,return_exceptions=True)
    assert len(attempts)==2 and attempts[0] is not attempts[1]
    assert len(h.model.requests)==4


@pytest.mark.parametrize('choice',['continue','defer','cancel'])
async def test_target_stop_during_source_preparation_never_creates_a_new_effect(handoff,monkeypatch,choice):
    h,c = handoff,handoff.c
    await decide(h,choice)
    apply = wait_effects._apply
    before = []

    async def stopped(manager,actor,event_id):
        await manager.request_cancel(c.main,c.owner.job_id,system_key='stop-before-effect',reason='test')
        before.append(await manager.get_job(c.main,c.owner.job_id))
        return await apply(manager,actor,event_id)

    with monkeypatch.context() as patch:
        patch.setattr(wait_effects,'_apply',stopped)
        if choice=='cancel':
            reply = await wait_effects.apply(c.manager,c.main,h.event_id)
            assert reply['status']=='stop_requested'
        else:
            with pytest.raises(JobError,match='cancel_requested|intake_closed'):
                await wait_effects.apply(c.manager,c.main,h.event_id)
    assert h.model.calls==2 and before[0].cancel_requested_at is not None
    assert not await c.store.inbox_items(c.sid)


@pytest.mark.parametrize('stage',['prepared','first_write','before_commit','after_commit'])
@pytest.mark.parametrize('choice',['continue','defer','cancel'])
async def test_real_effect_process_death_does_not_repeat_a_model_or_leave_partial_input(handoff,tmp_path,stage,choice):
    h,c = handoff,handoff.c
    await decide(h,choice)
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine=='sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    config = {'now':c.now[0],'actor':asdict(c.main),'event_id':h.event_id}
    marker = tmp_path/'effect.ready'
    process = await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._wait_effect_tree_worker',
        dsn,c.manager.prefix,json.dumps(config),stage,str(marker),
        cwd=Path(__file__).resolve().parents[2],stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        output,_ = await asyncio.wait_for(process.communicate(),30)
        assert process.returncode==83,output.decode()
        assert marker.read_text()==stage
        committed = stage=='after_commit'
        effects = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='wait_effect'")
        assert len(effects)==int(committed)
        assert len(await c.store.inbox_items(c.sid))==int(committed and choice=='continue')
        cold = JobManager(await c.connect(),now_ms=c.manager.now)
        reply = await wait_effects.apply(cold,c.main,h.event_id)
        rows = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await wait_effects.apply(cold,c.main,h.event_id)==reply
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")==rows
        assert reply['decision']==choice and not reply['physical_stop_confirmed']
        assert len(await c.store.inbox_items(c.sid))==int(choice=='continue')
        assert h.model.calls==2
    finally:
        await stop_worker(process)
