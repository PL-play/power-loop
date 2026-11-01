"""A retained controller inspects review originals before publishing its tree."""
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

from power_loop.runtime import owner_waits
from power_loop.runtime.jobs import JobError, JobLease, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_job_registry_192 import registry_store
from .test_owner_wait_recovery_192 import continue_original, paused
from .test_owner_waits_192 import original

__all__ = ['registry_store','original','paused']
pytestmark = [pytest.mark.unit,pytest.mark.asyncio]


@asynccontextmanager
async def observe(manager,monkeypatch):
    rows = await manager.db.fetchall(f"SELECT job_id,root_job_id FROM {manager.jobs}")
    roots = {row['job_id']:row['root_job_id'] for row in rows}
    write,one,many = manager._write,ControlTransaction.fetchone,ControlTransaction.fetchall
    footprints = []

    def note(tx,rows):
        tx.__dict__.setdefault('_owner_transition_reads',set()).update(
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
        jobs = tx.touched_jobs|tx.__dict__.get('_owner_transition_reads',set())
        footprints.append({roots[job] for job in jobs})

    with monkeypatch.context() as patch:
        patch.setattr(manager,'_write',scoped)
        patch.setattr(ControlTransaction,'fetchone',fetchone)
        patch.setattr(ControlTransaction,'fetchall',fetchall)
        yield footprints


@pytest.fixture
async def entry_pending(original,monkeypatch):
    c = original
    c.state.wait_ms = 20000
    enter = owner_waits._enter
    entered,release = asyncio.Event(),asyncio.Event()
    captured = {}

    async def held(manager,actor,lease,raw):
        captured.update(actor=actor,lease=lease,raw=raw)
        entered.set()
        await release.wait()
        return await enter(manager,actor,lease,raw)

    monkeypatch.setattr(owner_waits,'_enter',held)
    await c.loop.send('Start original finite task',c.state.parent)
    await asyncio.wait_for(entered.wait(),3)
    [registration] = c.state.registrations
    try:
        yield SimpleNamespace(c=c,registration=registration,enter=enter,release=release,**captured)
    finally:
        release.set()


async def test_first_owner_wait_entry_prepares_another_main_review_before_its_own_writer(entry_pending,monkeypatch):
    p,c = entry_pending,entry_pending.c
    await continue_original(c,p.registration)
    calls = c.model.calls.copy()
    pending = owner_waits._pending

    async def outside(manager,reader,*args):
        assert manager._control_intents.get() is None
        return await pending(manager,reader,*args)

    monkeypatch.setattr(owner_waits,'_pending',outside)
    async with observe(c.manager,monkeypatch) as footprints:
        identity,saved = await p.enter(c.manager,p.actor,p.lease,p.raw)
    assert saved['raw']==p.raw and saved['owner']['job_id']==p.lease.job_id
    assert (await c.manager.get_job(p.actor,p.lease.job_id)).status=='waiting'
    assert c.model.calls==calls
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_id=?",(identity,)))==1
    assert footprints and all(len(scope)<=1 for scope in footprints),footprints


@pytest.mark.parametrize('stage',['idle','resume','history'])
async def test_owner_wait_step_writes_only_a_new_own_tree_transition(paused,monkeypatch,stage):
    p,c = paused,paused.c
    if stage!='idle':
        await continue_original(c,p.registration)
    if stage=='history':
        assert await p.transition()=='input'
    before = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
    calls = c.model.calls.copy()
    async with observe(c.manager,monkeypatch) as footprints:
        result = await p.transition()
    assert result==(None if stage=='idle' else 'input')
    assert c.model.calls==calls
    assert all(len(scope)<=1 for scope in footprints),footprints
    assert len(footprints)==int(stage=='resume'),footprints
    if stage!='resume':
        assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")==before


@pytest.mark.parametrize('reviewed',[False,True])
async def test_original_wait_entry_replay_is_readonly_even_after_another_main_review(paused,monkeypatch,reviewed):
    p,c = paused,paused.c
    if reviewed:
        await continue_original(c,p.registration)
    calls = c.model.calls.copy()
    before = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
    async with observe(c.manager,monkeypatch) as footprints:
        result = await owner_waits._enter(c.manager,c.runtime.actor(c.state.parent),JobLease(**p.saved['owner']),p.saved['raw'])
    assert result==(p.event['event_id'],p.saved)
    assert c.model.calls==calls
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")==before
    assert not footprints,footprints


async def test_owner_stop_after_preparation_prevents_resume_publication(paused,monkeypatch):
    p,c = paused,paused.c
    await continue_original(c,p.registration)
    publish,attempts = owner_waits._publish_transition,[]
    other = JobManager(await c.connect(),now_ms=c.manager.now)

    async def stopped(manager,actor,lease,transition,**kwargs):
        attempts.append(transition)
        await other.request_cancel(c.runtime.actor(c.state.parent),p.owner.job_id,system_key='stop-after-prepared-resume',reason='test')
        return await publish(manager,actor,lease,transition,**kwargs)

    monkeypatch.setattr(owner_waits,'_publish_transition',stopped)
    with pytest.raises(JobError,match='cancel_requested'):
        await p.transition()
    assert len(attempts)==1 and c.model.calls['child']==2
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")


async def test_changed_required_timer_reprepares_before_resume(paused,monkeypatch):
    p,c = paused,paused.c
    await continue_original(c,p.registration)
    publish,prepare = owner_waits._publish_transition,owner_waits._prepare_step
    plans = []

    async def observed(*args):
        plans.append(True)
        return await prepare(*args)

    async def changed(manager,actor,lease,transition,**kwargs):
        if len(plans)==1:
            await c.manager.request_cancel(c.runtime.actor(c.state.parent),p.registration.job.job_id,
                system_key='stop-timer-after-prepared-resume',reason='test')
        return await publish(manager,actor,lease,transition,**kwargs)

    monkeypatch.setattr(owner_waits,'_prepare_step',observed)
    monkeypatch.setattr(owner_waits,'_publish_transition',changed)
    assert await p.transition()=='input'
    assert len(plans)==2 and c.model.calls['child']==2
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_resumed'"))==1
    assert (await c.manager.get_job(c.runtime.actor(c.state.parent),p.registration.job.job_id)).cancel_requested_at is not None


async def test_deadline_crossing_after_preparation_publishes_only_expiry(paused,monkeypatch):
    p,c = paused,paused.c
    await continue_original(c,p.registration)
    deadline = p.saved['deadline_at_ms']
    monkeypatch.setattr(c.manager,'now',lambda:deadline-1)
    publish,attempts = owner_waits._publish_transition,[]

    async def crossed(manager,actor,lease,transition,**kwargs):
        attempts.append(transition.event['event_type'])
        monkeypatch.setattr(c.manager,'now',lambda:deadline)
        return await publish(manager,actor,lease,transition,**kwargs)

    monkeypatch.setattr(owner_waits,'_publish_transition',crossed)
    assert await p.transition()=='expired'
    assert attempts==['owner_wait_resumed','owner_wait_expired']
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_expired'"))==1
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_resumed'")
    assert c.model.calls['child']==2


async def test_two_prepared_expiry_publishers_keep_one_original_without_a_version_write(paused,monkeypatch):
    p,c = paused,paused.c
    monkeypatch.setattr(c.manager,'now',lambda:p.saved['deadline_at_ms'])
    other = JobManager(await c.connect(),now_ms=c.manager.now)
    publish = owner_waits._publish_transition
    ready,release = asyncio.Event(),asyncio.Event()
    plans = []

    async def held(manager,actor,lease,transition,**kwargs):
        plans.append(transition)
        if len(plans)==2:
            ready.set()
        await release.wait()
        return await publish(manager,actor,lease,transition,**kwargs)

    monkeypatch.setattr(owner_waits,'_publish_transition',held)
    before = await c.manager.get_job(c.runtime.actor(c.state.parent),p.owner.job_id)
    actions = [asyncio.create_task(p.transition()),asyncio.create_task(p.step(other,
        c.runtime.actor(c.state.parent),JobLease(**p.saved['owner']),p.event['event_id'],p.saved))]
    try:
        await asyncio.wait_for(ready.wait(),3)
        release.set()
        assert await asyncio.wait_for(asyncio.gather(*actions),10)==['expired','expired']
    finally:
        release.set()
        for action in actions:
            if not action.done():
                action.cancel()
        await asyncio.gather(*actions,return_exceptions=True)
    assert len(plans)==2 and c.model.calls['child']==2
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='owner_wait_expired'"))==1
    assert await c.manager.get_job(c.runtime.actor(c.state.parent),p.owner.job_id)==before


async def test_slow_resource_proof_does_not_hold_up_a_third_tree(paused,monkeypatch):
    p,c = paused,paused.c
    await continue_original(c,p.registration)
    other = JobManager(await c.connect(),now_ms=c.manager.now)
    actor = c.runtime.actor(c.state.parent)
    third = await other.register_job(actor,kind='send',session_id=actor.session_id,system_key='third-owner-wait-root')
    quiescent = owner_waits._quiescent
    ready,release = asyncio.Event(),asyncio.Event()

    async def held(manager,reader,owner):
        assert manager._control_intents.get() is None
        ready.set()
        await release.wait()
        return await quiescent(manager,reader,owner)

    monkeypatch.setattr(owner_waits,'_quiescent',held)
    action = asyncio.create_task(p.transition())
    try:
        await asyncio.wait_for(ready.wait(),3)
        await asyncio.wait_for(other.request_cancel(actor,third.job.job_id,system_key='stop-third-owner-wait-root',reason='test'),3)
    finally:
        release.set()
        assert await asyncio.wait_for(action,5)=='input'
    assert (await other.get_job(actor,third.job.job_id)).cancel_requested_at is not None
    assert c.model.calls['child']==2


async def crash_transition(p,tmp_path,stage,operation,monkeypatch):
    c = p.c
    actor = p.actor if operation=='enter' else c.runtime.actor(c.state.parent)
    lease = p.lease if operation=='enter' else JobLease(**p.saved['owner'])
    if operation=='expire':
        monkeypatch.setattr(c.manager,'now',lambda:p.saved['deadline_at_ms'])
    config = {'now':c.manager.now(),'actor':asdict(actor),'lease':asdict(lease),'operation':operation}
    if operation=='enter':
        config['raw'] = p.raw
    else:
        config.update(waiting_id=p.event['event_id'],saved=p.saved)
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine=='sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    calls = c.model.calls.copy()
    messages = await c.manager.db.fetchall(f"SELECT * FROM {c.store.t.messages} ORDER BY id")
    marker = tmp_path/'owner-transition.ready'
    process = await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._owner_wait_transition_tree_worker',
        dsn,c.manager.prefix,json.dumps(config),stage,str(marker),
        cwd=Path(__file__).resolve().parents[2],stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        output,_ = await asyncio.wait_for(process.communicate(),30)
        assert process.returncode==90,output.decode()
        assert marker.read_text()==stage
        kind = {'enter':'owner_waiting','resume':'owner_wait_resumed','expire':'owner_wait_expired'}[operation]
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type=?",(kind,)))==int(stage=='after_commit')
        cold = JobManager(await c.connect(),now_ms=c.manager.now)

        async def recover():
            if operation=='enter':
                return await p.enter(cold,actor,lease,p.raw)
            return await p.step(cold,actor,lease,p.event['event_id'],p.saved)

        result = await recover()
        events = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await recover()==result
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")==events
        assert await cold.db.fetchall(f"SELECT * FROM {c.store.t.messages} ORDER BY id")==messages
        assert len(await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type=?",(kind,)))==1
        assert c.model.calls==calls
        if operation!='enter':
            assert result==('input' if operation=='resume' else 'expired')
    finally:
        await stop_worker(process)


@pytest.mark.parametrize('stage',['prepared','first_write','before_commit','after_commit'])
async def test_real_owner_wait_entry_death_replays_original_metadata_without_a_model(entry_pending,tmp_path,monkeypatch,stage):
    p = entry_pending
    await continue_original(p.c,p.registration)
    await crash_transition(p,tmp_path,stage,'enter',monkeypatch)


@pytest.mark.parametrize('stage',['prepared','first_write','before_commit','after_commit'])
async def test_real_owner_wait_resume_death_replays_one_accepted_input(paused,tmp_path,monkeypatch,stage):
    p = paused
    await continue_original(p.c,p.registration)
    await crash_transition(p,tmp_path,stage,'resume',monkeypatch)


@pytest.mark.parametrize('stage',['prepared','first_write','before_commit','after_commit'])
async def test_real_owner_wait_expiry_death_replays_one_fact(paused,tmp_path,monkeypatch,stage):
    await crash_transition(paused,tmp_path,stage,'expire',monkeypatch)
