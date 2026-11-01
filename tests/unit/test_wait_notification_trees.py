"""Only the original timer tree publishes its main-session notification."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop.runtime import timer_waits, wait_decision_admission, wait_effects, wait_notifications
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_wait_effects_192 import decide
from .test_wait_handoff_192 import handoff, registry_store, runtime, waiting

__all__ = ['handoff','registry_store','runtime','waiting']
pytestmark = [pytest.mark.unit,pytest.mark.asyncio]


async def test_notification_after_review_uses_only_one_tree_per_writer(handoff,monkeypatch):
    h,c = handoff,handoff.c
    await decide(h,'defer')
    await wait_effects.apply(c.manager,c.main,h.event_id)
    c.now[0] += 1001
    due = await timer_waits.prepare_due(c.manager,c.main,h.registration.job.job_id)
    write,one,many = c.manager._write,ControlTransaction.fetchone,ControlTransaction.fetchall
    footprints = []

    def note(tx,rows):
        tx.__dict__.setdefault('_notification_reads',set()).update(
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
            jobs = tx.touched_jobs|tx.__dict__.get('_notification_reads',set())
            footprints.append({roots[job] for job in jobs})

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',scoped)
        patch.setattr(ControlTransaction,'fetchone',fetchone)
        patch.setattr(ControlTransaction,'fetchall',fetchall)
        result = await wait_notifications.enqueue_notification(h.loop,c.main,due['event_id'])
    assert result['status']=='enqueued' and h.model.calls==2
    assert footprints and all(len(roots)<=1 for roots in footprints),footprints


async def ready(h):
    c = h.c
    await decide(h,'defer')
    await wait_effects.apply(c.manager,c.main,h.event_id)
    c.now[0] += 1001
    return (await timer_waits.prepare_due(c.manager,c.main,h.registration.job.job_id))['event_id']


async def test_replayed_notification_is_a_readonly_fact_and_never_starts_main(handoff,monkeypatch):
    h,c = handoff,handoff.c
    event_id = await ready(h)
    first = await wait_notifications.enqueue_notification(h.loop,c.main,event_id)

    @asynccontextmanager
    async def forbidden():
        pytest.fail('existing inbox/original facts must not acquire a writer')
        yield

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',forbidden)
        assert await wait_notifications.enqueue_notification(h.loop,c.main,event_id)==first
    assert h.model.calls==2


async def test_notification_main_sources_are_verified_before_its_own_writer(handoff,monkeypatch):
    h,c = handoff,handoff.c
    event_id = await ready(h)
    stored,seen = wait_decision_admission._stored_request,[]

    async def outside(manager,*args,**kwargs):
        assert manager._control_intents.get() is None
        seen.append(args[2]['event_id'])
        return await stored(manager,*args,**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(wait_decision_admission,'_stored_request',outside)
        await wait_notifications.enqueue_notification(h.loop,c.main,event_id)
    assert len(seen)==1 and h.model.calls==2


async def test_slow_notification_source_proof_leaves_unrelated_root_free(handoff,monkeypatch):
    h,c = handoff,handoff.c
    event_id = await ready(h)
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
        action = asyncio.create_task(wait_notifications.enqueue_notification(h.loop,c.main,event_id))
        try:
            await asyncio.wait_for(entered.wait(),3)
            await asyncio.wait_for(other.request_cancel(c.main,third.job.job_id,system_key='third-stop',reason='test'),3)
            assert (await other.get_job(c.main,third.job.job_id)).cancel_requested_at is not None
        finally:
            release.set()
            await asyncio.wait_for(action,5)
    assert h.model.calls==2


async def test_owner_stop_after_preparation_never_enqueues_a_new_notification(handoff,monkeypatch):
    h,c = handoff,handoff.c
    event_id = await ready(h)
    publish = wait_notifications._publish_notification

    async def stopped(manager,loop,actor,current):
        await manager.request_cancel(c.main,c.owner.job_id,system_key='stop-before-note',reason='test')
        return await publish(manager,loop,actor,current)

    with monkeypatch.context() as patch:
        patch.setattr(wait_notifications,'_publish_notification',stopped)
        with pytest.raises(JobError,match='cancel_requested|intake_closed'):
            await wait_notifications.enqueue_notification(h.loop,c.main,event_id)
    key = wait_notifications.notification_id(c.manager,event_id)
    assert not await c.manager.db.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_id=?",(key,))
    assert all(item['item_id']!=key for item in await c.store.inbox_items(c.main.session_id))
    assert h.model.calls==2


async def test_two_notification_publishers_enqueue_one_exact_original_item(handoff):
    from types import SimpleNamespace

    from power_loop.runtime.task_runtime import TaskRuntime

    h,c = handoff,handoff.c
    event_id = await ready(h)
    other = JobManager(await c.connect(),now_ms=c.manager.now)
    runtime = TaskRuntime(other,tenant_id=c.main.tenant_id,principal_id=c.main.principal_id,
        executor_id='main-executor',boot_id='main-boot')
    peer_loop = SimpleNamespace(task_runtime=runtime,tool_registry=h.loop.tool_registry,_note_committed_inbox=lambda sid:None)
    left,right = await asyncio.gather(wait_notifications.enqueue_notification(h.loop,c.main,event_id),
        wait_notifications.enqueue_notification(peer_loop,c.main,event_id))
    assert left==right
    key = wait_notifications.notification_id(c.manager,event_id)
    assert len([item for item in await c.store.inbox_items(c.main.session_id) if item['item_id']==key])==1
    assert h.model.calls==2


@pytest.mark.parametrize('stage',['prepared','first_write','after_inbox','before_commit','after_commit'])
async def test_real_notification_death_cannot_leave_an_inbox_without_its_original(handoff,tmp_path,stage):
    h,c = handoff,handoff.c
    event_id = await ready(h)
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine=='sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    config = {'now':c.now[0],'actor':asdict(c.main),'event_id':event_id}
    marker = tmp_path/'notification.ready'
    key = wait_notifications.notification_id(c.manager,event_id)
    process = await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._wait_notification_tree_worker',
        dsn,c.manager.prefix,json.dumps(config),stage,str(marker),
        cwd=Path(__file__).resolve().parents[2],stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        output,_ = await asyncio.wait_for(process.communicate(),30)
        assert process.returncode==85,output.decode()
        assert marker.read_text()==stage
        committed = stage=='after_commit'
        event = await c.manager.db.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_id=?",(key,))
        assert (event is not None)==committed
        assert len([item for item in await c.store.inbox_items(c.main.session_id) if item['item_id']==key])==int(committed)
        result = await wait_notifications.enqueue_notification(h.loop,c.main,event_id)
        rows = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
        assert await wait_notifications.enqueue_notification(h.loop,c.main,event_id)==result
        assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")==rows
        [item] = [item for item in await c.store.inbox_items(c.main.session_id) if item['item_id']==key]
        assert item['status']=='pending' and h.model.calls==2
    finally:
        await stop_worker(process)
