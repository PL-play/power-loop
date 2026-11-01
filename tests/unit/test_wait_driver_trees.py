"""Driver inspections stay readonly and dispatch only local publications."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop.runtime import wait_effects, wait_notifications, wait_reviews
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_wait_dispositions_192 import exited
from .test_wait_effects_192 import decide
from .test_wait_handoff_192 import handoff, registry_store, runtime, waiting

__all__ = ['handoff','registry_store','runtime','waiting']
pytestmark = [pytest.mark.unit,pytest.mark.asyncio]


@pytest.mark.parametrize('stage',['armed','notification','effect','negative'])
async def test_driver_uses_writers_only_for_its_actual_own_tree_publications(handoff,monkeypatch,stage):
    h,c = handoff,handoff.c
    await decide(h,'defer' if stage in ('armed','notification') else 'continue')
    if stage in ('armed','notification'):
        await wait_effects.apply(c.manager,c.main,h.event_id)
    if stage=='notification':
        c.now[0] += 1001
    if stage=='negative':
        await exited(h)
    write,one,many = c.manager._write,ControlTransaction.fetchone,ControlTransaction.fetchall
    footprints = []

    def note(tx,rows):
        tx.__dict__.setdefault('_wait_driver_reads',set()).update(
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
            jobs = tx.touched_jobs|tx.__dict__.get('_wait_driver_reads',set())
            footprints.append({roots[job] for job in jobs})

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',scoped)
        patch.setattr(ControlTransaction,'fetchone',fetchone)
        patch.setattr(ControlTransaction,'fetchall',fetchall)
        result = await wait_reviews.drive(h.loop,c.main,h.registration.job.job_id)
    assert result['status']=={'armed':'armed','notification':'enqueued','effect':'input_enqueued','negative':'owner_terminated_without_effect'}[stage]
    assert h.model.calls==2
    assert all(len(roots)<=1 for roots in footprints),footprints
    assert len(footprints)=={'armed':0,'notification':2,'effect':1,'negative':1}[stage],footprints


async def test_real_driver_death_between_due_and_notification_replays_without_another_model(handoff,tmp_path):
    h,c = handoff,handoff.c
    await decide(h,'defer')
    await wait_effects.apply(c.manager,c.main,h.event_id)
    c.now[0] += 1001
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine=='sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    marker = tmp_path/'due.ready'
    config = {'now':c.now[0],'actor':asdict(c.main),'job_id':h.registration.job.job_id}
    process = await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._wait_driver_tree_worker',
        dsn,c.manager.prefix,json.dumps(config),str(marker),
        cwd=Path(__file__).resolve().parents[2],stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        output,_ = await asyncio.wait_for(process.communicate(),30)
        assert process.returncode==87,output.decode()
        event_id = marker.read_text()
        assert event_id!=h.event_id
        key = wait_notifications.notification_id(c.manager,event_id)
        assert not await c.manager.db.fetchone(f"SELECT event_id FROM {c.manager.events} WHERE event_id=?",(key,))
        first = await wait_reviews.drive(h.loop,c.main,h.registration.job.job_id)
        events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
        assert await wait_reviews.drive(h.loop,c.main,h.registration.job.job_id)==first
        assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")==events
        assert first['event_id']==event_id and first['status']=='enqueued'
        [item] = [item for item in await c.store.inbox_items(c.main.session_id) if item['item_id']==key]
        assert item['status']=='pending' and h.model.calls==2
    finally:
        await stop_worker(process)
