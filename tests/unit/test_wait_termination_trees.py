"""Negative-owner originals are prepared before a logical wait's own writer."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop.runtime import wait_terminal
from power_loop.runtime.jobs import JobLease, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_wait_dispositions_192 import exited
from .test_wait_effects_192 import decide
from .test_wait_handoff_192 import handoff, registry_store, runtime, waiting

__all__ = ['handoff','registry_store','runtime','waiting']
pytestmark = [pytest.mark.unit,pytest.mark.asyncio]


@pytest.mark.parametrize('choice',['continue','defer','cancel'])
async def test_wait_termination_uses_one_tree_per_writer(handoff,monkeypatch,choice):
    h,c = handoff,handoff.c
    await decide(h,choice)
    await exited(h)
    write,one,many = c.manager._write,ControlTransaction.fetchone,ControlTransaction.fetchall
    footprints = []

    def note(tx,rows):
        tx.__dict__.setdefault('_termination_reads',set()).update(
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
            jobs = tx.touched_jobs|tx.__dict__.get('_termination_reads',set())
            footprints.append({roots[job] for job in jobs})

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',scoped)
        patch.setattr(ControlTransaction,'fetchone',fetchone)
        patch.setattr(ControlTransaction,'fetchall',fetchall)
        reply = await wait_terminal.terminate(c.manager,c.main,h.registration.job.job_id)
    assert reply['status']=='terminated' and h.model.calls==2
    assert footprints and all(len(roots)<=1 for roots in footprints),footprints


async def ready(h,choice='continue'):
    await decide(h,choice)
    await exited(h)


async def test_complete_termination_replay_is_readonly(handoff,monkeypatch):
    h,c = handoff,handoff.c
    await ready(h)
    result = await wait_terminal.terminate(c.manager,c.main,h.registration.job.job_id)

    @asynccontextmanager
    async def forbidden():
        pytest.fail('complete original terminal facts must not reacquire a writer')
        yield

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',forbidden)
        assert await wait_terminal.terminate(c.manager,c.main,h.registration.job.job_id)==result
    assert h.model.calls==2


async def test_owner_exit_proof_and_disposition_are_prepared_outside_the_writer(handoff,monkeypatch):
    h,c = handoff,handoff.c
    await ready(h)
    prepare,seen = wait_terminal._prepare_termination,[]

    async def outside(manager,reader,actor,job_id,prepared):
        assert manager._control_intents.get() is None
        seen.append(job_id)
        return await prepare(manager,reader,actor,job_id,prepared)

    with monkeypatch.context() as patch:
        patch.setattr(wait_terminal,'_prepare_termination',outside)
        await wait_terminal.terminate(c.manager,c.main,h.registration.job.job_id)
    assert seen==[h.registration.job.job_id] and h.model.calls==2


async def test_slow_exit_originals_do_not_block_an_unrelated_root(handoff,monkeypatch):
    h,c = handoff,handoff.c
    await ready(h)
    other = JobManager(await c.connect(),now_ms=c.manager.now)
    third = await other.register_job(c.main,kind='send',session_id=c.main.session_id,system_key='third-root')
    prepare = wait_terminal._prepare_termination
    entered,release = asyncio.Event(),asyncio.Event()

    async def pause(manager,reader,actor,job_id,prepared):
        assert manager._control_intents.get() is None
        entered.set()
        await release.wait()
        return await prepare(manager,reader,actor,job_id,prepared)

    with monkeypatch.context() as patch:
        patch.setattr(wait_terminal,'_prepare_termination',pause)
        action = asyncio.create_task(wait_terminal.terminate(c.manager,c.main,h.registration.job.job_id))
        try:
            await asyncio.wait_for(entered.wait(),3)
            await asyncio.wait_for(other.request_cancel(c.main,third.job.job_id,system_key='third-stop',reason='test'),3)
            assert (await other.get_job(c.main,third.job.job_id)).cancel_requested_at is not None
        finally:
            release.set()
            await asyncio.wait_for(action,5)
    assert h.model.calls==2


async def test_actual_same_tree_change_reprepares_before_termination(handoff,monkeypatch):
    h,c = handoff,handoff.c
    await ready(h)
    publish,attempts = wait_terminal._publish_snapshot,[]
    root = await c.manager._row(c.manager.db,c.root.job.job_id)
    lease = JobLease(root['job_id'],root['executor_id'],root['boot_id'],root['generation'])

    async def changed(manager,tree):
        attempts.append(tree)
        if len(attempts)==1:
            sibling = await manager.register_job(c.main,kind='background_tool',parent=lease,system_key='sibling-after-plan')
            await manager.request_cancel(c.main,sibling.job.job_id,system_key='sibling-stop',reason='test')
        return await publish(manager,tree)

    with monkeypatch.context() as patch:
        patch.setattr(wait_terminal,'_publish_snapshot',changed)
        await wait_terminal.terminate(c.manager,c.main,h.registration.job.job_id)
    assert len(attempts)==2 and attempts[0].version_sum!=attempts[1].version_sum
    assert h.model.calls==2


@pytest.mark.parametrize('stage',['prepared','first_write','before_commit','after_commit'])
@pytest.mark.parametrize('choice',['continue','defer','cancel'])
async def test_real_termination_death_preserves_original_exit_and_never_executes_intent(handoff,tmp_path,stage,choice):
    h,c = handoff,handoff.c
    await ready(h,choice)
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine=='sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    config = {'now':c.now[0],'actor':asdict(c.main),'job_id':h.registration.job.job_id}
    marker = tmp_path/'termination.ready'
    before = await c.manager.get_job(c.main,c.owner.job_id)
    process = await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._wait_termination_tree_worker',
        dsn,c.manager.prefix,json.dumps(config),stage,str(marker),
        cwd=Path(__file__).resolve().parents[2],stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        output,_ = await asyncio.wait_for(process.communicate(),30)
        assert process.returncode==86,output.decode()
        assert marker.read_text()==stage
        assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='wait_terminated'"))==int(stage=='after_commit')
        assert await c.manager.get_job(c.main,c.owner.job_id)==before
        cold = JobManager(await c.connect(),now_ms=c.manager.now)
        result = await wait_terminal.terminate(cold,c.main,h.registration.job.job_id)
        after = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await wait_terminal.terminate(cold,c.main,h.registration.job.job_id)==result
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")==after
        assert not await cold.db.fetchall(f"SELECT event_id FROM {cold.events} WHERE event_type='wait_effect'")
        assert not await c.store.inbox_items(c.sid)
        assert await cold.get_job(c.main,c.owner.job_id)==before and h.model.calls==2
    finally:
        await stop_worker(process)
