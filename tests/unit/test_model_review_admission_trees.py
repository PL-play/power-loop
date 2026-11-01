"""A new model's control admission does not inspect foreign review trees."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop.runtime import abort_outcomes, control_runtime, delegation
from power_loop.runtime.controller_handoff import transfer_review
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_controller_handoff_192 import new_main
from .test_executor_drain_trees import stop_worker
from .test_job_registry_192 import registry_store
from .test_original_owner_review_192 import original_owner_case, wait_owned

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit,pytest.mark.asyncio]


@asynccontextmanager
async def _review_setup(registry_store,kind,*,transfer):
    c = await original_owner_case(registry_store,kind,allowed_calls=1)
    initial = set(delegation._TASKS)
    loop = runtime = None
    try:
        await c.loop.send('Launch original work once.',c.state.root)
        await wait_owned(c,kind,initial)
        await c.runtime.shutdown(timeout_s=3)
        loop,runtime = await new_main(c,registry_store)
        if transfer:
            await transfer_review(loop,runtime.actor(c.state.root),c.state.source)
        yield c,loop,runtime
    finally:
        c.state.release.set()
        if runtime is not None:
            await runtime.shutdown(timeout_s=3)
            await loop.aclose(drain_timeout_s=.1)
        await c.runtime.shutdown(timeout_s=3)
        await c.loop.aclose(drain_timeout_s=.1)


@pytest.fixture(params=['subagent','workflow_node'])
async def review_case(registry_store,request):
    async with _review_setup(registry_store,request.param,transfer=True) as case:
        yield case


@pytest.fixture(params=['subagent','workflow_node'])
async def untransferred_review_case(registry_store,request):
    async with _review_setup(registry_store,request.param,transfer=False) as case:
        yield case


async def test_new_model_admission_keeps_original_review_proof_outside_its_writer(review_case,monkeypatch):
    c,loop,runtime = review_case
    manager = runtime.manager
    write,one,many = manager._write,ControlTransaction.fetchone,ControlTransaction.fetchall
    footprints = []

    def note(tx,rows):
        tx.__dict__.setdefault('_model_review_reads',set()).update(
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
            rows = await tx._tx.fetchall(f"SELECT job_id,root_job_id FROM {manager.jobs}")
            roots = {row['job_id']:row['root_job_id'] for row in rows}
            jobs = tx.touched_jobs|tx.__dict__.get('_model_review_reads',set())
            footprints.append({roots[job] for job in jobs})

    before = await manager.get_job(runtime.actor(c.state.root),c.state.source)
    with monkeypatch.context() as patch:
        patch.setattr(manager,'_write',scoped)
        patch.setattr(ControlTransaction,'fetchone',fetchone)
        patch.setattr(ControlTransaction,'fetchall',fetchall)
        result = await loop.send('Handle the original negative result without repeating work.',c.state.root,tools=[])
    after = await manager.get_job(runtime.actor(c.state.root),c.state.source)
    assert result.status=='completed' and after.result_status=='handled'
    assert after.result==before.result and not c.state.effects and c.state.counts['BUILD']==1
    assert footprints and all(len(scope)<=1 for scope in footprints),footprints


async def test_paused_original_review_proof_allows_a_third_tree_to_stop(review_case,registry_store,monkeypatch):
    c,loop,runtime = review_case
    manager,actor = runtime.manager,runtime.actor(c.state.root)
    other = JobManager(await registry_store[1](),now_ms=manager.now)
    third = await other.register_job(actor,kind='send',session_id=actor.session_id,system_key='third-review-source-root')
    pending = abort_outcomes.pending_reviews
    ready,release = asyncio.Event(),asyncio.Event()

    async def held(current,reader,actual_actor):
        assert current._control_intents.get() is None
        result = await pending(current,reader,actual_actor)
        if current is manager:
            ready.set()
            await release.wait()
        return result

    monkeypatch.setattr(abort_outcomes,'pending_reviews',held)
    action = asyncio.create_task(loop.send('Handle the retained result once.',c.state.root,tools=[]))
    try:
        await asyncio.wait_for(ready.wait(),3)
        await asyncio.wait_for(other.request_cancel(actor,third.job.job_id,system_key='stop-third-review-source-root',reason='test'),3)
    finally:
        release.set()
        result = await asyncio.wait_for(action,10)
    assert result.status=='completed'
    assert (await other.get_job(actor,third.job.job_id)).cancel_requested_at is not None
    assert c.state.counts['BUILD']==1 and not c.state.effects


async def test_new_handoff_between_review_proof_and_admission_reprepares_without_a_second_send(untransferred_review_case,monkeypatch):
    c,loop,runtime = untransferred_review_case
    manager,actor = runtime.manager,runtime.actor(c.state.root)
    prepare,open_control = control_runtime.prepare_review_sources,control_runtime.open_control
    snapshots,admitted = [],[]
    before = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='send' AND parent_job_id IS NULL AND session_id=?",(actor.session_id,))

    async def transferred(current,actual_actor):
        snapshot = await prepare(current,actual_actor)
        snapshots.append(snapshot)
        if len(snapshots)==1:
            await transfer_review(loop,actor,c.state.source)
        return snapshot

    async def observed(binding,**kwargs):
        already_open = binding.resource_evidence.get('_control_state') is not None
        state = await open_control(binding,**kwargs)
        if not already_open:
            admitted.append(binding.resource_evidence['_control_review_pending'])
        return state

    monkeypatch.setattr(control_runtime,'prepare_review_sources',transferred)
    monkeypatch.setattr(control_runtime,'open_control',observed)
    result = await loop.send('Handle the original outcome once;perform no business work.',actor.session_id,tools=[])
    after = await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='send' AND parent_job_id IS NULL AND session_id=?",(actor.session_id,))
    assert result.status=='completed' and len(snapshots)==2
    assert snapshots[0].version!=snapshots[1].version and snapshots[1].pending
    assert len(after)==len(before)+1
    assert admitted==[True]
    assert c.state.counts['BUILD']==1 and not c.state.effects


async def test_real_review_preparation_death_keeps_originals_and_recovery_never_repeats_business_work(review_case,tmp_path):
    c,loop,runtime = review_case
    manager,actor = runtime.manager,runtime.actor(c.state.root)
    if manager.db.dialect.name=='sqlite':
        rows = await manager.db.fetchall('PRAGMA database_list')
        dsn = next(row['file'] for row in rows if row['name']=='main')
    else:
        dsn = os.environ['POWER_LOOP_TASK_PG_DSN' if manager.db.dialect.name=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    before = await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id")
    marker = tmp_path/'review-source.ready'
    process = await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._model_review_source_worker',
        dsn,manager.prefix,json.dumps(asdict(actor)),str(marker),cwd=Path(__file__).resolve().parents[2],
        stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        output,_ = await asyncio.wait_for(process.communicate(),30)
        assert process.returncode==91,output.decode()
        assert json.loads(marker.read_text())['pending']
        assert await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id")==before
        result = await loop.send('Handle the original negative result after recovery.',actor.session_id,tools=[])
        assert result.status=='completed'
        assert (await manager.get_job(actor,c.state.source)).result_status=='handled'
        assert c.state.counts['BUILD']==1 and not c.state.effects
    finally:
        await stop_worker(process)
