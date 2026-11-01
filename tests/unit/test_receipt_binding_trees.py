"""Attaching an existing raw receipt never holds its caller and target trees."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.session_controls import ControlTransaction

from .test_executor_drain_trees import stop_worker
from .test_job_registry_192 import registry_store
from .test_stop_request_trees import cross_control

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit,pytest.mark.asyncio]


@pytest.fixture
async def receipt_case(registry_store):
    case = await cross_control(*registry_store,now=lambda:1_000_000)
    case.operation,_ = await case.stop()
    seq = await case.store.append_message(case.actor.session_id,role='tool',
        tool_call_id='stop-old-child',content='accepted',send_index=2,round_index=0)
    case.receipt = await case.store.message_ref(case.actor.session_id,seq)
    return case


@pytest.mark.parametrize('port',['live','historical'])
async def test_receipt_binding_uses_only_its_target_tree_writer(receipt_case,monkeypatch,port):
    c = receipt_case
    rows = await c.manager.db.fetchall(f"SELECT job_id,root_job_id FROM {c.manager.jobs}")
    roots = {row['job_id']:row['root_job_id'] for row in rows}
    write,one,many = c.manager._write,ControlTransaction.fetchone,ControlTransaction.fetchall
    footprints = []

    def note(tx,rows):
        tx.__dict__.setdefault('_receipt_reads',set()).update(
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
        jobs = tx.touched_jobs|tx.__dict__.get('_receipt_reads',set())
        footprints.append({roots[job] for job in jobs})

    before = await c.manager.get_job(c.actor,c.target.job.job_id)
    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',scoped)
        patch.setattr(ControlTransaction,'fetchone',fetchone)
        patch.setattr(ControlTransaction,'fetchall',fetchall)
        if port=='live':
            result = await c.manager.bind_receipt(c.actor,c.operation.operation_id,c.receipt,caller=c.caller)
        else:
            result = await c.manager.reconcile_receipt(c.actor,c.operation.operation_id,c.receipt)
    assert result.receipt_ref==c.receipt.to_dict()
    assert await c.manager.get_job(c.actor,c.target.job.job_id)==before
    assert footprints and all(len(scope)<=1 for scope in footprints),footprints


async def test_live_receipt_source_proof_is_readonly(receipt_case,monkeypatch):
    c = receipt_case
    fence,seen = c.manager._fence,[]

    async def outside(reader,actor,lease):
        assert c.manager._control_intents.get() is None
        seen.append(lease.job_id)
        return await fence(reader,actor,lease)

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_fence',outside)
        result = await c.manager.bind_receipt(c.actor,c.operation.operation_id,c.receipt,caller=c.caller)
    assert seen==[c.caller.job_id] and result.receipt_ref==c.receipt.to_dict()


@pytest.mark.parametrize('close_source',[False,True])
async def test_paused_receipt_source_proof_allows_another_root_to_stop(receipt_case,monkeypatch,close_source):
    c = receipt_case
    other = JobManager(await c.connect(),now_ms=c.manager.now)
    third = await other.register_job(c.actor,kind='send',session_id=c.actor.session_id,system_key='third-receipt-root')
    fence = c.manager._fence
    entered,release = asyncio.Event(),asyncio.Event()

    async def paused(reader,actor,lease):
        result = await fence(reader,actor,lease)
        entered.set()
        await release.wait()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_fence',paused)
        action = asyncio.create_task(c.manager.bind_receipt(c.actor,c.operation.operation_id,c.receipt,caller=c.caller))
        try:
            await asyncio.wait_for(entered.wait(),3)
            await asyncio.wait_for(other.request_cancel(c.actor,third.job.job_id,system_key='stop-third-receipt-root',reason='test'),3)
            if close_source:
                await asyncio.wait_for(other.close_model_execution(c.actor,c.caller),3)
        finally:
            release.set()
            result = await asyncio.wait_for(action,5)
    assert result.receipt_ref==c.receipt.to_dict()
    assert (await other.get_job(c.actor,third.job.job_id)).cancel_requested_at is not None
    if close_source:
        assert (await other.get_job(c.actor,c.caller.job_id)).status=='model_closed'
        with pytest.raises(JobError,match='closed|not_running'):
            await other.register_job(c.actor,kind='background_tool',parent=c.caller,system_key='receipt-is-not-permission')


async def test_stale_live_source_fails_before_a_target_writer(receipt_case,monkeypatch):
    c = receipt_case

    @asynccontextmanager
    async def forbidden():
        pytest.fail('stale source must be rejected before entering a target writer')
        yield

    with monkeypatch.context() as patch:
        patch.setattr(c.manager,'_write',forbidden)
        with pytest.raises(JobError,match='stale_execution'):
            await c.manager.bind_receipt(c.actor,c.operation.operation_id,c.receipt,
                caller=replace(c.caller,generation=c.caller.generation-1))
    assert (await c.manager.get_operation(c.actor,c.operation.operation_id)).receipt_ref is None


@pytest.mark.parametrize('stage',['prepared','receipt_update','before_commit','after_commit'])
async def test_real_receipt_binding_death_preserves_the_raw_fact(receipt_case,tmp_path,stage):
    c = receipt_case
    engine = c.manager.db.dialect.name
    dsn = str(tmp_path/'registry.db') if engine=='sqlite' else os.environ[
        'POWER_LOOP_TASK_PG_DSN' if engine=='postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    config = {'now':c.manager.now(),'actor':asdict(c.actor),'caller':asdict(c.caller),
        'operation_id':c.operation.operation_id,'receipt':c.receipt.to_dict()}
    marker = tmp_path/'receipt.ready'
    before = await c.manager.get_job(c.actor,c.target.job.job_id)
    process = await asyncio.create_subprocess_exec(sys.executable,'-m','tests.unit._receipt_binding_tree_worker',
        dsn,c.manager.prefix,json.dumps(config),stage,str(marker),
        cwd=Path(__file__).resolve().parents[2],stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        output,_ = await asyncio.wait_for(process.communicate(),30)
        assert process.returncode==88,output.decode()
        assert marker.read_text()==stage
        operation = await c.manager.get_operation(c.actor,c.operation.operation_id)
        assert operation.receipt_ref==(c.receipt.to_dict() if stage=='after_commit' else None)
        assert await c.store.message_ref(c.actor.session_id,c.receipt.seq)==c.receipt
        cold = JobManager(await c.connect(),now_ms=c.manager.now)
        await cold.close_model_execution(c.actor,c.caller)
        bound = await cold.reconcile_receipt(c.actor,c.operation.operation_id,c.receipt)
        events = await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")
        assert await cold.reconcile_receipt(c.actor,c.operation.operation_id,c.receipt)==bound
        assert await cold.db.fetchall(f"SELECT * FROM {cold.events} ORDER BY event_id")==events
        assert bound.receipt_ref==c.receipt.to_dict()
        assert await cold.get_job(c.actor,c.target.job.job_id)==before
    finally:
        await stop_worker(process)
