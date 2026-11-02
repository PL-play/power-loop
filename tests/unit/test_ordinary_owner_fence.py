"""Ordinary messages obey every ancestor generation,not only the root/leaf."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from power_loop.runtime import workflow_resume
from power_loop.runtime.invocation import current_invocation
from power_loop.runtime.jobs import JobManager, _hash
from power_loop.runtime.store.factory import _make_database
from power_loop.runtime.task_runtime import _SEND, TaskRuntime

from .test_budget_policy_admission import wait_for_admission_lock
from .test_executor_drain_trees import stop_worker, wait_ready
from .test_job_registry_192 import registry_store
from .test_node_nudge_runtime_192 import client
from .test_workflow_task_runtime_192 import managed_workflow, settle

__all__ = ['registry_store', 'managed_workflow']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('warm', [False, True])
async def test_direct_execution_captures_write_proofs_for_cold_and_warm_appends(registry_store, warm):
    store, _ = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id='direct-writer', principal_id='host', executor_id='e', boot_id='b')
    sid = await store.create_session()
    actor = runtime.actor(sid)
    registration = await manager.register_job(actor, kind='send', session_id=sid, system_key='direct')
    binding = await runtime.claim(actor, registration)
    async with runtime.execution(binding):
        if warm:
            await store.get_state(sid)
        assert binding.resource_evidence['_message_write_proofs'] == ((binding.lease.job_id, 1, 'e', 'b'),)
        seq = await store.append_message(sid, role='assistant', content='valid direct owner')
        assert seq == 1
    assert [row.content for row in await store.load_all_messages(sid)] == ['valid direct owner']


@pytest.mark.parametrize('overlap', [False, True])
async def test_workflow_two_fences_a_real_old_leaf_one_root_one_process(
        managed_workflow, monkeypatch, tmp_path, overlap):
    c = await client(managed_workflow, monkeypatch)
    c.state['on_complete'] = None
    engine = c.manager.db.dialect.name
    dsn = next(row['file'] for row in await c.manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main') \
        if engine == 'sqlite' else os.environ['POWER_LOOP_TASK_PG_DSN' if engine == 'postgres' else 'POWER_LOOP_TASK_MYSQL_DSN']
    observer = await _make_database(dsn)
    complete = c.model.complete
    processes, captured = [], {}

    async def capture(request, **kwargs):
        binding = _SEND.get()
        if 'ROLE-BUILD' in (request.system_prompt or '') and not captured:
            source = binding.resource_evidence['_control_source']
            ids = binding.resource_evidence['_control_ancestors']
            rows = await c.manager.db.fetchall(f'SELECT job_id,generation,executor_id,boot_id '
                f'FROM {c.manager.jobs} WHERE job_id IN ('+','.join('?' for _ in ids)+')', tuple(ids))
            by_id = {row['job_id']:row for row in rows}
            proofs = [(job, *(by_id[job][key] for key in ('generation','executor_id','boot_id'))) for job in ids]
            captured.update(job_id=source['job_id'], sid=binding.actor.session_id, root=source['root_job_id'], proofs=proofs)
            config = dict(actor=asdict(binding.actor), lease=asdict(binding.lease), proofs=proofs,
                send_index=binding.resource_evidence['_control_create_request']['send_index'])
            process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'tests.unit._ordinary_owner_worker',
                dsn, c.store.table_prefix, json.dumps(config), str(tmp_path),
                cwd=Path(__file__).resolve().parents[2], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            processes.append(process)
            await wait_ready(process, tmp_path/'owner.ready')
        return await complete(request, **kwargs)

    monkeypatch.setattr(c.model, 'complete', capture)
    commit = asyncio.Event()
    changed = asyncio.Event()
    issuer = waiter = None
    try:
        assert (await c.loop.send('launch', c.parent)).status == 'completed'
        await asyncio.wait_for(c.model.entered['BUILD'].wait(), 5)
        await settle(managed_workflow)
        assert (await c.loop.flush_follow_ups(c.parent)).status == 'completed'
        [process] = processes
        (tmp_path/'owner.probe').touch()
        await wait_ready(process, tmp_path/'owner.probed')
        prior_state = await c.store.get_state(captured['sid'])
        assert json.loads((tmp_path/'owner.probed').read_text())['next_seq'] == prior_state.next_seq
        workflow = c.state['handles'][0].job_id
        plan = await c.manager.get_workflow_restore_plan(c.runtime.actor(c.parent), workflow,
            generation=1, node_id='BUILD', message='compute again')
        write = c.manager._write

        @asynccontextmanager
        async def paused_handoff():
            async with write() as tx:
                execute = tx._execute
                async def statement(sql, params=()):
                    result = await execute(sql, params)
                    if overlap and sql.startswith(f'UPDATE {c.manager.jobs} SET generation=') and params[-4] == workflow:
                        changed.set()
                        await asyncio.wait_for(commit.wait(), 10)
                    return result
                tx._execute = statement
                yield tx

        async def restore():
            source = _SEND.get()
            seq, call_id = current_invocation(c.parent)
            op, lease = await workflow_resume.admit(c.manager, c.runtime.actor(c.parent), workflow,
                caller=source.lease, call_ref=await c.store.message_ref(c.parent, seq), tool_call_id=call_id,
                generation=1, node_id='BUILD', message='compute again', executor_id=c.runtime.executor_id,
                boot_id=c.runtime.boot_id, lease_ms=c.runtime.lease_ms, plan_hash=_hash(plan))
            assert lease.generation == 2
            return {'operation_id':op.operation_id,'generation':2}

        with monkeypatch.context() as patch:
            patch.setattr(c.manager, '_write', paused_handoff)
            issuer = asyncio.create_task(c.issue(handler=restore))
            if overlap:
                await asyncio.wait_for(changed.wait(), 10)
                (tmp_path/'owner.append').touch()
                await wait_ready(process, tmp_path/'owner.appending')
                async def done():
                    await wait_ready(process, tmp_path/'owner.done')
                waiter = asyncio.create_task(done())
                if engine != 'sqlite':
                    await wait_for_admission_lock(c.manager, observer, waiter)
                else:
                    await asyncio.sleep(.05)
                    assert not waiter.done()
                commit.set()
            assert (await asyncio.wait_for(issuer, 15)).status == 'completed'
        generations = await c.manager.db.fetchall(f'SELECT job_id,generation FROM {c.manager.jobs} '
            'WHERE job_id IN (?,?,?)', (workflow, captured['job_id'], captured['root']))
        assert {row['job_id']:row['generation'] for row in generations} == {
            workflow:2, captured['job_id']:1, captured['root']:1}
        (tmp_path/'owner.append').touch()
        stdout, _ = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 0, stdout.decode()
        assert json.loads((tmp_path/'owner.done').read_text()) == {'error':'SessionLeaseLost'}
        assert await c.store.get_state(captured['sid']) == prior_state
        assert all(row.content != 'stale-process' for row in await c.store.load_all_messages(captured['sid']))
    finally:
        commit.set()
        for path in ('owner.probe','owner.append'):
            (tmp_path/path).touch()
        for task in (issuer, waiter):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        for process in processes:
            await stop_worker(process)
        await observer.close()
