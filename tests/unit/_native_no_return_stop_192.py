"""An admitted original child stays controllable without a callback receipt."""
import asyncio
import json
import os
import sys
from dataclasses import asdict
from urllib.parse import urlparse

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime import delegation
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.pending_calls import recover_runtime
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.runtime.tool_contracts import RuntimeTool
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent


async def exercise(registry_store, model, record_property, *, target='child', real=False):
    store, _ = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id='no-return', principal_id='host',
                          executor_id='original-worker', boot_id='original-boot', lease_ms=1000)
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy='single_level', max_rounds=3, max_tokens=2048))
    sid = await loop.new_session()
    if real:
        model.root_sid, model.hold_children = sid, True
        entered, release = model.child_entered, model.release_children
    else:
        model.root = sid
        entered, release = model.entered, model.release
    original = registry.get('spawn_agent')
    callbacks = []

    class LostReturn(RuntimeTool):
        @property
        def runtime_contract(self):
            return original.handler.runtime_contract

        def dispatch_policy(self, arguments, *, runtime=None):
            return original.handler.dispatch_policy(arguments, runtime=runtime)

        async def __call__(self, **arguments):
            accepted = await original.handler(**arguments)
            callbacks.append(accepted)
            await asyncio.wait_for(entered.wait(), 10)
            # The actual original spawn has committed and started the child.
            # Interrupt before returning any string to the parent pipeline.
            raise asyncio.CancelledError('original callback interrupted before return')

    registry.register(original.definition, LostReturn(), overwrite=True)
    before = set(delegation._TASKS)
    process = None
    try:
        with pytest.raises(asyncio.CancelledError):
            await loop.send('Use spawn_agent exactly once to calculate 19+23. Set tools=[] and ask '
                            'the child to return only the integer. Do not use any other tools.', sid)
        await runtime.flush_control_checkpoints()
        actor = runtime.actor(sid)
        rows = await manager.db.fetchall(f'SELECT job_id,kind FROM {manager.jobs} WHERE created_by_session_id=?', (sid,))
        children = [row['job_id'] for row in rows if row['kind'] == 'subagent']
        assert len(children) == len(callbacks) == 1
        child_id = children[0]
        child = await manager.get_job(actor, child_id)
        root_id = child.parent_job_id
        pending = (await store.get_state(sid)).pending
        raw = await manager.db.fetchall(f'SELECT * FROM {store.t.messages} WHERE session_id=? ORDER BY seq', (sid,))
        assert pending and not any(r['role'] == 'tool' for r in raw)
        assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='native_callback_returned'", (root_id,))
        assert child.resource_state == 'running'
        tasks = set(delegation._TASKS) - before
        assert len(tasks) == 1 and all(not task.done() for task in tasks)
        calls = model.calls if real else model.child_calls
        if manager.db.dialect.name == 'sqlite':
            dsn = next(row['file'] for row in await manager.db.fetchall('PRAGMA database_list') if row['name'] == 'main')
        else:
            dsn = os.environ['POWER_LOOP_TASK_PG_DSN']
            assert urlparse(dsn).path.endswith('_test')
        code = '''import asyncio,json,sys
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.jobs import JobActor,JobManager
async def main():
 d=json.load(sys.stdin)
 store=await open_store(d['dsn'],table_prefix=d['prefix'],schema='tasks')
 try:
  manager=JobManager(store);actor=JobActor(**d['actor'])
  child=await manager.get_job(actor,d['child'])
  assert child.parent_job_id==d['root'] and child.resource_state=='running'
  first=await manager.request_cancel(actor,d['target'],system_key='original-no-return-stop',reason='Stop original admitted work')
  second=await manager.request_cancel(actor,d['target'],system_key='original-no-return-stop',reason='Stop original admitted work')
  assert first.operation_id==second.operation_id
  print(json.dumps({'child':child.job_id,'operation_id':first.operation_id}))
 finally: await store.close()
asyncio.run(main())'''
        process = await asyncio.create_subprocess_exec(sys.executable, '-c', code,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        record_property('original_no_return_stop_pid', process.pid)
        output, error = await asyncio.wait_for(process.communicate(json.dumps({'dsn': dsn,
            'prefix': manager.prefix, 'actor': asdict(actor), 'child': child_id, 'root': root_id,
            'target': root_id if target == 'root' else child_id}).encode()), 30)
        assert process.returncode == 0, error[-1000:]
        assert json.loads(output)['child'] == child_id
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
        await runtime.flush_control_checkpoints()
        child = await manager.get_job(actor, child_id)
        # Actual local task exit is proven below. A cancelled model attempt
        # has no provider completion proof, so resource certainty stays unknown.
        assert child.resource_state == 'unknown' and child.outcome == 'cancelled', child
        assert child.supervision_status == 'open'
        with pytest.raises(JobError):
            await manager.finalize_supervision(actor, child_id, generation=1)
        assert all(t.done() for t in tasks) and not release.is_set()
        assert len(callbacks) == 1 and (model.calls if real else model.child_calls) == calls
        assert (await recover_runtime(runtime, actor, root_id))['state'] == 'unconfirmed'
        assert (await store.get_state(sid)).pending == pending
        assert (await manager.get_job(actor, root_id)).supervision_status == 'open'
        with pytest.raises(JobError):
            await manager.finalize_supervision(actor, root_id, generation=1)
        assert await manager.db.fetchall(f'SELECT * FROM {store.t.messages} WHERE session_id=? ORDER BY seq', (sid,)) == raw
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        release.set()
        tasks = set(delegation._TASKS) - before
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
        await runtime.flush_control_checkpoints()
        await loop.aclose(drain_timeout_s=.1)
