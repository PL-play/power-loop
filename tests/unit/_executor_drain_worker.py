"""Owned process for durable drain boundaries; only dedicated _test servers."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from power_loop import AgentLoopConfig, LLMResponse, LLMService, StatefulAgentLoop
from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime


async def main():
    dsn, prefix, tenant, principal, executor, boot, sid, clock, mode, ready, release, output = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    store = await open_store(dsn, table_prefix=prefix, schema='compatible')
    manager = JobManager(store, now_ms=lambda: int(clock))
    identity = registry.ExecutorIdentity(tenant, principal, executor, boot)
    try:
        if mode == 'new-send':
            class Model(LLMService):
                calls = 0
                async def complete(self, request, **kwargs):
                    self.calls += 1
                    return LLMResponse(raw_text='must not enter the model')
                def stream(self, request):
                    raise AssertionError('nonstream fixture')
                async def close(self):
                    pass
            model = Model()
            # This boot was registered by the parent host. Task admission must
            # respect its durable state without replaying host bootstrap here.
            runtime = TaskRuntime(manager, tenant_id=tenant, principal_id=principal,
                                  executor_id=executor, boot_id=boot)
            loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime,
                config=AgentLoopConfig(delegation_policy='single_level', max_rounds=2, compactor=None))
            try:
                try:
                    result = await loop.send('new send during drain', sid)
                    packet = {'status': result.status, 'result': str(result), 'calls': model.calls}
                except JobError as exc:
                    packet = {'error': exc.code, 'calls': model.calls}
                Path(output).write_text(json.dumps(packet))
            finally:
                await loop.aclose(drain_timeout_s=.1)
            return
        if mode in ('opposite-forward', 'opposite-reverse'):
            request = {'action': 'drain', 'reason': 'shutdown'}
            operation = registry._operation_key(manager, identity, 'drain', 'process-stop')
            await registry._drain_barrier(manager, identity, operation, request)
            scope = await registry._drain_scope(manager, identity, operation, request)
            rows = await manager.db.fetchall(f'SELECT job_id,root_job_id,created_by_session_id FROM {manager.jobs}')
            jobs = {row['job_id']: row for row in rows}
            creators = {row['job_id']: row['created_by_session_id'] for row in rows}
            trees = {}
            for target in scope['effects']['executions']:
                trees.setdefault(jobs[target['job_id']]['root_job_id'], []).append(target)
            order = sorted(trees, reverse=mode == 'opposite-reverse')
            assert len(order) == 2
            peer = Path(ready).with_name(
                ('opposite-reverse' if mode == 'opposite-forward' else 'opposite-forward') + '.ready')
            for index, root_id in enumerate(order):
                stops, incomplete = await registry._drain_tree(
                    manager, identity, operation, 'shutdown', trees[root_id], creators)
                assert not incomplete and len(stops) == 1
                if index == 0:
                    Path(ready).write_text(json.dumps({'root': root_id, 'stops': stops}))
                    # No transaction remains open across the opposite tree's
                    # first commit. A multi-tree outer lock would deadlock here.
                    async with asyncio.timeout(15):
                        while not peer.exists():
                            await asyncio.sleep(.01)
            receipt = await registry.drain(manager, identity, system_key='process-stop', reason='shutdown')
            Path(output).write_text(json.dumps({'receipt': receipt, 'tree_order': order}))
            return
        original = registry._drain_tree
        calls = 0
        async def tree(*args, **kwargs):
            nonlocal calls
            if mode == 'hold-before-first' and calls == 0:
                Path(ready).write_text(json.dumps({'state': 'barrier-and-scope-committed'}))
                while not Path(release).exists():
                    await asyncio.sleep(.01)
            result = await original(*args, **kwargs)
            calls += 1
            if mode == 'crash-after-first':
                Path(ready).write_text(json.dumps({'state': 'first-tree-committed', 'stop_ids': result[0]}))
                os._exit(73)
            return result
        registry._drain_tree = tree
        receipt = await registry.drain(manager, identity, system_key='process-stop', reason='shutdown')
        Path(output).write_text(json.dumps(receipt))
    finally:
        await store.close()


asyncio.run(main())
