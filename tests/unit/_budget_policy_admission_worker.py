"""Pause a real send's entry transaction without the former store mutex."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.runtime import control_runtime
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.response_capture import ResponseCaptureReceipt
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime

from .test_model_budgets_192 import Host


async def wait_file(path):
    async with asyncio.timeout(20):
        while not path.exists():
            await asyncio.sleep(.01)


def make_loop(store, *, executor, budgeted, model_release=None, model_ready=None, registered=True):
    class Model(LLMService):
        calls = 0

        async def complete(self, request, **kwargs):
            self.calls += 1
            if model_ready is not None:
                model_ready.write_text(json.dumps({'pid': os.getpid(), 'calls': self.calls}))
                await wait_file(model_release)
            return LLMResponse(raw_text='done', token_usage=LLMTokenUsage(
                prompt_tokens=3, completion_tokens=2, total_tokens=5))

        def stream(self, request):
            raise AssertionError('nonstream fixture')

        async def close(self):
            pass

    model, host = Model(), Host()
    host.service = model

    async def capture(packet):
        return ResponseCaptureReceipt(packet.capture_id, packet.fingerprint, 'fixture-original')

    host.capture_response = capture
    config = BudgetConfig('first-policy', (LocalBudgetScope('shared', max_calls=10,
        max_tokens=10000, max_inflight=10),), host, max_pending=16)
    runtime = TaskRuntime(JobManager(store), tenant_id='policy-admission', principal_id='host',
        executor_id=executor, boot_id='boot', boot_contract_hash='a'*64 if registered else None,
        budget_config=config if budgeted else None)
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime,
        config=AgentLoopConfig(compactor=None, delegation_policy='single_level', max_tokens=20))
    return loop, runtime, model, host


async def main():
    dsn, prefix, sid, paid, directory, *extra = sys.argv[1:]
    mode = extra[0] if extra else 'send'
    registered = mode != 'send-unregistered'
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    directory = Path(directory)
    store = await open_store(dsn, table_prefix=prefix, schema='compatible')
    loop, runtime, model, host = make_loop(store, executor='worker', budgeted=paid == '1',
        model_release=directory/'model.release', model_ready=directory/'model.ready', registered=registered)
    await runtime.ensure_started()
    opening_entry = False

    @asynccontextmanager
    async def native():
        nonlocal opening_entry
        async with runtime.manager.db.transaction() as tx:
            yield tx
        if opening_entry and mode == 'crash-after-commit':
            opening_entry = False
            (directory/'entry.committed').write_text(json.dumps({'pid': os.getpid()}))
            await wait_file(directory/'commit.release')

    runtime.manager._control_transaction = native
    opening = control_runtime.open_control

    async def hold(*args, **kwargs):
        nonlocal opening_entry
        value = await opening(*args, **kwargs)
        opening_entry = True
        (directory/'entry.ready').write_text(json.dumps({'pid': os.getpid(), 'budgeted': paid == '1'}))
        await wait_file(directory/'entry.release')
        return value

    control_runtime.open_control = hold
    try:
        result = await loop.send('worker once', sid)
        (directory/'worker.done').write_text(json.dumps({'status': result.status, 'text': result.final_text,
            'calls': model.calls, 'reserve': host.reserve_calls, 'apply': host.apply_calls}))
    finally:
        if registered:
            await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)
        await store.close()


if __name__ == '__main__':
    asyncio.run(main())
