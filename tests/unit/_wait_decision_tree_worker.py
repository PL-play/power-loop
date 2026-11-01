"""Exit before/after each independently committed original wait-decision step."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime import control_requests, wait_decision_admission, wait_decisions
from power_loop.runtime.execution_state import CheckpointPolicy
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.wait_handoff import register_wait_handoff_tool

from .test_wait_handoff_192 import MainModel


async def main():
    dsn, prefix, raw, mode, evidence, marker = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname=='127.0.0.1' and address.path.endswith('_test')
    config = json.loads(raw)
    phase, committed = mode.split(':')
    committed = committed=='1'
    store = await open_store(dsn, table_prefix=prefix, schema='compatible')
    manager = JobManager(store, now_ms=lambda:config['now'])
    runtime = TaskRuntime(manager, tenant_id=config['tenant'], principal_id=config['principal'],
        executor_id='main-executor', boot_id='main-boot', checkpoint_policy=CheckpointPolicy(interval_s=300))
    registry, model = ToolRegistry(), MainModel()
    register_wait_handoff_tool(registry)
    model.options = {'event_id':config['event_id'], 'decision':'continue'}
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(delegation_policy='single_level',max_rounds=3,max_tokens=20))
    record, transaction = wait_decisions.record_decision, manager.db.transaction

    async def retain(*args, **kwargs):
        Path(evidence).write_text(json.dumps({**kwargs,'caller':asdict(kwargs['caller']),
            'call_ref':kwargs['call_ref'].to_dict(),'model_calls':model.calls}))
        return await record(*args, **kwargs)

    def die():
        Path(marker).write_text(mode)
        os._exit(82)

    @asynccontextmanager
    async def boundary():
        async with transaction() as tx:
            yield tx
            rows = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='task_control_request'")
            grants = [row for row in rows if (lambda data:data['phase']=='accepted' and data['action']==wait_decision_admission.ACTION)(json.loads(row['payload_json']))]
            current = None
            if grants:
                if any(row['event_id']==control_requests.completion_id(grants[0]['event_id']) for row in rows):
                    current = 'completed'
                elif await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_type='wait_decision'"):
                    current = 'target'
                else:
                    current = 'accepted'
            hit = current==phase
            if hit and not committed:
                die()
        if hit:
            die()

    manager.db.transaction = boundary
    wait_decisions.record_decision = retain
    await loop.flush_follow_ups(config['sid'])
    raise AssertionError('wait-decision process boundary was not reached')


asyncio.run(main())
