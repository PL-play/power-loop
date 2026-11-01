"""Exit after the driver commits a due but before its notification publication."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime import timer_waits, wait_reviews
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.wait_handoff import register_wait_handoff_tool

from .test_wait_handoff_192 import MainModel


async def main():
    dsn,prefix,raw,marker = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname=='127.0.0.1' and address.path.endswith('_test')
    config = json.loads(raw)
    store = await open_store(dsn,table_prefix=prefix,schema='compatible')
    manager = JobManager(store,now_ms=lambda:config['now'])
    actor = JobActor(**config['actor'])
    runtime = TaskRuntime(manager,tenant_id=actor.tenant_id,principal_id=actor.principal_id,
        executor_id='main-executor',boot_id='main-boot')
    tools = ToolRegistry()
    register_wait_handoff_tool(tools)
    loop = StatefulAgentLoop(llm=MainModel(),store=store,task_runtime=runtime,tool_registry=tools,
        config=AgentLoopConfig(delegation_policy='single_level',max_rounds=3,max_tokens=20))
    due = timer_waits.prepare_due

    async def exit_after_due(*args,**kwargs):
        result = await due(*args,**kwargs)
        assert result['status']=='decision_pending'
        Path(marker).write_text(result['event_id'])
        os._exit(87)

    timer_waits.prepare_due = exit_after_due
    await wait_reviews.drive(loop,actor,config['job_id'])
    raise AssertionError('driver after-due boundary was not reached')


asyncio.run(main())
