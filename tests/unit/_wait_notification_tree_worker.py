"""Crash around a reviewed timer's single-tree inbox/original publication."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime import wait_notifications
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.wait_handoff import register_wait_handoff_tool

from .test_wait_handoff_192 import MainModel


async def main():
    dsn,prefix,raw,stage,marker = sys.argv[1:]
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
    publish,transaction = wait_notifications._publish_notification,manager.db.transaction

    def die():
        Path(marker).write_text(stage)
        os._exit(85)

    async def prepared(*args,**kwargs):
        if stage=='prepared':
            die()
        return await publish(*args,**kwargs)

    class AtWrite:
        def __init__(self,tx):
            self.tx,self.written = tx,False

        def __getattr__(self,name):
            return getattr(self.tx,name)

        async def execute(self,sql,params=()):
            result = await self.tx.execute(sql,params)
            if sql.startswith(('INSERT','UPDATE','DELETE')):
                self.written = True
                if stage=='first_write':
                    die()
                if stage=='after_inbox' and sql.startswith('INSERT') and store.t.follow_up_queue in sql:
                    die()
            return result

    @asynccontextmanager
    async def boundary():
        wrapped = None
        async with transaction() as tx:
            wrapped = AtWrite(tx)
            yield wrapped
            if wrapped.written and stage=='before_commit':
                die()
        if wrapped.written and stage=='after_commit':
            die()

    wait_notifications._publish_notification = prepared
    manager.db.transaction = boundary
    await wait_notifications.enqueue_notification(loop,actor,config['event_id'])
    raise AssertionError('notification process boundary was not reached')


asyncio.run(main())
