"""Kill the actual cold controller publisher at native transaction boundaries."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import controller_recovery
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.session_controls import ControlTransaction
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import TaskRuntime


async def main():
    dsn, prefix, raw, stage, marker = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    config = json.loads(raw)
    store = await open_store(dsn, table_prefix=prefix, schema='compatible')
    manager = JobManager(store, now_ms=lambda: config['now'])
    actor = JobActor(**config['actor'])
    runtime = TaskRuntime(manager, tenant_id=actor.tenant_id, principal_id=actor.principal_id,
        executor_id='original-recovery-crash', boot_id='recovery-crash')
    publish, transaction = controller_recovery._publish_recovery, manager.db.transaction
    append, active = ControlTransaction.append_event, False

    def die():
        Path(marker).write_text(stage)
        os._exit(92)

    async def prepared(*args, **kwargs):
        nonlocal active
        if stage == 'prepared':
            die()
        active = True
        try:
            return await publish(*args, **kwargs)
        finally:
            active = False

    async def appended(tx, columns, values, **kwargs):
        result = await append(tx, columns, values, **kwargs)
        event = dict(zip(columns, values, strict=True))
        if active and event['job_id'] == config['job_id'] and (
                stage == 'result' and event['event_type'] == 'result' or
                stage == 'stopped' and event['event_type'] == 'controller_stopped'):
            die()
        return result

    class AtWrite:
        def __init__(self, tx):
            self.tx, self.written = tx, False

        def __getattr__(self, name):
            return getattr(self.tx, name)

        async def execute(self, sql, params=()):
            result = await self.tx.execute(sql, params)
            if active and result:
                self.written = True
                if stage == 'inbox' and sql.startswith('INSERT') and manager.store.t.follow_up_queue in sql:
                    die()
            return result

    @asynccontextmanager
    async def boundary():
        async with transaction() as tx:
            wrapped = AtWrite(tx)
            yield wrapped
            if wrapped.written and stage == 'before_commit':
                die()
        if wrapped.written and stage == 'after_commit':
            die()

    controller_recovery._publish_recovery = prepared
    ControlTransaction.append_event = appended
    manager.db.transaction = boundary
    await controller_recovery.reconcile(runtime, actor, config['job_id'])
    raise AssertionError('controller recovery publication boundary was not reached')


asyncio.run(main())
