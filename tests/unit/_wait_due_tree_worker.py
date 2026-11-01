"""Crash after timer source preparation or around its own due publication."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import timer_waits
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.store.factory import open_store


async def main():
    dsn,prefix,raw,stage,marker = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname=='127.0.0.1' and address.path.endswith('_test')
    config = json.loads(raw)
    store = await open_store(dsn,table_prefix=prefix,schema='compatible')
    manager = JobManager(store,now_ms=lambda:config['now'])
    actor = JobActor(**config['actor'])
    write,transaction = manager._write,manager.db.transaction

    def die():
        Path(marker).write_text(stage)
        os._exit(84)

    @asynccontextmanager
    async def writer():
        if stage=='prepared':
            die()
        async with write() as tx:
            yield tx

    class AtWrite:
        def __init__(self,tx):
            self.tx,self.written = tx,False

        def __getattr__(self,name):
            return getattr(self.tx,name)

        async def execute(self,sql,params=()):
            result = await self.tx.execute(sql,params)
            if sql.startswith('UPDATE '+manager.jobs+' ') or sql.startswith('INSERT INTO '+manager.events+' '):
                self.written = True
                if stage=='first_write':
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

    manager._write = writer
    manager.db.transaction = boundary
    await timer_waits.prepare_due(manager,actor,config['job_id'])
    raise AssertionError('timer due process boundary was not reached')


asyncio.run(main())
