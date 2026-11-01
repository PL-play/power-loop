"""Exit around an original stopped wait's typed own-tree termination plan."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import wait_terminal
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
    publish,transaction = wait_terminal._publish_snapshot,manager.db.transaction

    def die():
        Path(marker).write_text(stage)
        os._exit(86)

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

    wait_terminal._publish_snapshot = prepared
    manager.db.transaction = boundary
    await wait_terminal.terminate(manager,actor,config['job_id'])
    raise AssertionError('wait termination process boundary was not reached')


asyncio.run(main())
