"""Exit at an actual retained owner wait's single-tree publication boundaries."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import owner_waits
from power_loop.runtime.jobs import JobActor, JobLease, JobManager
from power_loop.runtime.store.factory import open_store


async def main():
    dsn,prefix,raw,stage,marker = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname=='127.0.0.1' and address.path.endswith('_test')
    config = json.loads(raw)
    store = await open_store(dsn,table_prefix=prefix,schema='compatible')
    manager = JobManager(store,now_ms=lambda:config['now'])
    actor,lease = JobActor(**config['actor']),JobLease(**config['lease'])
    publish,transaction = owner_waits._publish_transition,manager.db.transaction

    def die():
        Path(marker).write_text(stage)
        os._exit(90)

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
            if result and sql.startswith(('INSERT INTO '+manager.events,'UPDATE '+manager.jobs)):
                self.written = True
                if stage=='first_write':
                    die()
            return result

    @asynccontextmanager
    async def boundary():
        async with transaction() as tx:
            wrapped = AtWrite(tx)
            yield wrapped
            if wrapped.written and stage=='before_commit':
                die()
        if wrapped.written and stage=='after_commit':
            die()

    owner_waits._publish_transition = prepared
    manager.db.transaction = boundary
    if config['operation']=='enter':
        await owner_waits._enter(manager,actor,lease,config['raw'])
    else:
        await owner_waits._step(manager,actor,lease,config['waiting_id'],config['saved'])
    raise AssertionError('owner wait publication boundary was not reached')


asyncio.run(main())
