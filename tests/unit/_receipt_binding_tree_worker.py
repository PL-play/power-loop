"""Exit around a historical raw receipt's target-tree binding transaction."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime.jobs import JobActor, JobLease, JobManager
from power_loop.runtime.managed_effects import _ref
from power_loop.runtime.store.factory import open_store


async def main():
    dsn,prefix,raw,stage,marker = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname=='127.0.0.1' and address.path.endswith('_test')
    config = json.loads(raw)
    store = await open_store(dsn,table_prefix=prefix,schema="verify")
    manager = JobManager(store,now_ms=lambda:config['now'])
    actor,caller = JobActor(**config['actor']),JobLease(**config['caller'])
    receipt = _ref(config['receipt'])
    transaction,write = manager.db.transaction,manager._write

    def die():
        Path(marker).write_text(stage)
        os._exit(88)

    @asynccontextmanager
    async def prepared():
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
            if sql.startswith('UPDATE') and 'SET receipt_ref_json=' in sql:
                self.written = True
                if stage=='receipt_update':
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

    manager._write = prepared
    manager.db.transaction = boundary
    await manager.bind_receipt(actor,config['operation_id'],receipt,caller=caller)
    raise AssertionError('receipt binding boundary was not reached')


asyncio.run(main())
