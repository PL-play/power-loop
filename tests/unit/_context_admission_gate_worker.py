"""Hold a real context admission or original stop before its transaction ends."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import root_control
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.session_controls import ControlTransaction
from power_loop.runtime.store.factory import open_store


async def main():
    dsn, prefix, raw, mode, ready, release, output = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    config = json.loads(raw)
    store = await open_store(dsn, table_prefix=prefix, schema='compatible')
    manager = JobManager(store)
    actor = JobActor(**config['actor'])

    async def hold():
        Path(ready).write_text(json.dumps({'mode': mode, 'pid': os.getpid()}))
        async with asyncio.timeout(20):
            while not Path(release).exists():
                await asyncio.sleep(.01)

    try:
        if mode == 'hold-admission':
            async with manager.db.transaction() as native:
                tx = ControlTransaction(native, manager)
                await root_control._admit(manager, tx, actor, config['document'])
                await hold()
            result = {'state': 'admitted'}
        else:
            save = root_control._save
            async def saved(current, tx, actual, key, value):
                result = await save(current, tx, actual, key, value)
                if key.startswith('power-loop.root-stop.v1:'):
                    await hold()
                return result
            root_control._save = saved
            result = await root_control.cancel(manager, actor, namespace=config['document']['namespace'],
                payload=config['document']['payload'], system_key='original-cancel', reason='user stop')
        Path(output).write_text(json.dumps(result))
    finally:
        await store.close()


asyncio.run(main())
