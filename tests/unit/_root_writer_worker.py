"""Hold a real control root while another process uses public registry ports."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.store.factory import open_store


async def main():
    dsn, prefix, raw, ready, release = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    config = json.loads(raw)
    store = await open_store(dsn, table_prefix=prefix, schema='compatible')
    manager = JobManager(store, now_ms=lambda: config['now'])
    try:
        async with manager._write() as tx:
            row = await manager._authorize(tx, JobActor(**config['actor']), config['job_id'])
            Path(ready).write_text(json.dumps({'pid':os.getpid(),'root':row['root_job_id']}))
            async with asyncio.timeout(20):
                while not Path(release).exists():
                    await asyncio.sleep(.01)
    finally:
        await store.close()


asyncio.run(main())
