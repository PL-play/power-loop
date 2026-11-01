"""Exit after cold review proof and before any new model admission."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime.control_runtime import prepare_review_sources
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.store.factory import open_store


async def main():
    dsn,prefix,raw,marker = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname=='127.0.0.1' and address.path.endswith('_test')
    store = await open_store(dsn,table_prefix=prefix,schema='compatible')
    prepared = await prepare_review_sources(JobManager(store),JobActor(**json.loads(raw)))
    assert prepared.pending
    Path(marker).write_text(json.dumps({'version':prepared.version,'pending':prepared.pending}))
    os._exit(91)


asyncio.run(main())
