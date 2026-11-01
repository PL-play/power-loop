"""Exit after a real dispatch permission/target/receipt transaction commit."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import control_requests, resume_dispatch
from power_loop.runtime.jobs import JobActor, JobLease, JobManager
from power_loop.runtime.store.factory import open_store


async def main():
    dsn, prefix, raw, mode, ready, output = sys.argv[1:]
    phase, committed = mode.split(':')
    committed = committed == '1'
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    context = json.loads(raw)
    store = await open_store(dsn, table_prefix=prefix, schema='compatible')
    manager = JobManager(store, now_ms=lambda:context['now'])
    transaction = manager.db.transaction

    def die():
        Path(ready).write_text(mode)
        os._exit(79)

    @asynccontextmanager
    async def boundary():
        async with transaction() as tx:
            yield tx
            rows = await tx.fetchall(f"SELECT * FROM {manager.events} WHERE event_type='task_control_request'")
            accepted = [row for row in rows if (lambda data:data['phase']=='accepted' and data['action']==resume_dispatch.ACTION)(
                json.loads(row['payload_json']))]
            current = None
            if accepted:
                if any(row['event_id']==control_requests.completion_id(accepted[0]['event_id']) for row in rows):
                    current = 'completed'
                elif await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_type='dispatch'"):
                    current = 'target'
                else:
                    current = 'accepted'
            hit = current == phase
            if hit and not committed:
                die()
        if hit:
            die()

    manager.db.transaction = boundary
    try:
        result = await resume_dispatch.claim(manager, JobActor(**context['actor']), context['operation_id'],
            caller=JobLease(**context['caller']), token=context['token'])
        Path(output).write_text(json.dumps(result))
    finally:
        await store.close()


asyncio.run(main())
