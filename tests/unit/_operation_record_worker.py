"""Exit at actual source and target operation-record transaction boundaries."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import control_requests, operation_records
from power_loop.runtime.jobs import JobActor, JobLease, JobManager
from power_loop.runtime.session_controls import ControlTransaction
from power_loop.runtime.store.factory import open_store


async def main():
    dsn, prefix, raw, stage, marker = sys.argv[1:]
    if '://' in dsn:
        address = urlparse(dsn)
        assert address.hostname == '127.0.0.1' and address.path.endswith('_test')
    config = json.loads(raw)
    store = await open_store(dsn, table_prefix=prefix, schema='compatible')
    manager = JobManager(store, now_ms=lambda: config['now'])
    apply, insert, complete, append = operation_records._apply, manager._insert_operation, control_requests.complete, ControlTransaction.append_event

    def die():
        Path(marker).write_text(json.dumps({'stage': stage, 'pid': os.getpid()}))
        os._exit(94)

    async def applied(*args, **kwargs):
        if stage == 'accepted':
            die()
        value = await apply(*args, **kwargs)
        if stage == 'target-committed':
            die()
        return value

    async def inserted(*args, **kwargs):
        value = await insert(*args, **kwargs)
        if stage == 'target-uncommitted':
            die()
        return value

    async def appended(tx, columns, values, **kwargs):
        value = await append(tx, columns, values, **kwargs)
        event = dict(zip(columns, values, strict=True))
        if stage == 'source-uncommitted' and event['event_id'].startswith('control_done_'):
            die()
        return value

    async def completed(*args, **kwargs):
        value = await complete(*args, **kwargs)
        if stage == 'source-committed':
            die()
        return value

    operation_records._apply = applied
    manager._insert_operation = inserted
    ControlTransaction.append_event = appended
    control_requests.complete = completed
    await manager.register_operation(JobActor(**config['actor']), JobLease(**config['target']),
        caller=JobLease(**config['caller']), action=config['action'],
        call_ref=operation_records._call_ref(config['ref']), tool_call_id='actual-record',
        request={'message': 'record metadata'}, result={'status': 'accepted', 'saved': {'value': 'original'}})
    raise AssertionError('operation record crash boundary not reached')


asyncio.run(main())
