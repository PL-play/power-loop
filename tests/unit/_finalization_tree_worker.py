"""Kill a real closure process at its snapshot or publication boundaries."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import finalization
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.session_controls import ControlTransaction
from power_loop.runtime.store.factory import open_store


async def main():
    dsn, prefix, raw, phase, ready, output = sys.argv[1:]
    if "://" in dsn:
        address = urlparse(dsn)
        assert address.hostname == "127.0.0.1" and address.path.endswith("_test")
    context = json.loads(raw)
    store = await open_store(dsn, table_prefix=prefix, schema="compatible")
    manager = JobManager(store, now_ms=lambda: context["now"])
    actor = JobActor(**context["actor"])

    def die():
        Path(ready).write_text(phase)
        os._exit(79)

    publish = finalization._publish_plan

    async def prepared(owner, plans, tree):
        assert owner._control_intents.get() is None
        if phase == "prepared":
            die()
        return await publish(owner, plans, tree)

    finalization._publish_plan = prepared
    append, write = ControlTransaction.append_event_if_tree_current, ControlTransaction.write_job
    append_event = ControlTransaction.append_event

    async def original(tx, event, tree):
        count = await append(tx, event, tree)
        if phase == "original":
            assert count == 1
            die()
        return count

    async def projection(tx, job_id, sql, params=(), **kwargs):
        count = await write(tx, job_id, sql, params, **kwargs)
        if phase == "projection" and "supervision_status='closed'" in sql:
            assert count == 1
            die()
        return count

    ControlTransaction.append_event_if_tree_current = original
    ControlTransaction.write_job = projection

    async def bookkeeping(tx, columns, values, **kwargs):
        count = await append_event(tx, columns, values, **kwargs)
        if phase == "bookkeeping" and dict(zip(columns, values, strict=True))["event_type"] in (
                "input_disposition", "wait_terminated"):
            die()
        return count

    ControlTransaction.append_event = bookkeeping
    transaction = manager.db.transaction

    @asynccontextmanager
    async def committed():
        async with transaction() as tx:
            yield tx
            row = await tx.fetchone(f"SELECT supervision_status FROM {manager.jobs} WHERE job_id=?", (context["root"],))
            hit = row["supervision_status"] == "closed"
        if phase == "committed" and hit:
            die()

    manager.db.transaction = committed
    try:
        ref = await manager.finalize_supervision(actor, context["root"], generation=context["generation"])
        Path(output).write_text(json.dumps(ref))
    finally:
        await store.close()


asyncio.run(main())
