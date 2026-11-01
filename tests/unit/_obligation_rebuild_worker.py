"""Actually exit during repeatable repair of two original task trees."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from power_loop.runtime import session_obligations as obligations
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.store.factory import open_store

from .test_obligation_rebuild_trees import _originals, _result, _rows


async def main(dsn, prefix, evidence, point):
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")
    manager = JobManager(store)
    actor = JobActor("tenant", "host", await store.create_session())
    await _result(manager, actor, "process-first")
    await _result(manager, actor, "process-second")
    Path(evidence).write_text(json.dumps({"originals": await _originals(manager), "pending": await _rows(manager),
        "actor": {"tenant_id": actor.tenant_id, "principal_id": actor.principal_id, "session_id": actor.session_id}}))
    await manager.db.execute(f"DELETE FROM {manager.prefix}session_obligations")
    entries, transaction = obligations._write_entries, manager.db.transaction
    roots = []

    async def boundary(owner, tx, row, pending):
        result = await entries(owner, tx, row, pending)
        roots.append(row["root_job_id"])
        if point == "before_first_commit" and len(roots) == 1:
            os._exit(82)
        if point == "before_second_commit" and len(roots) == 2:
            os._exit(84)
        return result

    @asynccontextmanager
    async def after_commit():
        async with transaction() as tx:
            yield tx
        if point == "after_first_commit" and len(roots) == 1:
            os._exit(83)

    obligations._write_entries, manager.db.transaction = boundary, after_commit
    await obligations.rebuild(manager)
    raise AssertionError("process did not reach its repair commit boundary")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
