"""Die around the original transaction that opens or consumes a real review."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from power_loop.runtime import session_obligations
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.tool_call_plans import reconcile_pending

from .test_aborted_call_outcomes_192 import pending_batch
from .test_tool_call_plans_192 import controller


async def main(dsn, prefix, evidence, mode, point):
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")

    async def connect(**kwargs):
        return await open_store(dsn, table_prefix=prefix, schema="tasks")

    c, source = await pending_batch((store, connect))
    loop, model = await controller(c)
    if mode == "consume":
        await reconcile_pending(loop, c.actor, source["job_id"])
    original_replace, original_transaction = session_obligations._replace, store._db.transaction
    armed = False

    async def replace(manager, tx, row, **kwargs):
        nonlocal armed
        entries = await original_replace(manager, tx, row, **kwargs)
        if armed or row["job_id"] != source["job_id"]:
            return entries
        pending = await tx.fetchone(f"SELECT source_id FROM {manager.prefix}session_obligations WHERE job_id=? AND kind='abort'",
                                   (row["job_id"],))
        reached = (row["result_status"] == "pending_delivery" if mode == "produce"
                   else row["result_status"] == "handled" and pending is None)
        if reached:
            armed = True
            Path(evidence).write_text(json.dumps({"source_job_id": source["job_id"], "actor": {
                "tenant_id": c.actor.tenant_id, "principal_id": c.actor.principal_id, "session_id": c.sid},
                "model_calls": c.model.calls, "review_calls": model.calls,
                "dispatched": c.state.dispatched, "reserve_calls": c.host.reserve_calls,
                "apply_calls": c.host.apply_calls}))
            if point == "before_commit":
                os._exit(152)
        return entries

    @asynccontextmanager
    async def transaction():
        async with original_transaction() as tx:
            yield tx
        if armed:
            os._exit(153)

    session_obligations._replace, store._db.transaction = replace, transaction
    if mode == "produce":
        await reconcile_pending(loop, c.actor, source["job_id"])
    else:
        await loop.send("Review the original result without repeating any action.", c.sid)
    raise AssertionError("did not cross the original obligation boundary")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
