"""Die after logical decisions commit but before an actual model-exit record."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from power_loop.runtime.store.factory import open_store
from power_loop.runtime.tool_call_plans import reconcile_pending

from .test_aborted_call_outcomes_192 import pending_batch
from .test_tool_call_plans_192 import controller


async def main(dsn, prefix, evidence):
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")

    async def connect(**kwargs):
        return await open_store(dsn, table_prefix=prefix, schema="tasks")

    c, source = await pending_batch((store, connect))
    loop, model = await controller(c)
    await reconcile_pending(loop, c.actor, source["job_id"])
    manager = c.manager
    transaction = manager.db.transaction

    @asynccontextmanager
    async def die_after_decision():
        async with transaction() as tx:
            yield tx
            original = await tx.fetchone(f"SELECT * FROM {manager.events} WHERE event_type='processing_decision'")
        if original is None:
            return
        payload = json.loads(original["payload_json"])
        assert original["job_id"] == payload["handler_job_id"]
        assert not await manager.db.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?",
            ("model_closed_"+original["job_id"],))
        fees = {}
        for suffix in ("task_budget_attempts", "task_budget_scopes", "task_budget_policies"):
            table = manager.prefix+suffix
            fees[table] = await manager.db.fetchall(f"SELECT * FROM {table} ORDER BY 1")
        Path(evidence).write_text(json.dumps({"actor": {"tenant_id": c.actor.tenant_id,
            "principal_id": c.actor.principal_id, "session_id": c.sid}, "handler_job_id": original["job_id"],
            "source_job_id": source["job_id"], "events": await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id"),
            "messages": await manager.db.fetchall(f"SELECT * FROM {store.t.messages} ORDER BY id"),
            "fees": fees, "model_calls": c.model.calls, "review_calls": model.calls,
            "dispatched": c.state.dispatched, "reserve_calls": c.host.reserve_calls,
            "apply_calls": c.host.apply_calls}))
        os._exit(77)

    manager.db.transaction = die_after_decision
    await loop.send("Review only the original outcome without repeating any action.", c.sid)
    raise AssertionError("did not reach the committed processing-decision boundary")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
