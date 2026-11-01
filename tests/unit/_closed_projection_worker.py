"""Die after the original clean exit commits, before another root is touched."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from power_loop.runtime import session_obligations as obligations
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

    async def die(manager, decisions):
        if not decisions:
            return
        events = await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id")
        handlers = [event["job_id"] for event in events if event["event_type"] == "model_closed" and
            set(json.loads(event["payload_json"])["evidence"].get("processing_decisions", ())) == set(decisions)]
        assert len(handlers) == 1
        fees = {}
        for suffix in ("task_budget_attempts", "task_budget_scopes", "task_budget_policies"):
            table = manager.prefix+suffix
            fees[table] = await manager.db.fetchall(f"SELECT * FROM {table} ORDER BY 1")
        Path(evidence).write_text(json.dumps({"actor": {"tenant_id": c.actor.tenant_id,
            "principal_id": c.actor.principal_id, "session_id": c.sid}, "handler_job_id": handlers[0],
            "source_job_id": source["job_id"], "events": events,
            "messages": await manager.db.fetchall(f"SELECT * FROM {store.t.messages} ORDER BY id"),
            "fees": fees, "model_calls": c.model.calls, "review_calls": model.calls,
            "dispatched": c.state.dispatched, "reserve_calls": c.host.reserve_calls,
            "apply_calls": c.host.apply_calls}))
        os._exit(76)

    obligations.refresh_closed_decisions = die
    await loop.send("Review only the original outcome without repeating any action.", c.sid)
    raise AssertionError("did not reach the committed clean-exit boundary")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
