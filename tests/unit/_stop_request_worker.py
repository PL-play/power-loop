"""Actual stop-request commit, crash and opposite-direction process boundaries."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import task_control
from power_loop.runtime.jobs import JobActor, JobLease, JobManager
from power_loop.runtime.managed_effects import _ref
from power_loop.runtime.store.factory import open_store


async def main():
    dsn, prefix, context, mode, ready, release, output = sys.argv[1:]
    if "://" in dsn:
        address = urlparse(dsn)
        assert address.hostname == "127.0.0.1" and address.path.endswith("_test")
    context = json.loads(context)
    actor, caller = JobActor(**context["actor"]), JobLease(**context["caller"])
    store = await open_store(dsn, table_prefix=prefix, schema="compatible")
    manager = JobManager(store, now_ms=lambda: context["now"])
    transaction = manager.db.transaction

    async def phase(tx):
        rows = await tx.fetchall(f"SELECT payload_json FROM {manager.events} "
            "WHERE job_id=? AND event_type='task_control_request'", (caller.job_id,))
        phases = {json.loads(row["payload_json"])["phase"] for row in rows}
        if "completed" in phases:
            return "completed"
        if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND action='stop'", (context["target"],)):
            return "target"
        return "accepted" if phases else None

    @asynccontextmanager
    async def boundary():
        async with transaction() as tx:
            yield tx
            current = await phase(tx)
            if mode == "uncommitted-"+str(current):
                Path(ready).write_text(json.dumps({"phase": current, "committed": False}))
                os._exit(77)
        if mode == current:
            Path(ready).write_text(json.dumps({"phase": current, "committed": True}))
            os._exit(77)
        if mode == "hold-accepted" and current == "accepted":
            Path(ready).write_text(json.dumps({"phase": current, "committed": True}))
            async with asyncio.timeout(30):
                while not Path(release).exists():
                    await asyncio.sleep(.01)

    manager.db.transaction = boundary
    try:
        operation, replayed = await task_control.request_stop(manager, actor, context["target"],
            caller=caller, call_ref=_ref(context["ref"]), tool_call_id=context["call_id"])
        Path(output).write_text(json.dumps({"operation_id": operation.operation_id, "replayed": replayed,
            "response": operation.result["response"]}))
    finally:
        await store.close()


asyncio.run(main())
