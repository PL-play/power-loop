"""Real input-request process boundaries; never reconstruct a model or callback."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from power_loop.runtime import control_requests, nudges
from power_loop.runtime.jobs import JobActor, JobLease, JobManager
from power_loop.runtime.managed_effects import _ref
from power_loop.runtime.store.factory import open_store


async def main():
    dsn, prefix, context, mode, ready, release, output = sys.argv[1:]
    if "://" in dsn:
        address = urlparse(dsn)
        assert address.hostname == "127.0.0.1" and address.path.endswith("_test")
    context = json.loads(context)
    actor, caller, target = JobActor(**context["actor"]), JobLease(**context["caller"]), JobLease(**context["target"])
    store = await open_store(dsn, table_prefix=prefix, schema="compatible")
    manager = JobManager(store, now_ms=lambda: context["now"])
    ref = _ref(context["ref"])
    key = manager._origin_key(actor, "nudge", ref, context["call_id"], None)
    request_id = control_requests.identity("nudge", key)
    transaction = manager.db.transaction

    async def phase(tx):
        if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (control_requests.completion_id(request_id),)):
            return "completed"
        if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE operation_key=? OR event_id=?",
                (key, control_requests.rejection_id(request_id))):
            return "target"
        if await tx.fetchone(f"SELECT event_id FROM {manager.events} WHERE event_id=?", (request_id,)):
            return "accepted"
        return None

    @asynccontextmanager
    async def boundary():
        async with transaction() as tx:
            yield tx
            current = await phase(tx)
            if mode == "uncommitted-"+str(current):
                Path(ready).write_text(json.dumps({"phase": current, "committed": False}))
                os._exit(78)
        if mode == current:
            Path(ready).write_text(json.dumps({"phase": current, "committed": True}))
            os._exit(78)
        if mode == "hold-accepted" and current == "accepted":
            Path(ready).write_text(json.dumps({"phase": current, "committed": True}))
            async with asyncio.timeout(30):
                while not Path(release).exists():
                    await asyncio.sleep(.01)

    manager.db.transaction = boundary
    apply = nudges._apply_nudge_request

    async def stop_before_apply(owner, actor, event, data):
        await owner.request_cancel(actor, target.job_id, system_key="actual-stop-before-input", reason="stop")
        return await apply(owner, actor, event, data)

    if context.get("stop_target"):
        nudges._apply_nudge_request = stop_before_apply
    try:
        try:
            operation, replayed = await nudges.accept(manager, actor, target, caller=caller, call_ref=ref,
                tool_call_id=context["call_id"], session_id=context["sid"], message=context["message"])
            packet = {"operation_id": operation.operation_id, "replayed": replayed}
        except Exception as exc:
            from power_loop.runtime.jobs import JobError
            if not isinstance(exc, JobError):
                raise
            packet = {"error": exc.code}
        Path(output).write_text(json.dumps(packet))
    finally:
        await store.close()


asyncio.run(main())
