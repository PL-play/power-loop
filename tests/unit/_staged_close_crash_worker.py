"""Crash on either side of a transaction containing an open and closed batch."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from power_loop.runtime import model_attempts
from power_loop.runtime.control_runtime import current, flush_binding
from power_loop.runtime.store.factory import open_store

from .test_late_original_checkpoint import case
from .test_model_budgets_192 import new_send, start


async def main(dsn, prefix, output, point):
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")
    c = case(store)
    await new_send(c)
    attempt = await start(c)
    await model_attempts.observe(attempt, usage={"total_tokens": 7}, estimated=False)
    state = current(c.binding).get("budget", attempt.attempt_id)
    Path(output).write_text(json.dumps({"sid": c.sid, "job_id": c.binding.lease.job_id,
        "attempt_id": attempt.attempt_id, "submission": state["submission"], "ack": state["ack"],
        "reserve_calls": c.host.reserve_calls, "apply_calls": c.host.apply_calls}))
    async with c.manager.joined_write() as tx:
        await flush_binding(c.binding, transaction=tx)
        await flush_binding(c.binding, close=True, transaction=tx)
        if point == "before_commit":
            os._exit(152)
    os._exit(153)


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
