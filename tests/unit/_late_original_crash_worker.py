"""Exit after authentic late model evidence, before its accounting projection."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from power_loop.runtime import model_attempts, model_budgets
from power_loop.runtime.store.factory import open_store

from .test_budget_reconcile_192 import reconcile
from .test_budget_revisions_192 import evidence, stage
from .test_late_original_checkpoint import case
from .test_model_budgets_192 import new_send, start


async def main(dsn, prefix, output, point):
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")
    c = case(store)
    await new_send(c)
    attempt = await start(c)
    await stage(c, attempt, evidence(12), expected=0)
    await reconcile(c.runtime, c, attempt)
    state = await model_budgets.get_budget(c.manager, c.actor, attempt.attempt_id)
    Path(output).write_text(json.dumps({"sid": c.sid, "job_id": c.binding.lease.job_id,
        "attempt_id": attempt.attempt_id, "submission": state["submission"], "ack": state["ack"],
        "reserve_calls": c.host.reserve_calls, "apply_calls": c.host.apply_calls}))
    if point == "after_original":
        async def die(*args):
            os._exit(94)
        model_budgets.observe = die
    else:
        save = model_budgets._save
        async def die_in_projection(*args, **kwargs):
            await save(*args, **kwargs)
            os._exit(95)
        model_budgets._save = die_in_projection
    await model_attempts.observe(attempt, usage={"total_tokens": 7}, estimated=False)
    raise AssertionError("late original crash point was not reached")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
