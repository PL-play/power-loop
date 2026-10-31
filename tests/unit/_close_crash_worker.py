"""Terminate at the two row transitions now joined into one closing write."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from power_loop import LLMResponse, LLMService
from power_loop.runtime import finalization
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.task_runtime import _SEND

from .test_round_change_signals import fixture_loop


async def main(dsn, prefix, evidence, stage):
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")

    class Model(LLMService):
        async def complete(self, request, **kwargs):
            binding = _SEND.get()
            Path(evidence).write_text(json.dumps({"sid": binding.actor.session_id,
                "job_id": binding.lease.job_id, "generation": binding.lease.generation, "calls": 1}))
            return LLMResponse(raw_text="done")

        def stream(self, request):
            raise AssertionError("nonstream fixture")

        async def close(self):
            pass

    runtime, loop = fixture_loop(store, Model())
    closing = False
    publish = finalization._publish

    async def stop(*args):
        nonlocal closing
        closing = True
        if stage == "before_publish":
            os._exit(84)
        await publish(*args)
        if stage == "after_publish":
            os._exit(85)

    finalization._publish = stop
    transaction = store._db.transaction

    @asynccontextmanager
    async def die_after_commit():
        async with transaction() as tx:
            yield tx
        if closing:
            os._exit(86)

    if stage == "after_commit":
        store._db.transaction = die_after_commit
    await loop.send("close once", await loop.new_session())
    raise AssertionError("closing crash boundary was not reached")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
