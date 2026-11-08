"""Die inside a signal/payload transaction, without interpreter cleanup."""
from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from power_loop.runtime.store.factory import open_store


async def write(store, sid, kind):
    if kind == "inbox":
        await store.inbox_put(sid, [{"item_id": "late-input", "content": "once"}])
    elif kind == "background":
        await store.upsert_background_task(sid, task_id="late-result", command="fixture", status="completed", output_tail="once")
    else:
        await store.set_runtime_state(sid, "todo", {"rendered": "once"})


async def main(dsn, prefix, sid, kind, stage):
    store = await open_store(dsn, table_prefix=prefix, schema="verify")
    if stage == "after_signal":
        signal = store.signal_changes
        async def stop(*args):
            await signal(*args)
            os._exit(91)
        store.signal_changes = stop
    else:
        transaction = store._db.transaction
        @asynccontextmanager
        async def stop_before_commit():
            async with transaction() as tx:
                yield tx
                os._exit(92)
        store._db.transaction = stop_before_commit
    await write(store, sid, kind)
    raise AssertionError("crash point not reached")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
