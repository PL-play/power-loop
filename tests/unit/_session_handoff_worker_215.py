"""Real-process ownership worker; DSN is an argument and is never printed."""
from __future__ import annotations

import asyncio
import json
import sys

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import TargetRef
from power_loop.contracts.errors import SessionLeaseLost
from power_loop.runtime.store.factory import open_store
from tests.unit.session_native_fixture import ACTOR


def emit(**payload):
    print(json.dumps(payload), flush=True)


async def main():
    dsn, prefix, session_id, role = sys.argv[1:]
    store = await open_store(dsn, table_prefix=prefix, schema="verify")
    port = RuntimeInputStore(store)
    target = TargetRef(store.store_id, session_id)
    try:
        if role == "old":
            batch = await port.claim(target, node_id="old-process", boot_id="old-boot")
            emit(event="claimed", send_index=batch.send_index)
            await asyncio.wait_for(asyncio.to_thread(sys.stdin.readline), 15)
            try:
                await port.final(batch, final_text="stale")
            except SessionLeaseLost:
                emit(event="stale")
            else:
                raise AssertionError("stale owner finalized")
        else:
            await port.accept(target, "new", message_id="new", actor=ACTOR)
            assert (await port.inspect(target, actor=ACTOR)).execution_state == "recovering"
            recovered = await port.recover(target, node_id="new-process", boot_id="new-boot")
            assert recovered.execution_state == "queued"
            batch = await port.claim(target, node_id="new-process", boot_id="new-boot")
            await port.final(batch, final_text="new process")
            emit(event="completed", send_index=batch.send_index)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
