"""Die while a stale turn tries to append after public lease takeover."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.runtime.store import store as store_module
from power_loop.runtime.store.factory import open_store

from .test_round_change_signals import Model


async def main(dsn, prefix, sid, evidence, stage):
    store = await open_store(dsn, table_prefix=prefix, schema="tasks")
    other = await open_store(dsn, table_prefix=prefix, schema="compatible")
    now = [1000]
    store_module._now_ms = lambda: now[0]
    loop = StatefulAgentLoop(llm=Model(), store=store,
        config=AgentLoopConfig(distributed_sessions=True, session_lease_ttl_s=300, compactor=None))
    async with loop._session_lease(sid):
        calls = [{"id": "original", "type": "function", "function": {"name": "work", "arguments": "{}"}}]
        await store.append_message(sid, role="assistant", tool_calls=calls, pending_tool_calls=True)
        if stage == "after_takeover":
            now[0] += 300001
            assert await other.acquire_session_lease(sid, owner_id="replacement", ttl_ms=60000)
        Path(evidence).write_text(json.dumps({"now": 301001, "pending": (await other.get_state(sid)).pending}))
        # Exercise MySQL's locking-read path, whose INSERT precedes allocation.
        store._holding(sid).next_seq = None
        transaction = store._db.transaction

        @asynccontextmanager
        async def crash():
            async with transaction() as tx:
                execute, fetchone = tx.execute, tx.fetchone

                async def after_insert(sql, params=()):
                    result = await execute(sql, params)
                    if sql.startswith(f"INSERT INTO {store.t.messages} "):
                        os._exit(88)
                    return result

                async def after_guard(sql, params=()):
                    result = await fetchone(sql, params)
                    if sql.startswith(f"UPDATE {store.t.session_state} "):
                        os._exit(87)
                    return result

                tx.execute, tx.fetchone = after_insert, after_guard
                yield tx

        store._db.transaction = crash
        await store.append_message(sid, role="assistant", content="stale result")
    raise AssertionError("stale append crash boundary not reached")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
