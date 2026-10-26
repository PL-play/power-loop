"""Cancellation during an actual model response write retains observed usage."""
import asyncio
import threading

import pytest

from power_loop.runtime.model_attempt_query import list_attempts

from .test_real_model_attempts_192 import model_runtime

__all__ = ['model_runtime']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_response_write_cancellation_rolls_back_without_model_retry(model_runtime, monkeypatch):
    store, manager, runtime, provider, loop, sid = model_runtime
    db = store._db
    gate, release = threading.Event(), threading.Event()
    original = db._b_exec
    def blocked(sql, params):
        value = original(sql, params)
        if sql.startswith(f'INSERT INTO {store.t.messages} ') and params[2] == 'assistant' and not gate.is_set():
            gate.set()
            assert release.wait(15)
        return value
    monkeypatch.setattr(db, '_b_exec', blocked)
    sending = asyncio.create_task(loop.send('Reply HELLO in plain assistant text without tools.', sid, tools=[]))
    try:
        async with asyncio.timeout(100):
            while not gate.is_set():
                if sending.done():
                    await sending
                    pytest.fail('actual assistant write not reached')
                await asyncio.sleep(.01)
            sending.cancel()
            await asyncio.sleep(.01)
            sending.cancel()
            await asyncio.sleep(.01)
            assert not sending.done() and db._lock.locked()
            release.set()
            await asyncio.gather(sending, return_exceptions=True)
            assert sending.cancelled()
            assert provider.calls == len(provider.responses) == 1
            assert not db._conn.in_transaction
            assert not any(row.role == 'assistant' for row in await store.load_all_messages(sid))
            [job] = await manager.db.fetchall(f'SELECT * FROM {manager.jobs}')
            page = await list_attempts(manager, runtime.actor(sid), job['job_id'])
            [attempt] = page['items']
            assert attempt['status'] == 'returned' and attempt['response_received']
            assert attempt['observed_usage']['total_tokens'] == provider.responses[0].token_usage.total_tokens > 0
            # Still usable after rollback, without replaying the paid request.
            assert await db.fetchone('SELECT 1 AS value') == {'value': 1}
            assert provider.calls == 1
    finally:
        release.set()
        await asyncio.gather(sending, return_exceptions=True)
        await loop.aclose(drain_timeout_s=.1)
