"""Existing message readers tolerate holes left by unused sequence reservations."""
from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Model(LLMService):
    def __init__(self):
        self.seen = []

    async def complete(self, request, **kwargs):
        self.seen.append(str(request.messages))
        return LLMResponse(raw_text='done')

    def stream(self, request):
        raise AssertionError('nonstream fixture')

    async def close(self):
        pass


async def make_loop(runtime_agent, store, *, target=None):
    model = Model()
    runtime = TaskRuntime(JobManager(store), tenant_id='reserved-gap', principal_id='host', executor_id='e', boot_id='b')
    return await runtime_agent(model=model, store=store, task_runtime=runtime, target=target,
        config=AgentLoopConfig(max_rounds=2, compactor=None, delegation_policy='single_level')), model


async def test_warm_cache_delta_reader_message_refs_and_cold_entry_tolerate_unused_slots(registry_store, runtime_agent):
    store, connect = registry_store
    loop, model = await make_loop(runtime_agent, store)
    cold_loop = None
    sid = loop.target.session_id
    try:
        assert (await loop.send('before reserved holes')).status == 'completed'
        first = await store.get_state(sid)
        # Simulate unused committed reservation positions before implementing
        # allocation changes;the message reader itself must already accept them.
        await store._db.execute(f'UPDATE {store.t.session_state} SET next_seq=next_seq+3 WHERE session_id=?', (sid,))
        assert (await loop.send('after reserved holes')).status == 'completed'
        rows = await store.load_all_messages(sid)
        assert [row.seq for row in rows] == [1, 2, first.next_seq+3, first.next_seq+4]
        assert [row.seq for row in await store.load_active_messages(sid, after_seq=first.next_seq)] == [first.next_seq+3, first.next_seq+4]
        assert (await store.message_ref(sid, first.next_seq+3)).role == 'user'
        assert 'before reserved holes' in model.seen[-1] and 'after reserved holes' in model.seen[-1]
        await loop.aclose()
        cold = await connect()
        cold_loop, cold_model = await make_loop(runtime_agent, cold, target=loop.target)
        assert (await cold_loop.send('cold after holes', message_id='cold')).status == 'completed'
        assert all(text in cold_model.seen[-1] for text in ('before reserved holes','after reserved holes','cold after holes'))
        assert (await cold.get_state(sid)).next_seq == first.next_seq+7
    finally:
        if cold_loop is not None:
            await cold_loop.aclose()
        await loop.aclose()
