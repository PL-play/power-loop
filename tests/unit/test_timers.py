"""Default timer behavior through the current RuntimeClient and store backend."""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentHooks,
    AgentLoopConfig,
    ExecutionEnvironment,
    HookPoint,
    InputRef,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeRequestError,
    StoreTimerBackend,
    TimerContext,
    TimerRunner,
    create_default_tool_registry,
    get_tool_definitions,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.tools import DEFAULT_TOOL_HANDLERS
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']

@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done", content_text="done")
        r = self.responses[self._idx]
        self._idx += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


def _tool_resp(call_id: str, name: str, args: str) -> LLMResponse:
    return LLMResponse(
        raw_text="",
        tool_calls=[{"id": call_id, "type": "function",
                     "function": {"name": name, "arguments": args}}],
    )



@asynccontextmanager
async def timers(store, model=None, *, hooks=None):
    from .test_runtime_messaging_203 import fixture
    _, target, actor = await fixture(store)
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        return ExecutionEnvironment(model or _Scripted(), None, AgentLoopConfig(compactor=None))
    factory.register('test.agent@1', build)
    runtime = Runtime(store=store, factory=factory, node_id='timer-test', hooks=hooks)
    await runtime.worker.start()
    # Explicit scans make time/races deterministic; lifecycle itself is tested below.
    await runtime.timer_driver.stop()
    try:
        yield runtime, target, actor, TimerContext(store.store_id, target.session_id, actor)
    finally:
        await runtime.worker.shutdown(timeout_s=2, cancel_running=True)
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_store_schedule_list_cancel_limits_and_future(registry_store):
    store, _ = registry_store
    async with timers(store) as (runtime, target, actor, source):
        backend = runtime.timer_backend
        for i in range(10):
            await backend.create(source, delay_s=3600, note=f'future {i}')
        with pytest.raises(RuntimeRequestError, match='timer_limit_reached'):
            await backend.create(source, delay_s=3600, note='overflow')
        assert await runtime.timer_driver.scan_once() == 0
        assert len(await backend.list(source)) == 10
        assert await backend.cancel(source, 1)
        assert not await backend.cancel(source, 1)
        assert len(await backend.list(source)) == 9
        for kwargs in ({'note':'missing'}, {'delay_s':1,'due_at_ms':2,'note':'both'}, {'delay_s':1,'note':' '}):
            with pytest.raises(RuntimeRequestError, match='invalid_options'):
                await backend.create(source, **kwargs)


async def test_default_tool_schedule_list_cancel_and_time(registry_store):
    store, _ = registry_store
    registry = create_default_tool_registry(preset='core', bind=False)
    for definition in get_tool_definitions(include=['schedule_wakeup','current_time']):
        registry.register(definition, DEFAULT_TOOL_HANDLERS[definition.name], overwrite=True)
    model = _Scripted(responses=[
        _tool_resp('create','schedule_wakeup','{"action":"schedule","delay_seconds":3600,"note":"check"}'),
        _tool_resp('list','schedule_wakeup','{"action":"list"}'),
        _tool_resp('cancel','schedule_wakeup','{"action":"cancel","timer_id":1}'),
        _tool_resp('time','current_time','{}'),
    ])
    agent = await RunningAgent(store, model, registry, AgentLoopConfig(max_rounds=6)).start()
    try:
        result = await agent.send('schedule/list/cancel and time')
        assert result.status == 'completed'
        rows = await store.load_active_messages(agent.target.session_id)
        outputs = [row for row in rows if row.role=='tool']
        assert len(outputs) == 4 and all('Error:' not in str(r.content) and '"error"' not in str(r.content) for r in outputs), [r.content for r in outputs]
        assert (await store.get_timer(agent.target.session_id,1)).status == 'cancelled'
    finally:
        await agent.aclose()


async def test_periodic_fixed_delay_and_unknown_occurrence_not_replayed(registry_store):
    store, _ = registry_store
    async with timers(store) as (runtime, target, actor, source):
        timer = await runtime.timer_backend.create(source, due_at_ms=1, note='repeat', interval_s=60)
        before = int(time.time()*1000)
        assert await runtime.timer_driver.scan_once() == 1
        row = await store.get_timer(target.session_id,timer.timer_id)
        assert row.status == 'armed' and row.fire_count == 1 and row.due_at >= before+60000
        assert await runtime.timer_driver.scan_once() == 0
        # Restart abandons a stale unknown occurrence; it never submits it again.
        one = await runtime.timer_backend.create(source, due_at_ms=1, note='unknown')
        for item in (timer,one):
            await store.transition_timer(target.session_id,item.timer_id,from_status='armed',to_status='firing')
        await store._db.execute(f"UPDATE {store.t.timers} SET updated_at=1 WHERE status='firing'")
        recovered = TimerRunner(store=store,client=runtime.client)
        assert await recovered.scan_once() == 0
        periodic = await store.get_timer(target.session_id,timer.timer_id)
        assert periodic.status == 'armed' and periodic.fire_count == 2
        assert (await store.get_timer(target.session_id,one.timer_id)).status == 'cancelled'
        rows = await store._db.fetchall(f'SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?',(target.session_id,))
        assert len(rows) == 1


async def test_default_timer_hook_failure_is_visible_and_scan_continues(registry_store):
    store, _ = registry_store
    hooks = AgentHooks()

    def fail_before_append(ctx):
        if ctx.note == 'bad':
            raise RuntimeError('test timer hook failure')

    class RecordingModel(_Scripted):
        async def complete(self, request, **kwargs):
            requests.append(str(request.messages))
            return await super().complete(request, **kwargs)

    requests = []
    hooks.register(HookPoint.TIMER_FIRE, fail_before_append)
    async with timers(store, RecordingModel(), hooks=hooks) as (runtime, target, actor, source):
        backend = runtime.timer_backend
        bad = await backend.create(source, due_at_ms=1, note='bad')
        assert await runtime.timer_driver.scan_once() == 0
        [row] = await backend.list(source, statuses=('unknown',))
        assert row.timer_id == bad.timer_id and row.last_fired_at is None
        assert not await store._db.fetchall(f'SELECT item_id FROM {store.t.follow_up_queue}')
        error = runtime.status()['timer']['last_error']
        assert f'/{bad.timer_id} unknown: RuntimeError: test timer hook failure' in error
        assert await runtime.timer_driver.scan_once() == 0
        assert runtime.status()['timer']['last_error'] == error

        recurring = await backend.create(source, due_at_ms=1, note='bad', interval_s=60)
        good = await backend.create(source, due_at_ms=2, note='GOOD_TIMER_NOTE')
        assert await runtime.timer_driver.scan_once() == 1
        [rearmed] = await backend.list(source)
        assert (rearmed.timer_id, rearmed.fire_count) == (recurring.timer_id, 1)
        assert rearmed.due_at > int(time.time()*1000)
        ref = InputRef(store.store_id, target.session_id, f'timer-{good.timer_id}-{good.created_at}-0')
        result = await runtime.client.wait(ref, actor=actor, deadline_ms=int(time.time()*1000)+10000)
        assert result.status == 'completed'
        assert len(requests) == 1 and 'GOOD_TIMER_NOTE' in requests[0]
        assert await store.prune_timers(target.session_id, statuses=('unknown',)) == 1


async def test_busy_append_and_driver_stop_do_not_wait_for_or_cancel_model(registry_store):
    store, _ = registry_store
    class Blocking(_Scripted):
        def __init__(self):
            self.entered, self.release = asyncio.Event(), asyncio.Event()
            self.calls = []
        async def complete(self, request, **kwargs):
            self.calls.append(str(request.messages))
            if len(self.calls)==1:
                self.entered.set()
                await self.release.wait()
            return LLMResponse(raw_text='done')
    model = Blocking()
    async with timers(store,model) as (runtime,target,actor,source):
        receipt = await runtime.client.send(target,'start',message_id='original',actor=actor)
        await asyncio.wait_for(model.entered.wait(),10)
        await runtime.timer_backend.create(source,due_at_ms=1,note='WAKE_NOTE')
        try:
            assert await asyncio.wait_for(runtime.timer_driver.scan_once(),5) == 1
            await asyncio.wait_for(runtime.timer_driver.stop(),1)
            assert not model.release.is_set()
            model.release.set()
            result = await runtime.client.wait(receipt.input_ref,actor=actor,deadline_ms=int(time.time()*1000)+10000)
            assert result.status=='completed' and len(model.calls)==2 and 'WAKE_NOTE' in model.calls[1]
            jobs = await store._db.fetchall(f"SELECT job_id FROM {store.table_prefix}jobs WHERE kind='send'")
            assert len(jobs)==1
        finally:
            model.release.set()


async def test_runtime_lifecycle_reopen_and_session_cleanup(registry_store):
    store,connect = registry_store
    async with timers(store) as (runtime,target,actor,source):
        await runtime.timer_backend.create(source,delay_s=3600,note='survives runtime')
        await runtime.timer_driver.start()
        assert runtime.timer_driver.status().ready
    assert not runtime.timer_driver.status().ready
    other = await connect()
    assert len(await StoreTimerBackend(other).list(source))==1
    async with timers(other) as (reopened,_,_,_):
        assert reopened.timer_driver.status().ready is False  # helper stopped it after start
        await reopened.timer_driver.start()
        assert reopened.timer_driver.status().ready
        await reopened.timer_driver.stop()
    assert await store.close_session(target.session_id)==1
    assert await store.list_timers(target.session_id,statuses=('armed','fired','cancelled'))==[]
