"""Timer driver routes through RuntimeClient; host policy lives in the backend/driver."""
import asyncio

import pytest

from power_loop import AgentHooks, HookDirective, HookPoint, InputRef, TimerRunner
from power_loop.agent.runtime_store import now_ms

from .test_job_registry_192 import registry_store
from .test_timers import _Scripted, timers

__all__ = ['registry_store']
pytestmark = pytest.mark.asyncio


async def test_delivery_waits_for_acceptance_without_holding_model(registry_store):
    store, _ = registry_store
    started, release = asyncio.Event(), asyncio.Event()
    class BlockedModel(_Scripted):
        async def complete(self, request, **kwargs):
            started.set()
            await release.wait()
            return await super().complete(request, **kwargs)
    async with timers(store, BlockedModel()) as (runtime, target, actor, source):
        timer = await runtime.timer_backend.create(source, delay_s=-1, note='route me')
        try:
            assert await asyncio.wait_for(runtime.timer_driver.scan_once(), 3) == 1
            await asyncio.wait_for(started.wait(), 3)
            assert (await store.get_timer(target.session_id, timer.timer_id)).status == 'fired'
            ref = InputRef(store.store_id, target.session_id, f'timer-{timer.timer_id}-{timer.created_at}-0')
            assert not (await runtime.client.inspect(ref, actor=actor)).result_available
            release.set()
            assert (await runtime.client.wait(ref, actor=actor, deadline_ms=now_ms()+10000)).status == 'completed'
        finally:
            release.set()


@pytest.mark.parametrize('skip', [False, True])
async def test_hook_rewrite_or_skip_applies_before_client_acceptance(registry_store, skip):
    store, _ = registry_store
    async with timers(store) as (runtime, target, actor, source):
        hooks = AgentHooks()
        def policy(ctx):
            ctx.message = 'REWRITTEN: ' + ctx.note
            if skip:
                ctx.directive = HookDirective.SKIP
        hooks.register(HookPoint.TIMER_FIRE, policy)
        driver = TimerRunner(store=store, client=runtime.client, hooks=hooks)
        timer = await runtime.timer_backend.create(source, delay_s=-1, note='original')
        assert await driver.scan_once() == 1
        assert (await store.get_timer(target.session_id, timer.timer_id)).status == 'fired'
        if skip:
            assert await store.load_all_messages(target.session_id) == []
        else:
            ref = InputRef(store.store_id, target.session_id, f'timer-{timer.timer_id}-{timer.created_at}-0')
            await runtime.client.wait(ref, actor=actor, deadline_ms=now_ms()+10000)
            assert any(row.content == 'REWRITTEN: original' for row in await store.load_all_messages(target.session_id))


async def test_unknown_acceptance_is_not_replayed_and_next_timer_can_fire(registry_store, monkeypatch):
    store, _ = registry_store
    async with timers(store) as (runtime, target, actor, source):
        original = runtime.client.append
        calls = []
        async def lose_reply(*args, **kwargs):
            calls.append(kwargs['message_id'])
            await original(*args, **kwargs)
            raise ConnectionError('acceptance reply lost')
        monkeypatch.setattr(runtime.client, 'append', lose_reply)
        timer = await runtime.timer_backend.create(source, delay_s=-1, note='unknown original')
        assert await runtime.timer_driver.scan_once() == 0
        assert (await store.get_timer(target.session_id, timer.timer_id)).status == 'unknown'
        assert 'acceptance reply lost' in runtime.status()['timer']['last_error']
        ref = InputRef(store.store_id, target.session_id, calls[0])
        assert (await runtime.client.wait(ref, actor=actor, deadline_ms=now_ms()+10000)).status == 'completed'
        assert await runtime.timer_driver.scan_once() == 0
        assert len(calls) == 1
        monkeypatch.setattr(runtime.client, 'append', original)
        next_timer = await runtime.timer_backend.create(source, delay_s=-1, note='new invocation')
        assert await runtime.timer_driver.scan_once() == 1
        ref = InputRef(store.store_id, target.session_id, f'timer-{next_timer.timer_id}-{next_timer.created_at}-0')
        assert (await runtime.client.wait(ref, actor=actor, deadline_ms=now_ms()+10000)).status == 'completed'
