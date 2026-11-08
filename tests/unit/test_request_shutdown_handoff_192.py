"""Runtime shutdown/cancel stop original requests; cancelling a waiter does not."""
import asyncio

import pytest

from power_loop import LLMResponse, LLMService
from power_loop.agent.runtime_store import now_ms
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('stop_kind', ['shutdown', 'cancel'])
async def test_request_cancellation_runs_provider_cleanup_once(registry_store, stop_kind):
    store, _ = registry_store
    entered, cleaned, signals = asyncio.Event(), asyncio.Event(), []
    class Model(LLMService):
        async def complete(self, request, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                signals.append('request-cancelled')
                try:
                    await asyncio.sleep(.02)
                    cleaned.set()
                except asyncio.CancelledError:
                    signals.append('cleanup-cancelled-again')
                    raise
                raise
        async def close(self):
            pass
    agent = await RunningAgent(store, Model()).start()
    try:
        receipt = await agent.runtime.client.send(agent.target, 'original request', message_id='original', actor=agent.actor)
        await asyncio.wait_for(entered.wait(), 5)
        if stop_kind == 'shutdown':
            await agent.runtime.worker.shutdown(timeout_s=2, cancel_running=True)
        else:
            await agent.runtime.client.cancel(receipt.input_ref, actor=agent.actor, reason='user stop')
        result = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+10000)
        assert result.status == 'cancelled'
        assert signals == ['request-cancelled'] and cleaned.is_set()
        assert (await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+10000)) == result
        assert signals == ['request-cancelled']
    finally:
        await agent.aclose()


async def test_cancelled_waiter_does_not_cancel_or_replace_original_model(registry_store):
    store, _ = registry_store
    entered, release = asyncio.Event(), asyncio.Event()
    class Model(LLMService):
        calls = 0
        cancelled = False
        async def complete(self, request, **kwargs):
            self.calls += 1
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return LLMResponse(raw_text='original answer')
        async def close(self):
            pass
    model = Model()
    agent = await RunningAgent(store, model).start()
    try:
        receipt = await agent.runtime.client.send(agent.target, 'original request', message_id='original', actor=agent.actor)
        await asyncio.wait_for(entered.wait(), 5)
        waiter = asyncio.create_task(agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+10000))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not model.cancelled and model.calls == 1
        release.set()
        result = await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+10000)
        assert result.status == 'completed' and result.final_text == 'original answer'
        assert model.calls == 1 and not model.cancelled
        duplicate = await agent.runtime.client.send(agent.target, 'original request', message_id='original', actor=agent.actor)
        assert not duplicate.newly_accepted
        assert (await agent.runtime.client.wait(duplicate.input_ref, actor=agent.actor, deadline_ms=now_ms()+10000)) == result
        assert model.calls == 1
    finally:
        release.set()
        await agent.aclose()
