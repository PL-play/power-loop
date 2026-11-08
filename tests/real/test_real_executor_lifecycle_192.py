"""Actual provider work under Runtime startup and bounded worker shutdown."""
import asyncio

import pytest

from power_loop import AgentLoopConfig
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import RuntimeRequestError

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_shutdown_retains_known_usage_and_rejects_new_work(runtime_agent, record_property):
    provider = BoundedProvider(limit=3)
    agent = await runtime_agent(model=provider,
        config=AgentLoopConfig(max_rounds=2, max_tokens=2048, compactor=None))
    try:
        result = await agent.send('Reply exactly 42 in plain assistant text, without tools or pass_turn.')
        assert result.final_text.strip() == '42' and provider.calls == 1
        assert result.usage['total_tokens'] > 0
        stopped = await agent.runtime.worker.shutdown(timeout_s=5)
        assert stopped.drained
        with pytest.raises(RuntimeRequestError, match='session_closed'):
            await agent.send('Must not dispatch.')
        saved = await agent.runtime.client.wait(result.input_ref, actor=agent.actor, deadline_ms=now_ms()+1000)
        assert saved == result and provider.calls == 1
    finally:
        provider.report(record_property)


async def test_real_shutdown_cancels_original_inflight_request(runtime_agent, record_property, monkeypatch):
    provider, entered = BoundedProvider(limit=3), asyncio.Event()
    complete = provider.complete
    async def started(request, **kwargs):
        work = asyncio.create_task(complete(request, **kwargs))
        while provider.calls == 0:
            await asyncio.sleep(0)
        entered.set()
        return await work
    monkeypatch.setattr(provider, 'complete', started)
    agent = await runtime_agent(model=provider,
        config=AgentLoopConfig(max_rounds=2, max_tokens=2048, compactor=None))
    client = agent.runtime.client
    try:
        receipt = await client.send(agent.target, 'Write a detailed arithmetic explanation without tools.',
            message_id='original', actor=agent.actor)
        await asyncio.wait_for(entered.wait(), 30)
        status = await client.inspect(receipt.input_ref, actor=agent.actor)
        assert status.state == 'running'
        await agent.runtime.worker.shutdown(timeout_s=5, cancel_running=True)
        result = await client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+1000)
        assert result.status == 'cancelled' and provider.calls == 1
        with pytest.raises(RuntimeRequestError, match='session_closed'):
            await client.send(agent.target, 'Do not replace the original request.', message_id='rejected', actor=agent.actor)
        assert provider.calls == 1
    finally:
        provider.report(record_property)
