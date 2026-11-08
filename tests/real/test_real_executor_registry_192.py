"""Worker shutdown refuses further provider work and leaves original results readable."""
import asyncio

import pytest

from power_loop import AgentLoopConfig, RuntimeRequestError
from power_loop.agent.runtime_store import now_ms

from ._bounded import BoundedProvider

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


async def test_real_worker_drain_rejects_another_request_without_spending(runtime_agent,record_property):
    provider=BoundedProvider(limit=3)
    agent=await runtime_agent(model=provider,config=AgentLoopConfig(compactor=None,max_rounds=2,max_tokens=2048))
    try:
        result=await agent.send('Reply exactly 42 as plain assistant text without tools.')
        assert result.status=='completed' and result.final_text.strip()=='42' and provider.calls==1
        assert (await agent.runtime.worker.shutdown(timeout_s=3)).drained
        with pytest.raises(RuntimeRequestError,match='session_closed'):
            await agent.runtime.client.send(agent.target,'must not spend',message_id='refused',actor=agent.actor)
        assert await agent.runtime.client.wait(result.input_ref,actor=agent.actor,deadline_ms=now_ms()+1000)==result
        assert provider.calls==1
    finally:
        provider.report(record_property)


async def test_real_inflight_shutdown_stops_owned_request_without_replay(runtime_agent,record_property,monkeypatch):
    provider=BoundedProvider(limit=3)
    entered=asyncio.Event()
    complete=provider.complete
    async def started(request,**kwargs):
        task=asyncio.create_task(complete(request,**kwargs))
        while provider.calls==0:
            await asyncio.sleep(0)
        entered.set()
        return await task
    monkeypatch.setattr(provider,'complete',started)
    agent=await runtime_agent(model=provider,config=AgentLoopConfig(compactor=None,max_rounds=2,max_tokens=2048))
    try:
        receipt=await agent.runtime.client.send(agent.target,'Write a detailed arithmetic explanation.',message_id='original',actor=agent.actor)
        await asyncio.wait_for(entered.wait(),30)
        assert (await agent.runtime.worker.shutdown(timeout_s=5,cancel_running=True)).drained
        result=await agent.runtime.client.wait(receipt.input_ref,actor=agent.actor,deadline_ms=now_ms()+1000)
        assert result.status=='cancelled' and provider.calls==1
        with pytest.raises(RuntimeRequestError,match='session_closed'):
            await agent.runtime.client.send(agent.target,'must not spend',message_id='next',actor=agent.actor)
        assert provider.calls==1
        record_property('provider_server_receipt_confirmed',False)
    finally:
        provider.report(record_property)
