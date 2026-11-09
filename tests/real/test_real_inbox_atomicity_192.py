"""Withdrawn current inputs never reach a real provider or stored model history."""
import asyncio
import contextvars
import json

import pytest

from power_loop import AgentHooks, AgentLoopConfig, HookPoint
from power_loop.agent.runtime_store import now_ms

from ._bounded import BoundedProvider

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


@pytest.mark.parametrize('phase,keep',[('idle',True),('terminal',False),('terminal',True)])
async def test_real_withdrawn_input_never_reaches_provider(runtime_agent,record_property,phase,keep):
    requests=[]
    class Provider(BoundedProvider):
        async def complete(self,request,**kwargs):
            requests.append(request.messages)
            response=await super().complete(request,**kwargs)
            if self.calls==1 and phase=='terminal':
                async def add_and_cancel():
                    drop=await client.send(agent.target,'WITHDRAWN-DO-NOT-READ',message_id='drop',actor=agent.actor)
                    if keep:
                        await client.send(agent.target,'Return READY again.',message_id='keep',actor=agent.actor)
                    ctx=agent.runtime.worker._active[agent.target.session_id]
                    async with asyncio.timeout(5):
                        while len(ctx.append_refs)<1+int(keep):
                            await asyncio.sleep(.005)
                    assert (await client.cancel(drop.input_ref,actor=agent.actor)).job_ref is None
                await asyncio.create_task(add_and_cancel(),context=contextvars.Context())
            return response
    provider=Provider(limit=3)
    hooks=AgentHooks()
    async def no_tools(ctx):
        ctx.tools=[]
    hooks.register(HookPoint.LLM_BEFORE,no_tools,name='test.reply-only')
    agent=await runtime_agent(model=provider,hooks=hooks,config=AgentLoopConfig(compactor=None,max_rounds=3,max_tokens=2048,
        system_prompt='Return READY in assistant text.'))
    client=agent.runtime.client
    try:
        if phase=='idle':
            gate=await agent.runtime.coordinator.acquire((agent.runtime.store.store_id,agent.target.session_id),owner_id='test-gate',ttl_s=30,on_lost=lambda:None)
            try:
                drop=await client.send(agent.target,'WITHDRAWN-DO-NOT-READ',message_id='drop',actor=agent.actor)
                survivor=await client.send(agent.target,'Return READY.',message_id='keep',actor=agent.actor)
                assert (await client.cancel(drop.input_ref,actor=agent.actor)).job_ref is None
            finally:
                await agent.runtime.coordinator.release(gate)
            result=await client.wait(survivor.input_ref,actor=agent.actor,deadline_ms=now_ms()+120000)
        else:
            result=await agent.send('Return READY.')
        assert result.status=='completed' and result.final_text
        await agent.drain()
        assert provider.calls==(2 if phase=='terminal' and keep else 1)
        assert 'WITHDRAWN-DO-NOT-READ' not in json.dumps(requests)
        assert all('_inbox' not in message for request in requests for message in request)
        store=agent.runtime.store
        assert 'WITHDRAWN-DO-NOT-READ' not in str([r.content for r in await store.load_all_messages(agent.target.session_id)])
        assert (await store.inbox_counts(agent.target.session_id))['pending']==0
    finally:
        provider.report(record_property)
