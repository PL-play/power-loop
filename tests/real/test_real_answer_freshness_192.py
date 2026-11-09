"""A corrected node produces a fresh answer; its earlier answer cannot satisfy the correction."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, TargetRef
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import get_session_id

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_corrected_node_does_not_deliver_old_text_after_empty_pass(runtime_agent, record_property, monkeypatch):
    provider=BoundedProvider(limit=12)
    agent=await runtime_agent(model=provider,config=AgentLoopConfig(max_rounds=5,max_tokens=2048,compactor=None,
        system_prompt='Report the computed workflow answer in assistant text, without tools.'))
    node_sids=[]
    complete=provider.complete
    async def corrected(request,**kwargs):
        response=await complete(request,**kwargs)
        sid=get_session_id()
        if sid!=agent.target.session_id and not node_sids:
            assert response.raw_text.strip()=='42' and not response.get_tool_calls()
            node_sids.append(sid)
            await agent.runtime.client.send(TargetRef(agent.runtime.store.store_id,sid),
                "Correction: the current answer is 43. Call pass_turn exactly once with reason='Computed 43' and no assistant text. On a later final-answer request write 43 as assistant text.",
                message_id='correction',actor=agent.actor)
        return response
    monkeypatch.setattr(provider,'complete',corrected)
    spec={'name':'answer-freshness','root':{'type':'agent','id':'answer',
        'input':'Return exactly 42 as plain assistant text without tools. Follow any later correction.',
        'spec':{'name':'answer','system_prompt':'Follow each new instruction. The latest numeric answer replaces earlier ones. When asked for a final answer with no tools, write the latest number only.','tools':[],'max_rounds':6}}}
    try:
        async with asyncio.timeout(180):
            handle=await agent.runtime.client.workflow(agent.target,spec,message_id='one-workflow',actor=agent.actor)
            result=await handle.wait(deadline_ms=now_ms()+150000)
            await agent.drain()
            assert result.status=='completed' and result.results['answer'].text.strip()=='43'
            [sid]=node_sids
            rows=await agent.runtime.store.load_all_messages(sid)
            index=next(i for i,m in enumerate(rows) if m.role=='user' and m.name=='follow_up' and 'Correction:' in str(m.content))
            assert any(m.role=='assistant' and (m.content or '').strip()=='42' for m in rows[:index])
            first=next(m for m in rows[index+1:] if m.role=='assistant')
            assert not (first.content or '').strip()
            assert [c['function']['name'] for c in first.tool_calls or []]==['pass_turn']
            assert any(m.role=='assistant' and (m.content or '').strip()=='43' for m in rows[index+1:])
            record_property('node_transcript',json.dumps([{'role':m.role,'content':m.content,'tool_calls':m.tool_calls} for m in rows]))
    finally:
        await agent.aclose()
        provider.report(record_property)
