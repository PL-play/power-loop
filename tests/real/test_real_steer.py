"""A real provider accepts history repaired after model/tool steering."""
import asyncio
from typing import Any

import pytest

from power_loop import AgentLoopConfig, ToolDefinition, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.pricing import Free

from ._bounded import BoundedProvider
from ._live_fixture import live_agent, settle

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


async def test_real_steer_cuts_a_long_generation(tmp_path):
    model=BoundedProvider(limit=3)
    started=asyncio.Event()
    chars=0
    complete=model.complete
    async def stream(request,**kwargs):
        callback=kwargs.get('on_chunk_delta_text')
        def delta(text):
            nonlocal chars
            chars+=len(text or '')
            if chars>=300:
                started.set()
            if callback:
                return callback(text)
        return await complete(request,**{**kwargs,'on_chunk_delta_text':delta})
    model.complete=stream
    async with live_agent(tmp_path,model,config=AgentLoopConfig(max_rounds=3,max_tokens=6000,compactor=None)) as runner:
        receipt=await runner.runtime.client.send(runner.target,'写一篇至少5000字的小说，一直写下去。',message_id='novel',actor=runner.actor)
        await asyncio.wait_for(started.wait(),60)
        status=await runner.runtime.client.inspect(receipt.input_ref,actor=runner.actor)
        await runner.runtime.client.send(status.job_ref, '停止小说。只回答2+2等于几，只回一个数字。', urgent=True, message_id='steer',actor=runner.actor)
        result=await runner.runtime.client.wait(receipt.input_ref,actor=runner.actor,deadline_ms=now_ms()+120000)
        assert result.status=='completed' and '4' in result.final_text
        rows=await runner.runtime.store.load_all_messages(runner.target.session_id)
        answers=[row for row in rows if row.role=='assistant']
        assert len(answers)==1 and len(answers[0].content or '')<200


def _slow_registry(interrupt: str, state: dict[str, Any]) -> ToolRegistry:
    reg = ToolRegistry()

    async def research(topic: str = "", **kw: Any) -> str:
        state["calls"] = state.get("calls", 0) + 1
        state["started"].set()
        try:
            await asyncio.sleep(4)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        state["finished"] = True
        return f"Research on {topic}: the market grew 12% in 2025."

    reg.register(ToolDefinition(
        name="research", description="Slow web research on a topic (takes a while).",
        input_schema={"type": "object", "properties": {"topic": {"type": "string"}},
                      "required": ["topic"]},
        interrupt=interrupt, pricing=Free("test_fixture")), research)
    return reg



@pytest.mark.parametrize('interrupt',['background','abort'])
async def test_real_steer_preserves_tool_policy_and_provider_format(tmp_path,interrupt):
    model=BoundedProvider(limit=6)
    state={'started':asyncio.Event()}
    registry=_slow_registry(interrupt,state)
    config=AgentLoopConfig(system_prompt='Use research when requested; honor corrections without repeating the original call.',
        max_rounds=4,max_tokens=2048,compactor=None)
    async with live_agent(tmp_path,model,registry,config) as runner:
        receipt=await runner.runtime.client.send(runner.target,'Research electric bikes.',message_id='research',actor=runner.actor)
        await asyncio.wait_for(state['started'].wait(),60)
        status=await runner.runtime.client.inspect(receipt.input_ref,actor=runner.actor)
        await runner.runtime.client.send(status.job_ref, 'Do not repeat research. What is the capital of France? Reply briefly.', urgent=True, message_id='steer',actor=runner.actor)
        result=await runner.runtime.client.wait(receipt.input_ref,actor=runner.actor,deadline_ms=now_ms()+120000)
        assert result.status=='completed' and 'paris' in result.final_text.lower()
        await settle(runner)
        assert state['calls']==1
        if interrupt=='background':
            assert state.get('finished') and not state.get('cancelled')
        else:
            assert state.get('cancelled') and not state.get('finished')
            rows=await runner.runtime.store.load_all_messages(runner.target.session_id)
            assert any(row.role=='tool' and 'interrupted' in str(row.content) for row in rows)
