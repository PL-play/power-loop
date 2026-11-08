"""Live providers exercise asynchronous tool results and accepted message control."""
import asyncio
import secrets

import pytest

from power_loop import AgentLoopConfig, JobRef, ToolDefinition, create_default_tool_registry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.pricing import Free
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.tools.spawn_agent import register_spawn_agent

from ._bounded import BoundedProvider
from ._live_fixture import live_agent, settle, tool_receipt

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


@pytest.mark.parametrize('ending',['natural','pass_turn'])
async def test_real_catalog_child_result_returns_after_parent_yields(tmp_path,ending,record_property):
    model=BoundedProvider(limit=7)
    registry=create_default_tool_registry(include=['recall_send'],workspace_dir=tmp_path)
    register_spawn_agent(registry)
    config=AgentLoopConfig(system_prompt='Follow the requested sequence. On subagent_result report its answer without starting another task.',
        max_rounds=4,max_tokens=2048,representation=ProjectedRepresentation())
    async with live_agent(tmp_path,model,registry,config) as runner:
        model.root_sid,model.hold_children=runner.target.session_id,True
        instruction='call pass_turn' if ending=='pass_turn' else 'finish with a short text and no tools'
        try:
            first=await runner.send('Use spawn_agent exactly once to calculate 19+23. Give the child tools=[] and tell it to return only the number. After acceptance '+instruction+'.')
            await asyncio.wait_for(model.child_entered.wait(),10)
            assert first.status=='completed'
            receipt=await tool_receipt(runner,'spawn_agent')
            ref=JobRef(**receipt['job_ref'])
            assert not (await runner.runtime.client.inspect(ref,actor=runner.actor)).result_available
            model.release_children.set()
            result=await runner.runtime.client.wait(ref,actor=runner.actor,deadline_ms=now_ms()+120000)
            assert result.status=='completed' and '42' in result.final_text and result.usage['total_tokens']>0
            await settle(runner)
            rows=await runner.runtime.store.load_all_messages(runner.target.session_id)
            assert sum('[subagent_result]' in str(row.content) for row in rows if row.role=='user')==1
            assert '42' in [row.content for row in rows if row.role=='assistant' and row.content][-1]
            assert len(await runner.runtime.store.list_children(runner.target.session_id))==1
        finally:
            model.release_children.set()
            model.report(record_property)


async def test_real_catalog_background_result_is_consumed_after_pass(tmp_path,record_property):
    model=BoundedProvider(limit=6)
    entered,release=asyncio.Event(),asyncio.Event()
    proof='LOCAL-PROOF-'+secrets.token_hex(6)
    registry=create_default_tool_registry(include=['background_run'],workspace_dir=tmp_path)
    async def evidence():
        entered.set()
        await release.wait()
        return proof
    registry.register(ToolDefinition('evidence','Read local evidence.',async_capable=True,pricing=Free('fixture')),evidence)
    config=AgentLoopConfig(system_prompt='Follow instructions; on the background result report its exact code.',max_rounds=4,max_tokens=2048)
    async with live_agent(tmp_path,model,registry,config) as runner:
        try:
            first=await runner.send('Start evidence using background_run action=tool, tool=evidence, args={}. After receipt call pass_turn. Do not call evidence directly or poll.')
            assert first.status=='completed'
            await asyncio.wait_for(entered.wait(),10)
            receipt=await tool_receipt(runner,'background_run')
            release.set()
            result=await runner.runtime.client.wait(JobRef(**receipt['job_ref']),actor=runner.actor,deadline_ms=now_ms()+10000)
            assert result.status=='completed' and result.final_text==proof
            await settle(runner)
            rows=await runner.runtime.store.load_all_messages(runner.target.session_id)
            assert proof in [row.content for row in rows if row.role=='assistant' and row.content][-1]
        finally:
            release.set()
            model.report(record_property)


async def test_real_child_consumes_input_arriving_inside_pass(tmp_path,record_property):
    from power_loop import TargetRef
    from power_loop.core.agent_context import get_session_id
    from power_loop.core.pass_turn import request_pass
    model=BoundedProvider(limit=7)
    proof='CORRECTION-'+secrets.token_hex(6)
    registry=create_default_tool_registry(include=[],bind=False)
    async def boundary_pass():
        request_pass('waiting for correction')
        await runner.runtime.client.append(TargetRef(runner.runtime.store.store_id,get_session_id()),proof,
            message_id='correction',actor=runner.actor)
        return 'Passed control.'
    registry.register(ToolDefinition('boundary_pass','Yield for correction.',pricing=Free('fixture')),boundary_pass)
    register_spawn_agent(registry)
    async with live_agent(tmp_path,model,registry,AgentLoopConfig(max_rounds=5,max_tokens=2048)) as runner:
        await runner.send('Use spawn_agent once. Give the child only boundary_pass. Tell it to call boundary_pass once, then reply with the exact correction code it receives. After acceptance end your turn.')
        receipt=await tool_receipt(runner,'spawn_agent')
        result=await runner.runtime.client.wait(JobRef(**receipt['job_ref']),actor=runner.actor,deadline_ms=now_ms()+120000)
        assert result.status=='completed' and proof in result.final_text
        await settle(runner)
        model.report(record_property)


async def test_real_stop_mid_generation_preserves_usage_and_can_continue(tmp_path,record_property):
    model=BoundedProvider(limit=3)
    started=asyncio.Event()
    complete=model.complete
    async def streamed(request,**kwargs):
        callback=kwargs.get('on_chunk_delta_text')
        def on_text(text):
            if text:
                started.set()
            if callback:
                return callback(text)
        return await complete(request,**{**kwargs,'on_chunk_delta_text':on_text})
    model.complete=streamed
    async with live_agent(tmp_path,model,config=AgentLoopConfig(max_rounds=2,max_tokens=2048)) as runner:
        receipt=await runner.runtime.client.send(runner.target,'Write all integers 1 through 10000, one per line. Do not summarize.',message_id='long',actor=runner.actor)
        await asyncio.wait_for(started.wait(),60)
        await runner.runtime.client.cancel(receipt.input_ref,actor=runner.actor)
        result=await runner.runtime.client.wait(receipt.input_ref,actor=runner.actor,deadline_ms=now_ms()+15000)
        assert result.status=='cancelled' and result.usage['total_tokens']>0
        await settle(runner)
        follow=await runner.send('Stop listing. Reply only OK.')
        assert follow.status=='completed' and 'OK' in follow.final_text.upper()
        model.report(record_property)
