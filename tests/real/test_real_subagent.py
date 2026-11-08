"""Real delegation and structured results through current worker composition."""
import json

import pytest

from power_loop import AgentLoopConfig, AgentSpec, JobRef, SubagentLifecycle, ToolDefinition, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import get_current_loop
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import run_agent_spec
from power_loop.tools.spawn_agent import register_spawn_agent

from ._bounded import BoundedProvider
from ._live_fixture import live_agent, settle, tool_receipt

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


async def test_parent_can_spawn_subagent_via_tool(tmp_path):
    model=BoundedProvider(limit=7)
    registry=ToolRegistry()
    register_spawn_agent(registry)
    async with live_agent(tmp_path,model,registry,AgentLoopConfig(system_prompt=
        'Delegate once with spawn_agent; after subagent_result report the answer.',max_rounds=5,max_tokens=2048,compactor=None)) as runner:
        await runner.send('Delegate this and report back: what is the capital of Japan? Give the child tools=[]; after acceptance end the turn.')
        receipt=await tool_receipt(runner,'spawn_agent')
        result=await runner.runtime.client.wait(JobRef(**receipt['job_ref']),actor=runner.actor,deadline_ms=now_ms()+120000)
        assert result.status=='completed' and any(word in result.final_text.lower() for word in ('tokyo','东京'))
        await settle(runner)
        children=await runner.runtime.store.list_children(runner.target.session_id)
        assert len(children)==1 and children[0].lifecycle is SubagentLifecycle.LINKED
        rows=await runner.runtime.store.load_all_messages(runner.target.session_id)
        assert any(word in [row.content for row in rows if row.role=='assistant' and row.content][-1].lower() for word in ('tokyo','东京'))


async def test_run_agent_spec_with_linked_lifecycle_persists_child(tmp_path):
    model=BoundedProvider(limit=2)
    registry=ToolRegistry()
    async def delegate():
        return await run_agent_spec(AgentSpec(name='math-helper',system_prompt='Reply with the number only.',
            tools=[],max_rounds=1,max_tokens=128,lifecycle='linked'),'What is 17+25?',parent_loop=get_current_loop())
    registry.register(ToolDefinition('delegate','Host delegation',pricing=Free('composite')),delegate)
    async with live_agent(tmp_path,model,registry,AgentLoopConfig(compactor=None),command=('delegate',{})) as runner:
        result=await runner.send('explicit host command')
        assert result.status=='completed' and '42' in result.final_text
        children=await runner.runtime.store.list_children(runner.target.session_id)
        assert len(children)==1 and children[0].lifecycle is SubagentLifecycle.LINKED
        rows=await runner.runtime.store.load_all_messages(children[0].session_id)
        assert any(row.role=='assistant' and '42' in str(row.content) for row in rows)


async def test_spawn_agent_returns_parsed_json_without_native_json_schema(tmp_path):
    model=BoundedProvider(limit=6)
    # The provider still executes a real request; this fixture uses prompt-based
    # schema output to exercise the local parser instead of provider validation.
    complete=model.complete
    async def no_native(request,**kwargs):
        request.response_format=None
        return await complete(request,**kwargs)
    model.complete=no_native
    schema={'type':'object','properties':{'priority':{'type':'string'},'team':{'type':'string'}},'required':['priority','team']}
    registry=ToolRegistry()
    register_spawn_agent(registry)
    async with live_agent(tmp_path,model,registry,AgentLoopConfig(max_rounds=5,max_tokens=2048,compactor=None)) as runner:
        await runner.send('Use spawn_agent once. Ask the child to triage checkout returning 500 for every user; payments are down. Tell it to return JSON with urgent priority and an owning team. Supply output_schema='+json.dumps(schema)+'. Give it tools=[] and finish after acceptance.')
        receipt=await tool_receipt(runner,'spawn_agent')
        result=await runner.runtime.client.wait(JobRef(**receipt['job_ref']),actor=runner.actor,deadline_ms=now_ms()+120000)
        assert result.status=='completed',result
        assert result.structured_output['priority']=='urgent'
        assert result.structured_output['team'].strip()
