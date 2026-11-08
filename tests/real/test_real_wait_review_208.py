"""Real model decisions use the original finite child and current runtime ports."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, JobRef, SessionStore, ToolRegistry
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.wait_review import WaitPolicy
from power_loop.tools.spawn_agent import register_spawn_agent
from power_loop.tools.wait_handoff import register_wait_handoff_tool
from tests.runtime_fixture import RunningAgent

from ._bounded import BoundedProvider

pytestmark=[pytest.mark.real_llm,pytest.mark.asyncio]


@pytest.mark.parametrize('choice',['continue','cancel','defer'])
async def test_real_original_wait_decision_and_result(tmp_path,record_property,choice):
    store=await SessionStore.open(str(tmp_path/'wait-review.db'),schema='tasks')
    provider=BoundedProvider(limit=16)
    registry=ToolRegistry()
    register_spawn_agent(registry)
    deadline=now_ms()+120000
    register_wait_handoff_tool(registry,policy=WaitPolicy(deadline))
    prompt=('Initially create exactly one child using spawn_agent. Then reply WAITING. '
        'For each later Task data notification use wait_handoff decide with the exact original '
        'job_id, generation and event_id. Never replace or respawn the child. '
        +('First defer to now_ms plus 1000, and continue on the next notification.' if choice=='defer'
          else 'Choose '+choice+'.')+' After each accepted decision reply ACKNOWLEDGED.')
    runner=await RunningAgent(store,provider,registry,AgentLoopConfig(compactor=None,
        max_rounds=6,max_tokens=2048,system_prompt=prompt),capacity=1).start()
    try:
        due=now_ms()+3000
        task=(f'Create one child, tools=["wait_handoff"], max_rounds=4, task="Compute 19+23", '
            f'system_prompt="First call wait_handoff action=request, due_at_ms={due}, '
            'note=Review arithmetic, continuation=Finish the original 19+23 computation. '
            'After its answer, reply 42 without more tool calls." Then reply WAITING.')
        async with asyncio.timeout(140):
            await runner.send(task)
            await runner.drain()
            [child]=await store._db.fetchall(f"SELECT * FROM {store.table_prefix}jobs WHERE kind='subagent'")
            ref=JobRef(store.store_id,child['job_id'],child['generation'])
            result=await runner.runtime.client.wait(ref,actor=runner.actor,deadline_ms=deadline)
            await runner.drain()
            assert result.status==('cancelled' if choice=='cancel' else 'completed')
            if choice!='cancel':
                assert '42' in result.final_text
            assert (await store._db.fetchone(f"SELECT COUNT(*) AS n FROM {store.table_prefix}jobs WHERE kind='subagent'"))['n']==1
            history=await store.load_all_messages(runner.target.session_id)
            decisions=[json.loads(m.content) for m in history if m.role=='tool' and m.name=='wait_handoff']
            assert [d['decision'] for d in decisions if 'decision' in d]==(['defer','continue'] if choice=='defer' else [choice])
            record_property('original_job_ref',json.dumps(ref.__dict__))
    finally:
        await runner.aclose()
        provider.report(record_property)
        await provider.close()
        await store.close()
