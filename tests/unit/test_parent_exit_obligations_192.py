"""A child remains queryable and controllable after its parent's execution ends."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, JobRef, LLMResponse, LLMService, LLMTokenUsage
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.budget_contracts import LocalBudgetScope
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_budget_tree_192 import tree_case
from .test_job_registry_192 import registry_store

__all__ = ['registry_store', 'tree_case']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('tree_case', [{'scopes':(LocalBudgetScope('authority', max_calls=20),)}], indirect=True)
@pytest.mark.parametrize('exit_kind', ['hit_round_limit', 'context_checkpoint', 'budget_exceeded', 'provider_error'])
@pytest.mark.parametrize('child_outcome', ['completed', 'cancelled'])
async def test_parent_exit_preserves_child_control_result_and_known_fees(tree_case, runtime_agent, exit_kind, child_outcome):
    c = tree_case
    entered, release = asyncio.Event(), asyncio.Event()
    class Model(LLMService):
        root = None
        root_calls = 0
        child_calls = 0
        async def complete(self, request, **kwargs):
            if get_session_id() == self.root:
                self.root_calls += 1
                if self.root_calls == 1:
                    return LLMResponse(raw_text='', tool_calls=[{'id':'child', 'type':'function', 'function':{
                        'name':'spawn_agent', 'arguments':json.dumps({'task':'work', 'tools':[], 'max_rounds':2})}}],
                        token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
                if exit_kind == 'provider_error':
                    raise RuntimeError('original parent provider failed')
                return LLMResponse(raw_text='parent closed')
            self.child_calls += 1
            entered.set()
            await release.wait()
            return LLMResponse(raw_text='child result', token_usage=LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
        async def close(self):
            pass
    model, registry = Model(), ToolRegistry()
    c.host.service = model
    register_spawn_agent(registry)
    agent = await runtime_agent(store=c.store, model=model, registry=registry, task_runtime=c.runtime,
        config=AgentLoopConfig(max_rounds=1 if exit_kind=='hit_round_limit' else 3,
            max_tokens=20, compactor=None, context_checkpoint_tokens=1 if exit_kind=='context_checkpoint' else None,
            max_tokens_per_run=1 if exit_kind=='budget_exceeded' else None))
    model.root = agent.target.session_id
    try:
        parent = await agent.send('Launch child once.')
        assert parent.status == ('failed' if exit_kind=='provider_error' else 'interrupted'), parent
        if exit_kind != 'provider_error':
            assert parent.error == exit_kind
        await asyncio.wait_for(entered.wait(), 10)
        rows = await c.store.load_all_messages(model.root)
        receipt = json.loads(next(row.content for row in rows if row.tool_call_id=='child'))
        ref = JobRef(**receipt['job_ref'])
        running = await agent.runtime.client.inspect(ref, actor=agent.actor)
        assert running.state == 'running'
        applied = dict(c.host.applied)
        assert parent.usage['total_tokens'] >= 10 and applied
        if child_outcome == 'cancelled':
            await agent.runtime.client.cancel(running.job_ref, actor=agent.actor, reason='stop after parent exit')
        else:
            release.set()
        result = await agent.runtime.client.wait(ref, actor=agent.actor, deadline_ms=now_ms()+15000)
        assert result.status == child_outcome and model.child_calls == 1
        if child_outcome == 'completed':
            assert result.final_text == 'child result' and result.usage['total_tokens'] == 10
        await agent.drain()
        assert (await agent.runtime.client.wait(parent.input_ref, actor=agent.actor, deadline_ms=now_ms()+1000)) == parent
        assert all(c.host.applied[key] == value for key, value in applied.items())
        assert len(c.host.applied) >= len(applied) + int(child_outcome=='completed')
    finally:
        release.set()
        await agent.aclose()
