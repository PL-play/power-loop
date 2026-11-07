"""Actual model input and normal consumption, without post-fault proofs."""
import asyncio
import json

import pytest

from power_loop import AgentLoopConfig, LLMResponse
from power_loop.contracts.hooks import HookDirective, HookPoint
from power_loop.core.hooks import AgentHooks
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_subagent_runtime_203 import ParentAndChild

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('mode', ['included', 'copied', 'removed', 'rewritten', 'short_circuit', 'break'])
async def test_only_actual_result_input_is_consumed_and_later_request_can_continue(registry_store, mode):
    store, _ = registry_store
    model, registry, hooks = ParentAndChild(), ToolRegistry(), AgentHooks()
    model.child_gate = asyncio.Event()
    register_spawn_agent(registry)
    state = {'filter': True}

    def filter_result(ctx):
        if not state['filter']:
            return
        sources = [m for m in ctx.messages if '[subagent_result]' in str(m.get('content', ''))]
        if not sources:
            return
        if mode == 'copied':
            ctx.messages = [dict(m) for m in ctx.messages]
        elif mode == 'removed':
            ctx.messages = [m for m in ctx.messages if m not in sources]
        elif mode == 'rewritten':
            ctx.messages = [{**m, 'content': 'result excluded'} if m in sources else m for m in ctx.messages]
        elif mode == 'short_circuit':
            ctx.output = LLMResponse(raw_text='no provider request')
            ctx.directive = HookDirective.SHORT_CIRCUIT
        elif mode == 'break':
            ctx.directive = HookDirective.BREAK

    hooks.register(HookPoint.LLM_BEFORE, filter_result)
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None), hooks=hooks).start()
    try:
        await runner.send('delegate')
        await asyncio.wait_for(model.child_started.wait(), 5)
        table = store.table_prefix + 'jobs'
        child = await store._db.fetchone(f"SELECT * FROM {table} WHERE kind='subagent'")
        parent = await store._db.fetchone(f"SELECT * FROM {table} WHERE job_id=?", (child['parent_job_id'],))
        assert child['resource_state'] == 'running' and child['supervision_status'] == 'open'
        assert parent['supervision_status'] == 'open'
        model.child_gate.set()
        await asyncio.wait_for(runner.drain(), 15)
        child = await store._db.fetchone(f"SELECT * FROM {table} WHERE job_id=?", (child['job_id'],))
        assert child['result_status'] == ('handled' if mode in {'included', 'copied'} else 'available')
        assert child['supervision_status'] == ('closed' if mode in {'included', 'copied'} else 'open')
        assert not await store._db.fetchall(f"SELECT event_id FROM {store.table_prefix}job_events "
            "WHERE event_type IN ('processing_request','processing_decision')")
        state['filter'] = False
        result = await runner.send('Read the original result and continue')
        assert result.status == 'completed'
        child = await store._db.fetchone(f"SELECT * FROM {table} WHERE job_id=?", (child['job_id'],))
        assert child['result_status'] == 'handled' and child['supervision_status'] == 'closed'
        [original] = await store._db.fetchall(f"SELECT payload_json FROM {store.table_prefix}job_events "
            "WHERE job_id=? AND event_type='result'", (child['job_id'],))
        assert json.loads(original['payload_json'])['generation'] == 1
        messages = await store.load_all_messages(runner.target.session_id)
        assert sum('[subagent_result]' in (m.content or '') for m in messages) == 1
    finally:
        model.child_gate.set()
        await runner.aclose()
