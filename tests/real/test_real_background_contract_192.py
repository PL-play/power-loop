"""A live model uses an aliased scheduler and the implementation-owned schema."""

import asyncio
import json
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, HookPoint, JobRef, SessionStore, ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.tool_context import ToolContextConfig
from power_loop.tools import create_default_tool_registry
from power_loop.tools.registry import ToolRegistry

from ._bounded import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('alias', ['defer_work', 'submit_original'])
async def test_real_renamed_scheduler_uses_actual_contract(tmp_path, record_property, alias, runtime_agent):
    provider = BoundedProvider(limit=3)
    store = await SessionStore.open(str(tmp_path / 'alias.db'))
    defaults, registry = create_default_tool_registry(bind=False), ToolRegistry()
    source = defaults.get('background_run')
    definition = replace(source.definition, name=alias)
    registry.register(definition, source.handler)
    calls = []

    async def worker(**args):
        calls.append(args)
        await asyncio.sleep(0)
        return 'WORK_OK'

    # This plain tool happens to share an orchestration entry's customary name.
    registry.register(ToolDefinition('workflow', '', async_capable=True, pricing=Free("test_fixture")), worker)
    loop = await runtime_agent(model=provider, store=store, registry=registry,
        config=AgentLoopConfig(max_rounds=3, max_tokens=2048, retry_policy=LLMRetryPolicy(max_attempts=1),
                               tool_context=ToolContextConfig(capacity_enabled=False)))

    async def schema(ctx):
        scheduler = next(s for s in ctx.tools if s['function']['name'] == alias)
        assert 'command' not in scheduler['function']['parameters']['properties']
        assert 'run' not in scheduler['function']['parameters']['properties']['action']['enum']
        ctx.tools = [scheduler] if provider.calls == 0 else []

    loop.hooks.register(HookPoint.LLM_BEFORE, schema, name='test.aliased-scheduler')
    try:
        async with asyncio.timeout(150):
            sid = loop.target.session_id
            result = await loop.send(
                f'Call {alias} exactly once with action="tool", tool="workflow", args={{"value": 7}}. '
                'When it returns a task_id or a final result notification, reply exactly DONE. Do not call anything else.')
            rows = await store.load_all_messages(sid)
            receipt = json.loads(next(row.content for row in rows if row.role == 'tool'))
            from power_loop.agent.runtime_store import now_ms
            background = await loop.runtime.client.wait(JobRef(**receipt['job_ref']), actor=loop.actor, deadline_ms=now_ms()+10000)
            await loop.drain()
            assert result.final_text.strip() == 'DONE' and provider.calls in {2, 3} and provider.refused_calls == 0
            assert calls == [{'value': 7}]
            assert background.status == 'completed' and background.final_text == 'WORK_OK'
            assert 'command' in definition.input_schema['properties']  # raw registration was not rewritten
    finally:
        provider.report(record_property)
        record_property('original_worker_executions', len(calls))
        await loop.aclose()
        await store.close()
