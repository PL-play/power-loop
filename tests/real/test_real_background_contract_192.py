"""A live model uses an aliased scheduler and the implementation-owned schema."""

import asyncio
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.tool_context import ToolContextConfig
from power_loop.tools import create_default_tool_registry
from power_loop.tools import default_tools as dt
from power_loop.tools.registry import ToolRegistry

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('alias', ['defer_work', 'submit_original'])
async def test_real_renamed_scheduler_uses_actual_contract(tmp_path, record_property, monkeypatch, alias):
    provider = BoundedProvider(limit=2)
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
    manager = dt.BackgroundManager()
    monkeypatch.setattr(dt, 'BG', manager)
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
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
            sid = await loop.new_session()
            result = await loop.send(
                f'Call {alias} exactly once with action="tool", tool="workflow", args={{"value": 7}}. '
                'When it returns a task_id, reply exactly DONE. Do not call anything else.', sid)
            await asyncio.gather(*manager._workers.values())
            assert result.final_text.strip() == 'DONE' and provider.calls == 2
            assert calls == [{'value': 7}]
            assert all(row['status'] == 'completed' for row in manager.tasks.values())
            assert 'command' in definition.input_schema['properties']  # raw registration was not rewritten
    finally:
        provider.report(record_property)
        record_property('original_worker_executions', len(calls))
        for task in list(manager._workers.values()):
            if not task.done():
                task.cancel()
        await asyncio.gather(*manager._workers.values(), return_exceptions=True)
        try:
            await loop.aclose(drain_timeout_s=0.1)
        finally:
            await provider.close()
            await store.close()
