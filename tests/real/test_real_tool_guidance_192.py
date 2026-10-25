"""The actual provider sees implementation-owned hints and reports actual progress."""
import asyncio

import pytest

from power_loop import AgentEventType, AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract
from power_loop.tools.registry import ToolRegistry

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('name,declared', [('write_plan', True), ('todo', False)])
async def test_real_progress_and_prompt_follow_actual_declared_tool(tmp_path, record_property, name, declared):
    provider, store = BoundedProvider(limit=2), await SessionStore.open(str(tmp_path / 'guidance.db'))
    calls, events, registry = [], [], ToolRegistry()

    async def handler(**arguments):
        calls.append(arguments)
        assert arguments == {}
        return 'PLAN_RECORDED'

    implementation = FunctionRuntimeTool(handler, ToolRuntimeContract('example.progress',
        capabilities=('task.progress.v1',))) if declared else handler
    registry.register(ToolDefinition(name, 'Record the requested plan once. No arguments.',
        usage_hint='IMPLEMENTATION_PLAN_GUIDANCE: call this tool once with an empty object.'), implementation)
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=3, max_tokens=2048, retry_policy=LLMRetryPolicy(max_attempts=1)))
    loop.event_bus.subscribe(None, events.append)

    async def schema(ctx):
        assert 'IMPLEMENTATION_PLAN_GUIDANCE' in ctx.system_prompt
        assert 'persistent session' not in ctx.system_prompt
        ctx.tools = registry.to_openai_tools() if provider.calls == 0 else []

    loop.hooks.register(HookPoint.LLM_BEFORE, schema, name='test.actual-guidance')
    try:
        async with asyncio.timeout(150):
            sid = await loop.new_session()
            result = await loop.send(f'Call {name} exactly once with an empty object. '
                'After PLAN_RECORDED reply exactly DONE. Do not call other tools.', sid)
            assert result.status == 'completed' and result.final_text.strip() == 'DONE'
            assert calls == [{}] and provider.calls == 2
            rounds = [e for e in events if e.type == AgentEventType.ROUND_COMPLETED and e.payload.get('has_tools')]
            assert len(rounds) == 1 and rounds[0].payload['used_todo'] == declared
    finally:
        provider.report(record_property)
        record_property('actual_plan_writes', len(calls))
        try:
            await loop.aclose(drain_timeout_s=0.1)
        finally:
            await provider.close()
            await store.close()
