"""Live skill alias guidance and semantic yield statistics need no conventional name."""
import asyncio
from dataclasses import replace

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop
from power_loop.core import pass_turn
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.skills import LOAD_SKILL_DEFINITION, SkillLoader
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.skill_contract import SkillTool

from .test_real_async_boundaries_192 import BoundedProvider

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_model_reads_skill_using_actual_registered_alias(tmp_path, record_property):
    skills = tmp_path / 'skills'
    example = skills / 'example'
    example.mkdir(parents=True)
    (example / 'SKILL.md').write_text('---\nname: example\ndescription: The requested local evidence\n---\nSECRET_SKILL_VALUE=S42\n')
    loader, registry = SkillLoader(skills), ToolRegistry()
    calls = []

    def read(name):
        calls.append(name)
        return loader.handle_load_skill(name)

    registry.register(replace(LOAD_SKILL_DEFINITION, name='read_handbook'), SkillTool(read))
    provider = BoundedProvider(limit=2)
    store = await SessionStore.open(str(tmp_path / 'skill.db'))
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
        config=AgentLoopConfig(skills_dir=str(skills), max_tokens=2048, max_rounds=3, compactor=None,
                               retry_policy=LLMRetryPolicy(max_attempts=1)))

    async def prompt(ctx):
        assert 'Read skill instructions with read_handbook;' in ctx.system_prompt
        assert 'Use load_skill(name)' not in ctx.system_prompt
        if provider.calls:
            ctx.tools = []

    loop.hooks.register(HookPoint.LLM_BEFORE, prompt, name='test.skill-alias-guide')
    try:
        async with asyncio.timeout(150):
            sid = await loop.new_session()
            result = await loop.send('Read the example skill exactly once with its available reader, then reply only with its SECRET_SKILL_VALUE.', sid)
            assert result.status == 'completed' and result.final_text.strip() == 'S42'
            assert calls == ['example'] and provider.calls == 2
    finally:
        provider.report(record_property)
        try:
            await loop.aclose(drain_timeout_s=.1)
        finally:
            await provider.close()
            await store.close()


@pytest.mark.parametrize('implemented', [True, False])
async def test_real_yield_requires_actual_effect_and_statistics_capability(tmp_path, implemented, record_property):
    registry = ToolRegistry()
    original = pass_turn.ensure_pass_turn(ToolRegistry()).get('pass_turn')
    name = 'finish_stage' if implemented else 'pass_turn'
    seen = []

    def handler(reason=''):
        seen.append(pass_turn.pass_progress())
        return pass_turn.default_handler(reason) if implemented else 'CONTINUE'

    registry.register(replace(original.definition, name=name),
        FunctionRuntimeTool(handler, ToolRuntimeContract('example.yield', capabilities=('task.yield.v1',))) if implemented else handler)
    provider, store = BoundedProvider(limit=1 if implemented else 2), await SessionStore.open(str(tmp_path / 'yield.db'))
    loop = StatefulAgentLoop(llm=provider, store=store, tool_registry=registry,
        config=AgentLoopConfig(max_tokens=2048, max_rounds=3, compactor=None, retry_policy=LLMRetryPolicy(max_attempts=1)))

    async def schema(ctx):
        ctx.tools = [registry.get(name).definition.to_openai_tool()] if provider.calls == 0 else []

    loop.hooks.register(HookPoint.LLM_BEFORE, schema, name='test.yield-contract')
    try:
        async with asyncio.timeout(150):
            sid = await loop.new_session()
            result = await loop.send(f'Call {name} exactly once with reason="done", with no assistant text before the call. '
                                     'If it returns CONTINUE, reply only DONE without any more tools.', sid)
            assert result.status == 'completed' and result.passed == implemented
            assert len(seen) == 1 and seen[0]['work_tools'] == (0 if implemented else 1)
            assert provider.calls == (1 if implemented else 2)
            if not implemented:
                assert result.final_text.strip() == 'DONE'
    finally:
        provider.report(record_property)
        try:
            await loop.aclose(drain_timeout_s=.1)
        finally:
            await provider.close()
            await store.close()
