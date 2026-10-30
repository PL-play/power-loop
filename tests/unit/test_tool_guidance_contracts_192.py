"""Aliases and lookalikes exercise real presentation and progress paths."""

from dataclasses import replace

import pytest

from power_loop import AgentEventBus, AgentEventType, ToolDefinition, create_default_tool_registry
from power_loop.agent.system_prompt import (
    SystemPromptContext,
    resolve_runtime_system_prompt,
    section_todo_discipline,
    section_tool_guide,
)
from power_loop.runtime.pricing import Free
from power_loop.runtime.tool_contracts import (
    FunctionRuntimeTool,
    ToolContractError,
    ToolRuntimeContract,
    policy,
    resolve,
)
from power_loop.tools.registry import ToolRegistry

from .test_pipeline_tier2_fixes import _pipeline, _Scripted, _tool_call

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('name', ['todo', 'write_progress', 'send_message'])
@pytest.mark.parametrize('declared', [False, True])
@pytest.mark.parametrize('failure', [False, True])
async def test_progress_tracking_uses_successful_actual_contract_not_name(name, declared, failure):
    calls, events, registry = [], [], ToolRegistry()

    async def handler(**arguments):
        calls.append(arguments)
        if failure:
            raise ValueError('progress_write_failed')
        return 'OK'

    implementation = FunctionRuntimeTool(handler, ToolRuntimeContract('example.progress',
        capabilities=('task.progress.v1',))) if declared else handler
    registry.register(ToolDefinition(name, 'progress probe', pricing=Free("test_fixture")), implementation)
    bus = AgentEventBus()
    bus.subscribe(None, events.append)
    pipeline = _pipeline(llm=_Scripted(responses=[_tool_call('original', name)]), tool_registry=registry, bus=bus)
    pipeline.rounds_since_todo = 8
    result = await pipeline.run([{'role': 'user', 'content': 'write progress once'}])
    assert result.status == 'completed' and calls == [{}]
    rounds = [event for event in events if event.type == AgentEventType.ROUND_COMPLETED and event.payload.get('has_tools')]
    assert len(rounds) == 1 and rounds[0].payload['used_todo'] == (declared and not failure)
    assert pipeline.rounds_since_todo == (0 if declared and not failure else 9)


@pytest.mark.parametrize('name', ['read_file', 'custom_read', 'recall_send'])
@pytest.mark.parametrize('show', [False, True])
async def test_background_hint_is_explicit_metadata_independent_of_concurrency_or_admission(name, show):
    defaults, registry = create_default_tool_registry(bind=False), ToolRegistry()
    scheduler = defaults.get('background_run')
    registry.register(replace(scheduler.definition, name='defer_work'), scheduler.handler)
    definition = ToolDefinition(name, 'read something', async_capable=True, background_hint=show, pricing=Free("test_fixture"))
    registry.register(definition, lambda **args: 'OK')
    spec = next(s['function'] for s in registry.to_openai_tools() if s['function']['name'] == name)
    assert ('可异步' in spec['description']) == show
    assert definition.async_capable is True
    assert not policy(registry.get(name), {}).native_admitted
    assert registry.invoke(name, {}) == 'OK'


async def test_default_quick_reads_keep_concurrency_and_quiet_hint_under_aliases():
    defaults, registry = create_default_tool_registry(bind=False), ToolRegistry()
    scheduler = defaults.get('background_run')
    registry.register(replace(scheduler.definition, name='defer_work'), scheduler.handler)
    for name in ['read_file', 'glob', 'grep', 'recall_send', 'recall_compacted', 'load_skill']:
        source = defaults.get(name)
        registry.register(replace(source.definition, name='alias_' + name), source.handler)
        assert source.definition.async_capable
    assert '可异步' not in str(registry.to_openai_tools())


async def test_usage_guide_and_default_runtime_prompt_follow_actual_definitions_without_same_name_fallback():
    defaults, registry = create_default_tool_registry(bind=False), ToolRegistry()
    source = defaults.get('read_file')
    registry.register(replace(source.definition, name='read_notes'), source.handler)
    registry.register(ToolDefinition('bash', 'plain probe', usage_hint='HOST_PROBE_GUIDANCE', pricing=Free("test_fixture")), lambda **args: '')
    registry.register(ToolDefinition('background_run', 'plain other probe', pricing=Free("test_fixture")), lambda **args: '')
    guide = section_tool_guide(SystemPromptContext(tool_definitions=registry))
    assert 'read_notes:' in guide and 'returns numbered lines' in guide
    assert '- bash: HOST_PROBE_GUIDANCE' in guide
    assert 'persistent session' not in guide and '- background_run:' not in guide
    filtered = section_tool_guide(SystemPromptContext(tool_definitions=registry, tool_names=['bash']))
    assert 'read_notes:' not in filtered and 'HOST_PROBE_GUIDANCE' in filtered
    empty = section_tool_guide(SystemPromptContext(tool_definitions=[]))
    assert '- bash:' not in empty and '- read_file:' not in empty
    prompt = resolve_runtime_system_prompt(None, inject_tool_descriptions=True, tool_catalog_header='TOOLS',
        tool_registry=registry, skills_dir=None)
    assert 'HOST_PROBE_GUIDANCE' in prompt and 'persistent session' not in prompt
    assert '- background_run:' not in prompt and '# Todo Discipline' not in prompt
    assert prompt.count('TOOLS') == 1 and prompt.count('**bash**:') == 1
    # Only an explicit built-in preview without a registry uses built-in hints.
    assert 'persistent session' in section_tool_guide(SystemPromptContext(tool_names=['bash']))


@pytest.mark.parametrize('field,value', [('usage_hint', 4), ('usage_hint', 'x' * 4097),
                                       ('background_hint', 1), ('background_hint', None)])
async def test_presentation_metadata_rejects_ambiguous_types(field, value):
    with pytest.raises(TypeError):
        ToolDefinition('probe', '', **{field: value}, pricing=Free("test_fixture"))


@pytest.mark.parametrize('field,value', [('usage_hint', 'changed'), ('background_hint', False)])
async def test_registered_contract_freezes_presentation_metadata_too(field, value):
    implementation = FunctionRuntimeTool(lambda **args: '', ToolRuntimeContract('example.frozen'))
    registry = ToolRegistry()
    definition = ToolDefinition('probe', '', pricing=Free("test_fixture"))
    registry.register(definition, implementation)
    object.__setattr__(definition, field, value)
    with pytest.raises(ToolContractError, match='binding_changed'):
        resolve(registry.subset(['probe']).get('probe'))


async def test_progress_prompt_names_only_actual_capabilities_and_respects_filtered_tools():
    defaults, registry = create_default_tool_registry(bind=False), ToolRegistry()
    original = defaults.get('todo')
    registry.register(replace(original.definition, name='record_plan'), original.handler)
    registry.register(ToolDefinition('todo', 'unrelated', pricing=Free("test_fixture")), lambda **args: '')
    guide = section_todo_discipline(SystemPromptContext(tool_definitions=registry))
    assert 'record_plan' in guide and 'Track tasks with todo' not in guide
    assert section_todo_discipline(SystemPromptContext(tool_definitions=registry, tool_names=['todo'])) is None
    assert section_todo_discipline(SystemPromptContext(tool_definitions=[])) is None
