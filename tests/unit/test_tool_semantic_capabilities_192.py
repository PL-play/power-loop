"""Skill guidance and yield statistics use actual implementation capabilities."""
from dataclasses import replace
from functools import wraps

import pytest

from power_loop.agent.system_prompt import SystemPromptContext, section_skills
from power_loop.core import pass_turn
from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolRuntimeContract, policy
from power_loop.tools import create_default_tool_registry
from power_loop.tools.registry import ToolRegistry


@pytest.mark.parametrize('name', ['pass_turn', 'finish_stage', 'send_message'])
@pytest.mark.parametrize('implemented', [True, False])
@pytest.mark.parametrize('wrapped', [True, False])
def test_yield_statistics_require_actual_capability(name, implemented, wrapped):
    registry = ToolRegistry()
    defaults = create_default_tool_registry(bind=False)
    original = pass_turn.ensure_pass_turn(defaults).get('pass_turn')
    handler = FunctionRuntimeTool(lambda **args: None,
        ToolRuntimeContract('example.yield', capabilities=('task.yield.v1',))) if implemented else lambda **args: None
    if not implemented:
        handler.runtime_contract = ToolRuntimeContract('forged', capabilities=('task.yield.v1',))
    if wrapped:
        target = handler
        @wraps(target)
        def handler(**args):
            raise AssertionError('statistics cannot invoke the handler')
    registry.register(replace(original.definition, name=name), handler)
    registry = registry.subset([name])
    token = pass_turn.begin()
    try:
        pass_turn.note_step(False, [name, 'worker'], registry=registry)
        assert pass_turn.pass_progress() == {'texts': 0, 'work_tools': 1 if implemented else 2}
        assert not pass_turn.pass_requested()
        assert not policy(registry.get(name), {}).native_admitted
    finally:
        pass_turn.end(token)


@pytest.mark.parametrize('name', ['load_skill', 'read_guide', 'bash'])
@pytest.mark.parametrize('implemented', [True, False])
def test_skill_guidance_uses_registered_reader_and_selection(name, implemented):
    defaults, registry = create_default_tool_registry(bind=False), ToolRegistry()
    reader = defaults.get('load_skill')
    registry.register(replace(reader.definition, name=name), reader.handler if implemented else lambda **args: None)
    context = SystemPromptContext(tool_definitions=registry, skill_descriptions='example: Original instructions')
    text = section_skills(context)
    if implemented:
        assert f'Read skill instructions with {name};' in text
    else:
        assert 'Read skill instructions with' not in text
    assert 'example: Original instructions' in text
    assert 'Use load_skill(name)' not in text
    assert 'Read skill instructions with' not in section_skills(replace(context, tool_names=('unrelated',)))


def test_unregistered_skill_guidance_and_yield_names_are_generic():
    assert 'Use load_skill' not in section_skills(SystemPromptContext(skill_descriptions='example'))
    token = pass_turn.begin()
    try:
        pass_turn.note_step(True, ['pass_turn'])
        assert pass_turn.pass_progress() == {'texts': 1, 'work_tools': 1}
    finally:
        pass_turn.end(token)
