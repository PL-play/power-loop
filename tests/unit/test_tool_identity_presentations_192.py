"""Host presentation uses original registration and message metadata."""

import json
from dataclasses import replace

import pytest

from power_loop import ToolDefinition, validate_tool_args
from power_loop.contracts.context_policy import ToolContextPolicy
from power_loop.runtime.pricing import Free
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.types import MessageRow, MessageState
from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine, ToolSource
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_completion_uses_aliased_original_definition_and_explicit_override(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    registry = ToolRegistry()
    register_spawn_agent(registry)
    original = registry.get('spawn_agent')
    registry.register(replace(original.definition, name='delegate_notes',
        context_policy=ToolContextPolicy(result_max_chars=1700)), original.handler)
    config = ToolContextConfig(overrides={'spawn_agent':{'result_max_chars':20},'delegate_notes':{'result_max_chars':1300}})
    engine = ToolContextEngine(config=config,store=store,owner_session_id=sid,owner_send_index=1,
        registry=registry,llm=None,model='unused')
    result = {'status':'completed','session_id':'original-child','run_id':'original-run','final_text':'A'*3000}
    source = ToolSource(sid,1,2,'original','delegate_notes',{},json.dumps(result))
    assert engine.policy(source).result_max_chars == 1300
    output = json.loads(await engine.distill(source,'general'))
    assert output == {**result,'final_text':'A'*1300+'…[truncated]'}
    assert json.loads(source.result) == result


def test_timeline_projection_uses_metadata_and_does_not_infer_from_a_tool_name():
    first = MessageRow(session_id='sid',seq=1,role='user',name='background_tool_results',content='actual task',
        tool_calls=None,tool_call_id=None,round_index=0,state=MessageState.ACTIVE,meta={},created_at=0,send_index=1)
    notice = replace(first,seq=2,name='unrelated_notice_name',content='completed child',meta={'context_timeline':True})
    projected = ProjectedRepresentation().project_send([first,notice],send_index=1,tool_registry=None)
    assert projected.rows[0].content == {'input':['actual task']}
    assert any(tool['name']=='__user__' and tool['text']=='completed child' for tool in projected.rows[1].content['tools'])


@pytest.mark.parametrize('name', ['read_file','generate_image','unrelated'])
def test_explicit_definition_parameter_validation_overrides_legacy_catalog_names(name):
    definition = ToolDefinition(name,'',input_schema={'type':'object','required':['value']}, pricing=Free("test_fixture"))
    assert validate_tool_args(definition,{'value':1}) is None
    assert 'value' in validate_tool_args(definition,{})
