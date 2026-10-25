"""Host presentation uses original registration and message metadata."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import AgentLoopConfig, ToolDefinition, validate_tool_args
from power_loop.contracts.context_policy import ToolContextPolicy
from power_loop.runtime.delegation import _completion_cap, completion_view
from power_loop.runtime.invocation import invocation_context
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.store.types import MessageRow, MessageState
from power_loop.runtime.tool_context import ToolContextConfig
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_completion_uses_aliased_original_definition_and_explicit_override(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    registry = ToolRegistry()
    registry.register(ToolDefinition('delegate_notes', '', context_policy=ToolContextPolicy(result_max_chars=1700)), lambda **kw: '')
    config = AgentLoopConfig(tool_context=ToolContextConfig(overrides={'spawn_agent': {'result_max_chars':20}, 'delegate_notes': {'result_max_chars':1300}}))
    loop = SimpleNamespace(config=config, tool_registry=registry)
    seq = await store.append_message(sid, role='assistant', tool_calls=[{'id':'original', 'type':'function',
        'function':{'name':'delegate_notes','arguments':'{}'}}])
    with invocation_context(sid, seq, 'original'):
        cap = await _completion_cap(loop, store, sid)
    assert cap == 1300
    output = completion_view(loop, {'status':'completed','session_id':'original-child','run_id':'original-run','final_text':'A'*3000}, max_chars=cap)
    assert json.loads(output)['final_text'] == 'A'*1300 + '…[truncated]'
    with invocation_context(sid, seq, 'unrelated'):
        assert await _completion_cap(loop, store, sid) is None
    assert len(completion_view(loop, {'status':'completed','session_id':'original-child','run_id':'original-run','final_text':'A'*3000})) > 20


def test_timeline_projection_uses_metadata_and_does_not_infer_from_a_tool_name():
    first = MessageRow(session_id='sid',seq=1,role='user',name='background_tool_results',content='actual task',
        tool_calls=None,tool_call_id=None,round_index=0,state=MessageState.ACTIVE,meta={},created_at=0,send_index=1)
    notice = replace(first,seq=2,name='unrelated_notice_name',content='completed child',meta={'context_timeline':True})
    projected = ProjectedRepresentation().project_send([first,notice],send_index=1,tool_registry=None)
    assert projected.rows[0].content == {'input':['actual task']}
    assert any(tool['name']=='__user__' and tool['text']=='completed child' for tool in projected.rows[1].content['tools'])


@pytest.mark.parametrize('name', ['read_file','generate_image','unrelated'])
def test_explicit_definition_parameter_validation_overrides_legacy_catalog_names(name):
    definition = ToolDefinition(name,'',input_schema={'type':'object','required':['value']})
    assert validate_tool_args(definition,{'value':1}) is None
    assert 'value' in validate_tool_args(definition,{})
