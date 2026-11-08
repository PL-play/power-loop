"""Structured child and workflow output through current client/worker execution."""
import json
from typing import Any

import pytest

from power_loop import AgentLoopConfig, AgentSpec, AgentSpecError, JobRef, LLMResponse
from power_loop.agent.runtime_store import now_ms
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.spec import normalize_output_schema, output_response_format
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import (
    _RAW_TEXT_CAP,
    SPAWN_AGENT_DEFINITION,
    _format_structured_result,
    register_spawn_agent,
)
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_timers import _Scripted, _tool_resp

__all__ = ['registry_store']
LABEL_SCHEMA = {'type':'object','required':['label'],'properties':{'label':{'type':'string'}}}

def test_tool_schema_exposes_optional_output_schema() -> None:
    props = SPAWN_AGENT_DEFINITION.input_schema["properties"]
    assert props["output_schema"]["type"] == "object"
    assert "JSON Schema" in props["output_schema"]["description"]
    assert "output_schema" not in SPAWN_AGENT_DEFINITION.input_schema["required"]
    assert SPAWN_AGENT_DEFINITION.required_params == ()

def test_normalize_wrapped_form_keeps_name_and_declared_strict() -> None:
    out = normalize_output_schema(
        {"name": "Plan", "schema": LABEL_SCHEMA, "strict": True}, default_strict=False
    )
    assert out == {"name": "Plan", "schema": LABEL_SCHEMA, "strict": True}
    # no strict declared → the caller's default
    out = normalize_output_schema({"schema": LABEL_SCHEMA}, default_strict=False)
    assert out == {"name": "Output", "schema": LABEL_SCHEMA, "strict": False}

def test_normalize_bare_schema() -> None:
    assert normalize_output_schema(LABEL_SCHEMA, default_strict=False) == {
        "name": "Output", "schema": LABEL_SCHEMA, "strict": False,
    }
    assert normalize_output_schema(LABEL_SCHEMA)["strict"] is True  # library default

def test_normalize_json_string_of_either_form() -> None:
    assert normalize_output_schema(json.dumps(LABEL_SCHEMA))["schema"] == LABEL_SCHEMA
    wrapped = json.dumps({"name": "C", "schema": LABEL_SCHEMA, "strict": False})
    assert normalize_output_schema(wrapped) == {"name": "C", "schema": LABEL_SCHEMA, "strict": False}

def test_normalize_sanitizes_name_for_native_providers() -> None:
    out = normalize_output_schema({"name": "用户 列表", "schema": LABEL_SCHEMA})
    assert out["name"] == "_____"
    assert len(normalize_output_schema({"name": "x" * 100, "schema": LABEL_SCHEMA})["name"]) == 64

@pytest.mark.parametrize(
    ("value", "needle"),
    [
        ({"type": "array", "items": {"type": "string"}}, "根必须是"),
        ({"name": "X", "schema": {"type": "string"}}, "根必须是"),
        ({"properties": {"a": {"type": "string"}}}, "根必须是"),
        ("{not json", "不是合法的 JSON"),
        ("[1, 2]", "要是一个 JSON Schema 对象"),
        (42, "要是一个 JSON Schema 对象"),
        ({"schema": LABEL_SCHEMA, "strict": "false"}, "strict"),
    ],
)
def test_normalize_rejects_bad_shapes(value: Any, needle: str) -> None:
    with pytest.raises(AgentSpecError, match=needle):
        normalize_output_schema(value)

def test_structured_failure_reasons_and_cap() -> None:
    out = _format_structured_result('{"other": 1}', LABEL_SCHEMA)
    assert out == '结构化结果解析失败（缺少必填字段 label），原文：{"other": 1}'
    out = _format_structured_result('{"label": urgent}', LABEL_SCHEMA)
    assert out.startswith("结构化结果解析失败（JSON 格式不对）")
    long = "字" * (_RAW_TEXT_CAP + 500)
    out = _format_structured_result(long, LABEL_SCHEMA)
    assert f"原文共 {len(long)} 字" in out
    assert len(out) < len(long)

def test_output_response_format_strict_default_and_opt_out() -> None:
    rf = output_response_format({"name": "C", "schema": LABEL_SCHEMA})
    assert rf["json_schema"]["strict"] is True
    rf = output_response_format({"name": "C", "schema": LABEL_SCHEMA, "strict": False})
    assert "strict" not in rf["json_schema"]
    rf = output_response_format(LABEL_SCHEMA)  # bare schema
    assert rf["json_schema"] == {"name": "Output", "schema": LABEL_SCHEMA, "strict": True}

def test_agent_spec_rejects_non_boolean_strict() -> None:
    with pytest.raises(AgentSpecError, match="strict must be a boolean"):
        AgentSpec(name="n", system_prompt="p",
                  output_schema={"name": "C", "schema": LABEL_SCHEMA, "strict": "no"})

def _node_spec(output_schema: Any) -> dict[str, Any]:
    return {"name": "w", "root": {"type": "agent", "id": "leaf",
                                  "spec": {"name": "leaf", "system_prompt": "p"},
                                  "output_schema": output_schema}}

def test_workflow_node_accepts_strict_and_round_trips_it() -> None:
    from power_loop.workflow import WorkflowSpec

    parsed = WorkflowSpec.from_json(_node_spec({"name": "C", "schema": LABEL_SCHEMA, "strict": False}))
    assert parsed.root.output_schema == {"name": "C", "schema": LABEL_SCHEMA, "strict": False}
    again = WorkflowSpec.from_json(parsed.to_dict())  # resume rebuilds nodes from the journal
    assert again.root.output_schema["strict"] is False

@pytest.mark.parametrize("bad", [{"strict": "no"}, {"extra": 1}])
def test_workflow_node_rejects_non_boolean_strict_and_unknown_keys(bad: dict[str, Any]) -> None:
    from power_loop.workflow import WorkflowSpec

    with pytest.raises(Exception, match="output_schema"):
        WorkflowSpec.from_json(_node_spec({"name": "C", "schema": LABEL_SCHEMA, **bad}))

@pytest.mark.parametrize('reply,expected', [('{"label":"urgent"}', {'label':'urgent'}),
    ('好的：\n```json\n{"label":"urgent","score":3,}\n```', {'label':'urgent','score':3}),
    ('plain text', None)])
@pytest.mark.parametrize('strict',[False,True])
async def test_child_format_and_persisted_parent_completion(registry_store, reply, expected, strict):
    store,_=registry_store
    class Model(_Scripted):
        parent=None
        sent=False
        child_requests=[]
        async def complete(self,request,**kwargs):
            if get_session_id()!=self.parent:
                self.child_requests.append(request)
                return LLMResponse(raw_text=reply)
            if not self.sent:
                self.sent=True
                return _tool_resp('spawn','spawn_agent',json.dumps({'task':'classify','tools':[],
                    'output_schema':{'name':'Label','schema':LABEL_SCHEMA,'strict':strict}}))
            return LLMResponse(raw_text='parent done')
    model,registry=Model(),ToolRegistry()
    model.child_requests=[]
    register_spawn_agent(registry)
    agent=await RunningAgent(store,model,registry,AgentLoopConfig(max_rounds=4,compactor=None)).start()
    model.parent=agent.target.session_id
    try:
        await agent.send('classify')
        rows=await store.load_all_messages(agent.target.session_id)
        receipt=json.loads(next(r.content for r in rows if r.tool_call_id=='spawn'))
        result=await agent.runtime.client.wait(JobRef(**receipt['job_ref']),actor=agent.actor,deadline_ms=now_ms()+10000)
        assert result.status==('completed' if expected is not None else 'failed')
        assert result.structured_output==expected and result.final_text==reply
        if expected is None:
            assert 'structured_output' in result.error
        response_format=model.child_requests[0].response_format['json_schema']
        assert response_format['name']=='Label' and response_format['schema']==LABEL_SCHEMA
        assert response_format.get('strict',False) is strict
        await agent.drain()
        rows=await store.load_all_messages(agent.target.session_id)
        notifications=[r.content for r in rows if '[subagent_result]' in str(r.content)]
        assert len(notifications)==1
        payload=json.loads(notifications[0].split('\n',1)[1])
        assert payload['structured_output']==expected and payload['status']==result.status
    finally:
        await agent.aclose()


@pytest.mark.parametrize('arguments,reason', [({'task':'x','output_schema':{'type':'array'}},'output_schema'),
    ({'task':'x','tools':'bash'},'参数有误')])
async def test_invalid_child_contract_creates_no_child(registry_store,arguments,reason):
    store,_=registry_store
    registry=ToolRegistry()
    register_spawn_agent(registry)
    agent=await RunningAgent(store,_Scripted([_tool_resp('spawn','spawn_agent',json.dumps(arguments))]),registry).start()
    try:
        await agent.send('invalid')
        rows=await store.load_all_messages(agent.target.session_id)
        output=next(r.content for r in rows if r.tool_call_id=='spawn')
        assert output.startswith('Error:') and reason in output
        assert await store.list_children(agent.target.session_id)==[]
    finally:
        await agent.aclose()


@pytest.mark.parametrize('strict',[False,True])
async def test_workflow_worker_receives_schema_and_strict(registry_store,strict):
    store,_=registry_store
    class Model(_Scripted):
        seen=[]
        async def complete(self,request,**kwargs):
            self.seen.append(request)
            return LLMResponse(raw_text='{"label":"a"}')
    model=Model()
    model.seen=[]
    agent=await RunningAgent(store,model,ToolRegistry()).start()
    try:
        spec=_node_spec({'name':'C','schema':LABEL_SCHEMA,'strict':strict})
        handle=await agent.runtime.client.workflow(agent.target,spec,message_id='strict',actor=agent.actor)
        result=await agent.runtime.client.wait(JobRef(store.store_id,handle.job_id,0),actor=agent.actor,deadline_ms=now_ms()+10000)
        assert result.status=='completed',result
        leaf_request=next(r for r in model.seen if r.response_format and r.response_format.get('type')=='json_schema')
        assert leaf_request.response_format['json_schema'].get('strict',False) is strict
        nodes=await store._db.fetchall(f"SELECT job_id FROM {store.table_prefix}jobs WHERE kind='workflow_node'")
        assert len(nodes)==1
        child=await agent.runtime.client.wait(JobRef(store.store_id,nodes[0]['job_id'],0),actor=agent.actor,deadline_ms=now_ms()+10000)
        assert child.structured_output=={'label':'a'}
    finally:
        await agent.aclose()
