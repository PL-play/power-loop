"""design/126 §2: ``spawn_agent(output_schema=…)`` and the ``strict`` flag on
``AgentSpec.output_schema`` (run_agent_spec + the isolated workflow worker)."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import AgentLoopConfig, AgentSpec, AgentSpecError, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.spec import (
    normalize_output_schema,
    output_response_format,
    run_agent_spec,
)
from power_loop.runtime.store.store import MAX_SPAWN_DEPTH
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import (
    _RAW_TEXT_CAP,
    SPAWN_AGENT_DEFINITION,
    _format_structured_result,
    _handle_spawn_agent,
    register_spawn_agent,
)
from power_loop.workflow import WorkerBootstrap, run_spec_isolated

LABEL_SCHEMA = {
    "type": "object",
    "required": ["label"],
    "properties": {"label": {"type": "string"}},
}


@dataclass
class _Capturing(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    seen: list[LLMRequest] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        self.seen.append(request)
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done")
        r = self.responses[self._idx]
        self._idx += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


async def _call_tool(loop: StatefulAgentLoop, parent_sid: str, **kwargs: Any) -> str:
    """Invoke the spawn_agent handler as if from inside ``loop``'s run on ``parent_sid``."""
    tok_loop = set_current_loop(loop)
    tok_sid = set_session_id(parent_sid)
    try:
        return await _handle_spawn_agent(**kwargs)
    finally:
        reset_current_loop(tok_loop)
        reset_session_id(tok_sid)


# ── tool schema ─────────────────────────────────────────────────────────


def test_tool_schema_exposes_optional_output_schema() -> None:
    props = SPAWN_AGENT_DEFINITION.input_schema["properties"]
    assert props["output_schema"]["type"] == "object"
    assert "JSON Schema" in props["output_schema"]["description"]
    assert "output_schema" not in SPAWN_AGENT_DEFINITION.input_schema["required"]
    assert SPAWN_AGENT_DEFINITION.required_params == ("task",)


# ── normalization ───────────────────────────────────────────────────────


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


# ── spawn_agent handler ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_schema_is_reported_without_spawning(store: SessionStore) -> None:
    llm = _Capturing()
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=3))
    parent = await loop.new_session()
    out = await _call_tool(loop, parent, task="classify", output_schema={"type": "array"})
    assert out.startswith("Error: output_schema 无效，子 agent 没有启动")
    assert "根必须是" in out
    assert llm.seen == []
    assert await store.list_children(parent) == []


@pytest.mark.asyncio
async def test_bad_spec_args_are_reported_not_raised(store: SessionStore) -> None:
    llm = _Capturing()
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=3))
    parent = await loop.new_session()
    out = await _call_tool(loop, parent, task="x", tools="bash")  # must be a list
    assert out.startswith("Error: spawn_agent 参数有误")
    assert llm.seen == []


@pytest.mark.asyncio
async def test_child_request_carries_non_strict_response_format(store: SessionStore) -> None:
    llm = _Capturing(responses=[LLMResponse(raw_text='{"label": "紧急"}')])
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=3))
    parent = await loop.new_session()
    out = await _call_tool(loop, parent, task="classify", output_schema=LABEL_SCHEMA)
    rf = llm.seen[0].response_format
    assert rf == {"type": "json_schema", "json_schema": {"name": "Output", "schema": LABEL_SCHEMA}}
    assert "strict" not in rf["json_schema"]
    assert out == '结构化结果：{"label":"紧急"}'


@pytest.mark.asyncio
async def test_declared_strict_true_is_honoured(store: SessionStore) -> None:
    llm = _Capturing(responses=[LLMResponse(raw_text='{"label": "a"}')])
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=3))
    parent = await loop.new_session()
    await _call_tool(
        loop, parent, task="classify",
        output_schema=json.dumps({"name": "Label", "schema": LABEL_SCHEMA, "strict": True}),
    )
    js = llm.seen[0].response_format["json_schema"]
    assert js["name"] == "Label" and js["strict"] is True


@pytest.mark.asyncio
async def test_structured_success_from_fenced_prose(store: SessionStore) -> None:
    reply = '好的：\n```json\n{"label": "urgent", "score": 3,}\n```'
    llm = _Capturing(responses=[LLMResponse(raw_text=reply)])
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=3))
    parent = await loop.new_session()
    out = await _call_tool(loop, parent, task="classify", output_schema=LABEL_SCHEMA)
    assert out == '结构化结果：{"label":"urgent","score":3}'


@pytest.mark.asyncio
async def test_structured_failure_returns_reason_and_text(store: SessionStore) -> None:
    llm = _Capturing(responses=[LLMResponse(raw_text="这张工单很紧急")])
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=3))
    parent = await loop.new_session()
    out = await _call_tool(loop, parent, task="classify", output_schema=LABEL_SCHEMA)
    assert out == "结构化结果解析失败（回复里没有 JSON 对象），原文：这张工单很紧急"


def test_structured_failure_reasons_and_cap() -> None:
    out = _format_structured_result('{"other": 1}', LABEL_SCHEMA)
    assert out == '结构化结果解析失败（缺少必填字段 label），原文：{"other": 1}'
    out = _format_structured_result('{"label": urgent}', LABEL_SCHEMA)
    assert out.startswith("结构化结果解析失败（JSON 格式不对）")
    long = "字" * (_RAW_TEXT_CAP + 500)
    out = _format_structured_result(long, LABEL_SCHEMA)
    assert f"原文共 {len(long)} 字" in out
    assert len(out) < len(long)


@pytest.mark.asyncio
async def test_non_completed_status_keeps_plain_format(store: SessionStore) -> None:
    sid = await store.create_session()
    cur = sid
    for _ in range(MAX_SPAWN_DEPTH):
        cur = await store.create_session(parent_session_id=cur)
    llm = _Capturing()
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=1))
    out = await _call_tool(loop, cur, task="x", output_schema=LABEL_SCHEMA)
    assert out.startswith("[sub-agent status=rejected]")
    assert "结构化结果" not in out


@pytest.mark.asyncio
async def test_parent_loop_receives_structured_tool_result(store: SessionStore) -> None:
    """End to end: the parent LLM passes output_schema as a JSON string (as models sometimes do);
    the tool result the parent sees next is the parsed object."""
    spawn = LLMResponse(
        raw_text="",
        tool_calls=[{
            "id": "tc1",
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "arguments": json.dumps({"task": "classify", "output_schema": json.dumps(LABEL_SCHEMA)}),
            },
        }],
    )
    llm = _Capturing(responses=[
        spawn,
        LLMResponse(raw_text='{"label": "urgent"}'),  # child
        LLMResponse(raw_text="parent done"),
    ])
    registry = ToolRegistry()
    register_spawn_agent(registry)
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=registry, config=AgentLoopConfig(max_rounds=4),
    )
    sid = await loop.new_session()
    r = await loop.send("go", session_id=sid)
    assert r.final_text == "parent done"
    assert llm.seen[0].response_format is None and llm.seen[2].response_format is None
    assert llm.seen[1].response_format["type"] == "json_schema"
    tool_msgs = [m for m in llm.seen[2].messages if m.get("role") == "tool"]
    assert any('结构化结果：{"label":"urgent"}' in str(m.get("content")) for m in tool_msgs)


# ── strict flag in run_agent_spec / the isolated worker ─────────────────


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


@pytest.mark.asyncio
@pytest.mark.parametrize(("declared", "expect_strict"), [({}, True), ({"strict": False}, False)])
async def test_run_agent_spec_honours_strict(
    store: SessionStore, declared: dict[str, Any], expect_strict: bool
) -> None:
    llm = _Capturing(responses=[LLMResponse(raw_text='{"label": "a"}')])
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=3))
    await run_agent_spec(
        AgentSpec(name="kid", system_prompt="p",
                  output_schema={"name": "C", "schema": LABEL_SCHEMA, **declared}),
        "task", parent_loop=loop,
    )
    js = llm.seen[0].response_format["json_schema"]
    assert js["name"] == "C" and js["schema"] == LABEL_SCHEMA
    assert js.get("strict", False) is expect_strict


@pytest.mark.asyncio
@pytest.mark.parametrize(("declared", "expect_strict"), [({}, True), ({"strict": False}, False)])
async def test_isolated_worker_honours_strict(declared: dict[str, Any], expect_strict: bool) -> None:
    llm = _Capturing(responses=[LLMResponse(raw_text='{"label": "a"}')])
    db = tempfile.mktemp(suffix=".db")
    try:
        result = await run_spec_isolated(
            {"name": "leaf", "system_prompt": "p",
             "output_schema": {"name": "C", "schema": LABEL_SCHEMA, **declared}},
            "task", bootstrap=WorkerBootstrap(llm_factory=lambda: llm), db_path=db,
        )
    finally:
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(db + suffix):
                os.remove(db + suffix)
    assert result["status"] == "completed"
    assert llm.seen[0].response_format["json_schema"].get("strict", False) is expect_strict


# ── workflow agent node: output_schema may carry strict ──────────────────────


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


@pytest.mark.asyncio
@pytest.mark.parametrize(("declared", "expect_strict"), [({}, True), ({"strict": False}, False)])
async def test_workflow_node_strict_reaches_the_request(
    store: SessionStore, declared: dict[str, Any], expect_strict: bool
) -> None:
    from power_loop.workflow import WorkflowSpec, create_workflow

    llm = _Capturing(responses=[LLMResponse(raw_text='{"label": "a"}')])
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=3))
    psid = await loop.new_session()
    spec = WorkflowSpec.from_json(_node_spec({"name": "C", "schema": LABEL_SCHEMA, **declared}))
    res = await create_workflow(spec, parent_loop=loop, parent_session_id=psid).start(detached=False)
    assert res.status == "completed"
    assert llm.seen[0].response_format["json_schema"].get("strict", False) is expect_strict
