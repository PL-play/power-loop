from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentLoopConfig,
    AgentSpec,
    AgentSpecError,
    SessionStore,
    SubagentLifecycle,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.runtime.spec import filtered_registry, run_agent_spec
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
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

    async def close(self) -> None: return None


def _spawn_call(call_id: str, task: str) -> LLMResponse:
    import json
    return LLMResponse(
        raw_text="",
        tool_calls=[{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "arguments": json.dumps({"task": task}),
            },
        }],
    )


def _spawn_call_kwargs(call_id: str, args: dict) -> LLMResponse:
    import json
    return LLMResponse(
        raw_text="",
        tool_calls=[{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "arguments": json.dumps(args),
            },
        }],
    )


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


# ── AgentSpec validation ────────────────────────────────────────────────


def test_agent_spec_rejects_unknown_fields() -> None:
    with pytest.raises(AgentSpecError, match="unknown AgentSpec field"):
        AgentSpec.from_dict({"name": "x", "system_prompt": "y", "evil": True})


def test_agent_spec_rejects_empty_name() -> None:
    with pytest.raises(AgentSpecError):
        AgentSpec(name="", system_prompt="ok")


def test_agent_spec_rejects_bad_lifecycle() -> None:
    with pytest.raises(AgentSpecError, match="unknown lifecycle"):
        AgentSpec(name="n", system_prompt="p", lifecycle="forever")


def test_agent_spec_rejects_nonpositive_rounds() -> None:
    with pytest.raises(AgentSpecError):
        AgentSpec(name="n", system_prompt="p", max_rounds=0)
    with pytest.raises(AgentSpecError):
        AgentSpec(name="n", system_prompt="p", max_rounds=-1)
    # No upper cap: a large round budget is allowed (was previously rejected at >50).
    assert AgentSpec(name="n", system_prompt="p", max_rounds=999).max_rounds == 999


def test_agent_spec_accepts_minimal() -> None:
    s = AgentSpec(name="n", system_prompt="p")
    assert s.tools is None
    assert s.lifecycle == SubagentLifecycle.EPHEMERAL.value


def test_agent_spec_from_json_string() -> None:
    s = AgentSpec.from_json('{"name":"n","system_prompt":"p","max_rounds":3}')
    assert s.max_rounds == 3


# ── filtered_registry ──────────────────────────────────────────────────


def test_filtered_registry_none_returns_none() -> None:
    assert filtered_registry(None, ["x"]) is None


def test_filtered_registry_no_whitelist_returns_same() -> None:
    reg = ToolRegistry()
    assert filtered_registry(reg, None) is reg


def test_filtered_registry_whitelist_subset() -> None:
    parent = ToolRegistry()
    for name in ("a", "b", "c"):
        parent.register(
            ToolDefinition(name=name, description=name, input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")),
            lambda **_kw: name,
        )
    sub = filtered_registry(parent, ["a", "c", "missing"])
    assert sub is not None
    assert sub.has("a") and sub.has("c") and not sub.has("b")
    assert not sub.has("missing")


# ── spawn_agent / run_agent end-to-end ─────────────────────────────────


async def _child(runtime_agent, store, model, spec, *, registry=None, config=None, task_runtime=None):
    from power_loop.core.agent_context import get_current_loop
    registry = registry or ToolRegistry()
    async def delegate():
        return await run_agent_spec(spec, "do work", parent_loop=get_current_loop())
    registry.register(ToolDefinition(name="delegate", description="host delegation",
        input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), delegate)
    agent = await runtime_agent(model=model, store=store, registry=registry,
        config=config or AgentLoopConfig(compactor=None), task_runtime=task_runtime, command=("delegate", {}))
    result = await agent.send("go")
    assert result.status == "completed", result
    return agent, json.loads(result.final_text)


@pytest.mark.parametrize("arguments", [
    {"task": "summarize topic X"},
    {"task": "summarize", "name": "summarizer", "system_prompt": "You summarize.", "max_rounds": 2},
])
async def test_spawn_agent_creates_retained_child_with_frozen_settings(store, runtime_agent, arguments):
    from power_loop import JobRef
    from power_loop.agent.runtime_store import now_ms
    # Parent finishes after admission; the capacity-one worker then runs the child.
    model = _Scripted(responses=[_spawn_call_kwargs("spawn", arguments), LLMResponse(raw_text="parent done"),
                                LLMResponse(raw_text="child summary")])
    registry = ToolRegistry()
    register_spawn_agent(registry)
    agent = await runtime_agent(model=model, store=store, registry=registry, config=AgentLoopConfig(compactor=None))
    parent = await agent.send("go")
    assert parent.status == "completed"
    rows = await store.load_active_messages(agent.target.session_id)
    receipt = json.loads(next(row.content for row in rows if row.tool_call_id == "spawn"))
    assert receipt["status"] == "accepted" and receipt["mode"] == "create"
    result = await agent.runtime.client.wait(JobRef(**receipt["job_ref"]), actor=agent.actor,
        deadline_ms=now_ms()+10000)
    await agent.drain()
    assert result.status == "completed" and result.final_text == "child summary"
    child = await store.get_session(receipt["session_id"])
    assert child.parent_session_id == agent.target.session_id and child.lifecycle == SubagentLifecycle.LINKED
    if "system_prompt" in arguments:
        assert child.system_prompt == arguments["system_prompt"]
        assert child.metadata["runtime_descriptor"]["config"]["child_spec"]["name"] == "summarizer"


async def test_spawn_agent_with_missing_task_returns_error(store, runtime_agent):
    registry = ToolRegistry()
    register_spawn_agent(registry)
    agent = await runtime_agent(model=_Scripted(responses=[_spawn_call_kwargs("tc3", {"name": "x"}),
        LLMResponse(raw_text="parent recovered")]), store=store, registry=registry,
        config=AgentLoopConfig(max_rounds=2))
    result = await agent.send("go")
    assert result.final_text == "parent recovered"
    assert await store.list_children(agent.target.session_id) == []


async def test_subagent_parent_link_recorded_when_lifecycle_linked(store, runtime_agent):
    agent, child = await _child(runtime_agent, store, _Scripted(responses=[LLMResponse(raw_text="child output")]),
        AgentSpec(name="kept", system_prompt="P", lifecycle=SubagentLifecycle.LINKED.value))
    assert child["status"] == "completed"
    children = await store.list_children(agent.target.session_id)
    assert len(children) == 1 and children[0].session_id == child["session_id"]
    assert children[0].lifecycle is SubagentLifecycle.LINKED


async def test_subagent_failure_preserves_session_even_when_ephemeral(store, runtime_agent):
    _, result = await _child(runtime_agent, store, _Scripted(responses=[_spawn_call("tc", "forbidden")]),
        AgentSpec(name="hopeless", system_prompt="P", max_rounds=1))
    assert result["status"] != "completed"
    assert result["session_id"] and await store.get_session(result["session_id"]) is not None


async def test_spawn_depth_rejection(store, runtime_agent):
    # All Runtime children use the single-level policy, including host commands.
    from power_loop.core.agent_context import get_current_loop
    attempts = []
    registry = ToolRegistry()
    async def deeper():
        try:
            return await run_agent_spec(AgentSpec(name="grandchild", system_prompt="G"), "go",
                                        parent_loop=get_current_loop())
        except ValueError as exc:
            attempts.append(str(exc))
            raise
    registry.register(ToolDefinition(name="deeper", description="attempt nested delegation",
        input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), deeper)
    response = LLMResponse(raw_text="", tool_calls=[{"id":"nested", "type":"function",
        "function":{"name":"deeper", "arguments":"{}"}}])
    _, child = await _child(runtime_agent, store, _Scripted(responses=[response, LLMResponse(raw_text="recovered")]),
        AgentSpec(name="child", system_prompt="C", lifecycle="linked"), registry=registry)
    assert attempts == ["execution_role_forbids_subagent"]
    assert await store.list_children(child["session_id"]) == []


def test_spawn_agent_outside_loop_returns_clear_error() -> None:
    import asyncio

    from power_loop.tools.spawn_agent import _handle_spawn_agent
    out = asyncio.run(_handle_spawn_agent(task="x"))
    assert "must be invoked from inside" in out


async def test_subagent_empty_final_text_recovers_last_assistant_text(store, runtime_agent):
    reg = ToolRegistry()
    reg.register(ToolDefinition(name="echo", description="e",
        input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), lambda **k: "ok")
    model = _Scripted(responses=[LLMResponse(raw_text="the real review findings", tool_calls=[{
        "id":"c1", "type":"function", "function":{"name":"echo", "arguments":"{}"}}]),
        *[LLMResponse(raw_text="") for _ in range(4)]])
    _, child = await _child(runtime_agent, store, model, AgentSpec(name="quiet", system_prompt="P", max_rounds=5), registry=reg)
    assert child["status"] == "completed"
    assert child["final_text"] == "the real review findings"


async def test_subagent_unknown_model_action_is_not_replayed_and_new_call_works(store, runtime_agent):
    from power_loop import LLMRetryPolicy
    class Flaky(_Scripted):
        calls = 0
        async def complete(self, request, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("supplier outcome unknown")
            return LLMResponse(raw_text="child made it")
    model = Flaky()
    agent, first = await _child(runtime_agent, store, model, AgentSpec(name="resilient", system_prompt="P"),
        config=AgentLoopConfig(retry_policy=LLMRetryPolicy(max_attempts=3, backoff_initial=.001)))
    assert first["status"] == "failed" and model.calls == 1
    second = json.loads((await agent.send("new request")).final_text)
    assert second["status"] == "completed" and second["final_text"] == "child made it"
    assert model.calls == 2 and second["session_id"] != first["session_id"]


async def test_run_agent_spec_model_binding_uses_another_service(store, runtime_agent):
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.task_runtime import TaskRuntime
    runtime = TaskRuntime(JobManager(store), tenant_id="test", principal_id="host", executor_id="fixture", boot_id="boot")
    runtime.model_services = {"qa@1": _Scripted(responses=[LLMResponse(raw_text="override side")])}
    _, result = await _child(runtime_agent, store, _Scripted(responses=[LLMResponse(raw_text="parent side")]),
        AgentSpec(name="qa", system_prompt="P", metadata={"model_binding":"qa@1"}), task_runtime=runtime)
    assert result["status"] == "completed" and result["final_text"] == "override side"


async def test_subagent_inherits_parent_max_tokens(store, runtime_agent):
    seen = []
    class Recording(_Scripted):
        async def complete(self, request, **kwargs):
            seen.append(request.max_tokens)
            return LLMResponse(raw_text="child")
    spec = AgentSpec(name="inherits", system_prompt="P", max_rounds=1)
    assert spec.max_tokens is None
    await _child(runtime_agent, store, Recording(), spec, config=AgentLoopConfig(max_tokens=40000))
    assert seen == [40000]


@pytest.mark.asyncio
async def test_think_only_round_is_not_retried_verbatim(store: SessionStore, runtime_agent) -> None:
    """只有思考、没有正文也没有工具调用 → 不当「打嗝」原样重试。

    输入不变，输出不会变：重试只会再烧一轮同样的预算。正确做法是改变输入——
    往历史里补一条「你上一轮只产出了思考」的提示，让它直接给结论。
    流式响应常常取不到 finish_reason 与 usage，所以这条判据不能只靠它们。
    """
    from power_loop.core.pipeline import _THINK_ONLY_NOTICE

    llm = _Scripted(responses=[
        LLMResponse(raw_text="", think="想了很久但什么都没写"),   # 第 1 轮：只有思考
        LLMResponse(raw_text="这是结论"),                          # 第 2 轮：被提示后给出正文
    ])
    loop = await runtime_agent(model=llm, store=store, config=AgentLoopConfig(max_rounds=4))
    sid = loop.target.session_id
    r = await loop.send("问题")
    assert r.final_text == "这是结论"
    rows = await store.load_active_messages(sid)
    assert any(r0.role == "user" and _THINK_ONLY_NOTICE in str(r0.content or "") for r0 in rows), \
        "应当补一条提示改变输入，而不是原样重试"


@pytest.mark.asyncio
async def test_reasoning_content_is_echoed_back(store: SessionStore, runtime_agent) -> None:
    """带过思考的 assistant 消息，**下一次请求**要把 reasoning_content 原样带回去。

    DeepSeek 的思考模式明确要求这一点，少了就是 400（硬错：重试同样的历史只会再错一遍），
    最后 retry_exhausted 整个 run 降级。这里验的是真正的契约——发出去的 messages 里有没有，
    而不是落库那一份（落库是给重启后重放用的，另一回事）。
    """
    seen: list[list[dict]] = []

    class _Recording(_Scripted):
        async def complete(self, request, *, on_chunk_delta_text=None,
                           on_chunk_think=None, on_stream_end=None):
            seen.append([dict(m) for m in (getattr(request, "messages", None) or [])])
            return await super().complete(request)

    # 用真实工具轮触发下一次请求；无正文、无工具的空 assistant 已由请求整理删除。
    from tests.unit.test_stateful_loop import _echo_registry
    llm = _Recording(responses=[
        LLMResponse(raw_text="", think="推演过程", tool_calls=[{
            "id": "echo-think", "type": "function", "function": {
                "name": "echo", "arguments": '{"text":"ok"}'}}]),
        LLMResponse(raw_text="答案"),
    ])
    loop = await runtime_agent(model=llm, store=store, registry=_echo_registry(),
        config=AgentLoopConfig(max_rounds=3))
    await loop.send("问题")
    assert len(seen) >= 2, "应当有第二次请求"
    echoed = [m for m in seen[-1]
              if m.get("role") == "assistant" and m.get("reasoning_content") == "推演过程"]
    assert echoed, f"下一次请求里没带回思考内容：{[m.get('role') for m in seen[-1]]}"


def test_store_strips_nul_bytes() -> None:
    """模型输出里的 0x00 不能让整程工作丢掉。

    Postgres 的 text/jsonb 存不下 NUL（invalid byte sequence for encoding "UTF8": 0x00），
    这是硬约束。真实事故：一次正常的 assistant 回复里夹了一个 NUL，落库直接抛，
    整个 run 失败——用户那边看到的是会话自己停了。
    """
    from power_loop.runtime.store.store import _dumps, _nul_safe

    assert _nul_safe("好\x00的") == "好的"
    assert _nul_safe(None) is None
    assert _nul_safe("没有控制字符") == "没有控制字符"
    # 别的字符一个都不能动（制表符、换行、Unicode 都是合法内容）
    assert _nul_safe("a\tb\nc—😀") == "a\tb\nc—😀"
    # JSON 那一路同样要干净（tool_calls / meta 都走它）
    assert "\x00" not in (_dumps({"k": "v\x00w"}) or "")


async def test_missing_child_model_binding_is_visible_and_stops_incapable_node(store, runtime_agent):
    agent, result = await _child(runtime_agent, store, _Scripted(),
        AgentSpec(name="qa", system_prompt="P", metadata={"model_binding":"missing@1"}))
    assert result["status"] == "failed" and result["error_code"] == "resource_unavailable"
    assert "missing@1" in result["error"]
    health = agent.runtime.status()
    assert health["accepting"] is False
    assert health["last_error"]["code"] == "resource_unavailable"
