from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentEvent,
    AgentEventBus,
    AgentEventType,
    AgentHooks,
    AgentLoopConfig,
    HookPoint,
    RuntimeProjector,
    SessionStore,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.runtime.pricing import Free
from tests.unit.test_job_registry_192 import registry_store

__all__ = ["registry_store"]


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    calls: list[list[dict[str, Any]]] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(request.messages))
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done")
        response = self.responses[self._idx]
        self._idx += 1
        return response

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


def _tool_resp(call_id: str, name: str, args: str = "{}") -> LLMResponse:
    return LLMResponse(
        raw_text="",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    )


class _InspectProjector(RuntimeProjector):
    async def project(self, *, store: Any, session_id: str, round_index: int, context: Any) -> list[dict[str, Any]]:
        state = await store.get_runtime_state(session_id, "inspect", default={}) or {}
        if not state:
            return []
        return [{"role": "user", "name": "inspect_state", "content": f"<inspect_state>{state}</inspect_state>"}]


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


def test_tool_runtime_context_outside_loop_is_explicit() -> None:
    ctx = get_tool_runtime_context()
    assert ctx.session_id is None
    assert ctx.store is None
    assert ctx.loop is None
    assert ctx.config is None
    assert ctx.has_session_store is False

    with pytest.raises(RuntimeError, match="No active StatefulAgentLoop"):
        get_tool_runtime_context(required=True)


@pytest.mark.asyncio
async def test_custom_tool_queries_session_and_projects_state(store: SessionStore, runtime_agent) -> None:
    reg = ToolRegistry()

    async def inspect_session(**_: Any) -> str:
        runtime = get_tool_runtime_context(required=True)
        session = await runtime.store.get_session(runtime.session_id)
        messages = await runtime.store.load_active_messages(runtime.session_id)
        await runtime.store.set_runtime_state(
            runtime.session_id,
            "inspect",
            {
                "owner": session.metadata["principal_id"],
                "message_count": len(messages),
            },
        )
        return "session inspected"

    reg.register(
        ToolDefinition(name="inspect_session", description="Inspect current session", pricing=Free("test_fixture")),
        inspect_session,
    )
    llm = _Scripted(
        responses=[
            _tool_resp("tc-inspect", "inspect_session"),
            LLMResponse(raw_text="saw inspect state"),
        ]
    )
    loop = await runtime_agent(
        model=llm,
        store=store,
        registry=reg,
        config=AgentLoopConfig(
            system_prompt="S",
            max_rounds=3,
            compactor=None,
            runtime_projectors=(_InspectProjector(),),
        ),
    )
    sid = loop.target.session_id

    await loop.send("inspect this session")

    state = await store.get_runtime_state(sid, "inspect")
    assert state == {"owner": "host", "message_count": 2}
    assert any("host" in str(msg.get("content", "")) for msg in llm.calls[1])
    assert any("message_count" in str(msg.get("content", "")) for msg in llm.calls[1])


@pytest.mark.asyncio
async def test_hooks_can_modify_tool_args_and_write_runtime_state(store: SessionStore, runtime_agent) -> None:
    reg = ToolRegistry()

    async def save_value(value: str) -> str:
        runtime = get_tool_runtime_context(required=True)
        await runtime.store.set_runtime_state(runtime.session_id, "inspect", {"value": value})
        return f"saved {value}"

    reg.register(
        ToolDefinition(
            name="save_value",
            description="Save a runtime value",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            required_params=("value",),
        pricing=Free("test_fixture")),
        save_value,
    )
    hooks = AgentHooks()

    def before_tool(ctx: Any) -> None:
        if ctx.tool_name == "save_value":
            ctx.tool_args["value"] = "hooked-value"

    async def after_tool(ctx: Any) -> None:
        if ctx.tool_name == "save_value":
            runtime = get_tool_runtime_context(required=True)
            await runtime.store.set_runtime_state(runtime.session_id, "after_hook", {"output": ctx.output})

    hooks.register(HookPoint.TOOL_BEFORE, before_tool)
    hooks.register(HookPoint.TOOL_AFTER, after_tool)

    llm = _Scripted(
        responses=[
            _tool_resp("tc-save", "save_value", '{"value":"model-value"}'),
            LLMResponse(raw_text="done"),
        ]
    )
    loop = await runtime_agent(
        model=llm,
        store=store,
        registry=reg,
        hooks=hooks,
        config=AgentLoopConfig(system_prompt="S", max_rounds=3, compactor=None),
    )
    sid = loop.target.session_id

    result = await loop.send("save a value")
    assert result.status == "completed", result
    assert await store.get_runtime_state(sid, "inspect") is not None, [(r.role,r.content) for r in await store.load_all_messages(sid)]

    assert await store.get_runtime_state(sid, "inspect") == {"value": "hooked-value"}
    assert await store.get_runtime_state(sid, "after_hook") == {"output": "saved hooked-value"}


@pytest.mark.asyncio
async def test_events_observe_custom_tool_lifecycle(store: SessionStore, runtime_agent) -> None:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="ping_runtime", description="Ping runtime", pricing=Free("test_fixture")),
        lambda **_: "pong",
    )
    bus = AgentEventBus()
    seen: list[tuple[AgentEventType, str | None]] = []

    def on_event(event: AgentEvent) -> None:
        if event.type in {AgentEventType.TOOL_CALL_STARTED, AgentEventType.TOOL_CALL_COMPLETED}:
            seen.append((event.type, event.data.name))

    bus.subscribe(None, on_event)
    llm = _Scripted(
        responses=[
            _tool_resp("tc-ping", "ping_runtime"),
            LLMResponse(raw_text="done"),
        ]
    )
    loop = await runtime_agent(
        model=llm,
        store=store,
        registry=reg,
        event_bus=bus,
        config=AgentLoopConfig(system_prompt="S", max_rounds=3, compactor=None),
    )

    await loop.send("ping")

    assert seen == [
        (AgentEventType.TOOL_CALL_STARTED, "ping_runtime"),
        (AgentEventType.TOOL_CALL_COMPLETED, "ping_runtime"),
    ]


async def test_hook_arguments_price_actual_call_and_retain_original_history(registry_store, runtime_agent):
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.pricing import Free
    from power_loop.runtime.task_runtime import TaskRuntime

    store, _ = registry_store
    executed = []
    async def save(value):
        executed.append(value)
        return value
    model = _Scripted([_tool_resp('original', 'save', '{"value":"x"}'), LLMResponse(raw_text='done')])
    task_runtime = TaskRuntime(JobManager(store), tenant_id='hook-price', principal_id='host', executor_id='hook',
        boot_id='boot', boot_contract_hash='a'*64)
    registry = ToolRegistry()
    registry.register(ToolDefinition('save', 'save', required_params=('value',), pricing=Free('fixture')), save)
    hooks = AgentHooks()
    def replace(ctx):
        ctx.tool_args['value'] = 'hello'
    hooks.register(HookPoint.TOOL_BEFORE, replace)
    agent = await runtime_agent(model=model, store=store, registry=registry, hooks=hooks, task_runtime=task_runtime,
        config=AgentLoopConfig(model="test", max_tokens=20, max_rounds=3, compactor=None))
    result = await agent.send('save')
    assert result.status == 'completed', result
    assert executed == ['hello']
    rows = await store.load_all_messages(agent.target.session_id)
    call = next(row for row in rows if row.tool_calls)
    assert call.tool_calls[0]['id'] == 'original' and call.tool_calls[0]['function']['arguments'] == '{"value":"x"}'
    assert next(row.content for row in rows if row.tool_call_id == 'original') == 'hello'
