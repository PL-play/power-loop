from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Generator
from dataclasses import dataclass, field
from typing import Any

import pytest

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import (
    AgentEvent,
    AgentEventBus,
    AgentEventType,
    AgentHooks,
    AgentLoopConfig,
    HookPoint,
    RuntimeProjector,
    SessionStore,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
    get_tool_runtime_context,
)


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
    def project(self, *, store: Any, session_id: str, round_index: int, context: Any) -> list[dict[str, Any]]:
        state = store.get_runtime_state(session_id, "inspect", default={}) or {}
        if not state:
            return []
        return [{"role": "user", "name": "inspect_state", "content": f"<inspect_state>{state}</inspect_state>"}]


@pytest.fixture
def store() -> Generator[SessionStore, None, None]:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


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
async def test_custom_tool_queries_session_and_projects_state(store: SessionStore) -> None:
    reg = ToolRegistry()

    def inspect_session(**_: Any) -> str:
        runtime = get_tool_runtime_context(required=True)
        session = runtime.store.get_session(runtime.session_id)
        messages = runtime.loop.get_messages(runtime.session_id)
        runtime.store.set_runtime_state(
            runtime.session_id,
            "inspect",
            {
                "owner": session.metadata["owner"],
                "message_count": len(messages),
            },
        )
        return "session inspected"

    reg.register(
        ToolDefinition(name="inspect_session", description="Inspect current session"),
        inspect_session,
    )
    llm = _Scripted(
        responses=[
            _tool_resp("tc-inspect", "inspect_session"),
            LLMResponse(raw_text="saw inspect state"),
        ]
    )
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        tool_registry=reg,
        config=AgentLoopConfig(
            system_prompt="S",
            max_rounds=3,
            compactor=None,
            runtime_projectors=(_InspectProjector(),),
        ),
    )
    sid = loop.new_session(metadata={"owner": "alice"})

    await loop.send("inspect this session", session_id=sid)

    state = store.get_runtime_state(sid, "inspect")
    assert state == {"owner": "alice", "message_count": 2}
    assert any("alice" in str(msg.get("content", "")) for msg in llm.calls[1])
    assert any("message_count" in str(msg.get("content", "")) for msg in llm.calls[1])


@pytest.mark.asyncio
async def test_hooks_can_modify_tool_args_and_write_runtime_state(store: SessionStore) -> None:
    reg = ToolRegistry()

    def save_value(value: str) -> str:
        runtime = get_tool_runtime_context(required=True)
        runtime.store.set_runtime_state(runtime.session_id, "inspect", {"value": value})
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
        ),
        save_value,
    )
    hooks = AgentHooks()

    def before_tool(ctx: Any) -> None:
        if ctx.tool_name == "save_value":
            ctx.tool_args["value"] = "hooked-value"

    def after_tool(ctx: Any) -> None:
        if ctx.tool_name == "save_value":
            runtime = get_tool_runtime_context(required=True)
            runtime.store.set_runtime_state(runtime.session_id, "after_hook", {"output": ctx.output})

    hooks.register(HookPoint.TOOL_BEFORE, before_tool)
    hooks.register(HookPoint.TOOL_AFTER, after_tool)

    llm = _Scripted(
        responses=[
            _tool_resp("tc-save", "save_value", '{"value":"model-value"}'),
            LLMResponse(raw_text="done"),
        ]
    )
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        tool_registry=reg,
        hooks=hooks,
        config=AgentLoopConfig(system_prompt="S", max_rounds=3, compactor=None),
    )
    sid = loop.new_session()

    await loop.send("save a value", session_id=sid)

    assert store.get_runtime_state(sid, "inspect") == {"value": "hooked-value"}
    assert store.get_runtime_state(sid, "after_hook") == {"output": "saved hooked-value"}


@pytest.mark.asyncio
async def test_events_observe_custom_tool_lifecycle(store: SessionStore) -> None:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="ping_runtime", description="Ping runtime"),
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
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        tool_registry=reg,
        event_bus=bus,
        config=AgentLoopConfig(system_prompt="S", max_rounds=3, compactor=None),
    )
    sid = loop.new_session()

    await loop.send("ping", session_id=sid)

    assert seen == [
        (AgentEventType.TOOL_CALL_STARTED, "ping_runtime"),
        (AgentEventType.TOOL_CALL_COMPLETED, "ping_runtime"),
    ]
