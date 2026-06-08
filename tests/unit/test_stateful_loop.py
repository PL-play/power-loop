from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Generator
from dataclasses import dataclass, field
from typing import Any

import pytest

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import (
    AgentLoopConfig,
    FollowUpQueued,
    MessageState,
    RuntimeProjector,
    SessionPendingError,
    SessionStore,
    StatefulAgentLoop,
    create_default_tool_registry,
    get_tool_runtime_context,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry

# ── Fake LLM ────────────────────────────────────────────────────────────


@dataclass
class _Scripted(LLMService):
    """LLM that returns canned responses, one per call."""

    responses: list[LLMResponse] = field(default_factory=list)
    calls: list[list[dict]] = field(default_factory=list)
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


def _echo_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo",
            description="echo back",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            required_params=("text",),
        ),
        lambda **kw: kw.get("text", ""),
    )
    return reg


class _CustomProjector(RuntimeProjector):
    def project(self, *, store: Any, session_id: str, round_index: int, context: Any) -> list[dict[str, Any]]:
        state = store.get_runtime_state(session_id, "custom", default={}) or {}
        text = state.get("text", "")
        return [{"role": "user", "name": "custom_runtime", "content": f"<custom>{text}</custom>"}] if text else []


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> Generator[SessionStore, None, None]:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


@pytest.mark.asyncio
async def test_new_session_then_send_persists_user_message(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text="hi back")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, config=AgentLoopConfig(system_prompt="S", max_rounds=2),
    )
    sid = loop.new_session(metadata={"owner": "test"})
    row = store.get_session(sid)
    assert row is not None
    assert row.system_prompt == "S"
    assert row.metadata == {"owner": "test"}

    custom_sid = loop.new_session(system_prompt="custom", metadata={"owner": "custom"})
    custom_row = store.get_session(custom_sid)
    assert custom_row is not None
    assert custom_row.system_prompt == "custom"
    assert custom_row.metadata == {"owner": "custom"}

    r = await loop.send("hello", session_id=sid)
    assert r.session_id == sid
    assert r.status == "completed"
    assert r.final_text == "hi back"

    msgs = loop.get_messages(r.session_id)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["content"] == "hi back"


@pytest.mark.asyncio
async def test_todo_runtime_state_is_injected_without_persisting_virtual_message(store: SessionStore) -> None:
    llm = _Scripted(
        responses=[
            _tool_resp(
                "tc-todo",
                "todo",
                '{"items":[{"id":"one","text":"Implement runtime state","status":"in_progress"}]}',
            ),
            LLMResponse(raw_text="I see the runtime todo."),
        ]
    )
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        tool_registry=create_default_tool_registry(include=["todo"]),
        config=AgentLoopConfig(system_prompt="Use todos.", max_rounds=3, compactor=None),
    )
    sid = loop.new_session()
    result = await loop.send("make a todo", session_id=sid)

    assert result.status == "completed"
    todo_state = store.get_runtime_state(sid, "todo")
    assert todo_state["items"][0]["text"] == "Implement runtime state"
    assert any("<current_todos>" in str(msg.get("content", "")) for msg in llm.calls[1])
    persisted = loop.get_messages(sid)
    assert not any("<current_todos>" in str(msg.get("content", "")) for msg in persisted)


@pytest.mark.asyncio
async def test_todo_runtime_state_survives_new_loop_instance(store: SessionStore) -> None:
    sid = store.create_session(system_prompt="S")
    store.set_runtime_state(
        sid,
        "todo",
        {
            "items": [{"id": "persisted", "text": "Survive restart", "status": "in_progress"}],
            "rendered": "[>] #persisted: Survive restart\n\n(0/1 completed)",
            "counts": {"total": 1, "completed": 0},
        },
    )
    store.append_message(sid, role="user", content="previous")
    store.append_message(sid, role="assistant", content="previous answer")
    llm = _Scripted(responses=[LLMResponse(raw_text="runtime state visible")])

    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None),
    )
    result = await loop.send("continue", session_id=sid)

    assert result.status == "completed"
    assert any("Survive restart" in str(msg.get("content", "")) for msg in llm.calls[0])


@pytest.mark.asyncio
async def test_request_user_input_pauses_then_submit_resumes(store: SessionStore) -> None:
    llm = _Scripted(
        responses=[
            _tool_resp(
                "tc-input",
                "request_user_input",
                (
                    '{"kind":"confirm","prompt":"Approve access?",'
                    '"options":[{"id":"yes","label":"Yes"},{"id":"no","label":"No"}],'
                    '"metadata":{"scope":"contacts"}}'
                ),
            ),
            LLMResponse(raw_text="Approved path continued."),
        ]
    )
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        tool_registry=create_default_tool_registry(include=["request_user_input"]),
        config=AgentLoopConfig(system_prompt="Ask before continuing.", max_rounds=3, compactor=None),
    )
    sid = loop.new_session()

    result = await loop.send("needs approval", session_id=sid)

    assert result.status == "waiting_for_input"
    assert result.pending_tool_calls[0]["id"] == "tc-input"
    assert len(result.pending_interactions) == 1
    interaction = result.pending_interactions[0]
    assert interaction["tool_call_id"] == "tc-input"
    assert interaction["kind"] == "confirm"
    assert interaction["prompt"] == "Approve access?"
    assert interaction["options"][0]["id"] == "yes"
    assert interaction["metadata"] == {"scope": "contacts"}
    assert loop.get_pending(sid)["pending_interactions"][0]["interaction_id"] == interaction["interaction_id"]

    blocked = await loop.resume(sid)
    assert blocked.status == "waiting_for_input"
    assert blocked.pending_interactions[0]["interaction_id"] == interaction["interaction_id"]

    resumed = await loop.submit_input(sid, interaction["interaction_id"], {"choice": "yes"})

    assert resumed.status == "completed"
    assert resumed.final_text == "Approved path continued."
    assert loop.get_pending(sid) is None
    persisted = loop.get_messages(sid)
    tool_messages = [msg for msg in persisted if msg["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "tc-input"
    assert json.loads(str(tool_messages[0]["content"])) == {"choice": "yes"}
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "tc-input" for msg in llm.calls[1])


@pytest.mark.asyncio
async def test_request_user_input_survives_new_loop_instance(store: SessionStore) -> None:
    first_llm = _Scripted(
        responses=[
            _tool_resp(
                "tc-input",
                "request_user_input",
                '{"kind":"text","prompt":"What should I say?"}',
            )
        ]
    )
    first_loop = StatefulAgentLoop(
        llm=first_llm,
        store=store,
        tool_registry=create_default_tool_registry(include=["request_user_input"]),
        config=AgentLoopConfig(system_prompt="Ask externally.", max_rounds=2, compactor=None),
    )
    sid = first_loop.new_session()
    waiting = await first_loop.send("pause", session_id=sid)
    interaction_id = waiting.pending_interactions[0]["interaction_id"]

    second_llm = _Scripted(responses=[LLMResponse(raw_text="Second process continued.")])
    second_loop = StatefulAgentLoop(
        llm=second_llm,
        store=store,
        tool_registry=create_default_tool_registry(include=["request_user_input"]),
        config=AgentLoopConfig(system_prompt="Ask externally.", max_rounds=2, compactor=None),
    )

    resumed = await second_loop.submit_input(sid, interaction_id, "hello from user")

    assert resumed.status == "completed"
    assert resumed.final_text == "Second process continued."
    assert second_loop.get_pending(sid) is None
    assert any(
        msg.get("role") == "tool" and msg.get("content") == "hello from user"
        for msg in second_llm.calls[0]
    )


@pytest.mark.asyncio
async def test_background_runtime_updates_are_injected_once(store: SessionStore) -> None:
    sid = store.create_session(system_prompt="S")
    store.append_message(sid, role="user", content="start")
    store.append_message(sid, role="assistant", content="started")
    store.upsert_background_task(
        sid,
        task_id="bg1",
        command="printf done",
        status="completed",
        return_code=0,
        output_tail="done",
    )
    llm = _Scripted(responses=[LLMResponse(raw_text="saw background"), LLMResponse(raw_text="no duplicate")])
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None),
    )

    await loop.send("what happened?", session_id=sid)
    assert any("<background_updates>" in str(msg.get("content", "")) for msg in llm.calls[0])
    await loop.send("again?", session_id=sid)
    assert not any("<background_updates>" in str(msg.get("content", "")) for msg in llm.calls[1])


@pytest.mark.asyncio
async def test_custom_runtime_projector_is_configurable(store: SessionStore) -> None:
    sid = store.create_session(system_prompt="S")
    store.set_runtime_state(sid, "custom", {"text": "CUSTOM_RUNTIME_VISIBLE"})
    llm = _Scripted(responses=[LLMResponse(raw_text="custom visible")])
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        config=AgentLoopConfig(
            system_prompt="S",
            max_rounds=1,
            compactor=None,
            runtime_projectors=(_CustomProjector(),),
        ),
    )

    await loop.send("check custom runtime", session_id=sid)
    assert any("CUSTOM_RUNTIME_VISIBLE" in str(msg.get("content", "")) for msg in llm.calls[0])


@pytest.mark.asyncio
async def test_custom_tool_can_use_public_runtime_context(store: SessionStore) -> None:
    reg = ToolRegistry()

    def write_custom_state(**kw) -> str:
        runtime = get_tool_runtime_context(required=True)
        runtime.store.set_runtime_state(
            runtime.session_id,
            "custom",
            {"text": kw["text"]},
        )
        return "custom runtime written"

    reg.register(
        ToolDefinition(
            name="write_custom_state",
            description="Write custom runtime state",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            required_params=("text",),
        ),
        write_custom_state,
    )
    llm = _Scripted(
        responses=[
            _tool_resp("tc-custom", "write_custom_state", '{"text":"PUBLIC_RUNTIME_CONTEXT"}'),
            LLMResponse(raw_text="done"),
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
            runtime_projectors=(_CustomProjector(),),
        ),
    )
    sid = loop.new_session()

    await loop.send("write custom state", session_id=sid)
    assert store.get_runtime_state(sid, "custom") == {"text": "PUBLIC_RUNTIME_CONTEXT"}
    assert any("PUBLIC_RUNTIME_CONTEXT" in str(msg.get("content", "")) for msg in llm.calls[1])


@pytest.mark.asyncio
async def test_runtime_projectors_can_be_disabled(store: SessionStore) -> None:
    sid = store.create_session(system_prompt="S")
    store.set_runtime_state(
        sid,
        "todo",
        {
            "items": [{"id": "hidden", "text": "SHOULD_NOT_INJECT", "status": "in_progress"}],
            "rendered": "[>] #hidden: SHOULD_NOT_INJECT\n\n(0/1 completed)",
        },
    )
    llm = _Scripted(responses=[LLMResponse(raw_text="no runtime")])
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        config=AgentLoopConfig(
            system_prompt="S",
            max_rounds=1,
            compactor=None,
            runtime_projectors=(),
        ),
    )

    await loop.send("no runtime please", session_id=sid)
    assert not any("SHOULD_NOT_INJECT" in str(msg.get("content", "")) for msg in llm.calls[0])


@pytest.mark.asyncio
async def test_configured_skills_dir_is_in_prompt_and_load_skill(tmp_path, store: SessionStore) -> None:
    skill_root = tmp_path / "skills"
    skill_dir = skill_root / "runtime-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: Runtime test skill\n---\nUse the runtime skill instructions.",
        encoding="utf-8",
    )
    llm = _Scripted(
        responses=[
            _tool_resp("tc-skill", "load_skill", '{"name":"runtime-skill"}'),
            LLMResponse(raw_text="loaded"),
        ]
    )
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        tool_registry=create_default_tool_registry(include=["load_skill"]),
        config=AgentLoopConfig(
            system_prompt="Use configured skills.",
            skills_dir=str(skill_root),
            max_rounds=3,
            compactor=None,
        ),
    )
    sid = loop.new_session()
    resolved = loop.resolve_system_prompt(session_id=sid)
    assert "runtime-skill" in resolved
    assert str(skill_root) in resolved

    await loop.send("load the runtime skill", session_id=sid)
    tool_messages = [msg for msg in loop.get_messages(sid) if msg.get("role") == "tool"]
    assert any("Use the runtime skill instructions." in str(msg.get("content", "")) for msg in tool_messages)


@pytest.mark.asyncio
async def test_subsequent_send_reuses_history(store: SessionStore) -> None:
    llm = _Scripted(responses=[
        LLMResponse(raw_text="r1"),
        LLMResponse(raw_text="r2"),
    ])
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=2))
    sid = loop.new_session()
    r1 = await loop.send("first", session_id=sid)
    r2 = await loop.send("second", session_id=sid)
    assert r2.session_id == r1.session_id

    # Second LLM call must see four messages: [user1, assistant1, user2]
    msgs_seen = llm.calls[1]
    assert [m.get("role") for m in msgs_seen] == ["user", "assistant", "user"]
    assert msgs_seen[2]["content"] == "second"


@pytest.mark.asyncio
async def test_unknown_session_id_raises(store: SessionStore) -> None:
    loop = StatefulAgentLoop(llm=_Scripted(), store=store)
    from power_loop import SessionNotFoundError
    with pytest.raises(SessionNotFoundError):
        await loop.send("hello", session_id="sess_nope")


@pytest.mark.asyncio
async def test_send_requires_explicit_session_id(store: SessionStore) -> None:
    loop = StatefulAgentLoop(llm=_Scripted(), store=store)
    with pytest.raises(TypeError):
        await loop.send("hello")  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_tool_round_persists_assistant_and_tool_messages(store: SessionStore) -> None:
    llm = _Scripted(responses=[
        _tool_resp("tc1", "echo", '{"text": "hi"}'),
        LLMResponse(raw_text="final answer"),
    ])
    loop = StatefulAgentLoop(
        llm=llm, store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
        tool_registry=_echo_registry(),
    )
    sid = loop.new_session()
    r = await loop.send("go", session_id=sid)
    assert r.status == "completed"
    assert r.final_text == "final answer"

    roles = [m["role"] for m in loop.get_messages(r.session_id)]
    assert roles == ["user", "assistant", "tool", "assistant"]

    # Pending must have been cleared after the tool message landed.
    assert loop.get_pending(r.session_id) is None


@pytest.mark.asyncio
async def test_pending_state_blocks_followup_send(store: SessionStore) -> None:
    """Simulate a crash mid-tool by deferring tool execution.

    We emulate the crash by directly inserting an assistant(tool_calls)
    message and setting pending — the regular path would not get here
    without a real interrupt, but the invariant we test is the same.
    """
    loop = StatefulAgentLoop(llm=_Scripted(), store=store)
    sid = store.create_session(system_prompt="S")
    seq = store.append_message(
        sid, role="assistant",
        tool_calls=[{"id": "tc-stuck", "function": {"name": "echo", "arguments": "{}"}}],
        round_index=0,
    )
    store.set_pending(sid, {
        "assistant_seq": seq, "round_index": 0,
        "tool_call_ids": ["tc-stuck"],
        "tool_calls": [{"id": "tc-stuck", "function": {"name": "echo", "arguments": "{}"}}],
    })

    with pytest.raises(SessionPendingError) as exc:
        await loop.send("anything", session_id=sid)
    assert exc.value.session_id == sid
    assert exc.value.assistant_seq == seq
    assert len(exc.value.pending_tool_calls) == 1


@pytest.mark.asyncio
async def test_abort_pending_restores_session(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text="recovered")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(max_rounds=3),
    )
    sid = store.create_session()
    seq = store.append_message(
        sid, role="assistant",
        tool_calls=[{"id": "tc-x", "function": {"name": "echo", "arguments": "{}"}}],
        round_index=0,
    )
    store.set_pending(sid, {
        "assistant_seq": seq, "round_index": 0,
        "tool_call_ids": ["tc-x"],
        "tool_calls": [{"id": "tc-x", "function": {"name": "echo", "arguments": "{}"}}],
    })

    aborted = loop.abort_pending(sid, reason="user_cancelled")
    assert aborted == 1
    assert loop.get_pending(sid) is None

    msgs = loop.get_messages(sid)
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert "<aborted: user_cancelled>" in tool_msg["content"]

    # Now send works.
    r = await loop.send("next?", session_id=sid)
    assert r.status == "completed"
    assert r.final_text == "recovered"


@pytest.mark.asyncio
async def test_resume_executes_pending_tools(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text="after resume")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_echo_registry(),
        config=AgentLoopConfig(max_rounds=3),
    )
    sid = store.create_session()
    store.append_message(sid, role="user", content="kick off")
    asst_seq = store.append_message(
        sid, role="assistant",
        tool_calls=[{"id": "tc-y", "function": {"name": "echo", "arguments": "{\"text\":\"resumed\"}"}}],
        round_index=0,
    )
    store.set_pending(sid, {
        "assistant_seq": asst_seq, "round_index": 0,
        "tool_call_ids": ["tc-y"],
        "tool_calls": [{"id": "tc-y", "function": {"name": "echo", "arguments": "{\"text\":\"resumed\"}"}}],
    })

    r = await loop.resume(sid)
    assert r.status == "completed"
    assert r.final_text == "after resume"

    msgs = loop.get_messages(sid)
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert tool_msg["content"] == "resumed"
    assert loop.get_pending(sid) is None


@pytest.mark.asyncio
async def test_close_session_wipes_data(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text="x")])
    loop = StatefulAgentLoop(llm=llm, store=store)
    sid = loop.new_session()
    r = await loop.send("hi", session_id=sid)
    assert loop.get_messages(r.session_id)

    deleted = loop.close_session(r.session_id)
    assert deleted == 1
    assert store.get_session(r.session_id) is None


@pytest.mark.asyncio
async def test_pending_set_then_cleared_during_normal_tool_round(store: SessionStore) -> None:
    """Mid-round pending bookkeeping: we observe pending is set after the
    assistant emits tool_calls and cleared by the time the round ends."""
    seen_pending: list[dict | None] = []
    sid_box: dict[str, str] = {}

    reg = ToolRegistry()

    def _spy(**kw):
        # Snapshot pending while the tool is executing.
        seen_pending.append(store.get_state(sid_box["sid"]).pending)
        return "ok"

    reg.register(
        ToolDefinition(
            name="spy", description="spy",
            input_schema={"type": "object", "properties": {}}, required_params=(),
        ),
        _spy,
    )

    llm = _Scripted(responses=[
        _tool_resp("tc-spy", "spy", "{}"),
        LLMResponse(raw_text="done"),
    ])
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=reg)
    sid_box["sid"] = store.create_session(system_prompt=None)
    r = await loop.send("go", session_id=sid_box["sid"])
    assert r.status == "completed"
    assert seen_pending and seen_pending[0] is not None
    assert "tc-spy" in seen_pending[0]["tool_call_ids"]
    assert loop.get_pending(r.session_id) is None


def test_close_releases_owned_store(tmp_path) -> None:
    db = tmp_path / "s.db"
    loop = StatefulAgentLoop(llm=_Scripted(), db_path=str(db))
    assert loop._owns_store is True
    loop.close()


def test_does_not_close_external_store(store: SessionStore) -> None:
    loop = StatefulAgentLoop(llm=_Scripted(), store=store)
    loop.close()
    # store is still usable
    sid = store.create_session()
    assert store.get_session(sid) is not None


@pytest.mark.asyncio
async def test_compacted_messages_excluded_from_history(store: SessionStore) -> None:
    """Pipeline must see only state=active messages on resume."""
    sid = store.create_session(system_prompt="S")
    store.append_message(sid, role="user", content="old1")
    store.append_message(sid, role="assistant", content="old2")
    store.record_compaction(
        sid, from_seq=1, to_seq=2, note_content="summary of old1+old2",
        before_tokens=100, after_tokens=10, round_index=0,
    )
    store.append_message(sid, role="user", content="latest")

    active = store.load_active_messages(sid)
    assert [m.role for m in active] == ["system", "user"]
    assert [m.state for m in active] == [MessageState.ACTIVE, MessageState.ACTIVE]
    assert active[0].name == "compact_note"


# ── resolve_system_prompt (M1.10) ────────────────────────────────────────


def test_resolve_system_prompt_default(store: SessionStore) -> None:
    """No config.system_prompt → DEFAULT_AGENT_SYSTEM_PROMPT + catalog."""
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="lookup", description="Look up info",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            required_params=("q",),
        ),
        lambda **kw: "ok",
    )
    loop = StatefulAgentLoop(llm=_Scripted(), store=store, tool_registry=reg)
    resolved = loop.resolve_system_prompt()
    # Should contain the default prompt text
    assert "interactive coding agent" in resolved
    # Should contain the auto-injected tool catalog (name + description only)
    assert "# Available Tools" in resolved
    assert "- **lookup**: Look up info" in resolved
    # Parameter schema should NOT be in the catalog (it's in tools= API param)
    assert "q*(string)" not in resolved


def test_resolve_system_prompt_with_config(store: SessionStore) -> None:
    """Custom system_prompt + tool catalog appended."""
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="calc", description="Calculate",
            input_schema={"type": "object", "properties": {"expr": {"type": "string"}}},
            required_params=("expr",),
        ),
        lambda **kw: "42",
    )
    cfg = AgentLoopConfig(system_prompt="You are a math bot.")
    loop = StatefulAgentLoop(llm=_Scripted(), store=store, config=cfg, tool_registry=reg)
    resolved = loop.resolve_system_prompt()
    assert resolved.startswith("You are a math bot.")
    assert "# Available Tools" in resolved
    assert "- **calc**:" in resolved


def test_resolve_system_prompt_injection_disabled(store: SessionStore) -> None:
    """inject_tool_descriptions=False → no catalog appended."""
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="ping", description="Ping",
            input_schema={"type": "object", "properties": {}},
            required_params=(),
        ),
        lambda **kw: "pong",
    )
    cfg = AgentLoopConfig(system_prompt="Hello.", inject_tool_descriptions=False)
    loop = StatefulAgentLoop(llm=_Scripted(), store=store, config=cfg, tool_registry=reg)
    resolved = loop.resolve_system_prompt()
    assert resolved == "Hello."
    assert "# Available Tools" not in resolved


def test_resolve_system_prompt_no_registry(store: SessionStore) -> None:
    """No tool_registry → no catalog appended."""
    cfg = AgentLoopConfig(system_prompt="No tools here.")
    loop = StatefulAgentLoop(llm=_Scripted(), store=store, config=cfg, tool_registry=None)
    resolved = loop.resolve_system_prompt()
    assert resolved == "No tools here."


def test_resolve_system_prompt_custom_header(store: SessionStore) -> None:
    """Custom tool_catalog_header is respected."""
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="x", description="X tool",
            input_schema={"type": "object", "properties": {}},
            required_params=(),
        ),
        lambda **kw: "",
    )
    cfg = AgentLoopConfig(
        system_prompt="Hi.",
        tool_catalog_header="# Tool Reference",
    )
    loop = StatefulAgentLoop(llm=_Scripted(), store=store, config=cfg, tool_registry=reg)
    resolved = loop.resolve_system_prompt()
    assert "# Tool Reference" in resolved
    assert "# Available Tools" not in resolved


def test_resolve_system_prompt_session_override(store: SessionStore) -> None:
    """Session-level system_prompt overrides config-level prompt."""
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="t", description="T",
            input_schema={"type": "object", "properties": {}},
            required_params=(),
        ),
        lambda **kw: "",
    )
    cfg = AgentLoopConfig(system_prompt="Config prompt.")
    loop = StatefulAgentLoop(llm=_Scripted(), store=store, config=cfg, tool_registry=reg)
    # Create session with override
    sid = loop.new_session(system_prompt="Session prompt.")
    resolved = loop.resolve_system_prompt(session_id=sid)
    assert resolved.startswith("Session prompt.")
    assert "Config prompt." not in resolved
    # Catalog is still appended
    assert "# Available Tools" in resolved


class _GateLLM(_Scripted):
    """Blocks the first LLM call until released — simulates a long in-flight run."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.release_first = asyncio.Event()

    async def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        if self._idx == 0:
            await self.release_first.wait()
        return await super().complete(request, **kwargs)


@pytest.mark.asyncio
async def test_follow_up_while_running_queues_for_next_round(store: SessionStore) -> None:
    llm = _GateLLM(
        responses=[
            _tool_resp("c1", "echo", '{"text":"step1"}'),
            LLMResponse(raw_text="steered done"),
        ]
    )
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=4),
        tool_registry=_echo_registry(),
    )
    sid = loop.new_session()

    send_task = asyncio.create_task(loop.send("start task", sid))

    for _ in range(200):
        if loop._lock_for(sid).locked():
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("expected session lock to be held during send")

    queued = await loop.follow_up("please focus on feelings", sid)
    assert isinstance(queued, FollowUpQueued)
    assert queued.queue_depth == 1

    llm.release_first.set()
    result = await send_task
    assert result.status == "completed"
    assert result.final_text == "steered done"
    assert len(llm.calls) >= 2
    second_request = llm.calls[1]
    follow_contents = [
        str(m.get("content") or "")
        for m in second_request
        if m.get("name") == "follow_up" or "<follow_up>" in str(m.get("content") or "")
    ]
    assert any("please focus on feelings" in c for c in follow_contents)


@pytest.mark.asyncio
async def test_follow_up_when_idle_degrades_to_send(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text="idle reply")])
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        config=AgentLoopConfig(system_prompt="S", max_rounds=2),
    )
    sid = loop.new_session()
    result = await loop.follow_up("hello", sid)
    assert not isinstance(result, FollowUpQueued)
    assert result.final_text == "idle reply"
    rows = store.load_active_messages(sid)
    assert any(r.role == "user" and r.content == "hello" for r in rows)
