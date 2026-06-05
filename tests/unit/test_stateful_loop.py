from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from llm_client.interface import LLMResponse
from power_loop import (
    AgentLoopConfig,
    MessageState,
    SessionPendingError,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry

# ── Fake LLM ────────────────────────────────────────────────────────────


@dataclass
class _Scripted:
    """LLM that returns canned responses, one per call."""

    responses: list[LLMResponse] = field(default_factory=list)
    calls: list[list[dict]] = field(default_factory=list)
    _idx: int = 0

    async def complete(self, request, **kwargs):
        self.calls.append(list(request.messages))
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done")
        r = self.responses[self._idx]
        self._idx += 1
        return r

    async def stream(self, request):
        raise NotImplementedError

    async def close(self):
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


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> SessionStore:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


@pytest.mark.asyncio
async def test_send_creates_session_and_persists_user_message(store: SessionStore) -> None:
    llm = _Scripted(responses=[LLMResponse(raw_text="hi back")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, config=AgentLoopConfig(system_prompt="S", max_rounds=2),
    )
    r = await loop.send("hello")
    assert r.session_id.startswith("sess_")
    assert r.status == "completed"
    assert r.final_text == "hi back"

    msgs = loop.get_messages(r.session_id)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["content"] == "hi back"


@pytest.mark.asyncio
async def test_subsequent_send_reuses_history(store: SessionStore) -> None:
    llm = _Scripted(responses=[
        LLMResponse(raw_text="r1"),
        LLMResponse(raw_text="r2"),
    ])
    loop = StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=2))
    r1 = await loop.send("first")
    r2 = await loop.send("second", session_id=r1.session_id)
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
    r = await loop.send("go")
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
    r = await loop.send("hi")
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
