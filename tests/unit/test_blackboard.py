"""Scoped shared blackboard: default SqliteBlackboard + board_* tools."""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass

import pytest

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop import (
    AgentLoopConfig,
    BlackboardEntry,
    BlackboardError,
    SessionStore,
    SqliteBlackboard,
    StatefulAgentLoop,
    register_blackboard_tools,
    render_entries,
)
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


@pytest.fixture
def store() -> Generator[SessionStore, None, None]:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


# ── default SqliteBlackboard ──────────────────────────────────────────────────


async def test_post_read_update_remove(store: SessionStore) -> None:
    bb = SqliteBlackboard(store)
    e = await bb.post("b1", text="claim the plan", kind="task", status="open", author="alice")
    assert isinstance(e, BlackboardEntry) and e.id == 1 and e.kind == "task"

    entries = await bb.read("b1")
    assert len(entries) == 1 and entries[0].text == "claim the plan" and entries[0].author == "alice"

    await bb.update("b1", 1, status="done")
    assert (await bb.read("b1"))[0].status == "done"

    assert await bb.remove("b1", 1) is True
    assert await bb.read("b1") == []


async def test_boards_are_isolated_by_id(store: SessionStore) -> None:
    bb = SqliteBlackboard(store)
    await bb.post("run-A", text="A only", author="x")
    await bb.post("run-B", text="B only", author="y")
    assert [e.text for e in await bb.read("run-A")] == ["A only"]
    assert [e.text for e in await bb.read("run-B")] == ["B only"]


async def test_two_authors_share_one_board(store: SessionStore) -> None:
    bb = SqliteBlackboard(store)
    await bb.post("shared", text="from alice", author="alice")
    await bb.post("shared", text="from bob", author="bob")
    entries = await bb.read("shared")
    assert {e.author for e in entries} == {"alice", "bob"}
    assert [e.id for e in entries] == [1, 2]  # append-only, monotonic ids


async def test_caps_and_errors(store: SessionStore) -> None:
    bb = SqliteBlackboard(store, max_entries=2, max_text_len=10)
    await bb.post("b", text="one")
    await bb.post("b", text="two")
    with pytest.raises(BlackboardError, match="full"):
        await bb.post("b", text="three")
    with pytest.raises(BlackboardError, match="text is required"):
        await bb.post("b2", text="   ")
    with pytest.raises(BlackboardError, match="no board entry"):
        await bb.update("b", 999, status="done")
    # text is truncated to max_text_len
    e = await bb.post("b3", text="0123456789ABCDEF")
    assert e.text == "0123456789"


def test_render_entries() -> None:
    entries = [
        BlackboardEntry(id=1, text="do X", kind="task", status="open", author="alice"),
        BlackboardEntry(id=2, text="noted", kind="note", author="bob"),
    ]
    out = render_entries(entries, header="Board:")
    assert "Board:" in out
    assert "- #1 [task·open] (alice) do X" in out
    assert "- #2 [note] (bob) noted" in out
    assert render_entries([], header="Board:", empty="(empty)").endswith("(empty)")


# ── board_* tools (resolve board + author from RuntimeEnv / session) ──────────


@dataclass
class _LLM(LLMService):
    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        return LLMResponse(raw_text="x", content_text="x")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _tools() -> ToolRegistry:
    reg = ToolRegistry()
    register_blackboard_tools(reg)
    return reg


async def test_board_tools_end_to_end() -> None:
    loop = StatefulAgentLoop(llm=_LLM(), db_path=":memory:",
                             config=AgentLoopConfig(system_prompt="o", compactor=None))
    sid = loop.new_session(metadata={"spec_name": "alice"})
    bb = SqliteBlackboard(loop.store)
    reg = _tools()
    post = reg.get("board_post").handler
    read = reg.get("board_read").handler
    update = reg.get("board_update").handler

    with runtime_env_context(RuntimeEnv(blackboard=bb, blackboard_id="conv-7")):
        tl = set_current_loop(loop)
        ts = set_session_id(sid)
        try:
            out = await post(text="claim the intro", kind="task", status="open")
            assert "task·open" in out and "alice" in out and "claim the intro" in out
            await update(entry_id=1, status="done")
            snap = await read()
        finally:
            reset_current_loop(tl)
            reset_session_id(ts)

    assert "task·done" in snap
    # author was resolved from session metadata, persisted under the board id
    assert [e.author for e in await bb.read("conv-7")] == ["alice"]


async def test_board_tools_error_when_no_board_configured() -> None:
    reg = _tools()
    post = reg.get("board_post").handler
    with runtime_env_context(RuntimeEnv()):  # no blackboard / id
        out = await post(text="hi")
    assert "no shared board is configured" in out


async def test_board_post_validates_kind() -> None:
    loop = StatefulAgentLoop(llm=_LLM(), db_path=":memory:",
                             config=AgentLoopConfig(system_prompt="o", compactor=None))
    sid = loop.new_session()
    reg = _tools()
    post = reg.get("board_post").handler
    with runtime_env_context(RuntimeEnv(blackboard=SqliteBlackboard(loop.store), blackboard_id="b")):
        tl = set_current_loop(loop)
        ts = set_session_id(sid)
        try:
            out = await post(text="x", kind="bogus")
        finally:
            reset_current_loop(tl)
            reset_session_id(ts)
    assert "kind must be one of" in out
