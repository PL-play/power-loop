"""recall_send tool — re-expand a past send the projection layer summarized.

Returns that send's FULL pl_messages detail (assistant text + tool calls by name + their
results) by send_index, read-only, current session. The detail always exists because
pl_messages is the immutable audit log (the projection never deletes it).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.store.store import SessionStore
from power_loop.tools import create_default_tool_registry
from power_loop.tools.default_tools import run_recall_send

pytestmark = pytest.mark.unit


@contextmanager
def _active(store: SessionStore, sid: str):
    loop = SimpleNamespace(store=store, config=None)
    t_loop = set_current_loop(loop)  # type: ignore[arg-type]
    t_sid = set_session_id(sid)
    try:
        yield
    finally:
        reset_session_id(t_sid)
        reset_current_loop(t_loop)


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


async def _seed_two_sends(store: SessionStore, sid: str) -> None:
    await store.append_message(sid, role="user", content="do X", send_index=1)
    await store.append_message(
        sid, role="assistant",
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        send_index=1,
    )
    await store.append_message(
        sid, role="tool", content="exit 0\nthe-result", tool_call_id="c1", name="bash",
        send_index=1,
    )
    await store.append_message(sid, role="assistant", content="all done", send_index=1)
    await store.append_message(sid, role="user", content="next thing", send_index=2)


@pytest.mark.asyncio
async def test_recall_send_returns_full_detail(store: SessionStore) -> None:
    sid = await store.create_session(system_prompt="S")
    await _seed_two_sends(store, sid)
    with _active(store, sid):
        out = await run_recall_send(1)
    assert out.startswith("send #1")
    assert "do X" in out and "all done" in out          # user + assistant text
    assert "bash" in out and "the-result" in out         # tool call name + its result
    assert "next thing" not in out                       # only send 1, not send 2


@pytest.mark.asyncio
async def test_recall_send_accepts_string_index(store: SessionStore) -> None:
    sid = await store.create_session()
    await _seed_two_sends(store, sid)
    with _active(store, sid):
        out = await run_recall_send("1")  # tool args may arrive as strings
    assert out.startswith("send #1") and "do X" in out


@pytest.mark.asyncio
async def test_recall_send_missing_send(store: SessionStore) -> None:
    sid = await store.create_session()
    with _active(store, sid):
        out = await run_recall_send(99)
    assert "No messages found for send #99" in out


@pytest.mark.asyncio
async def test_recall_send_truncation_keeps_tool_calls_suffix(store: SessionStore) -> None:
    # A long body must not eat the [tool_calls: …] suffix — else a tool-bearing send looks
    # tool-free in the inspector (regression for #13: truncate body THEN append suffix).
    from power_loop.tools.default_tools import RECALL_SEND_CONTENT_CHARS
    sid = await store.create_session()
    await store.append_message(
        sid, role="assistant", content="x" * (RECALL_SEND_CONTENT_CHARS + 500),
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        send_index=1,
    )
    with _active(store, sid):
        out = await run_recall_send(1)
    assert "[tool_calls: bash]" in out and "…[truncated]" in out


@pytest.mark.asyncio
async def test_recall_send_tolerates_malformed_tool_calls(store: SessionStore) -> None:
    # A non-dict "function" must not raise (recall_send always returns a string).
    sid = await store.create_session()
    await store.append_message(
        sid, role="assistant", content="hmm",
        tool_calls=[{"id": "c1", "function": "not-a-dict", "name": "weird"}], send_index=1,
    )
    with _active(store, sid):
        out = await run_recall_send(1)
    assert "[tool_calls: weird]" in out


@pytest.mark.asyncio
async def test_recall_send_truncation_multiple_tool_calls(store: SessionStore) -> None:
    # The [tool_calls: a, b, c] suffix must list ALL tool names (comma-joined) and survive
    # truncation of a long body.
    from power_loop.tools.default_tools import RECALL_SEND_CONTENT_CHARS
    sid = await store.create_session()
    await store.append_message(
        sid, role="assistant", content="y" * (RECALL_SEND_CONTENT_CHARS + 200),
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "python", "arguments": "{}"}},
            {"id": "c3", "type": "function", "function": {"name": "node", "arguments": "{}"}},
        ],
        send_index=1,
    )
    with _active(store, sid):
        out = await run_recall_send(1)
    assert "[tool_calls: bash, python, node]" in out and "…[truncated]" in out


def test_session_pending_error_tolerates_malformed_function() -> None:
    # SessionPendingError name extraction must guard a non-dict "function" (always builds a msg).
    from power_loop.contracts.errors import SessionPendingError
    exc = SessionPendingError(
        "sid", assistant_seq=1,
        pending_tool_calls=[{"id": "c1", "function": "not-a-dict", "name": "weird"}],
    )
    assert "weird" in str(exc)


def test_recall_send_registered_in_default_tools() -> None:
    reg = create_default_tool_registry(include=["recall_send"], bind=False)
    assert reg.get("recall_send") is not None
    # and it is part of the "full" preset (manifest-driven)
    full = create_default_tool_registry(preset="full", bind=False)
    assert "recall_send" in full.names()
