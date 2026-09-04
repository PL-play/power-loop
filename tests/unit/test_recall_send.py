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


def _amake(rows):
    async def _f(_sid):
        return rows
    return _f


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
    assert "[tool_calls: bash]" in out and "…[truncated:" in out  # 体量与 seq 坐标现在写进截断标记里（/…[truncated: N chars — recall_send(…, seq=S)]）


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
    assert "[tool_calls: bash, python, node]" in out and "…[truncated:" in out


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


# ── 5.4.0: seq-level recall ────────────────────────────────────────────────


async def test_recall_send_seq_returns_single_row_in_full_with_its_call(store):
    sid = await store.create_session()
    await store.append_message(sid, role="user", content="do X", send_index=1)
    await store.append_message(
        sid, role="assistant",
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "load_skill", "arguments": '{"name": "x"}'}}],
        send_index=1,
    )
    big = "SKILL-BODY " * 1000  # 11k chars — past the 2000-char send-level cap
    await store.append_message(sid, role="tool", content=big, tool_call_id="c1", name="load_skill", send_index=1)
    await store.append_message(sid, role="user", content="next", send_index=2)
    rows = await store.load_all_messages(sid)
    tool_seq = next(r.seq for r in rows if r.role == "tool")
    with _active(store, sid):
        out = await run_recall_send(1, tool_seq)
        assert out.startswith(f"send #1 · seq {tool_seq} — original row ({len(big)} chars)")
        assert "[seq 2 · assistant call · load_skill]" in out and '{"name": "x"}' in out
        assert big in out  # NOT cut at 2000
        # send-level view still caps each body
        whole = await run_recall_send(1)
        assert big not in whole and "…[truncated:" in whole
        # wrong seq → says the span, doesn't guess
        bad = await run_recall_send(1, 999)
        assert bad.startswith("No row seq 999 in send #1") and "span seq" in bad
        assert (await run_recall_send(1, "zz")).startswith("Invalid seq")


async def test_recall_send_seq_caps_huge_row_and_counts_remainder(store, monkeypatch):
    import power_loop.tools.default_tools as dt

    monkeypatch.setattr(dt, "RECALL_SEND_ROW_CHARS", 50)
    sid = await store.create_session()
    await store.append_message(sid, role="user", content="go", send_index=1)
    await store.append_message(sid, role="assistant", content="x" * 200, send_index=1)
    rows = await store.load_all_messages(sid)
    seq = rows[-1].seq
    with _active(store, sid):
        out = await run_recall_send(1, seq)
    assert "…[truncated: 150 more chars of 200]" in out


@pytest.mark.asyncio
async def test_send_level_listing_fits_one_budget(monkeypatch, tmp_path):
    """列整份 send 是「地图」不是「载荷」。真实日志里不带 seq 的调用平均 31.5K 字符、带 seq 的
    1.5K——找一样东西的目录比东西本身还贵。所以整份列表共用一个预算：行全在（连同 seq 坐标和
    真实体量），正文一起缩到放得下。"""
    from power_loop.tools import default_tools as dt

    rows = [
        SimpleNamespace(send_index=3, seq=s, role="tool", name=f"t{s}", round_index=1,
                        tool_calls=None, tool_call_id=None, content="x" * 9000, meta=None)
        for s in range(1, 9)
    ]
    store = SimpleNamespace(load_all_messages=_amake(rows))
    monkeypatch.setattr(dt, "get_tool_runtime_context",
                        lambda **_: SimpleNamespace(store=store, session_id="s1"))

    out = await dt.run_recall_send(3)
    assert len(out) < dt.RECALL_SEND_TOTAL_CHARS * 1.6          # 一个预算，不是 8×9000
    assert all(f"[seq {s} ·" in out for s in range(1, 9))       # 每一行都还在
    assert "seq=1" in out and "9000 chars" in out               # 坐标 + 真实体量都给了
