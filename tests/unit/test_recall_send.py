"""recall_send tool — re-expand a past send the projection layer summarized.

Returns that send's FULL pl_messages detail (assistant text + tool calls by name + their
results) by send_index, read-only, current session. The detail always exists because
pl_messages is the immutable audit log (the projection never deletes it).
"""

from __future__ import annotations

import base64
import dataclasses
import json
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
from power_loop.runtime.recall import recall
from power_loop.runtime.store.store import SessionStore
from power_loop.tools import create_default_tool_registry
from power_loop.tools.default_tools import run_recall_send
from tests.unit.session_native_fixture import ACTOR as STORE_ACTOR
from tests.unit.session_native_fixture import DESCRIPTOR as STORE_DESCRIPTOR


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
    assistant_seq = await store.append_message(
        sid, role="assistant",
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        send_index=1,
    )
    await store.append_message(
        sid, role="tool", content="exit 0\nthe-result", tool_call_id="c1", name="bash",
        send_index=1, meta={"_assistant_seq": assistant_seq},
    )
    await store.append_message(sid, role="assistant", content="all done", send_index=1)
    await store.append_message(sid, role="user", content="next thing", send_index=2)


@pytest.mark.asyncio
async def test_recall_send_returns_full_detail(store: SessionStore) -> None:
    sid = await store.create_session(dataclasses.replace(STORE_DESCRIPTOR, config={"system_prompt": "S"}), actor=STORE_ACTOR)
    await _seed_two_sends(store, sid)
    with _active(store, sid):
        out = await run_recall_send(1)
    assert out.startswith("send #1")
    assert "do X" in out and "all done" in out          # user + assistant text
    assert "bash" in out and "the-result" in out         # tool call name + its result
    assert "next thing" not in out                       # only send 1, not send 2


@pytest.mark.asyncio
async def test_recall_send_accepts_string_index(store: SessionStore) -> None:
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    await _seed_two_sends(store, sid)
    with _active(store, sid):
        out = await run_recall_send("1")  # tool args may arrive as strings
    assert out.startswith("send #1") and "do X" in out


@pytest.mark.asyncio
async def test_recall_send_missing_send(store: SessionStore) -> None:
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    with _active(store, sid):
        out = await run_recall_send(99)
    assert "No messages found for send #99" in out


@pytest.mark.asyncio
async def test_recall_send_truncation_keeps_tool_calls_suffix(store: SessionStore) -> None:
    # A long body must not eat the [tool_calls: …] suffix — else a tool-bearing send looks
    # tool-free in the inspector (regression for #13: truncate body THEN append suffix).
    from power_loop.tools.default_tools import RECALL_SEND_CONTENT_CHARS
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
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
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
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
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
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


def test_recall_send_registered_in_default_tools() -> None:
    reg = create_default_tool_registry(include=["recall_send"], bind=False)
    assert reg.get("recall_send") is not None
    # and it is part of the "full" preset (manifest-driven)
    full = create_default_tool_registry(preset="full", bind=False)
    assert "recall_send" in full.names()


# ── 5.4.0: seq-level recall ────────────────────────────────────────────────


async def test_recall_send_seq_returns_single_row_in_full_with_its_call(store):
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    await store.append_message(sid, role="user", content="do X", send_index=1)
    assistant_seq = await store.append_message(
        sid, role="assistant",
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "load_skill", "arguments": '{"name": "x"}'}}],
        send_index=1,
    )
    big = "SKILL-BODY " * 1000  # 11k chars — past the 2000-char send-level cap
    await store.append_message(sid, role="tool", content=big, tool_call_id="c1", name="load_skill",
                               send_index=1, meta={"_assistant_seq": assistant_seq})
    await store.append_message(sid, role="user", content="next", send_index=2)
    rows = await store.load_all_messages(sid)
    tool_seq = next(r.seq for r in rows if r.role == "tool")
    with _active(store, sid):
        out = await run_recall_send(1, tool_seq, char_limit=20000)
        assert out.startswith(f"send #1 · seq {tool_seq} — original row ({len(big)} chars)")
        assert "[seq 2 · assistant call · load_skill]" in out and '{"name": "x"}' in out
        assert big in out  # NOT cut at 2000
        # send-level view still caps each body
        whole = await run_recall_send(1)
        assert big not in whole and '"truncated":true' in whole
        # wrong seq → says the span, doesn't guess
        bad = await run_recall_send(1, 999)
        assert bad.startswith("No row seq 999 in send #1") and "span seq" in bad
        assert (await run_recall_send(1, "zz")).startswith("Invalid seq")


async def test_recall_send_seq_caps_huge_row_and_counts_remainder(store, monkeypatch):
    import power_loop.tools.default_tools as dt

    monkeypatch.setattr(dt, "RECALL_SEND_ROW_CHARS", 50)
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    await store.append_message(sid, role="user", content="go", send_index=1)
    await store.append_message(sid, role="assistant", content="x" * 200, send_index=1)
    rows = await store.load_all_messages(sid)
    seq = rows[-1].seq
    with _active(store, sid):
        out = await run_recall_send(1, seq)
    assert "…[truncated: 150 more chars of 200]" in out


@pytest.mark.asyncio
async def test_send_level_listing_fits_one_budget(store):
    import json
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for i in range(8):
        await store.append_message(sid, role="tool", content="x" * 9000, name=f"t{i}", send_index=3)
    with _active(store, sid):
        out = await run_recall_send(3)
    data = json.loads(out)
    assert len(out) < 8000
    assert [r["seq"] for r in data["records"]] == list(range(1, 9))
    assert all(r["truncated"] and r["total_chars"] > 9000 for r in data["records"])


async def test_small_list_reads_only_the_target_send_page(store, monkeypatch):
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    await _seed_two_sends(store, sid)
    calls = []
    original = store.history_records

    async def records(*args, **kwargs):
        calls.append(kwargs)
        return await original(*args, **kwargs)

    async def no_full_read(_sid):
        raise AssertionError("small list must not load the full session")

    monkeypatch.setattr(store, "history_records", records)
    monkeypatch.setattr(store, "load_all_messages", no_full_read)
    with _active(store, sid):
        out = await run_recall_send(1)
    assert out.startswith("send #1") and "all done" in out and "next thing" not in out
    assert len(calls) == 1 and calls[0]["send_index"] == 1 and calls[0]["limit"] == 21


async def test_large_list_reads_limit_plus_one_before_core(store, monkeypatch):
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for index in range(25):
        await store.append_message(sid, role="tool", content=f"row {index}", send_index=1)
    calls = []
    original = store.history_records

    async def records(*args, **kwargs):
        calls.append(kwargs)
        return await original(*args, **kwargs)

    async def no_full_read(_sid):
        raise AssertionError("large list must not load the full session")

    monkeypatch.setattr(store, "history_records", records)
    monkeypatch.setattr(store, "load_all_messages", no_full_read)
    with _active(store, sid):
        out = json.loads(await run_recall_send(1, limit=20))
    assert len(out["records"]) == 20 and out["next_cursor"]
    assert [call["limit"] for call in calls] == [21, 21]
    assert all(call["send_index"] == 1 for call in calls)


async def test_targeted_bounds_and_exact_seq(store, monkeypatch):
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    await _seed_two_sends(store, sid)
    queries = []
    original = store._db.fetchone

    async def fetchone(sql, params=()):
        queries.append(" ".join(sql.split()))
        return await original(sql, params)

    monkeypatch.setattr(store._db, "fetchone", fetchone)
    assert await store.history_bounds(sid, send_index=1) == [1, 4]
    assert await store.history_bounds(sid, lo=2, hi=4) == [2, 4]
    targeted = [sql for sql in queries if "LIMIT 1" in sql]
    assert len(targeted) == 4 and all("min(" not in sql and "max(" not in sql for sql in targeted)
    assert sum("send_index=?" in sql for sql in targeted) == 2
    assert sum("ORDER BY seq ASC LIMIT 1" in sql for sql in targeted) == 2
    assert sum("ORDER BY seq DESC LIMIT 1" in sql for sql in targeted) == 2

    async def no_bounds(*args, **kwargs):
        raise AssertionError("exact seq must not query bounds")

    monkeypatch.setattr(store, "history_bounds", no_bounds)
    answer = json.loads(await recall(store, sid, seq=4))
    assert answer["status"] == "ok" and json.loads(answer["text"])["content"] == "all done"


async def test_ids_cursor_captures_actual_last_record(registry_store):
    store = registry_store[0]
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for content in ("one", "two", "three"):
        await store.append_message(sid, role="user", content=content, send_index=1)
    first = json.loads(await recall(store, sid, ids=[1, 100], limit=1))
    assert [row["seq"] for row in first["records"]] == [1]
    state = json.loads(base64.urlsafe_b64decode(first["next_cursor"].encode()))
    assert state["hi"] == 3
    assert await store.append_message(sid, role="user", content="four", send_index=1) == 4
    seen = [1]
    cursor = first["next_cursor"]
    while cursor is not None:
        page = json.loads(await recall(store, sid, ids=[1, 100], cursor=cursor, limit=1))
        assert page["status"] == "ok"
        seen.extend(row["seq"] for row in page["records"])
        cursor = page["next_cursor"]
    assert seen == [1, 2, 3]
    invalid = json.loads(await recall(store, sid, ids=[1, 100], cursor="not-a-cursor"))
    assert invalid["status"] == "error"


async def test_range_reads_stay_within_session(registry_store):
    store = registry_store[0]
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    other = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for index in range(1, 4):
        await store.append_message(sid, role="user", content=f"m{index}", send_index=1)
        await store.append_message(other, role="user", content=f"o{index}", send_index=1)
    assert await store.history_bounds(sid, send_index=1) == [1, 3]
    assert [row["seq"] for row in await store.history_records(
        sid, send_index=1, limit=10)] == [1, 2, 3]
    assert await store.history_bounds(sid, send_index=9) is None
    assert await store.history_records(sid, send_index=9, limit=10) == []
    assert await store.history_bounds(sid, lo=2, hi=100) == [2, 3]

    await store.upsert_project_message(sid, send_index=1, kind="project", content={"n": 1})
    await store.upsert_project_message(other, send_index=1, kind="project", content={"n": 2})
    await store.upsert_project_message(sid, send_index=2, kind="project", content={"n": 3})
    projection = [row["id"] for row in await store.history_records(
        sid, source="projection", lo=0, hi=None, limit=10)]
    assert len(projection) == 2
    assert await store.history_bounds(
        sid, source="projection", lo=1, hi=2**63 - 1) == [projection[0], projection[-1]]
    assert [row["id"] for row in await store.history_records(
        sid, source="projection", lo=projection[0], hi=projection[0], limit=10,
    )] == [projection[0]]
    assert await store.history_bounds(
        sid, source="projection", lo=2**63 - 1, hi=2**63 - 1) is None


async def test_projection_finite_range_and_filters(registry_store):
    store = registry_store[0]
    sessions = [await store.create_session(
        STORE_DESCRIPTOR, actor=STORE_ACTOR, session_id=f"range-{name}",
    ) for name in ("a", "m", "z", "empty")]
    left, sid, right, empty = sessions
    # Unrelated ids precede and interleave the target; session keys sort on both sides.
    for index, owner in enumerate((left, right, sid, left, sid, right, sid), 1):
        await store.upsert_project_message(
            owner, send_index=index, kind="compact" if index == 5 else "project",
            content={"n": index}, source_seq_hi=index, projector_version=1,
        )
    rows = await store.history_records(sid, source="projection", limit=20)
    ids = [row["id"] for row in rows]
    assert [row["content"] for row in rows] == [{"n": 3}, {"n": 5}, {"n": 7}]
    hi = 2**63 - 1
    assert await store.history_bounds(sid, source="projection", lo=1, hi=hi) == [ids[0], ids[-1]]
    first = await store.history_records(sid, source="projection", lo=1, hi=hi, limit=1)
    rest = await store.history_records(sid, source="projection", lo=1, hi=hi, after=ids[0], limit=20)
    assert first + rest == rows
    assert await store.history_records(
        sid, source="projection", lo=1, hi=hi, exclude_compact=True,
    ) == [rows[0], rows[2]]
    assert await store.history_bounds(
        sid, source="projection", lo=1, hi=hi, after_seq=3,
    ) == [ids[-1], ids[-1]]
    assert await store.history_bounds(empty, source="projection", lo=1, hi=hi) is None
    assert await store.history_records(empty, source="projection", lo=1, hi=hi) == []
    for after in (ids[-1], ids[-1] + 1):
        assert await store.history_records(sid, source="projection", lo=1, hi=ids[-1], after=after) == []
    assert await store.history_records(sid, source="projection", lo=ids[1], hi=ids[1]) == [rows[1]]
    assert await store.history_bounds(sid, source="projection", send_index=5) == [ids[1], ids[1]]
    assert await store.history_records(sid, source="projection", send_index=5) == [rows[1]]
    assert await store.history_records(sid, source="projection", lo=ids[1]) == rows[1:]
    assert await store.history_records(
        sid, source="projection", lo=1, hi=hi, record_ids=[ids[1]],
    ) == [rows[1]]


async def test_projection_cursor_keeps_captured_upper_bound(registry_store):
    store = registry_store[0]
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    for index in range(1, 4):
        await store.upsert_project_message(sid, send_index=index, kind="project", content={"n": index})
    rows = await store.history_records(sid, source="projection")
    ids = [1, 2**63 - 1]
    first = json.loads(await recall(store, sid, source="projection", ids=ids, limit=1))
    state = json.loads(base64.urlsafe_b64decode(first["next_cursor"].encode()))
    assert state["hi"] == rows[-1]["id"]
    await store.upsert_project_message(sid, send_index=4, kind="project", content={"n": 4})
    current = await store.history_records(sid, source="projection")
    assert current[-1]["id"] > state["hi"]
    seen = [first["records"][0]["id"]]
    cursor = first["next_cursor"]
    while cursor is not None:
        page = json.loads(await recall(store, sid, source="projection", ids=ids, cursor=cursor, limit=1))
        assert page["status"] == "ok"
        seen.extend(row["id"] for row in page["records"])
        cursor = page["next_cursor"]
    assert seen == [row["id"] for row in rows]


async def test_sink_recall_uses_actual_call_per_round(registry_store):
    from power_loop.agent.sink import SQLiteSink
    from power_loop.runtime.tool_context import ToolContextConfig, ToolContextEngine, active_engine

    store = registry_store[0]
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    sink = SQLiteSink(store, sid)
    results = []
    for round_index in range(2):
        calls = [{"id": f"reused-{i}", "type": "function", "function": {
            "name": "bash", "arguments": json.dumps({"marker": f"round-{round_index}-call-{i}"})}}
                 for i in range(2)]
        await sink.on_message_appended({"role": "assistant", "tool_calls": calls,
                                       "send_index": 1}, round_index=round_index)
        source_seq = sink.persisted_assistant_seq
        await sink.on_assistant_tool_calls(assistant_seq=source_seq, tool_calls=calls, round_index=round_index)
        for i in (1, 0):  # Results may arrive in a different order than the call array.
            message = {"role": "tool", "tool_call_id": f"reused-{i}", "name": "bash", "send_index": 1,
                       "content": ("success", "error", "hook skipped", "stopped")[round_index * 2 + i],
                       "meta": {"_assistant_seq": 999999, "result_status": "fixture"}}
            before = json.loads(json.dumps(message))
            await sink.on_message_appended(message, round_index=round_index)
            assert message == before
            row = await store.get_message(sid, sink.send_rows[-1].seq)
            assert row.meta == sink.send_rows[-1].meta
            assert row.meta["_assistant_seq"] == source_seq
            results.append((row, f"round-{round_index}-call-{i}", source_seq))
        assert (await store.get_state(sid)).pending is None
        assert sink.persisted_assistant_seq is None
    engine = ToolContextEngine(config=ToolContextConfig(), store=store, owner_session_id=sid,
                               owner_send_index=1, llm=None, model="test")
    await engine.start()
    with _active(store, sid):
        for row, marker, source_seq in results:
            text = await run_recall_send(1, row.seq)
            assert text.count("assistant call") == 1
            assert f"[seq {source_seq} · assistant call · bash]" in text
            assert marker in text and row.content in text
            other_round = "round-1" if marker.startswith("round-0") else "round-0"
            assert other_round not in text
            token = active_engine.set(engine)
            try:
                value = json.loads(await run_recall_send(1, row.seq))
            finally:
                active_engine.reset(token)
            assert json.loads(value["arguments"]) == {"marker": marker}
            assert value["result"] == row.content
        row, marker, _ = results[-1]
        chunks, offset = [], 1
        while offset is not None:
            page = json.loads(await run_recall_send(1, row.seq, char_offset=offset, char_limit=256))
            chunks.append(page["text"])
            offset = page["next_char_offset"]
        assert json.loads("{" + "".join(chunks)) == value
    # An unrelated raw result cannot smuggle a managed-sink source through external meta.
    await sink.on_message_appended({"role": "tool", "tool_call_id": "unmanaged", "send_index": 1,
                                   "meta": {"_assistant_seq": 999999}, "content": "raw"}, round_index=2)
    assert "_assistant_seq" not in sink.send_rows[-1].meta
    assert "_assistant_seq" not in (await store.get_message(sid, sink.send_rows[-1].seq)).meta


async def test_recall_missing_source_never_substitutes_another_call(registry_store):
    from power_loop.runtime.tool_archive import recall_original

    store = registry_store[0]
    sid = await store.create_session(STORE_DESCRIPTOR, actor=STORE_ACTOR)
    def call(marker):
        return [{"id": "same", "function": {"name": "bash", "arguments": marker}}]
    original = await store.append_message(sid, role="assistant", send_index=1, tool_calls=call("ORIGINAL"))
    await store.append_message(sid, role="assistant", send_index=1, tool_calls=call("RECENT"))
    row_seq = await store.append_message(sid, role="tool", send_index=1, tool_call_id="same",
                                         content="result", meta={"_assistant_seq": original})
    row = await store.get_message(sid, row_seq)
    assert json.loads(await recall_original(store, sid, 1, row))["arguments"] == "ORIGINAL"
    foreign = await store.append_message(sid, role="assistant", send_index=2, tool_calls=call("OTHER SEND"))
    wrong_id = await store.append_message(sid, role="assistant", send_index=1, tool_calls=[{"id": "different"}])
    plain = await store.append_message(sid, role="user", send_index=1, content="ordinary")
    for meta in ({}, {"_assistant_seq": foreign}, {"_assistant_seq": wrong_id}, {"_assistant_seq": plain}):
        seq = await store.append_message(sid, role="tool", send_index=1, tool_call_id="same",
                                         content="still readable", meta=meta)
        value = json.loads(await recall_original(store, sid, 1, await store.get_message(sid, seq)))
        assert value["arguments"] is None and value["result"] == "still readable"
        with _active(store, sid):
            text = await run_recall_send(1, seq)
        assert "Original call arguments unavailable." in text and "assistant call" not in text
    await store._db.execute(f"DELETE FROM {store.t.messages} WHERE session_id=? AND seq=?", (sid, original))
    assert json.loads(await recall_original(store, sid, 1, row))["arguments"] is None
    with _active(store, sid):
        assert "ordinary" in await run_recall_send(1, plain)
        assert "span seq" in await run_recall_send(1, foreign)
        assert "span seq" in await run_recall_send(1, 999999)
        assert "No messages found" in await run_recall_send(99, plain)
        assert (await run_recall_send(1, "invalid")).startswith("Invalid seq")
