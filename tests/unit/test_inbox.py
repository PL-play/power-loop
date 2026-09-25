"""The session inbox (design/124 §6): durable, deduplicated, delivered exactly once.

Covers the loop-level contract on top of the store tests in test_session_leases*.py:
- a cancelled run neither loses nor double-delivers what was waiting (the Z4 incident);
- what a person said never shares a message with a system wake-up (Z2);
- the idle path, the in-flight path and the terminal-window flush all go through the inbox;
- nothing private (the ``_inbox`` marker) ever reaches the provider;
- a v7 store migrates to v8 with its queued rows kept.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from power_loop import AgentLoopConfig, FollowUpQueued, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService
from power_loop.agent.follow_up import FOLLOW_UP_MESSAGE_NAME, InboxItem
from power_loop.contracts.errors import SessionPendingError
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.cancellation import CancellationToken
from power_loop.tools.registry import ToolRegistry


class _Gate(LLMService):
    """Scripted LLM; call #n blocks until ``gates[n]`` is set (if one is registered)."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.gates: dict[int, asyncio.Event] = {}

    def gate(self, n: int) -> asyncio.Event:
        self.gates[n] = asyncio.Event()
        return self.gates[n]

    async def complete(self, request: LLMRequest, **kwargs: Any) -> LLMResponse:
        n = len(self.calls)
        self.calls.append([dict(m) for m in request.messages])
        if n in self.gates:
            await self.gates[n].wait()
        return self.responses.pop(0) if self.responses else LLMResponse(raw_text="done")

    async def close(self) -> None:
        return None


def _tool_call(call_id: str) -> LLMResponse:
    return LLMResponse(raw_text="", tool_calls=[{
        "id": call_id, "type": "function",
        "function": {"name": "echo", "arguments": '{"text":"x"}'},
    }])


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="echo", description="Echo",
                       input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                       required_params=("text",)),
        lambda **kw: str(kw.get("text") or ""),
    )
    return reg


def _loop(store: SessionStore, llm: LLMService, **cfg: Any) -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=llm, store=store, tool_registry=_registry(),
        config=AgentLoopConfig(system_prompt="S", max_rounds=cfg.pop("max_rounds", 4),
                               compactor=None, **cfg),
    )


async def _wait_calls(llm: _Gate, n: int) -> None:
    for _ in range(400):
        if len(llm.calls) >= n:
            return
        await asyncio.sleep(0.005)
    pytest.fail(f"expected ≥{n} LLM calls, got {len(llm.calls)}")


def _user_texts(rows: list[Any]) -> list[str]:
    return [str(r.content or "") for r in rows if r.role == "user"]


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


# ── Z4: cancel neither loses nor duplicates ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_then_redeliver_puts_each_message_in_history_once(store: SessionStore) -> None:
    """The incident reproduced in design/124 phase 0: a follow-up waiting when the run was
    cancelled, plus the host re-sending the same batch, used to put the same user message in the
    transcript twice (and the stale copy last, as if just said). With item ids the re-send is a
    no-op and the waiting item is delivered once, by the next send."""
    llm = _Gate([_tool_call("c1"), LLMResponse(raw_text="ok")])
    gate = llm.gate(0)
    loop = _loop(store, llm)
    sid = await loop.new_session()
    tok = CancellationToken()

    run = asyncio.create_task(loop.deliver(
        InboxItem("帮我做调研", item_id="m1"), sid, stop_event=tok))
    await _wait_calls(llm, 1)
    queued = await loop.deliver(InboxItem("顺便查一下天气", item_id="m2"), sid)
    assert isinstance(queued, FollowUpQueued) and queued.accepted == 1
    tok.cancel("user stop")
    gate.set()
    res = await run
    assert res.status == "cancelled"
    await loop.abort_pending(sid, reason="cancelled by user")

    # waiting item survived the cancel
    assert (await loop.inbox_pending(sid))["pending"] == 1

    # host re-sends the whole batch + one new message
    res2 = await loop.deliver([InboxItem("帮我做调研", item_id="m1"),
                               InboxItem("顺便查一下天气", item_id="m2"),
                               InboxItem("新的一句", item_id="m3")], sid)
    assert res2.status == "completed"
    texts = "\n".join(_user_texts(await store.load_active_messages(sid)))
    assert texts.count("帮我做调研") == 1
    assert texts.count("顺便查一下天气") == 1
    assert texts.count("新的一句") == 1
    assert (await loop.inbox_pending(sid))["pending"] == 0


@pytest.mark.asyncio
async def test_duplicate_only_delivery_starts_no_run(store: SessionStore) -> None:
    llm = _Gate([LLMResponse(raw_text="first")])
    loop = _loop(store, llm)
    sid = await loop.new_session()
    await loop.deliver(InboxItem("hello", item_id="m1"), sid)
    n = len(llm.calls)
    again = await loop.deliver(InboxItem("hello", item_id="m1"), sid)
    assert isinstance(again, FollowUpQueued)
    assert again.accepted == 0 and again.duplicates == 1
    assert len(llm.calls) == n, "a duplicate must not wake the model"


# ── Z2: kinds never merge ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inflight_kinds_become_separate_messages(store: SessionStore) -> None:
    llm = _Gate([_tool_call("c1"), LLMResponse(raw_text="ok")])
    gate = llm.gate(0)
    loop = _loop(store, llm)
    sid = await loop.new_session()
    run = asyncio.create_task(loop.send("start", sid))
    await _wait_calls(llm, 1)
    await loop.deliver([InboxItem("用户：换个方向", item_id="u1"),
                        InboxItem("用户：再加一点", item_id="u2"),
                        InboxItem("[后台任务 t1 已完成]", kind="task_done", item_id="t1")], sid)
    gate.set()
    await run

    second = llm.calls[1]
    follow = [m for m in second if m.get("name") == FOLLOW_UP_MESSAGE_NAME]
    assert len(follow) == 2, "user items and the task wake-up must be two messages"
    assert "换个方向" in follow[0]["content"] and "再加一点" in follow[0]["content"]
    assert "后台任务" not in follow[0]["content"]
    assert "后台任务" in follow[1]["content"]

    rows = [r for r in await store.load_active_messages(sid) if r.name == FOLLOW_UP_MESSAGE_NAME]
    assert [r.meta.get("inbox", {}).get("kinds") for r in rows] == [["user"], ["task_done"]]
    assert rows[0].meta["inbox"]["item_ids"] == ["u1", "u2"]


@pytest.mark.asyncio
async def test_idle_multi_kind_first_group_is_the_input_rest_before_first_call(
    store: SessionStore,
) -> None:
    llm = _Gate([LLMResponse(raw_text="ok")])
    loop = _loop(store, llm)
    sid = await loop.new_session()
    s = await loop.ensure_store()
    # left waiting by earlier runs (e.g. a wake-up that arrived while the session was busy)
    await s.inbox_put(sid, [{"item_id": "t1", "kind": "task_done", "content": "任务完成"}])
    await loop.deliver(InboxItem("用户的新问题", item_id="u1"), sid)

    first_call = llm.calls[0]
    user_msgs = [m for m in first_call if m.get("role") == "user"]
    assert user_msgs[0]["content"] == "任务完成"          # send input: oldest group, no envelope
    assert "用户的新问题" in user_msgs[1]["content"]       # claimed by the round-0 drain
    assert user_msgs[1].get("name") == FOLLOW_UP_MESSAGE_NAME
    assert len(llm.calls) == 1, "both reached the model in the SAME first call"


# ── nothing private leaks, structured content survives ───────────────────────────────────


@pytest.mark.asyncio
async def test_inbox_marker_never_reaches_the_provider(store: SessionStore) -> None:
    llm = _Gate([_tool_call("c1"), LLMResponse(raw_text="ok")])
    gate = llm.gate(0)
    loop = _loop(store, llm)
    sid = await loop.new_session()
    run = asyncio.create_task(loop.deliver(InboxItem("go", item_id="a"), sid))
    await _wait_calls(llm, 1)
    await loop.deliver(InboxItem("steer", item_id="b"), sid)
    gate.set()
    await run
    for call in llm.calls:
        for m in call:
            assert "_inbox" not in m and "inbox" not in m


@pytest.mark.asyncio
async def test_image_blocks_survive_the_inbox(store: SessionStore) -> None:
    llm = _Gate([_tool_call("c1"), LLMResponse(raw_text="ok")])
    gate = llm.gate(0)
    loop = _loop(store, llm)
    sid = await loop.new_session()
    run = asyncio.create_task(loop.send("start", sid))
    await _wait_calls(llm, 1)
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    await loop.deliver(InboxItem({"role": "user", "content": [{"type": "text", "text": "看这张"},
                                                             img]}, item_id="img1"), sid)
    gate.set()
    await run
    follow = [m for m in llm.calls[1] if m.get("name") == FOLLOW_UP_MESSAGE_NAME][0]
    assert isinstance(follow["content"], list)
    assert img in follow["content"]
    assert "看这张" in follow["content"][0]["text"]


# ── steer event, pending safety, terminal flush ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_steer_event_set_on_accept_and_cleared_on_claim(store: SessionStore) -> None:
    llm = _Gate([_tool_call("c1"), LLMResponse(raw_text="ok")])
    gate = llm.gate(0)
    loop = _loop(store, llm)
    sid = await loop.new_session()
    run = asyncio.create_task(loop.send("start", sid))
    await _wait_calls(llm, 1)
    ev = loop.steer_event(sid)
    assert not ev.is_set()
    await loop.deliver(InboxItem("queued only", item_id="q"), sid)
    assert not ev.is_set(), "queue-mode items don't raise the steer flag"
    await loop.deliver(InboxItem("stop please", mode="steer", item_id="s"), sid)
    assert ev.is_set()
    assert (await loop.inbox_pending(sid)) == {"pending": 2, "steer": 1}
    gate.set()
    await run
    assert not ev.is_set()
    assert (await loop.inbox_pending(sid)) == {"pending": 0, "steer": 0}


@pytest.mark.asyncio
async def test_refused_idle_send_leaves_items_pending(store: SessionStore) -> None:
    """A session with stale tool_calls refuses a send; what was delivered must not be lost with
    it — it waits until the host heals the session."""
    llm = _Gate([_tool_call("c1")])
    loop = _loop(store, llm, max_rounds=1)
    sid = await loop.new_session()
    s = await loop.ensure_store()
    await loop.send("start", sid)  # round limit hit mid tool-call? force a pending state:
    await s.set_pending(sid, {"assistant_seq": 1, "round_index": 0, "tool_call_ids": ["zz"],
                              "tool_calls": [{"id": "zz", "type": "function",
                                              "function": {"name": "echo", "arguments": "{}"}}]})
    with pytest.raises(SessionPendingError):
        await loop.deliver(InboxItem("don't lose me", item_id="k"), sid)
    assert (await loop.inbox_pending(sid))["pending"] == 1
    await loop.abort_pending(sid, reason="heal")
    llm.responses.append(LLMResponse(raw_text="ok"))
    res = await loop.flush_follow_ups(sid)
    assert res is not None and res.status == "completed"
    assert "don't lose me" in "\n".join(_user_texts(await store.load_active_messages(sid)))


@pytest.mark.asyncio
async def test_terminal_window_item_is_flushed_once(store: SessionStore) -> None:
    llm = _Gate([LLMResponse(raw_text="a"), LLMResponse(raw_text="b")])
    loop = _loop(store, llm)
    sid = await loop.new_session()
    await loop.send("first", sid)
    # accepted as if the lock was still held at the very end of that run
    async with loop._lock_for(sid):
        q = await loop.follow_up("late card submission", sid, item_id="card")
    assert isinstance(q, FollowUpQueued)
    assert loop.pending_follow_up_count(sid) == 1
    res = await loop.flush_follow_ups(sid)
    assert res is not None and res.status == "completed"
    assert loop.pending_follow_up_count(sid) == 0
    assert await loop.flush_follow_ups(sid) is None
    texts = "\n".join(_user_texts(await store.load_active_messages(sid)))
    assert texts.count("late card submission") == 1


# ── schema v7 → v8 ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_v7_store_migrates_with_queued_rows_kept(tmp_path: Any) -> None:
    path = str(tmp_path / "v7.db")
    s = await SessionStore.open(path)
    sid = await s.create_session(system_prompt="S")
    db = s._db
    # rewind to the v7 shape: old queue table with two parked rows, version 7
    await db.execute(f"DROP TABLE {s.t.follow_up_queue}")
    await db.execute(
        f"CREATE TABLE {s.t.follow_up_queue} (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT NOT NULL, content TEXT NOT NULL, created_at INTEGER NOT NULL)")
    for i, text in enumerate(("parked one", "parked two")):
        await db.execute(
            f"INSERT INTO {s.t.follow_up_queue} (session_id, content, created_at) VALUES (?,?,?)",
            (sid, text, 1000 + i))
    prefix = s.t.sessions[: -len("sessions")]
    ver_table = f"{prefix}schema_migrations"
    await db.execute(f"UPDATE {ver_table} SET version=7")
    await s.close()

    s2 = await SessionStore.open(path)
    try:
        assert s2.schema_version == 8
        rows = await s2.inbox_items(sid)
        assert [(r["item_id"], r["kind"], r["mode"], r["status"], r["content"]) for r in rows] == [
            ("v7:1", "user", "queue", "pending", "parked one"),
            ("v7:2", "user", "queue", "pending", "parked two"),
        ]
        # and the migrated table behaves like a fresh one
        assert await s2.inbox_put(sid, [{"item_id": "v7:1", "content": "dup"}]) == []
    finally:
        await s2.close()
