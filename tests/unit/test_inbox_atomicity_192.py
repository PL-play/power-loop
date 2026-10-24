"""A stale claim cannot become a transcript row or a paid model input."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from power_loop import AgentEventBus, AgentHooks, AgentLoopConfig, FollowUpQueued, HookPoint
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.agent.follow_up import InboxItem
from power_loop.agent.sink import SQLiteSink
from power_loop.contracts.errors import InboxClaimChanged
from power_loop.core.pipeline import AgentPipeline
from power_loop.core.state import ContextManager

from .test_inbox import _Gate, _loop
from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("changed", ["void", "release", "reclaim"])
async def test_stale_claim_rolls_back_message_counter_and_all_delivery_markers(registry_store, changed):
    store, _ = registry_store
    sid = await store.create_session()
    await store.inbox_put(
        sid,
        [
            {"item_id": "drop", "kind": "relay", "content": "withdrawn instruction"},
            {"item_id": "keep", "kind": "relay", "content": "valid instruction"},
        ],
    )
    claimed = await store.inbox_claim(sid)
    token = claimed[0]["claim_token"]
    before = await store.get_state(sid)
    if changed == "void":
        assert await store.inbox_void(sid, ["drop"]) == 1
    elif changed == "release":
        assert await store.inbox_release(sid, token) == 2
    else:
        assert len(await store.inbox_claim(sid, stale_after_ms=-1)) == 2
    with pytest.raises(InboxClaimChanged, match="claim"):
        await store.append_message(
            sid,
            role="user",
            content="withdrawn instruction\nvalid instruction",
            inbox={"claim_token": token, "ids": [r["id"] for r in claimed]},
        )
    assert await store.load_all_messages(sid) == []
    assert (await store.get_state(sid)).next_seq == before.next_seq
    assert all(r["status"] != "delivered" for r in await store.inbox_items(sid))


async def test_committed_delivery_wins_against_later_void_and_remains_deduplicated(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    await store.inbox_put(sid, [{"item_id": "one", "content": "delivered"}])
    [item] = await store.inbox_claim(sid)
    seq = await store.append_message(
        sid, role="user", content="delivered", inbox={"claim_token": item["claim_token"], "ids": [item["id"]]}
    )
    assert await store.inbox_void(sid, ["one"]) == 0
    [row] = await store.inbox_items(sid)
    assert row["status"] == "delivered" and row["delivered_seq"] == seq
    assert await store.inbox_put(sid, [{"item_id": "one", "content": "replacement"}]) == []


@pytest.mark.parametrize("bad", ["empty", "duplicate", "bool", "missing", "foreign", "token"])
async def test_invalid_batch_cannot_partially_deliver_or_write_audit(registry_store, bad):
    store, _ = registry_store
    sid, other = await store.create_session(), await store.create_session()
    await store.inbox_put(sid, [{"item_id": "one", "content": "one"}])
    await store.inbox_put(other, [{"item_id": "other", "content": "foreign"}])
    [row], [foreign] = await store.inbox_claim(sid), await store.inbox_claim(other)
    inbox = {"ids": [row["id"]], "claim_token": row["claim_token"]}
    if bad == "empty":
        inbox["ids"] = []
    elif bad == "duplicate":
        inbox["ids"] *= 2
    elif bad == "bool":
        inbox["ids"] = [True]
    elif bad == "missing":
        inbox["ids"].append(2**63 - 1)
    elif bad == "foreign":
        inbox["ids"].append(foreign["id"])
    else:
        inbox["claim_token"] = ""
    with pytest.raises(InboxClaimChanged):
        await store.append_message(
            sid,
            role="user",
            content="must not exist",
            inbox=inbox,
            hook_injected={"hook_point": "llm_before", "kind": "test", "payload": {"text": "no"}},
        )
    assert await store.load_all_messages(sid) == []
    assert await store.list_hook_events(sid) == []
    assert (await store.get_state(sid)).next_seq == 1
    assert [r["status"] for r in await store.inbox_items(sid)] == ["claimed"]
    assert [r["status"] for r in await store.inbox_items(other)] == ["claimed"]


async def test_delivery_transaction_serializes_concurrent_void_from_another_connection(registry_store, monkeypatch):
    store, connect = registry_store
    other = await connect()
    sid = await store.create_session()
    await store.inbox_put(sid, [{"item_id": "one", "content": "one"}])
    [row] = await store.inbox_claim(sid)
    held, release, void_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = type(store._db).transaction
    execute_original = type(store._db).execute

    class HeldTransaction:
        def __init__(self, tx):
            self.tx = tx

        async def execute(self, sql, params=()):
            count = await self.tx.execute(sql, params)
            if "SET status='delivered'" in sql:
                held.set()
                await release.wait()
            return count

        def __getattr__(self, name):
            return getattr(self.tx, name)

    @asynccontextmanager
    async def transaction(db):
        async with original(db) as tx:
            yield HeldTransaction(tx) if db is store._db else tx

    async def execute(db, sql, params=()):
        if db is other._db and "SET status='void'" in sql:
            void_started.set()
        return await execute_original(db, sql, params)

    monkeypatch.setattr(type(store._db), "transaction", transaction)
    monkeypatch.setattr(type(store._db), "execute", execute)
    async with asyncio.timeout(15):
        delivery = asyncio.create_task(
            store.append_message(
                sid, role="user", content="one", inbox={"ids": [row["id"]], "claim_token": row["claim_token"]}
            )
        )
        void = None
        try:
            await held.wait()
            void = asyncio.create_task(other.inbox_void(sid, ["one"]))
            await void_started.wait()
            assert not void.done()
            release.set()
            seq, count = await asyncio.gather(delivery, void)
            assert count == 0
            [saved] = await store.inbox_items(sid)
            assert saved["status"] == "delivered" and saved["delivered_seq"] == seq
        finally:
            release.set()
            for task in [delivery, void]:
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*[t for t in [delivery, void] if t is not None], return_exceptions=True)


@pytest.mark.parametrize("keep", [False, True])
async def test_idle_reclaims_only_surviving_input_without_reusing_old_batch(registry_store, monkeypatch, keep):
    store, _ = registry_store
    llm = _Gate([LLMResponse(raw_text="done")])
    loop = _loop(store, llm)
    sid = await loop.new_session()
    original = type(store).append_message
    changed = False

    async def append(self, session_id, **kwargs):
        nonlocal changed
        if self is store and kwargs.get("inbox") and not changed:
            changed = True
            assert await store.inbox_void(sid, ["drop"]) == 1
        return await original(self, session_id, **kwargs)

    monkeypatch.setattr(type(store), "append_message", append)
    try:
        result = await loop.deliver(
            [InboxItem("WITHDRAWN", item_id="drop"), *([InboxItem("VALID", item_id="keep")] if keep else [])], sid
        )
        rows = await store.load_all_messages(sid)
        assert "WITHDRAWN" not in json.dumps([r.content for r in rows])
        assert "WITHDRAWN" not in json.dumps(llm.calls)
        assert len(llm.calls) == int(keep)
        if keep:
            assert result.status == "completed"
            assert "VALID" in json.dumps(llm.calls)
        else:
            assert isinstance(result, FollowUpQueued) and result.queue_depth == 0 and rows == []
            assert await loop.flush_follow_ups(sid) is None
        assert (await loop.inbox_pending(sid))["pending"] == 0
    finally:
        await loop.aclose()


@pytest.mark.parametrize("phase", ["round_start", "terminal", "pass_turn"])
@pytest.mark.parametrize("keep", [False, True])
async def test_inflight_retraction_cannot_reach_provider_or_fabricate_a_new_round(
    registry_store, monkeypatch, phase, keep
):
    store, _ = registry_store
    loop = None

    class EnqueueProvider(_Gate):
        async def complete(self, request, **kwargs):
            if not self.calls and phase != "round_start":
                await store.inbox_put(
                    sid,
                    [
                        {"item_id": "drop", "content": "WITHDRAWN"},
                        *([{"item_id": "keep", "content": "VALID"}] if keep else []),
                    ],
                )
                loop._note_committed_inbox(sid)
            return await super().complete(request, **kwargs)

    first = LLMResponse(raw_text="done")
    if phase == "pass_turn":
        first = LLMResponse(
            raw_text="",
            tool_calls=[
                {"id": "pass", "type": "function", "function": {"name": "pass_turn", "arguments": '{"reason":"done"}'}}
            ],
        )
    llm = EnqueueProvider([first, LLMResponse(raw_text="accepted")])
    loop = _loop(store, llm)
    sid = await loop.new_session()
    if phase == "round_start":
        await store.inbox_put(
            sid,
            [{"item_id": "drop", "content": "WITHDRAWN"}, *([{"item_id": "keep", "content": "VALID"}] if keep else [])],
        )
        loop._note_committed_inbox(sid)
    original, usage_original = type(store).append_message, SQLiteSink.on_round_ended
    changed, round_ends, usage_ends = False, [], []

    async def append(self, session_id, **kwargs):
        nonlocal changed
        if self is store and kwargs.get("inbox") and not changed:
            changed = True
            assert await store.inbox_void(sid, ["drop"]) == 1
        return await original(self, session_id, **kwargs)

    async def usage(self, round_index, **kwargs):
        usage_ends.append(round_index)
        return await usage_original(self, round_index, **kwargs)

    async def round_end(ctx):
        round_ends.append(ctx.round_index)

    monkeypatch.setattr(type(store), "append_message", append)
    monkeypatch.setattr(SQLiteSink, "on_round_ended", usage)
    loop.hooks.register(HookPoint.ROUND_END, round_end, name="test.round-end")
    try:
        result = await loop.send("original", sid)
        assert result.status == "completed"
        assert changed
        assert "WITHDRAWN" not in json.dumps(llm.calls)
        assert "WITHDRAWN" not in json.dumps([r.content for r in await store.load_all_messages(sid)])
        expected = 2 if phase == "terminal" and keep else 1
        assert len(llm.calls) == expected
        assert round_ends == usage_ends == list(range(expected))
        if phase == "pass_turn":
            assert bool(result.new_input_after_pass) == keep
        elif keep:
            assert "VALID" in json.dumps(llm.calls)
        assert (await loop.inbox_pending(sid))["pending"] == 0
    finally:
        await loop.aclose()


@pytest.mark.parametrize("after_commit", [False, True])
async def test_unknown_storage_outcome_is_not_retried_or_dispatched(registry_store, monkeypatch, after_commit):
    store, _ = registry_store
    llm, original, appends = _Gate([]), type(store).append_message, []
    loop = _loop(store, llm)
    sid = await loop.new_session()

    async def append(self, session_id, **kwargs):
        appends.append(kwargs)
        if after_commit:
            await original(self, session_id, **kwargs)
        raise OSError("storage outcome unknown")

    monkeypatch.setattr(type(store), "append_message", append)
    try:
        with pytest.raises(OSError, match="unknown"):
            await loop.deliver(InboxItem("original", item_id="one"), sid)
        assert len(appends) == 1 and not llm.calls
        assert len(await store.load_all_messages(sid)) == int(after_commit)
        [row] = await store.inbox_items(sid)
        assert row["status"] == ("delivered" if after_commit else "pending")
    finally:
        await loop.aclose()


async def test_retracted_image_leaves_no_memory_token_image_or_sink_history(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    await store.inbox_put(sid, [{"item_id": "image", "content": "withdrawn image"}])
    [item] = await store.inbox_claim(sid)
    await store.inbox_void(sid, ["image"])
    sink = SQLiteSink(store, sid)
    pipeline = AgentPipeline(
        llm=_Gate([]),
        config=AgentLoopConfig(),
        hooks=AgentHooks(),
        bus=AgentEventBus(),
        ctx=ContextManager(),
        session_id=sid,
        sink=sink,
    )
    pipeline._tok_len = 0
    before = (
        pipeline._tok_len,
        pipeline._tok_total,
        dict(pipeline._image_rounds),
        list(sink._history_seqs),
        list(sink._history_ord),
    )
    accepted = await pipeline._append_message(
        {
            "role": "user",
            "content": [{"type": "attachment", "name": "WITHDRAWN.png", "url": "https://invalid.test/image.png"}],
            "_inbox": {"ids": [item["id"]], "claim_token": item["claim_token"]},
        },
        round_index=2,
    )
    assert accepted is False and pipeline.history == []
    assert (
        pipeline._tok_len,
        pipeline._tok_total,
        pipeline._image_rounds,
        sink._history_seqs,
        sink._history_ord,
    ) == before
    assert await store.load_all_messages(sid) == []


async def test_repeated_claim_churn_is_bounded_and_retains_pending_input(registry_store, monkeypatch):
    store, _ = registry_store
    llm, attempts = _Gate([LLMResponse(raw_text="done")]), []
    loop = _loop(store, llm)
    sid = await loop.new_session()
    original = type(store).append_message

    async def append(self, session_id, **kwargs):
        if self is store and kwargs.get("inbox"):
            attempts.append(kwargs["inbox"]["claim_token"])
            await store.inbox_release(sid, attempts[-1])
        return await original(self, session_id, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(type(store), "append_message", append)
            await loop.deliver(InboxItem("VALID", item_id="one"), sid)
        assert len(attempts) == len(set(attempts)) == 3
        assert not llm.calls and await store.load_all_messages(sid) == []
        assert (await loop.inbox_pending(sid))["pending"] == 1
        assert loop.pending_follow_up_count(sid) > 0
        result = await loop.flush_follow_ups(sid)
        assert result.status == "completed" and len(llm.calls) == 1
        assert (await loop.inbox_pending(sid))["pending"] == 0
    finally:
        await loop.aclose()
