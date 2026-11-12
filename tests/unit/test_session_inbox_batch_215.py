"""Atomic session inbox acceptance and whole-snapshot batching."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import CapacityError, RuntimeRequestError

from .session_native_fixture import ACTOR, fail_next_commit, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_accept_is_idempotent_without_touching_idle_or_version(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    first = await port.accept(root, "one", message_id="m1", actor=ACTOR)
    duplicate = await port.accept(root, "one", message_id="m1", actor=ACTOR)
    assert first.newly_accepted and not duplicate.newly_accepted
    assert duplicate.state == "accepted" and duplicate.notification == "not_needed"
    expired_duplicate = await port.accept(root, "one", message_id="m1", actor=ACTOR, deadline_ms=1)
    assert expired_duplicate.state == "accepted"
    with pytest.raises(RuntimeRequestError) as deadline_error:
        await port.accept(root, "new", message_id="expired", actor=ACTOR, deadline_ms=1)
    assert deadline_error.value.code == "deadline_expired"
    state = await store._db.fetchone(
        f"SELECT execution_state,inbox_version FROM {store.t.session_state} WHERE session_id=?",
        (root.session_id,),
    )
    assert state == {"execution_state": "queued", "inbox_version": 1}
    await port.cancel(first.input_ref, actor=ACTOR)
    idle_duplicate = await port.accept(root, "one", message_id="m1", actor=ACTOR)
    assert idle_duplicate.state == "voided"
    state = await store._db.fetchone(
        f"SELECT execution_state,inbox_version FROM {store.t.session_state} WHERE session_id=?",
        (root.session_id,),
    )
    assert state == {"execution_state": "idle", "inbox_version": 1}
    with pytest.raises(RuntimeRequestError) as error:
        await port.accept(root, "different", message_id="m1", actor=ACTOR)
    assert error.value.code == "request_conflict"


async def test_accept_unknown_commit_retries_original_receipt(registry_store, monkeypatch):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    fail_next_commit(monkeypatch, store)
    with pytest.raises(Exception) as error:
        await port.accept(root, "one", message_id="m1", actor=ACTOR)
    assert getattr(error.value, "code", None) == "store_commit_unknown"
    monkeypatch.undo()
    receipt = await port.accept(root, "one", message_id="m1", actor=ACTOR)
    assert not receipt.newly_accepted and receipt.input_ref.message_id == "m1"
    assert len(await store._db.fetchall(f"SELECT * FROM {store.t.follow_up_queue}")) == 1


async def test_capacity_rolls_back_the_whole_accept(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store, inbox_item_limit=1, inbox_byte_limit=100)
    root = await target(port)
    await port.accept(root, "x" * 40, message_id="m1", actor=ACTOR)
    with pytest.raises(CapacityError):
        await port.accept(root, "second", message_id="m2", actor=ACTOR)
    rows = await store._db.fetchall(f"SELECT item_id FROM {store.t.follow_up_queue}")
    assert [row["item_id"] for row in rows] == ["m1"]
    byte_port = RuntimeInputStore(store, inbox_item_limit=10, inbox_byte_limit=1)
    with pytest.raises(CapacityError):
        await byte_port.accept(root, "too large", message_id="m3", actor=ACTOR)


async def test_atomic_snapshot_has_one_merged_message_and_late_input_stays_pending(
    registry_store, monkeypatch,
):
    store, connect = registry_store
    other = await connect(schema="verify")
    port, other_port = RuntimeInputStore(store), RuntimeInputStore(other)
    root = await target(port)
    for message_id, body, priority in (
        ("normal", {"content": [{"type": "text", "text": "plain"}]}, "normal"),
        ("urgent", "stop and redirect", "urgent"),
    ):
        await port.accept(root, body, message_id=message_id, actor=ACTOR, priority=priority)
    inserted, release = asyncio.Event(), asyncio.Event()
    original = port._insert_message

    async def paused_insert(*args, **kwargs):
        inserted.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(port, "_insert_message", paused_insert)
    claiming = asyncio.create_task(port.claim(root, node_id="a", boot_id="boot-a"))
    await inserted.wait()
    late_task = asyncio.create_task(other_port.accept(root, "late", message_id="late", actor=ACTOR))
    await asyncio.sleep(0)
    assert not late_task.done()
    release.set()
    batch = await claiming
    await late_task
    assert [ref.message_id for ref in batch.input_refs] == ["urgent", "normal"]
    assert len(batch.input_refs) == 2
    messages = await port.messages(root, actor=ACTOR)
    assert len(messages.items) == 1 and messages.items[0].role == "user"
    body = json.loads(messages.items[0].content)
    assert body["version"] == 1 and [item["message_id"] for item in body["items"]] == [
        "urgent", "normal"
    ]
    assert body["items"][1]["content"]["content"][0]["text"] == "plain"
    rows = await store._db.fetchall(
        f"SELECT item_id,status,delivered_send_index,delivered_seq FROM {store.t.follow_up_queue} "
        "ORDER BY item_id"
    )
    assert [(row["item_id"], row["status"]) for row in rows] == [
        ("late", "pending"), ("normal", "delivered"), ("urgent", "delivered")
    ]
    assert all(
        row["delivered_send_index"] == batch.send_index and row["delivered_seq"] == batch.message_seq
        for row in rows if row["status"] == "delivered"
    )
    assert await other_port.claim(root, node_id="b", boot_id="boot-b") is None


async def test_final_only_completes_consumed_and_never_claims_at_final_boundary(registry_store):
    store, connect = registry_store
    port = RuntimeInputStore(store)
    other = RuntimeInputStore(await connect(schema="verify"))
    root = await target(port)
    for message_id in ("one", "two"):
        await port.accept(root, message_id, message_id=message_id, actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    original = port._final_tx
    entered, release = asyncio.Event(), asyncio.Event()

    async def paused_final(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    port._final_tx = paused_final
    finishing = asyncio.create_task(port.final(
        batch, consumed=(batch.input_refs[0],), final_text="first result"
    ))
    await entered.wait()
    accepting = asyncio.create_task(other.accept(root, "late", message_id="late", actor=ACTOR))
    await asyncio.sleep(0)
    assert not accepting.done()
    release.set()
    final = await finishing
    await accepting
    assert final.execution_state == "queued"
    consumed, unconsumed = batch.input_refs
    assert (await other.inspect(consumed, actor=ACTOR)).terminal_status == "completed"
    pending = await other.inspect(unconsumed, actor=ACTOR)
    assert pending.input_state == "pending" and pending.terminal_status is None
    late = await other.get_result(
        type(consumed)(other.store.store_id, root.session_id, "late"), actor=ACTOR
    )
    assert late is None
    next_batch = await other.claim(root, node_id="b", boot_id="b")
    assert {ref.message_id for ref in next_batch.input_refs} == {unconsumed.message_id, "late"}
    assert next_batch.send_index == batch.send_index + 1
    messages = await other.messages(root, actor=ACTOR)
    assert len([row for row in messages.items if row.role == "assistant"]) == 1
