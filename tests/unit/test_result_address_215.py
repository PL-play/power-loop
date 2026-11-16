"""Canonical SDK result addresses and atomic unstarted cancellation."""
from __future__ import annotations

import json

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import (
    InputRef,
    Result,
    RuntimeRequestError,
    TrustedActor,
    result_message_id,
    sha256_canonical,
)

from .session_native_fixture import ACTOR, DESCRIPTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_result_message_id_is_canonical_bounded_and_kind_stable(registry_store):
    store, _ = registry_store
    child = await target(RuntimeInputStore(store), "address-child")
    start = InputRef(store.store_id, child.session_id, "start")
    kinds = (
        "agent_result", "workflow_result", "workflow_node_result",
        "background_result", "managed_effect_result",
    )
    for kind in kinds:
        unstarted = result_message_id(child, start, None, kind)
        started = result_message_id(child, start, 7, kind)
        assert unstarted == "res_" + sha256_canonical([
            store.store_id, child.session_id, f"input:{start.message_id}", kind,
        ])[:56]
        assert started == "res_" + sha256_canonical([
            store.store_id, child.session_id, "send:7", kind,
        ])[:56]
        assert len(unstarted) == len(started) == 60
        assert unstarted != started
    assert len({result_message_id(child, start, 7, kind) for kind in kinds}) == len(kinds)
    with pytest.raises(RuntimeRequestError):
        result_message_id(child, InputRef(store.store_id, "other", "start"), None, "agent_result")


async def _routed_child(port, name, *, parent=None):
    parent = parent or await target(port, name + "-parent")
    start = InputRef(port.store.store_id, name, "start")
    child = await port.create_session(
        DESCRIPTOR,
        actor=ACTOR,
        parent=parent,
        result_route={"target": parent, "input_ref": start, "result_kind": "agent_result"},
        options={"session_id": name},
    )
    return parent, child, start


async def test_unstarted_cancel_and_repeat_rebuild_missing_parent_result(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent, child, start = await _routed_child(port, "cancel-address")
    await port.accept(child, "run", message_id="start", actor=ACTOR)
    receipt = await port.cancel(start, actor=ACTOR, reason="before claim")
    assert receipt.state == "voided"
    result_id = result_message_id(child, start, None, "agent_result")
    parent_ref = InputRef(store.store_id, parent.session_id, result_id)
    assert (await port.inspect(parent_ref, actor=ACTOR)).input_state == "pending"
    await store._db.execute(
        f"DELETE FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (parent.session_id, result_id),
    )
    again = await port.cancel(start, actor=ACTOR, reason="duplicate")
    assert again.state == "already_finished"
    assert (await port.inspect(parent_ref, actor=ACTOR)).input_state == "pending"
    rows = await store._db.fetchall(
        f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?",
        (parent.session_id,),
    )
    assert [row["item_id"] for row in rows] == [result_id]


async def test_unstarted_cancel_records_parent_closed_without_rollback(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent, child, start = await _routed_child(port, "closed-address")
    await port.accept(child, "run", message_id="start", actor=ACTOR)
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET lifecycle='closed' WHERE session_id=?",
        (parent.session_id,),
    )
    await port.cancel(start, actor=ACTOR, reason="parent closed")
    result = await port.get_result(start, actor=ACTOR)
    assert result.status == "cancelled" and result.send_index is None
    assert result.result_delivery_error == "session_closed"
    assert (await port.inspect(start, actor=ACTOR)).result_delivery_error == "session_closed"


@pytest.mark.parametrize("delivery", ["open", "closed", "full"])
async def test_release_claim_matching_cancel_routes_result_once(registry_store, delivery):
    store, _ = registry_store
    port = RuntimeInputStore(store, inbox_item_limit=1)
    parent, child, start = await _routed_child(port, "release-cancel-" + delivery)
    await port.accept(child, "run", message_id="start", actor=ACTOR)
    if delivery == "closed":
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET lifecycle='closed' WHERE session_id=?",
            (parent.session_id,),
        )
    elif delivery == "full":
        await port.accept(parent, "occupies capacity", message_id="parent-input", actor=ACTOR)

    batch = await port.claim(child, node_id="A", boot_id="A")
    receipt = await port.cancel(start, actor=ACTOR, reason="stop build")
    assert receipt.state == "cancel_requested"
    await port.release_claim(batch)

    result = await port.get_result(start, actor=ACTOR)
    expected_error = {
        "open": None, "closed": "session_closed", "full": "capacity_exceeded",
    }[delivery]
    assert result.status == "cancelled" and result.send_index == 1
    assert result.error == "stop build" and result.error_code == "cancelled"
    assert result.result_delivery_error == expected_error
    child_row = await store._db.fetchone(
        f"SELECT status,terminal_status,error_detail,result_delivery_error "
        f"FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id='start'",
        (child.session_id,),
    )
    assert child_row == {
        "status": "delivered", "terminal_status": "cancelled",
        "error_detail": "stop build", "result_delivery_error": expected_error,
    }
    state = await store._db.fetchone(
        f"SELECT execution_state,active_send_index,cancel_send_index,owner_token "
        f"FROM {store.t.session_state} WHERE session_id=?",
        (child.session_id,),
    )
    assert state == {
        "execution_state": "idle", "active_send_index": None,
        "cancel_send_index": None, "owner_token": None,
    }

    result_id = result_message_id(child, start, 1, "agent_result")
    assert (await port.cancel(start, actor=ACTOR, reason="duplicate")).state == "already_finished"
    parent_rows = await store._db.fetchall(
        f"SELECT item_id,content_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND kind='agent_result'",
        (parent.session_id,),
    )
    if delivery == "open":
        assert len(parent_rows) == 1
        assert parent_rows[0]["item_id"] == result_id
        assert json.loads(parent_rows[0]["content_json"])["status"] == "cancelled"
    else:
        assert parent_rows == []


async def test_accept_result_uses_frozen_route_and_deduplicates(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent, child, start = await _routed_child(port, "trusted-address")
    await port.accept(child, "run", message_id="start", actor=ACTOR)
    batch = await port.claim(child, node_id="A", boot_id="A")
    assert batch.result_route == {
        "target": {"store_id": parent.store_id, "session_id": parent.session_id},
        "input_ref": {
            "store_id": start.store_id, "session_id": start.session_id,
            "message_id": start.message_id,
        },
        "result_kind": "agent_result",
    }
    result = Result(start, child, "completed", batch.send_index, 2, "done", {"total_tokens": 1})
    first = await port.accept_result(child, result, result_kind="agent_result")
    second = await port.accept_result(child, result, result_kind="agent_result")
    assert first[0] == second[0] and first[1] is None and second[1] is None
    changed = Result(start, child, "completed", batch.send_index, 2, "different")
    conflict = await port.accept_result(child, changed, result_kind="agent_result")
    assert conflict[0] == first[0] and conflict[1] == "request_conflict"
    rows = await store._db.fetchall(
        f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?",
        (parent.session_id,),
    )
    assert len(rows) == 1
    with pytest.raises(RuntimeRequestError):
        await port.accept(
            parent,
            {"forged": True},
            message_id=first[0].message_id,
            actor=TrustedActor("tenant", "host", "unrelated"),
            kind="agent_result",
            result_kind="agent_result",
        )
