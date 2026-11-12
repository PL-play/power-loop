"""Cold storage reads backing the unified session client (no main loop)."""
from __future__ import annotations

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import (
    InputRef,
    RuntimeDescriptor,
    RuntimeRequestError,
    result_message_id,
)

from .session_native_fixture import ACTOR, DESCRIPTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_pending_void_active_cancel_and_stale_input_cancel(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    pending = await port.accept(root, "pending", message_id="pending", actor=ACTOR)
    voided = await port.cancel(pending.input_ref, actor=ACTOR, reason="not needed")
    assert voided.state == "voided"
    assert (await port.inspect(root, actor=ACTOR)).execution_state == "idle"
    result = await port.get_result(pending.input_ref, actor=ACTOR)
    assert result.status == "cancelled" and result.send_index is None
    assert (await port.cancel(pending.input_ref, actor=ACTOR)).state == "already_finished"

    await port.accept(root, "first", message_id="first", actor=ACTOR)
    first = await port.claim(root, node_id="a", boot_id="a")
    requested = await port.cancel(first.input_refs[0], actor=ACTOR, reason="stop")
    assert requested.state == "cancel_requested"
    await port.final(first, final_text="must not be success")
    cancelled = await port.get_result(first.input_refs[0], actor=ACTOR)
    assert cancelled.status == "cancelled" and cancelled.error == "stop"

    await port.accept(root, "second", message_id="second", actor=ACTOR)
    second = await port.claim(root, node_id="b", boot_id="b")
    late = await port.cancel(first.input_refs[0], actor=ACTOR, reason="late")
    assert late.state == "already_finished"
    status = await port.inspect(second.input_refs[0], actor=ACTOR)
    assert not status.cancel_requested
    await port.final(second, final_text="second done")


async def test_messages_children_inspect_and_cold_result(registry_store):
    store, connect = registry_store
    port = RuntimeInputStore(store)
    root = await target(port, "root")
    child = await target(port, "child", parent=root)
    assert (await port.list_children(root, actor=ACTOR)).items == (child,)
    receipt = await port.accept(child, {"content": "hello"}, message_id="start", actor=ACTOR)
    batch = await port.claim(child, node_id="a", boot_id="a")
    await port.final(batch, final_text="saved", usage={"total_tokens": 3}, structured_output={"ok": True})
    target_status = await port.inspect(child, actor=ACTOR)
    assert target_status.execution_state == "idle" and target_status.send_index == 1
    assert target_status.input_state is None and not target_status.result_available
    input_status = await port.inspect(receipt.input_ref, actor=ACTOR)
    assert input_status.result_available and input_status.result_seq == 2
    page = await port.messages(child, actor=ACTOR, limit=1)
    assert len(page.items) == 1 and page.next_after_seq == 1
    tail = await port.messages(child, actor=ACTOR, after_seq=page.next_after_seq)
    assert [row.role for row in tail.items] == ["assistant"] and tail.next_after_seq is None

    reopened = RuntimeInputStore(await connect(schema="verify"))
    cold = await reopened.get_result(receipt.input_ref, actor=ACTOR)
    assert cold.final_text == "saved" and dict(cold.usage) == {"total_tokens": 3}
    assert dict(cold.structured_output) == {"ok": True}
    assert cold.send_index == 1 and cold.result_seq == 2


@pytest.mark.parametrize("failure", ("closed", "full"))
async def test_result_route_failure_is_persisted_without_fake_parent_input(
    registry_store, failure,
):
    store, connect = registry_store
    port = RuntimeInputStore(store, inbox_item_limit=1)
    parent = await target(port, "parent-" + failure)
    child_id = "child-" + failure
    start = InputRef(store.store_id, child_id, "start")
    child = await port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=parent,
        result_route={"target": parent, "input_ref": start, "result_kind": "agent_result"},
        options={"session_id": child_id},
    )
    await port.accept(child, "run", message_id="start", actor=ACTOR)
    if failure == "closed":
        await store._db.execute(
            f"UPDATE {store.t.sessions} SET lifecycle='closed' WHERE session_id=?",
            (parent.session_id,),
        )
    else:
        await port.accept(parent, "occupies capacity", message_id="parent-input", actor=ACTOR)
    batch = await port.claim(child, node_id="a", boot_id="a")
    final = await port.final(batch, final_text="child result")
    assert final.results[0].result_delivery_error == (
        "session_closed" if failure == "closed" else "capacity_exceeded"
    )
    cold_port = RuntimeInputStore(await connect(schema="verify"))
    cold = await cold_port.get_result(start, actor=ACTOR)
    assert cold.status == "completed" and cold.final_text == "child result"
    assert cold.result_delivery_error == final.results[0].result_delivery_error
    status = await cold_port.inspect(start, actor=ACTOR)
    assert status.result_delivery_error == cold.result_delivery_error
    messages = await cold_port.messages(child, actor=ACTOR)
    assert messages.items[-1].meta["result_delivery_error"] == cold.result_delivery_error
    parent_rows = await store._db.fetchall(
        f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?", (parent.session_id,)
    )
    expected = ["parent-input"] if failure == "full" else []
    assert [row["item_id"] for row in parent_rows] == expected


async def test_pending_child_cancel_routes_one_unstarted_result_atomically(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent = await target(port, "parent-cancel")
    start = InputRef(store.store_id, "child-cancel", "start")
    child = await port.create_session(
        DESCRIPTOR, actor=ACTOR, parent=parent,
        result_route={"target": parent, "input_ref": start, "result_kind": "agent_result"},
        options={"session_id": "child-cancel"},
    )
    await port.accept(child, "run", message_id="start", actor=ACTOR)
    receipt = await port.cancel(start, actor=ACTOR, reason="cancel before claim")
    assert receipt.state == "voided"
    result = await port.get_result(start, actor=ACTOR)
    assert result.status == "cancelled" and result.send_index is None
    result_ref = InputRef(
        store.store_id, parent.session_id,
        result_message_id(child, start, None, "agent_result"),
    )
    routed = await port.inspect(result_ref, actor=ACTOR)
    assert routed.input_state == "pending"
    await port.cancel(start, actor=ACTOR, reason="duplicate")
    rows = await store._db.fetchall(
        f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?", (parent.session_id,)
    )
    assert [row["item_id"] for row in rows] == [result_ref.message_id]


async def test_parent_actor_uses_target_lifecycle_and_child_result_route(registry_store):
    from power_loop.agent.runtime_types import TrustedActor

    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent = await target(port, "actor-parent")
    parent_actor = TrustedActor("tenant", "host", parent.session_id)
    start = InputRef(store.store_id, "actor-child", "start")
    child = await port.create_session(
        DESCRIPTOR, actor=parent_actor, parent=parent,
        result_route={"target": parent, "input_ref": start, "result_kind": "agent_result"},
        options={"session_id": "actor-child"},
    )
    await port.accept(child, "run", message_id="start", actor=parent_actor)
    await port.cancel(start, actor=parent_actor, reason="parent cancels child")
    result_ref = InputRef(
        store.store_id, parent.session_id,
        result_message_id(child, start, None, "agent_result"),
    )
    assert (await port.inspect(result_ref, actor=ACTOR)).input_state == "pending"

    await store._db.execute(
        f"UPDATE {store.t.sessions} SET lifecycle='closed' WHERE session_id=?",
        (child.session_id,),
    )
    with pytest.raises(RuntimeRequestError) as error:
        await port.accept(child, "closed", message_id="closed", actor=parent_actor)
    assert error.value.code == "session_closed"


async def test_result_address_and_descriptor_version_are_frozen(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    child = await target(port, "child")
    start = InputRef(store.store_id, child.session_id, "start")
    unstarted = result_message_id(child, start, None, "agent_result")
    assert len(unstarted) == 60
    assert unstarted == result_message_id(child, start, None, "agent_result")
    assert unstarted != result_message_id(child, start, 1, "agent_result")
    with pytest.raises(RuntimeRequestError) as error:
        RuntimeDescriptor(
            1, "agent", "test.agent@203", None, "test.tools@203", "test.hooks@203",
            "test.resources@203", {}, {},
        )
    assert error.value.code == "definition_unavailable"


async def test_read_scope_and_wrong_store_are_explicit(registry_store):
    from power_loop.agent.runtime_types import TargetRef, TrustedActor

    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    with pytest.raises(RuntimeRequestError) as error:
        await port.inspect(TargetRef("wrong", root.session_id), actor=ACTOR)
    assert error.value.code == "wrong_store"
    with pytest.raises(RuntimeRequestError) as error:
        await port.inspect(root, actor=TrustedActor("other", "host", None))
    assert error.value.code == "forbidden"
    with pytest.raises(RuntimeRequestError) as error:
        await port.inspect(root, actor=TrustedActor("tenant", "host", "unrelated"))
    assert error.value.code == "forbidden"
    with pytest.raises(RuntimeRequestError) as subtree_error:
        await port.cancel(root, actor=ACTOR, scope="subtree")
    assert subtree_error.value.code == "capability_unavailable"
    with pytest.raises(RuntimeRequestError) as input_subtree_error:
        await port.cancel(
            InputRef(store.store_id, root.session_id, "missing"), actor=ACTOR, scope="subtree"
        )
    assert input_subtree_error.value.code == "invalid_options"


async def test_void_and_close(registry_store):
    from power_loop.contracts.errors import SessionBusy

    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    receipt = await port.accept(root, "pending", message_id="pending", actor=ACTOR)
    with pytest.raises(SessionBusy):
        await port.close(root, actor=ACTOR)
    await port.cancel(receipt.input_ref, actor=ACTOR)
    await port.close(root, actor=ACTOR)
    with pytest.raises(RuntimeRequestError) as error:
        await port.accept(root, "closed", message_id="closed", actor=ACTOR)
    assert error.value.code == "session_closed"


async def test_inspect_during_cancel_uses_read_snapshot(registry_store, monkeypatch):
    import asyncio

    store, connect = registry_store
    writer = RuntimeInputStore(store)
    reader = RuntimeInputStore(await connect(schema="verify"))
    root = await target(writer)
    receipt = await writer.accept(root, "run", message_id="run", actor=ACTOR)
    await writer.claim(root, node_id="a", boot_id="a")
    entered, release = asyncio.Event(), asyncio.Event()
    original = writer._cancel_input_tx

    async def paused_cancel(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(writer, "_cancel_input_tx", paused_cancel)
    cancelling = asyncio.create_task(writer.cancel(receipt.input_ref, actor=ACTOR, reason="stop"))
    await entered.wait()
    status = await asyncio.wait_for(reader.inspect(receipt.input_ref, actor=ACTOR), 5)
    assert status.input_state == "delivered" and status.terminal_status is None
    release.set()
    assert (await cancelling).state == "cancel_requested"


async def test_message_rows_pagination_scope_export(registry_store):
    import json

    from power_loop.agent.runtime_types import TrustedActor

    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port, "history")
    other = await target(port, "other")
    receipt = await port.accept(
        root, {"content": "原请求"}, message_id="history-input", actor=ACTOR,
    )
    batch = await port.claim(root, node_id="a", boot_id="a")
    await port.final(batch, final_text="ORIGINAL-42", usage={"total_tokens": 4})
    await port.accept(other, "OTHER-PRIVATE", message_id="other", actor=ACTOR)
    first = await port.messages(root, actor=ACTOR, limit=1)
    assert first.items[0].id > 0 and first.items[0].seq == 1 and first.next_after_seq == 1
    tail = await port.messages(root, actor=ACTOR, after_seq=first.next_after_seq)
    assert [row.content for row in tail.items] == ["ORIGINAL-42"]
    with pytest.raises(RuntimeRequestError) as error:
        await port.messages(root, actor=TrustedActor("foreign", "host", None))
    assert error.value.code == "forbidden"

    exported = await store.export_session(root.session_id)
    assert exported["schema_version"] == 17
    assert json.loads(json.dumps(exported, ensure_ascii=False))["session_id"] == root.session_id
    assert [row["content"] for row in exported["tables"]["messages"]] == [
        first.items[0].content, "ORIGINAL-42"
    ]
    assert exported["tables"]["follow_up_queue"][0]["item_id"] == receipt.input_ref.message_id
    assert "OTHER-PRIVATE" not in json.dumps(exported, ensure_ascii=False)
