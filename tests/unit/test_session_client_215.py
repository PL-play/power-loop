"""Cold storage reads backing the unified session client (no main loop)."""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import (
    InputRef,
    RuntimeDescriptor,
    RuntimeRequestError,
    TrustedActor,
    canonical,
    result_message_id,
)

from .session_native_fixture import ACTOR, DESCRIPTOR, target
from .test_send_sql_profile import Profiled

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
    subtree = await port.cancel(root, actor=ACTOR, scope="subtree")
    assert subtree.state == "already_finished"
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
    assert exported["schema_version"] == 18
    assert json.loads(json.dumps(exported, ensure_ascii=False))["session_id"] == root.session_id
    assert [row["content"] for row in exported["tables"]["messages"]] == [
        first.items[0].content, "ORIGINAL-42"
    ]
    assert exported["tables"]["follow_up_queue"][0]["item_id"] == receipt.input_ref.message_id
    assert "OTHER-PRIVATE" not in json.dumps(exported, ensure_ascii=False)


@asynccontextmanager
async def _profiled(store):
    """Measure only the calls made inside the context."""
    original = store._db
    profiled = Profiled(original)
    store._db = profiled
    try:
        yield profiled
    finally:
        store._db = original


async def _terminal_input(port, name, *, final_status="completed", **final):
    session = await target(port, name)
    receipt = await port.accept(session, "run", message_id="start", actor=ACTOR)
    batch = await port.claim(session, node_id="a", boot_id="a")
    await port.final(batch, status=final_status, **final)
    return session, receipt.input_ref


async def test_result_read_is_one_statement_for_root_states(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)

    pending = await target(port, "read-pending")
    pending_receipt = await port.accept(pending, "run", message_id="start", actor=ACTOR)
    async with _profiled(store) as measured:
        assert await port.get_result(pending_receipt.input_ref, actor=ACTOR) is None
    assert (len(measured.records), measured.transactions) == (1, 0)

    _session, completed_ref = await _terminal_input(
        port, "read-completed", final_text="done", usage={"total_tokens": 4},
        structured_output={"answer": 42},
    )
    async with _profiled(store) as measured:
        completed = await port.get_result(completed_ref, actor=ACTOR)
    assert (len(measured.records), measured.transactions) == (1, 0)
    assert completed.status == "completed" and completed.final_text == "done"
    assert dict(completed.usage) == {"total_tokens": 4}
    assert dict(completed.structured_output) == {"answer": 42}

    _session, failed_ref = await _terminal_input(
        port, "read-failed", final_status="failed", error="boom", error_code="failed",
    )
    async with _profiled(store) as measured:
        failed = await port.get_result(failed_ref, actor=ACTOR)
    assert (len(measured.records), measured.transactions) == (1, 0)
    assert failed.status == "failed" and failed.result_seq is None
    assert failed.final_text == "" and failed.error == "boom"

    cancelled_session = await target(port, "read-cancelled")
    cancelled_receipt = await port.accept(
        cancelled_session, "run", message_id="start", actor=ACTOR,
    )
    await port.cancel(cancelled_receipt.input_ref, actor=ACTOR, reason="stop")
    async with _profiled(store) as measured:
        cancelled = await port.get_result(cancelled_receipt.input_ref, actor=ACTOR)
    assert (len(measured.records), measured.transactions) == (1, 0)
    assert cancelled.status == "cancelled" and cancelled.result_seq is None

    missing_ref = InputRef(store.store_id, cancelled_session.session_id, "missing")
    async with _profiled(store) as measured:
        state = await port.read_result_state(missing_ref, actor=ACTOR)
    assert (len(measured.records), measured.transactions) == (1, 0)
    assert state.record is None and state.resolved is None and state.result is None
    assert state.session["tenant_id"] == "tenant"
    async with _profiled(store) as measured:
        with pytest.raises(RuntimeRequestError) as error:
            await port.get_result(missing_ref, actor=ACTOR)
    assert error.value.code == "not_found"
    assert (len(measured.records), measured.transactions) == (1, 0)


async def test_result_read_keeps_authorization_and_counts_special_paths(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent = await target(port, "read-parent")
    child = await target(port, "read-child", parent=parent)
    receipt = await port.accept(child, "run", message_id="start", actor=ACTOR)
    batch = await port.claim(child, node_id="a", boot_id="a")
    await port.final(batch, final_text="child result")

    async with _profiled(store) as measured:
        own = await port.get_result(
            receipt.input_ref, actor=TrustedActor("tenant", "host", child.session_id),
        )
    assert (len(measured.records), measured.transactions) == (1, 0)
    assert own.final_text == "child result"

    async with _profiled(store) as measured:
        ancestor = await port.get_result(
            receipt.input_ref, actor=TrustedActor("tenant", "host", parent.session_id),
        )
    assert (len(measured.records), measured.transactions) == (1, 1)
    assert ancestor.final_text == "child result"

    for actor in (
        TrustedActor("other", "host", None),
        TrustedActor("tenant", "other", None),
    ):
        async with _profiled(store) as measured:
            with pytest.raises(RuntimeRequestError) as error:
                await port.get_result(receipt.input_ref, actor=actor)
        assert error.value.code == "forbidden"
        assert (len(measured.records), measured.transactions) == (1, 0)

    unrelated = TrustedActor("tenant", "host", "unrelated")
    async with _profiled(store) as measured:
        with pytest.raises(RuntimeRequestError) as error:
            await port.get_result(receipt.input_ref, actor=unrelated)
    assert error.value.code == "forbidden"
    assert (len(measured.records), measured.transactions) == (2, 1)

    async with _profiled(store) as measured:
        with pytest.raises(RuntimeRequestError) as error:
            await port.get_result(
                InputRef("wrong", child.session_id, "start"), actor=ACTOR,
            )
    assert error.value.code == "wrong_store"
    assert (len(measured.records), measured.transactions) == (0, 0)


async def test_result_read_follows_continuation_and_reports_extra_reads(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    session = await target(port, "read-continuation")
    receipt = await port.accept(session, "run", message_id="start", actor=ACTOR)
    batch = await port.claim(session, node_id="a", boot_id="a")
    await store._db.execute(
        f"UPDATE {store.t.follow_up_queue} SET status='waiting_continuation' "
        "WHERE session_id=? AND item_id=?",
        (session.session_id, "start"),
    )
    await store._db.execute(
        f"INSERT INTO {store.t.follow_up_queue} (session_id,item_id,inbox_version,kind,priority,wake,status,"
        "content_json,content_bytes,meta_json,request_hash,created_at,continuation_of,continuation_index) "
        "VALUES (?,?,2,'workflow_continue','normal',1,'pending',?,24,?,?,?,?,?)",
        (session.session_id, "wfc-1", canonical({"workflow_continue": {}}), canonical({}),
         "0" * 64, 1, "start", 1),
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET wake_pending_count=wake_pending_count+1,"
        "pending_count=pending_count+1 WHERE session_id=?",
        (session.session_id,),
    )
    await port.final(batch)
    continuation = await port.claim(session, node_id="b", boot_id="b")
    await port.final(continuation, final_text="continued", usage={"total_tokens": 7})

    async with _profiled(store) as measured:
        state = await port.read_result_state(receipt.input_ref, actor=ACTOR)
    assert (len(measured.records), measured.transactions) == (5, 1)
    assert state.record.ref == receipt.input_ref
    assert "content_json" not in state.record.row and "meta_json" not in state.record.row
    assert state.resolved.ref.message_id == "wfc-1"
    assert state.result.status == "completed" and state.result.final_text == "continued"
    assert state.result.send_index == 2
    assert state.result.input_ref == receipt.input_ref

    async with _profiled(store) as measured:
        result = await port.get_result(receipt.input_ref, actor=ACTOR)
    assert (len(measured.records), measured.transactions) == (5, 1)
    assert result.final_text == "continued"


async def test_result_read_follows_multihop_continuation(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    session = await target(port, "read-multihop")
    receipt = await port.accept(session, "run", message_id="start", actor=ACTOR)
    batch = await port.claim(session, node_id="a", boot_id="a")
    await store._db.execute(
        f"UPDATE {store.t.follow_up_queue} SET status='waiting_continuation',"
        "delivered_at=1,delivered_seq=1,delivered_send_index=1 "
        "WHERE session_id=? AND item_id='start'",
        (session.session_id,),
    )
    await store._db.execute(
        f"INSERT INTO {store.t.follow_up_queue} (session_id,item_id,inbox_version,kind,priority,wake,status,"
        "content_json,content_bytes,meta_json,request_hash,created_at,delivered_at,delivered_seq,"
        "delivered_send_index,continuation_of,continuation_index) "
        "VALUES (?,?,2,'workflow_continue','normal',1,'waiting_continuation',?,24,?,?,?,?,?,?,?,?)",
        (session.session_id, "wfc-1", canonical({"workflow_continue": {}}), canonical({}),
         "0" * 64, 1, 1, 1, 2, "start", 1),
    )
    await store._db.execute(
        f"INSERT INTO {store.t.follow_up_queue} (session_id,item_id,inbox_version,kind,priority,wake,status,"
        "content_json,content_bytes,meta_json,request_hash,created_at,continuation_of,continuation_index) "
        "VALUES (?,?,3,'workflow_continue','normal',1,'pending',?,24,?,?,?,?,?)",
        (session.session_id, "wfc-2", canonical({"workflow_continue": {}}), canonical({}),
         "0" * 64, 1, "wfc-1", 2),
    )
    await store._db.execute(
        f"UPDATE {store.t.sessions} SET wake_pending_count=wake_pending_count+1,"
        "pending_count=pending_count+1 WHERE session_id=?",
        (session.session_id,),
    )
    await port.final(batch)
    continuation = await port.claim(session, node_id="b", boot_id="b")
    await port.final(continuation, final_text="multihop done")

    async with _profiled(store) as measured:
        state = await port.read_result_state(receipt.input_ref, actor=ACTOR)
    assert (len(measured.records), measured.transactions) == (6, 1)
    assert state.record.ref == receipt.input_ref
    assert state.resolved.ref.message_id == "wfc-2"
    assert state.result.status == "completed" and state.result.final_text == "multihop done"
    assert state.result.send_index == 2
    assert state.result.input_ref == receipt.input_ref


async def test_result_read_recovers_cancelled_usage_without_exposing_meta_json(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    session = await target(port, "read-cancelled-usage")
    receipt = await port.accept(session, "run", message_id="start", actor=ACTOR)
    batch = await port.claim(session, node_id="a", boot_id="a")
    await port.final(
        batch,
        status="cancelled",
        usage={"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
    )

    state = await port.read_result_state(receipt.input_ref, actor=ACTOR)
    assert "content_json" not in state.record.row and "meta_json" not in state.record.row
    assert "content_json" not in state.resolved.row and "meta_json" not in state.resolved.row
    assert state.record.row["result_seq"] is None
    assert state.result is not None
    assert state.result.status == "cancelled"
    assert state.result.usage == {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16}
