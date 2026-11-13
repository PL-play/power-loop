"""Deterministic child creation, conflicts, cancellation races and uncertain receipts."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import RuntimeRequestError, TrustedActor, sha256_canonical
from power_loop.contracts.errors import CancellationRequested

from .session_native_fixture import ACTOR, DESCRIPTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def _parent_with_call(port, session_id="creation-parent"):
    parent = await target(port, session_id)
    seq = await port.store.append_message(
        parent.session_id,
        role="assistant",
        content="",
        tool_calls=[{
            "id": "call", "type": "function",
            "function": {"name": "spawn_agent", "arguments": "{}"},
        }],
    )
    ref = await port.store.message_ref(parent.session_id, seq)
    return parent, ref, TrustedActor("tenant", "host", parent.session_id)


async def test_deterministic_child_retry_and_creation_conflict(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent, call_ref, actor = await _parent_with_call(port)
    first = await port.create_child(
        DESCRIPTOR, parent=parent, actor=actor, call_ref=call_ref,
        tool_call_id="call", body="task",
    )
    second = await port.create_child(
        DESCRIPTOR, parent=parent, actor=actor, call_ref=call_ref,
        tool_call_id="call", body="task",
    )
    assert first[0] == second[0] and first[1] == second[1]
    assert first[2].newly_accepted is True and second[2].newly_accepted is False
    expected_session = "ses_" + sha256_canonical([
        "child", parent.store_id, parent.session_id, call_ref.to_dict(), "call",
    ])[:56]
    expected_start = "inp_" + sha256_canonical(["child_start", parent.store_id, expected_session])[:56]
    assert first[0].session_id == expected_session
    assert first[1].message_id == expected_start
    with pytest.raises(RuntimeRequestError) as conflict:
        await port.create_child(
            replace(DESCRIPTOR, config={"max_rounds": 5}), parent=parent, actor=actor,
            call_ref=call_ref, tool_call_id="call", body="task",
        )
    assert conflict.value.code == "creation_conflict"
    rows = await store._db.fetchall(
        f"SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=?",
        (parent.session_id,),
    )
    assert len(rows) == 1


async def test_creation_during_active_parent_cancel_does_not_add_child(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent, call_ref, actor = await _parent_with_call(port, "cancel-race-parent")
    receipt = await port.accept(parent, "run", message_id="parent-start", actor=ACTOR)
    await port.claim(parent, node_id="A", boot_id="A")
    await port.cancel(receipt.input_ref, actor=ACTOR, reason="stop before create")
    with pytest.raises(CancellationRequested):
        await port.create_child(
            DESCRIPTOR, parent=parent, actor=actor, call_ref=call_ref,
            tool_call_id="call", body="task",
        )
    assert (await port.list_children(parent, actor=ACTOR)).items == ()


async def test_committed_child_is_recovered_without_replaying_creation(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent, call_ref, actor = await _parent_with_call(port, "uncertain-parent")
    await port.accept(parent, "run", message_id="parent-start", actor=ACTOR)
    batch = await port.claim(parent, node_id="A", boot_id="A")
    assistant_seq = await store.append_message(
        parent.session_id,
        role="assistant",
        content="",
        tool_calls=[{
            "id": "call", "type": "function",
            "function": {"name": "spawn_agent", "arguments": "{}"},
        }],
        send_index=batch.send_index,
        round_index=0,
    )
    assistant_ref = await store.message_ref(parent.session_id, assistant_seq)
    child, start, _ = await port.create_child(
        DESCRIPTOR, parent=parent, actor=actor, call_ref=assistant_ref,
        tool_call_id="call", body="task",
    )
    await store.set_pending(parent.session_id, {
        "assistant_seq": assistant_seq,
        "round_index": 0,
        "tool_calls": [{
            "id": "call", "type": "function",
            "function": {"name": "spawn_agent", "arguments": "{}"},
        }],
    })
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?",
        (parent.session_id,),
    )
    recovered = await port.recover(parent, node_id="B", boot_id="B")
    assert recovered.results[0].status == "interrupted"
    tool_rows = await store._db.fetchall(
        f"SELECT content,meta_json,send_index FROM {store.t.messages} "
        "WHERE session_id=? AND role='tool' AND tool_call_id='call'",
        (parent.session_id,),
    )
    assert len(tool_rows) == 1
    payload = json.loads(tool_rows[0]["content"])
    assert payload["creation_result_uncertain"] is True
    assert payload["target_ref"]["session_id"] == child.session_id
    assert payload["input_ref"] == {"store_id": start.store_id, "session_id": start.session_id, "message_id": start.message_id}
    assert tool_rows[0]["send_index"] == batch.send_index
    children = await port.list_children(parent, actor=ACTOR)
    assert children.items == (child,)
