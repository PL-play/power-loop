"""P2 crash recovery: pending repair, no replay, owner fences, expired-run discovery."""
from __future__ import annotations

import json

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.contracts.errors import SessionLeaseLost

from .session_native_fixture import ACTOR, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def _assistant_with_pending(store, port, batch, tool_ids):
    """Simulate a crash right after the assistant tool-call append: the raw assistant
    row and pending_json are durable, but no tool result was written."""
    seq = batch.message_seq + 1
    tool_calls = [{"id": cid, "type": "function", "function": {"name": "echo", "arguments": "{}"}}
                  for cid in tool_ids]
    await store._db.execute(
        f"INSERT INTO {store.t.messages} (session_id,seq,role,content,tool_calls_json,round_index,"
        "state,meta_json,send_index,created_at) VALUES (?,?, 'assistant', '', ?, 0, 'active', ?, ?, 0)",
        (batch.target.session_id, seq, json.dumps(tool_calls), "{}", batch.send_index),
    )
    pending = {"assistant_seq": seq, "round_index": 0, "tool_call_ids": list(tool_ids),
               "tool_calls": tool_calls}
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET pending_json=?,next_seq=? WHERE session_id=?",
        (json.dumps(pending), seq + 1, batch.target.session_id),
    )
    return seq


async def _expire(store, session_id):
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET owner_lease_until_ms=1 WHERE session_id=?", (session_id,))


async def test_recover_repairs_missing_tools_without_replay(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    await _assistant_with_pending(store, port, batch, ["t1", "t2"])
    await _expire(store, root.session_id)

    recovered = await port.recover(root, node_id="b", boot_id="b")
    assert recovered is not None
    assert recovered.results[0].status == "interrupted"
    # Both missing tools got a fixed unknown result stamped with the INTERRUPTED send_index.
    tools = await store._db.fetchall(
        f"SELECT tool_call_id,content,send_index FROM {store.t.messages} "
        "WHERE session_id=? AND role='tool' ORDER BY seq", (root.session_id,))
    assert {t["tool_call_id"] for t in tools} == {"t1", "t2"}
    assert all(t["send_index"] == batch.send_index for t in tools)
    assert all("执行中断" in t["content"] for t in tools)
    state = await store.get_state(root.session_id)
    assert state.pending is None
    # The input is terminal interrupted; a cold wait sees it (no eternal wait).
    result = await port.get_result(batch.input_refs[0], actor=ACTOR)
    assert result.status == "interrupted"


async def test_existing_tool_result_is_not_duplicated(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    assistant_seq = await _assistant_with_pending(store, port, batch, ["t1", "t2"])
    # t1 already has a real result before the crash.
    await store._db.execute(
        f"INSERT INTO {store.t.messages} (session_id,seq,role,content,tool_call_id,round_index,"
        "state,meta_json,send_index,created_at) VALUES (?,?, 'tool', 'real result', 't1', 0, 'active', ?, ?, 0)",
        (root.session_id, assistant_seq + 1, "{}", batch.send_index),
    )
    await store._db.execute(
        f"UPDATE {store.t.session_state} SET next_seq=? WHERE session_id=?",
        (assistant_seq + 2, root.session_id),
    )
    await _expire(store, root.session_id)
    await port.recover(root, node_id="b", boot_id="b")
    tools = await store._db.fetchall(
        f"SELECT tool_call_id,content FROM {store.t.messages} WHERE session_id=? AND role='tool' "
        "ORDER BY seq", (root.session_id,))
    assert [t["tool_call_id"] for t in tools] == ["t1", "t2"]  # t1 not duplicated
    assert tools[0]["content"] == "real result"
    assert "执行中断" in tools[1]["content"]


async def test_claim_repairs_pending_before_new_user_message(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "first", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    await _assistant_with_pending(store, port, batch, ["t1"])
    # The interrupted run is finalized (pending left behind), session returns to idle.
    await port.final(batch, status="interrupted", error="crash")
    assert (await store.get_state(root.session_id)).pending is not None
    # A new send must repair the old pending BEFORE writing its merged user message.
    await port.accept(root, "second", message_id="m2", actor=ACTOR)
    second = await port.claim(root, node_id="b", boot_id="b")
    assert second.send_index == batch.send_index + 1
    tools = await store._db.fetchall(
        f"SELECT tool_call_id,send_index FROM {store.t.messages} WHERE session_id=? AND role='tool'",
        (root.session_id,))
    assert tools[0]["tool_call_id"] == "t1"
    assert tools[0]["send_index"] == batch.send_index  # repaired onto the OLD send
    assert (await store.get_state(root.session_id)).pending is None
    # The new merged user message carries the new send_index, after the repair.
    users = await store._db.fetchall(
        f"SELECT seq,send_index FROM {store.t.messages} WHERE session_id=? AND role='user' ORDER BY seq",
        (root.session_id,))
    assert users[-1]["send_index"] == second.send_index
    assert users[-1]["seq"] > tools[0]["send_index"]


async def test_running_expired_without_pending_input_terminates_wait(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    await _expire(store, root.session_id)
    # No pending inbox input, but the running lease expired: discovery must still find it.
    found = await port.discover_unfinished()
    assert root in found
    recovered = await port.recover(root, node_id="b", boot_id="b")
    assert recovered.results[0].status == "interrupted"
    state = await store._db.fetchone(
        f"SELECT execution_state,active_send_index FROM {store.t.session_state} WHERE session_id=?",
        (root.session_id,))
    assert state["execution_state"] == "idle" and state["active_send_index"] is None
    assert (await port.get_result(batch.input_refs[0], actor=ACTOR)).status == "interrupted"


async def test_stale_owner_final_is_rejected_after_takeover(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    await _expire(store, root.session_id)
    await port.recover(root, node_id="b", boot_id="b")  # new owner epoch
    with pytest.raises(SessionLeaseLost):
        await port.final(batch, final_text="stale result")
    # The stale owner's result message was never written.
    assistants = await store._db.fetchall(
        f"SELECT content FROM {store.t.messages} WHERE session_id=? AND role='assistant'",
        (root.session_id,))
    assert all(a["content"] != "stale result" for a in assistants)


async def test_final_ack_loss_recovers_without_duplicate_result(registry_store, monkeypatch):
    """design 215 §9.2: final commits but the ACK is lost — the retry reads the durable
    terminal state and returns the original result, never a second result message."""
    from .session_native_fixture import fail_next_commit
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    fail_next_commit(monkeypatch, store)
    recovered = await port.final(batch, consumed=batch.input_refs, final_text="the result",
                                 usage={"total_tokens": 3})
    assert recovered.results[0].status == "completed"
    assert recovered.results[0].final_text == "the result"
    assistants = await store._db.fetchall(
        f"SELECT content FROM {store.t.messages} WHERE session_id=? AND role='assistant'",
        (root.session_id,))
    assert [a["content"] for a in assistants] == ["the result"]  # exactly one
    assert (await port.get_result(batch.input_refs[0], actor=ACTOR)).final_text == "the result"


async def test_projection_write_failure_does_not_fail_send(registry_store, monkeypatch):
    """design 215 §8.2: the projection is derived/best-effort; a failed projection write
    never fails the send (raw messages remain the source of truth)."""
    from power_loop import AgentLoopConfig
    from power_loop.agent import stateful_loop as sl
    from tests.runtime_fixture import RunningAgent
    from .test_session_send_215 import EchoModel
    store, _ = registry_store

    async def broken_projection(self, *a, **k):
        raise RuntimeError("projection store unavailable")

    monkeypatch.setattr(sl._ExecutionLoop, "_write_session_projection", broken_projection)
    # Force projection mode so the end-of-send projection path is exercised.
    from power_loop.runtime.representation import ProjectedRepresentation
    agent = await RunningAgent(store, EchoModel(),
                               config=AgentLoopConfig(max_rounds=4, compactor=None,
                                                      representation=ProjectedRepresentation())).start()
    try:
        result = await agent.send("go")
        assert result.status == "completed" and result.final_text == "done"
        rows = await store.load_all_messages(agent.target.session_id)
        assert [r.role for r in rows] == ["user", "assistant"]  # raw intact
    finally:
        await agent.aclose()


async def test_recovering_with_followup_terminates_old_then_claims_new(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "old", message_id="old", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    await _expire(store, root.session_id)
    # A new input arrives while the old run is expired: accept turns it recovering.
    await port.accept(root, "new", message_id="new", actor=ACTOR)
    state = await store._db.fetchone(
        f"SELECT execution_state FROM {store.t.session_state} WHERE session_id=?", (root.session_id,))
    assert state["execution_state"] == "recovering"
    # Recovery terminates the old run, leaves the new input pending → queued.
    recovered = await port.recover(root, node_id="b", boot_id="b")
    assert recovered.results[0].status == "interrupted"
    assert recovered.execution_state == "queued"
    assert (await port.get_result(batch.input_refs[0], actor=ACTOR)).status == "interrupted"
    # The new input is claimed as a fresh send, not merged into the interrupted one.
    new_batch = await port.claim(root, node_id="b", boot_id="b")
    assert new_batch.send_index == batch.send_index + 1
    assert [r.message_id for r in new_batch.input_refs] == ["new"]
