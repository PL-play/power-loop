"""Deterministic child creation, conflicts, cancellation races and uncertain receipts."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import (
    CapacityError,
    InputRef,
    RuntimeRequestError,
    TargetRef,
    TrustedActor,
    canonical,
    sha256_canonical,
)
from power_loop.contracts.errors import CancellationRequested, SessionBusy

from .session_native_fixture import (
    ACTOR,
    DESCRIPTOR,
    UnknownCommit,
    fail_next_commit,
    target,
)
from .test_session_inbox_batch_215 import _await_database_lock_wait, _waiter_probe

ZERO_RETENTION = {
    "sessions": 0, "messages": 0, "inbox": 0, "runtime": 0, "projections": 0,
}

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
    state = await store._db.fetchone(
        f"SELECT execution_state,inbox_version,next_seq FROM {store.t.sessions} "
        "WHERE session_id=?", (first[0].session_id,),
    )
    # The first-create path initializes the known queued state directly instead of
    # inserting idle and updating it once more.
    assert state == {"execution_state": "queued", "inbox_version": 1, "next_seq": 1}
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


async def test_first_create_rejects_an_oversized_start_without_rows(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store, inbox_byte_limit=32)
    parent, call_ref, actor = await _parent_with_call(port, "oversize-parent")
    with pytest.raises(CapacityError):
        await port.create_child(
            DESCRIPTOR, parent=parent, actor=actor, call_ref=call_ref,
            tool_call_id="call", body="x" * 128,
        )
    assert (await port.list_children(parent, actor=ACTOR)).items == ()
    children = await store._db.fetchall(
        f"SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=?",
        (parent.session_id,),
    )
    assert children == []


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
        f"UPDATE {store.t.sessions} SET owner_lease_until_ms=1 WHERE session_id=?",
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


@asynccontextmanager
async def _gated_lock(store, *, locked):
    """Hold the first state lock acquired by ``store`` until ``release`` is set."""
    original = store._db.dialect.lock_state
    release = asyncio.Event()

    async def gated(tx, state_table, session_id):
        row = await original(tx, state_table, session_id)
        locked.set()
        await release.wait()
        return row

    store._db.dialect.lock_state = gated
    try:
        yield release
    finally:
        store._db.dialect.lock_state = original


def _workflow_node(port, name):
    return (
        TargetRef(port.store.store_id, name),
        InputRef(port.store.store_id, name, "node-start"),
    )


async def _create_entry(entry, port, parent, call_ref, actor, node, node_start):
    if entry == "child":
        return await port.create_child(
            DESCRIPTOR, parent=parent, actor=actor, call_ref=call_ref,
            tool_call_id="call", body="task",
        )
    return await port.create_workflow_session(
        DESCRIPTOR, parent=parent, actor=actor, target=node, start=node_start,
        body="task", result_kind="workflow_node_result",
    )


async def _children(store, parent):
    return await store._db.fetchall(
        f"SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=?",
        (parent.session_id,),
    )


async def _confirm_creation_lock_wait(store, probe_store, session_id, *, waiter_task):
    """Confirm the holder's state row lock before releasing it.

    The existing backend wait metadata is the first check. MySQL can sit in the
    optimizer's ``statistics`` state before it reports ``LOCK WAIT``; in that case a
    direct ``FOR UPDATE SKIP LOCKED`` probe from the third connection confirms the row
    is locked, and the still-running waiter task confirms the creator has not acquired
    it. This is a lock fact from the database, not a timeout-based guess.
    """
    try:
        await _await_database_lock_wait(store)
        return
    except TimeoutError:
        pass
    row = await probe_store._db.fetchone(
        f"SELECT * FROM {probe_store.t.sessions} WHERE session_id=? "
        "FOR UPDATE SKIP LOCKED", (session_id,))
    assert row is None, "holder did not hold the parent/target state lock"
    assert not waiter_task.done(), "waiter finished before the holder released the lock"


async def _warm_state_lock(store, session_id):
    """Run the same FOR UPDATE lookup once on the waiter connection and commit it.

    This loads the table's statistics/plan on that connection before the gated race;
    it changes no rows and is only test setup. SQLite's BEGIN IMMEDIATE serializes
    anyway, so the warm-up is skipped there.
    """
    if store._db.dialect.name == "sqlite":
        return
    async with store._db.transaction() as tx:
        await store._db.dialect.lock_state(tx, store.t.sessions, session_id)


@pytest.mark.parametrize("entry", ("child", "workflow_node"))
async def test_close_committed_while_create_waits_rejects_the_creation(
    registry_store, entry,
):
    """The parent lifecycle is read after the state lock: a close that won the lock
    rejects the waiting creation instead of leaving an open child under a closed parent."""
    store, connect = registry_store
    other = await connect(schema="verify")
    port, waiter = RuntimeInputStore(store), RuntimeInputStore(other)
    parent, call_ref, actor = await _parent_with_call(port, f"close-first-{entry}")
    node, node_start = _workflow_node(port, f"close-first-{entry}-node")
    # Warm the second pool so the waiter reaches the lock, not connection setup, before
    # the backend lock-wait metadata is checked.
    await waiter.inspect(parent, actor=ACTOR)
    await _warm_state_lock(other, parent.session_id)
    probe = await connect(schema="verify")
    locked, started, reached = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async with _gated_lock(store, locked=locked) as release:
        try:
            async with _waiter_probe(other, started=started, locked=reached):
                closing = asyncio.create_task(port.close(parent, actor=ACTOR))
                await asyncio.wait_for(locked.wait(), 5)
                creating = asyncio.create_task(
                    _create_entry(entry, waiter, parent, call_ref, actor, node, node_start))
                await asyncio.wait_for(started.wait(), 5)
                # PG/MySQL: the backend reports a real lock wait. SQLite: BEGIN IMMEDIATE
                # is the gate; no row-lock claim is made.
                await _confirm_creation_lock_wait(
                    store, probe, parent.session_id, waiter_task=creating)
                assert not reached.is_set()
                release.set()
                await asyncio.wait_for(closing, 5)
                with pytest.raises(RuntimeRequestError) as rejected:
                    await asyncio.wait_for(creating, 5)
        finally:
            release.set()

    assert rejected.value.code == "session_closed"
    lifecycle = await store._db.fetchone(
        f"SELECT lifecycle FROM {store.t.sessions} WHERE session_id=?",
        (parent.session_id,),
    )
    assert lifecycle == {"lifecycle": "closed"}
    assert await _children(store, parent) == []
    queue = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.follow_up_queue} WHERE session_id=?",
        (node.session_id if entry == "workflow_node" else "unused",),
    )
    # For the child entry the deterministic child id is not in the queue either.
    total_queue = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.follow_up_queue}")
    assert int(total_queue["n"]) == 0
    assert int(queue["n"]) == 0
    if entry == "workflow_node":
        journal = await store._db.fetchone(
            f"SELECT COUNT(*) AS n FROM {store.t.session_runtime_state} WHERE session_id=?",
            (node.session_id,),
        )
        assert int(journal["n"]) == 0


@pytest.mark.parametrize("entry", ("child", "workflow_node"))
async def test_linked_create_committed_while_close_waits_keeps_parent_busy(
    registry_store, entry,
):
    """A linked child created first makes the waiting close return SessionBusy; the
    parent stays open and the child/start remain."""
    store, connect = registry_store
    other = await connect(schema="verify")
    port, closer = RuntimeInputStore(store), RuntimeInputStore(other)
    parent, call_ref, actor = await _parent_with_call(port, f"create-first-{entry}")
    node, node_start = _workflow_node(port, f"create-first-{entry}-node")
    # Warm the second pool so the closer reaches the lock, not connection setup, before
    # the backend lock-wait metadata is checked.
    await closer.inspect(parent, actor=ACTOR)
    await _warm_state_lock(other, parent.session_id)
    probe = await connect(schema="verify")
    locked, started, reached = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async with _gated_lock(store, locked=locked) as release:
        try:
            async with _waiter_probe(other, started=started, locked=reached):
                creating = asyncio.create_task(
                    _create_entry(entry, port, parent, call_ref, actor, node, node_start))
                await asyncio.wait_for(locked.wait(), 5)
                closing = asyncio.create_task(closer.close(parent, actor=ACTOR))
                await asyncio.wait_for(started.wait(), 5)
                await _confirm_creation_lock_wait(
                    store, probe, parent.session_id, waiter_task=closing)
                assert not reached.is_set()
                release.set()
                target, start, receipt = await asyncio.wait_for(creating, 5)
                assert receipt.newly_accepted is True
                with pytest.raises(SessionBusy):
                    await asyncio.wait_for(closing, 5)
        finally:
            release.set()

    assert (await closer.inspect(parent, actor=ACTOR)).execution_state == "idle"
    parent_row = await store._db.fetchone(
        f"SELECT lifecycle FROM {store.t.sessions} WHERE session_id=?",
        (parent.session_id,),
    )
    assert parent_row == {"lifecycle": "open"}
    child_row = await store._db.fetchone(
        f"SELECT lifecycle FROM {store.t.sessions} WHERE session_id=?",
        (target.session_id,),
    )
    assert child_row == {"lifecycle": "open"}
    state = await store._db.fetchone(
        f"SELECT execution_state,inbox_version FROM {store.t.sessions} "
        "WHERE session_id=?", (target.session_id,),
    )
    assert state == {"execution_state": "queued", "inbox_version": 1}
    queue = await store._db.fetchone(
        f"SELECT status FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (target.session_id, start.message_id),
    )
    assert queue == {"status": "pending"}


async def test_closed_parent_rejects_retry_but_keeps_existing_independent_child(
    registry_store,
):
    """An independent child does not block close; afterwards a deterministic retry is
    rejected without deleting or changing the existing child and its start input."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent, call_ref, actor = await _parent_with_call(port, "independent-retry")
    child, start, first = await port.create_child(
        DESCRIPTOR, parent=parent, actor=actor, call_ref=call_ref,
        tool_call_id="call", body="task", cancel_propagation="independent",
    )
    assert first.newly_accepted is True
    await port.close(parent, actor=ACTOR)
    with pytest.raises(RuntimeRequestError) as rejected:
        await port.create_child(
            DESCRIPTOR, parent=parent, actor=actor, call_ref=call_ref,
            tool_call_id="call", body="task", cancel_propagation="independent",
        )
    assert rejected.value.code == "session_closed"
    lifecycle = await store._db.fetchone(
        f"SELECT lifecycle FROM {store.t.sessions} WHERE session_id=?",
        (child.session_id,),
    )
    assert lifecycle == {"lifecycle": "open"}
    state = await store._db.fetchone(
        f"SELECT execution_state FROM {store.t.sessions} WHERE session_id=?",
        (child.session_id,),
    )
    assert state == {"execution_state": "queued"}
    queue = await store._db.fetchone(
        f"SELECT status FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (child.session_id, start.message_id),
    )
    assert queue == {"status": "pending"}
    assert await port.get_result(start, actor=ACTOR) is None
    status = await port.inspect(child, actor=ACTOR)
    assert status.execution_state == "queued" and status.terminal_status is None
    assert len(await _children(store, parent)) == 1


@pytest.mark.parametrize("entry", ("child", "workflow_node"))
async def test_missing_or_unauthorized_parent_keeps_domain_errors(registry_store, entry):
    """Moving authorize after the lock still maps a missing parent to not_found and
    keeps an unauthorized actor on forbidden."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent, call_ref, _actor = await _parent_with_call(port, f"domain-{entry}")
    node, node_start = _workflow_node(port, f"domain-{entry}-node")
    missing = TargetRef(store.store_id, f"missing-{entry}-parent")
    missing_actor = TrustedActor("tenant", "host", missing.session_id)
    with pytest.raises(RuntimeRequestError) as absent:
        await _create_entry(
            entry, port, missing, call_ref, missing_actor, node, node_start)
    assert absent.value.code == "not_found"
    denied_actor = TrustedActor(
        "other", "host", parent.session_id if entry == "child" else None)
    with pytest.raises(RuntimeRequestError) as denied:
        await _create_entry(
            entry, port, parent, call_ref, denied_actor, node, node_start)
    assert denied.value.code == "forbidden"
    assert await _children(store, parent) == []


async def test_closed_parent_rejects_workflow_retry_but_keeps_existing_target(
    registry_store,
):
    """The workflow entry checks the parent lifecycle before the existing-target
    identity check; a closed parent does not disturb the independent target/start."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent = await target(port, "workflow-independent-parent")
    node, start = _workflow_node(port, "workflow-independent-node")
    first = await port.create_workflow_session(
        DESCRIPTOR, parent=parent, actor=ACTOR, target=node, start=start,
        body="task", result_kind="workflow_result", cancel_propagation="independent",
    )
    assert first[2].newly_accepted is True
    await port.close(parent, actor=ACTOR)
    with pytest.raises(RuntimeRequestError) as rejected:
        await port.create_workflow_session(
            DESCRIPTOR, parent=parent, actor=ACTOR, target=node, start=start,
            body="task", result_kind="workflow_result", cancel_propagation="independent",
        )
    assert rejected.value.code == "session_closed"
    lifecycle = await store._db.fetchone(
        f"SELECT lifecycle FROM {store.t.sessions} WHERE session_id=?",
        (node.session_id,),
    )
    assert lifecycle == {"lifecycle": "open"}
    state = await store._db.fetchone(
        f"SELECT execution_state,inbox_version FROM {store.t.sessions} "
        "WHERE session_id=?", (node.session_id,),
    )
    assert state == {"execution_state": "queued", "inbox_version": 1}
    queue = await store._db.fetchone(
        f"SELECT status FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (node.session_id, start.message_id),
    )
    assert queue == {"status": "pending"}
    assert (await port.inspect(node, actor=ACTOR)).execution_state == "queued"


@asynccontextmanager
async def _gated_existing_reads(stores, target_id, *, gate_reentry=False):
    """Gate the real pre-lock existing reads so both creators see the target absent.

    The gate wraps only the exact ``SELECT * FROM sessions WHERE session_id=?`` read;
    it does not change returned rows. With ``gate_reentry`` it also pauses the loser's
    first lock after the conditional INSERT affected zero rows, so a test can hold the
    child state from another connection and prove the loser does not take the parent.
    """
    sessions = stores[0].t.sessions
    state = {
        "rows": [], "both": asyncio.Event(), "lost_store": None,
        "reentry_lock": None, "reentry_attempted": asyncio.Event(),
        "allow_reentry": asyncio.Event(),
    }

    class Gate:
        def __init__(self, tx, store):
            self.tx, self.store = tx, store

        def __getattr__(self, name):
            return getattr(self.tx, name)

        async def fetchone(self, sql, params=()):
            row = await self.tx.fetchone(sql, params)
            if sql == f"SELECT * FROM {sessions} WHERE session_id=?" and tuple(params) == (target_id,):
                state["rows"].append(row is None)
                if len(state["rows"]) >= 2:
                    state["both"].set()
                await asyncio.wait_for(state["both"].wait(), 5)
            return row

        async def execute(self, sql, params=()):
            changed = await self.tx.execute(sql, params)
            if "WHERE NOT EXISTS" in sql and changed == 0:
                state["lost_store"] = self.store
            return changed

    originals = []
    try:
        for store in stores:
            original_tx = store._db.transaction
            original_lock = store._db.dialect.lock_state
            originals.append((store, original_tx, original_lock))

            @asynccontextmanager
            async def transaction(original_tx=original_tx, store=store):
                async with original_tx() as tx:
                    yield Gate(tx, store)

            store._db.transaction = transaction
            if gate_reentry:
                async def lock(tx, table, sid, original_lock=original_lock, store=store):
                    if state["lost_store"] is store and state["reentry_lock"] is None:
                        state["reentry_lock"] = sid
                        state["reentry_attempted"].set()
                        await state["allow_reentry"].wait()
                    return await original_lock(tx, table, sid)

                store._db.dialect.lock_state = lock
        yield state
    finally:
        state["allow_reentry"].set()
        for store, original_tx, original_lock in originals:
            store._db.transaction = original_tx
            store._db.dialect.lock_state = original_lock


def _race_target_id(entry, parent, call_ref, node):
    if entry == "child":
        return "ses_" + sha256_canonical([
            "child", parent.store_id, parent.session_id, call_ref.to_dict(), "call",
        ])[:56]
    return node.session_id


async def _race_setup(entry, ports):
    if entry == "child":
        parent, call_ref, actor = await _parent_with_call(ports[0], "race-parent")
        return parent, call_ref, actor, None, None, None, None
    parent = await target(ports[0], "race-parent")
    node, start = _workflow_node(ports[0], f"race-{entry}")
    kind = "workflow_node_result" if entry == "workflow_node" else "workflow_result"
    cancel = "linked" if entry == "workflow_node" else "independent"
    run_journal = (
        {"version": 1, "run_id": start.message_id, "status": "running"}
        if entry == "driver" else None
    )
    return parent, None, ACTOR, node, start, kind, (cancel, run_journal)


async def _race_create(entry, port, parent, call_ref, actor, node, start, kind,
                       cancel_run, descriptor, body):
    if entry == "child":
        return await port.create_child(
            descriptor, parent=parent, actor=actor, call_ref=call_ref,
            tool_call_id="call", body=body,
        )
    cancel, run_journal = cancel_run
    return await port.create_workflow_session(
        descriptor, parent=parent, actor=actor, target=node, start=start,
        body=body, result_kind=kind, cancel_propagation=cancel,
        run_journal=run_journal,
    )


@pytest.mark.parametrize("entry", ("child", "workflow_node", "driver"))
async def test_same_request_concurrent_creation_is_idempotent(registry_store, entry):
    store, connect = registry_store
    other = await connect(schema="verify")
    ports = [RuntimeInputStore(store), RuntimeInputStore(other)]
    parent, call_ref, actor, node, start, kind, cancel_run = await _race_setup(entry, ports)
    target_id = _race_target_id(entry, parent, call_ref, node)
    child_target = TargetRef(store.store_id, target_id)

    async def create(port):
        return await _race_create(
            entry, port, parent, call_ref, actor, node, start, kind, cancel_run,
            DESCRIPTOR, "same task",
        )

    if store._db.dialect.name == "sqlite":
        outcomes = await asyncio.wait_for(
            asyncio.gather(create(ports[0]), create(ports[1])), 10)
        gate_state = None
    else:
        async with _gated_existing_reads([store, other], target_id) as gate_state:
            outcomes = await asyncio.wait_for(
                asyncio.gather(create(ports[0]), create(ports[1])), 10)
    first, second = outcomes
    assert first[0] == second[0] == child_target
    assert first[1] == second[1]
    assert {first[2].newly_accepted, second[2].newly_accepted} == {True, False}
    if gate_state is not None:
        assert gate_state["rows"][:2] == [True, True]
        assert any(row is False for row in gate_state["rows"][2:])
    sessions = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.sessions} WHERE session_id=?", (target_id,))
    states = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.sessions} WHERE session_id=?", (target_id,))
    inputs = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.follow_up_queue} WHERE session_id=?", (target_id,))
    assert (int(sessions["n"]), int(states["n"]), int(inputs["n"])) == (1, 1, 1)
    if entry == "driver":
        journal = await store._db.fetchone(
            f"SELECT COUNT(*) AS n FROM {store.t.session_runtime_state} WHERE session_id=?",
            (target_id,),
        )
        assert int(journal["n"]) == 1


@pytest.mark.parametrize("entry", ("child", "workflow_node"))
@pytest.mark.parametrize("conflict", ("body", "descriptor"))
async def test_concurrent_creation_with_different_request_conflicts(
    registry_store, entry, conflict,
):
    store, connect = registry_store
    other = await connect(schema="verify")
    ports = [RuntimeInputStore(store), RuntimeInputStore(other)]
    parent, call_ref, actor, node, start, kind, cancel_run = await _race_setup(entry, ports)
    target_id = _race_target_id(entry, parent, call_ref, node)
    descriptors = (
        [DESCRIPTOR, replace(DESCRIPTOR, config={"max_rounds": 9})]
        if conflict == "descriptor" else [DESCRIPTOR, DESCRIPTOR]
    )
    bodies = ["task one", "task two"] if conflict == "body" else ["same task", "same task"]

    async def create(index, port):
        return await _race_create(
            entry, port, parent, call_ref, actor, node, start, kind, cancel_run,
            descriptors[index], bodies[index],
        )

    if store._db.dialect.name == "sqlite":
        outcomes = await asyncio.wait_for(
            asyncio.gather(create(0, ports[0]), create(1, ports[1]),
                           return_exceptions=True), 10)
    else:
        async with _gated_existing_reads([store, other], target_id) as gate_state:
            outcomes = await asyncio.wait_for(
                asyncio.gather(create(0, ports[0]), create(1, ports[1]),
                               return_exceptions=True), 10)
            assert gate_state["rows"][:2] == [True, True]
    successes = [i for i, value in enumerate(outcomes) if not isinstance(value, BaseException)]
    failures = [value for value in outcomes if isinstance(value, BaseException)]
    assert len(successes) == 1 and len(failures) == 1, outcomes
    winner = successes[0]
    failure = failures[0]
    assert isinstance(failure, RuntimeRequestError)
    assert failure.code == ("request_conflict" if conflict == "body" else "creation_conflict")
    session = await store._db.fetchone(
        f"SELECT descriptor_json FROM {store.t.sessions} WHERE session_id=?", (target_id,))
    assert session["descriptor_json"] == canonical(descriptors[winner])
    input_row = await store._db.fetchone(
        f"SELECT content_json FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id=?",
        (target_id, outcomes[winner][1].message_id))
    assert json.loads(input_row["content_json"]) == bodies[winner]
    sessions = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.sessions} WHERE session_id=?", (target_id,))
    states = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.sessions} WHERE session_id=?", (target_id,))
    inputs = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.follow_up_queue} WHERE session_id=?", (target_id,))
    assert (int(sessions["n"]), int(states["n"]), int(inputs["n"])) == (1, 1, 1)


async def test_create_commit_error_is_not_treated_as_a_concurrent_create(
    registry_store, monkeypatch,
):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    parent, call_ref, actor = await _parent_with_call(port, "commit-error-parent")
    fail_next_commit(monkeypatch, store)
    with pytest.raises(UnknownCommit):
        await port.create_child(
            DESCRIPTOR, parent=parent, actor=actor, call_ref=call_ref,
            tool_call_id="call", body="task",
        )
    sessions = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.sessions} WHERE parent_session_id=?",
        (parent.session_id,),
    )
    inputs = await store._db.fetchone(
        f"SELECT COUNT(*) AS n FROM {store.t.follow_up_queue} WHERE session_id IN "
        f"(SELECT session_id FROM {store.t.sessions} WHERE parent_session_id=?)",
        (parent.session_id,),
    )
    assert int(sessions["n"]) == 1
    assert int(inputs["n"]) == 1


@pytest.mark.parametrize("entry", ("child", "workflow_node"))
async def test_loser_reentry_does_not_hold_parent_while_child_is_locked(
    registry_store, entry,
):
    """With child_id < parent_id, the loser's re-entry must try the child first.

    The final holds the child state while the loser is released; the final can still
    lock the parent and complete. The recorded first re-entry lock and the backend
    LOCK WAIT metadata prove the order without relying on a timeout.
    """
    store, connect = registry_store
    if store._db.dialect.name == "sqlite":
        pytest.skip("SQLite serializes with BEGIN IMMEDIATE; the row-lock order is PG/MySQL")
    other = await connect(schema="verify")
    final_store = await connect(schema="verify")
    ports = [RuntimeInputStore(store), RuntimeInputStore(other)]
    final_port = RuntimeInputStore(final_store)
    parent_name = "zz-race-parent"
    if entry == "child":
        parent, call_ref, actor = await _parent_with_call(ports[0], parent_name)
        node = start = kind = None
        cancel_run = (None, None)
    else:
        parent = await target(ports[0], parent_name)
        call_ref, actor = None, ACTOR
        node, start = _workflow_node(ports[0], "aa-race-node")
        kind = "workflow_node_result"
        cancel_run = ("linked", None)
    target_id = _race_target_id(entry, parent, call_ref, node)
    child_target = TargetRef(store.store_id, target_id)
    tasks = []
    final_task = None
    release_final = asyncio.Event()
    original_final_lock = final_store._db.dialect.lock_state

    async def create(port):
        return await _race_create(
            entry, port, parent, call_ref, actor, node, start, kind, cancel_run,
            DESCRIPTOR, "same task",
        )

    try:
        async with _gated_existing_reads([store, other], target_id, gate_reentry=True) as gate_state:
            tasks = [asyncio.create_task(create(port)) for port in ports]
            await asyncio.wait_for(gate_state["reentry_attempted"].wait(), 10)
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED, timeout=10)
            winner_task = next(iter(done))
            winner = await winner_task
            assert winner[2].newly_accepted is True
            batch = await final_port.claim(child_target, node_id="final", boot_id="final")
            assert batch is not None and batch.target == child_target

            final_locked = asyncio.Event()
            final_state = {"first": None, "calls": 0}

            async def gated_final(tx, table, sid):
                row = await original_final_lock(tx, table, sid)
                final_state["calls"] += 1
                if final_state["calls"] == 1:
                    final_state["first"] = sid
                    final_locked.set()
                    await release_final.wait()
                return row

            final_store._db.dialect.lock_state = gated_final
            final_task = asyncio.create_task(
                final_port.final(batch, consumed=batch.input_refs, final_text="done"))
            await asyncio.wait_for(final_locked.wait(), 10)
            assert final_state["first"] == target_id
            gate_state["allow_reentry"].set()
            assert gate_state["reentry_lock"] == target_id
            loser_task = next(task for task in tasks if task is not winner_task)
            await _confirm_creation_lock_wait(
                gate_state["lost_store"], final_store, target_id, waiter_task=loser_task)
            release_final.set()
            final_batch = await asyncio.wait_for(final_task, 10)
            assert final_batch.execution_state == "idle"
            loser = await asyncio.wait_for(loser_task, 10)
            assert loser[0] == winner[0] and loser[1] == winner[1]
            assert loser[2].newly_accepted is False
    finally:
        release_final.set()
        final_store._db.dialect.lock_state = original_final_lock
        for task in tasks:
            if not task.done():
                task.cancel()
        if final_task is not None and not final_task.done():
            final_task.cancel()
        await asyncio.gather(*tasks, *([final_task] if final_task is not None else []),
                             return_exceptions=True)


@pytest.mark.parametrize("entry", ("child", "workflow_node"))
async def test_reentry_after_concurrent_creation_rejects_parent_closed_meanwhile(
    registry_store, entry,
):
    """The loser's re-entry checks the parent after its own sorted locks. If the parent
    closes while the loser waits, the creation is rejected and the winner's independent
    child/start remain untouched."""
    store, connect = registry_store
    if store._db.dialect.name == "sqlite":
        pytest.skip("SQLite has no zero-row conditional branch under BEGIN IMMEDIATE")
    other = await connect(schema="verify")
    third = await connect(schema="verify")
    ports = [RuntimeInputStore(store), RuntimeInputStore(other)]
    closer = RuntimeInputStore(third)
    parent_name = "zz-close-race-parent"
    if entry == "child":
        parent, call_ref, actor = await _parent_with_call(ports[0], parent_name)
        node = start = None
    else:
        parent = await target(ports[0], parent_name)
        call_ref, actor = None, ACTOR
        node, start = _workflow_node(ports[0], "aa-close-race-node")
    target_id = _race_target_id(entry, parent, call_ref, node)
    tasks = []

    async def create(port):
        if entry == "child":
            return await port.create_child(
                DESCRIPTOR, parent=parent, actor=actor, call_ref=call_ref,
                tool_call_id="call", body="same task",
                cancel_propagation="independent",
            )
        return await port.create_workflow_session(
            DESCRIPTOR, parent=parent, actor=actor, target=node, start=start,
            body="same task", result_kind="workflow_node_result",
            cancel_propagation="independent",
        )

    try:
        async with _gated_existing_reads([store, other], target_id, gate_reentry=True) as gate_state:
            tasks = [asyncio.create_task(create(port)) for port in ports]
            await asyncio.wait_for(gate_state["reentry_attempted"].wait(), 10)
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED, timeout=10)
            winner_task = next(iter(done))
            winner = await winner_task
            assert winner[2].newly_accepted is True
            await closer.close(parent, actor=ACTOR)
            gate_state["allow_reentry"].set()
            loser_task = next(task for task in tasks if task is not winner_task)
            with pytest.raises(RuntimeRequestError) as rejected:
                await asyncio.wait_for(loser_task, 10)
            assert rejected.value.code == "session_closed"
            child = winner[0].session_id
            lifecycle = await store._db.fetchone(
                f"SELECT lifecycle FROM {store.t.sessions} WHERE session_id=?",
                (child,),
            )
            assert lifecycle == {"lifecycle": "open"}
            state = await store._db.fetchone(
                f"SELECT execution_state,inbox_version FROM {store.t.sessions} "
                "WHERE session_id=?", (child,),
            )
            assert state == {"execution_state": "queued", "inbox_version": 1}
            start_row = await store._db.fetchone(
                f"SELECT status FROM {store.t.follow_up_queue} "
                "WHERE session_id=? AND item_id=?",
                (child, winner[1].message_id),
            )
            assert start_row == {"status": "pending"}
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("entry", ("child", "workflow_node"))
async def test_cleaned_target_before_reentry_lock_returns_creation_conflict(
    registry_store, entry,
):
    """The loser re-read the target, then public cancel/close/retention removed it
    before its target state lock. The lock failure must be the domain conflict, not a
    raw ValueError, and the target must not be recreated."""
    store, connect = registry_store
    if store._db.dialect.name == "sqlite":
        pytest.skip("BEGIN IMMEDIATE does not expose the re-entry read/lock window")
    other = await connect(schema="verify")
    cleaner = await connect(schema="verify")
    ports = [RuntimeInputStore(store), RuntimeInputStore(other)]
    cleaner_port = RuntimeInputStore(cleaner)
    parent, call_ref, actor, node, start, kind, cancel_run = await _race_setup(entry, ports)
    target_id = _race_target_id(entry, parent, call_ref, node)
    tasks = []

    async def create(port):
        return await _race_create(
            entry, port, parent, call_ref, actor, node, start, kind, cancel_run,
            DESCRIPTOR, "same task",
        )

    try:
        async with _gated_existing_reads([store, other], target_id, gate_reentry=True) as gate_state:
            tasks = [asyncio.create_task(create(port)) for port in ports]
            await asyncio.wait_for(gate_state["reentry_attempted"].wait(), 10)
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED, timeout=10)
            winner_task = next(iter(done))
            winner = await winner_task
            assert winner[2].newly_accepted is True
            # The second existing read saw the winner's row; the loser is paused
            # before its sorted state locks.
            assert gate_state["rows"][-1] is False
            assert gate_state["reentry_lock"] == min(parent.session_id, target_id)
            cancelled = await cleaner_port.cancel(winner[1], actor=ACTOR)
            assert cancelled.state == "voided"
            await cleaner_port.close(winner[0], actor=ACTOR)
            # Real current clock and zero retention; the sleep only lets the close's
            # updated_at be strictly before the purge cutoff.
            await asyncio.sleep(0.003)
            purged = await cleaner.purge_expired(
                time.time_ns() // 1_000_000, limit=20,
                retention_days=ZERO_RETENTION, session_id=target_id,
            )
            assert purged.get("sessions") == 1, purged
            assert purged.get("inbox") == 1, purged
            counts = {}
            for name, table in (
                ("session", store.t.sessions),
                ("input", store.t.follow_up_queue),
            ):
                row = await cleaner._db.fetchone(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE session_id=?", (target_id,))
                counts[name] = int(row["n"])
            assert counts == {"session": 0, "input": 0}
            gate_state["allow_reentry"].set()
            loser_task = next(task for task in tasks if task is not winner_task)
            with pytest.raises(RuntimeRequestError) as rejected:
                await asyncio.wait_for(loser_task, 10)
            assert rejected.value.code == "creation_conflict"
            parent_row = await cleaner._db.fetchone(
                f"SELECT lifecycle FROM {store.t.sessions} WHERE session_id=?",
                (parent.session_id,),
            )
            assert parent_row == {"lifecycle": "open"}
            row = await cleaner._db.fetchone(
                f"SELECT COUNT(*) AS n FROM {store.t.sessions} WHERE session_id=?",
                (target_id,),
            )
            assert int(row["n"]) == 0
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
