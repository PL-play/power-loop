"""Atomic session inbox acceptance and whole-snapshot batching."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from power_loop.agent.runtime_store import RuntimeInputStore
from power_loop.agent.runtime_types import CapacityError, RuntimeRequestError, StoreError
from power_loop.contracts.errors import SessionBusy

from .session_native_fixture import ACTOR, DESCRIPTOR, fail_next_commit, target

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@asynccontextmanager
async def _waiter_probe(store, *, started, locked):
    """Signal the waiter's transaction start and its sessions row-lock acquisition.

    ``started`` is set before the backend opens its transaction (SQLite's write lock is
    taken inside it); ``locked`` only after the first statement touching ``sessions``
    completes (or ``lock_state`` completes). A waiter blocked on the database therefore
    has ``started`` set and ``locked`` unset.
    """
    db = store._db
    original_transaction, original_lock = db.transaction, db.dialect.lock_state
    sessions_table = store.t.sessions

    def _is_locking_sessions(sql: str) -> bool:
        if sessions_table not in sql:
            return False
        upper = sql.lstrip().upper()
        if upper.startswith(("UPDATE", "INSERT", "DELETE")):
            return True
        return "FOR UPDATE" in upper

    class _ProbedTx:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def execute(self, sql, params=()):
            res = await self._inner.execute(sql, params)
            if _is_locking_sessions(sql):
                locked.set()
            return res

        async def fetchone(self, sql, params=()):
            res = await self._inner.fetchone(sql, params)
            if _is_locking_sessions(sql):
                locked.set()
            return res

        async def fetchall(self, sql, params=()):
            res = await self._inner.fetchall(sql, params)
            if _is_locking_sessions(sql):
                locked.set()
            return res

        async def execute_packed(self, sql, params=()):
            res = await self._inner.execute_packed(sql, params)
            if _is_locking_sessions(sql):
                locked.set()
            return res

    @asynccontextmanager
    async def transaction():
        started.set()
        async with original_transaction() as tx:
            yield _ProbedTx(tx)

    async def lock(tx, state_table, session_id):
        row = await original_lock(tx, state_table, session_id)
        locked.set()
        return row

    db.transaction = transaction
    db.dialect.lock_state = lock
    try:
        yield
    finally:
        db.transaction = original_transaction
        db.dialect.lock_state = original_lock


@asynccontextmanager
async def _hold_next_transaction(store, *, held, release):
    """Pause the next transaction right before commit while holding its DB locks."""
    db = store._db
    original_transaction = db.transaction
    armed = True

    @asynccontextmanager
    async def transaction():
        nonlocal armed
        async with original_transaction() as tx:
            yield tx
            if armed:
                armed = False
                held.set()
                await release.wait()

    db.transaction = transaction
    try:
        yield
    finally:
        release.set()
        db.transaction = original_transaction


async def _await_database_lock_wait(store):
    """Wait for the backend's own evidence that a statement is blocked on a lock."""
    backend = store._db.dialect.name
    if backend == "postgres":
        sql = (
            "SELECT COUNT(*) AS n FROM pg_stat_activity "
            "WHERE datname=current_database() AND wait_event_type='Lock' AND query LIKE ?"
        )
    elif backend == "mysql":
        sql = (
            "SELECT COUNT(*) AS n FROM information_schema.innodb_trx "
            "WHERE trx_state='LOCK WAIT' AND trx_query LIKE ?"
        )
    else:
        return
    param = f"%{store.table_prefix}sessions%"
    async with asyncio.timeout(5):
        while True:
            row = await store._db.fetchone(sql, (param,))
            if int(row["n"]):
                return
            await asyncio.sleep(.02)


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
        f"SELECT execution_state,inbox_version FROM {store.t.sessions} WHERE session_id=?",
        (root.session_id,),
    )
    assert state == {"execution_state": "queued", "inbox_version": 1}
    await port.cancel(first.input_ref, actor=ACTOR)
    idle_duplicate = await port.accept(root, "one", message_id="m1", actor=ACTOR)
    assert idle_duplicate.state == "accepted"
    assert not idle_duplicate.newly_accepted and idle_duplicate.notification == "not_needed"
    state = await store._db.fetchone(
        f"SELECT execution_state,inbox_version FROM {store.t.sessions} WHERE session_id=?",
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
    # A retry of the input that filled the inbox is still an idempotent hit, not a
    # capacity rejection.
    duplicate = await port.accept(root, "x" * 40, message_id="m1", actor=ACTOR)
    assert not duplicate.newly_accepted
    rows = await store._db.fetchall(f"SELECT item_id FROM {store.t.follow_up_queue}")
    assert [row["item_id"] for row in rows] == ["m1"]
    byte_port = RuntimeInputStore(store, inbox_item_limit=10, inbox_byte_limit=1)
    with pytest.raises(CapacityError):
        await byte_port.accept(root, "too large", message_id="m3", actor=ACTOR)


@pytest.mark.parametrize("boundary", ("items", "bytes"))
async def test_concurrent_accept_at_capacity_boundary_accepts_exactly_one(
    registry_store, boundary,
):
    store, connect = registry_store
    other = await connect(schema="verify")
    kwargs = ({"inbox_item_limit": 1} if boundary == "items"
              else {"inbox_item_limit": 10, "inbox_byte_limit": 100})
    port, other_port = RuntimeInputStore(store, **kwargs), RuntimeInputStore(other, **kwargs)
    root = await target(port)
    body = "x" * 40 if boundary == "bytes" else "entry"
    results = await asyncio.gather(
        port.accept(root, body, message_id="a", actor=ACTOR),
        other_port.accept(root, body, message_id="b", actor=ACTOR),
        return_exceptions=True,
    )
    accepted = [r for r in results if not isinstance(r, BaseException)]
    rejected = [r for r in results if isinstance(r, CapacityError)]
    assert len(accepted) == 1 and len(rejected) == 1, results
    rows = await store._db.fetchall(
        f"SELECT item_id FROM {store.t.follow_up_queue} ORDER BY item_id"
    )
    assert len(rows) == 1
    state = await store._db.fetchone(
        f"SELECT execution_state,inbox_version FROM {store.t.sessions} "
        "WHERE session_id=?", (root.session_id,),
    )
    assert state == {"execution_state": "queued", "inbox_version": 1}


async def test_accept_lock_wait_reads_the_committed_pending_count(registry_store):
    store, connect = registry_store
    other = await connect(schema="verify")
    port, waiter = RuntimeInputStore(store, inbox_item_limit=1), RuntimeInputStore(
        other, inbox_item_limit=1)
    root = await target(port)
    locked, release = asyncio.Event(), asyncio.Event()
    started, reached = asyncio.Event(), asyncio.Event()
    async with _hold_next_transaction(store, held=locked, release=release):
        async with _waiter_probe(other, started=started, locked=reached):
            first = asyncio.create_task(
                port.accept(root, "first", message_id="a", actor=ACTOR))
            await asyncio.wait_for(locked.wait(), 5)
            second = asyncio.create_task(
                waiter.accept(root, "second", message_id="b", actor=ACTOR))
            await asyncio.wait_for(started.wait(), 5)
            await _await_database_lock_wait(store)
            assert not reached.is_set()
            release.set()
            assert (await first).newly_accepted
            with pytest.raises(CapacityError):
                await second
    rows = await store._db.fetchall(
        f"SELECT item_id FROM {store.t.follow_up_queue} ORDER BY item_id")
    assert [row["item_id"] for row in rows] == ["a"]


@pytest.mark.parametrize("order", ("close-first", "accept-first"))
async def test_close_and_accept_serialize_on_the_state_lock(registry_store, order):
    store, connect = registry_store
    other = await connect(schema="verify")
    holder_store, waiter_store = (
        (store, other) if order == "close-first" else (other, store))
    holder, waiter = RuntimeInputStore(holder_store), RuntimeInputStore(waiter_store)
    root = await target(holder)
    held, release = asyncio.Event(), asyncio.Event()
    started, reached = asyncio.Event(), asyncio.Event()
    async with _hold_next_transaction(holder_store, held=held, release=release):
        async with _waiter_probe(waiter_store, started=started, locked=reached):
            if order == "close-first":
                first = asyncio.create_task(holder.close(root, actor=ACTOR))
            else:
                first = asyncio.create_task(
                    holder.accept(root, "race", message_id="race", actor=ACTOR))
            await asyncio.wait_for(held.wait(), 5)
            if order == "close-first":
                second = asyncio.create_task(
                    waiter.accept(root, "race", message_id="race", actor=ACTOR))
            else:
                second = asyncio.create_task(waiter.close(root, actor=ACTOR))
            await asyncio.wait_for(started.wait(), 5)
            await _await_database_lock_wait(holder_store)
            assert not reached.is_set()
            release.set()
            await asyncio.wait_for(first, 5)
            if order == "close-first":
                with pytest.raises(RuntimeRequestError) as error:
                    await asyncio.wait_for(second, 5)
                assert error.value.code == "session_closed"
            else:
                with pytest.raises(SessionBusy):
                    await asyncio.wait_for(second, 5)

    session = await store._db.fetchone(
        f"SELECT lifecycle FROM {store.t.sessions} WHERE session_id=?",
        (root.session_id,),
    )
    inputs = await store._db.fetchall(
        f"SELECT item_id FROM {store.t.follow_up_queue} WHERE session_id=?",
        (root.session_id,),
    )
    state = await store._db.fetchone(
        f"SELECT inbox_version FROM {store.t.sessions} WHERE session_id=?",
        (root.session_id,),
    )
    if order == "close-first":
        assert session == {"lifecycle": "closed"}
        assert inputs == []
        assert state == {"inbox_version": 0}
    else:
        assert session == {"lifecycle": "open"}
        assert [row["item_id"] for row in inputs] == ["race"]
        assert state == {"inbox_version": 1}


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

    late_started = asyncio.Event()
    transaction = other._db.transaction

    @asynccontextmanager
    async def signaled_transaction():
        late_started.set()
        async with transaction() as tx:
            yield tx

    other._db.transaction = signaled_transaction
    monkeypatch.setattr(port, "_insert_message", paused_insert)
    claiming = asyncio.create_task(port.claim(root, node_id="a", boot_id="boot-a"))
    await inserted.wait()
    late_task = asyncio.create_task(other_port.accept(root, "late", message_id="late", actor=ACTOR))
    await late_started.wait()
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


async def test_claim_returns_the_merged_user_row_and_identity(registry_store):
    """The claim hands the worker/loop the exact row it wrote (design 216 §6.1)."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    for message_id, body, priority in (
        ("normal", "plain", "normal"),
        ("urgent", "stop", "urgent"),
    ):
        await port.accept(root, body, message_id=message_id, actor=ACTOR, priority=priority)
    batch = await port.claim(
        root, node_id="a", boot_id="a", can_execute=lambda descriptor: True)
    assert batch.descriptor == DESCRIPTOR
    assert (batch.tenant_id, batch.principal_id, batch.parent_session_id) == (
        ACTOR.tenant_id, ACTOR.principal_id, None)
    row = batch.user_row
    assert (row.seq, row.send_index, row.role) == (batch.message_seq, batch.send_index, "user")
    assert [item["message_id"] for item in json.loads(row.content)["items"]] == [
        "urgent", "normal"]
    assert row.meta["input_refs"] == [
        {"store_id": store.store_id, "session_id": root.session_id, "message_id": message_id}
        for message_id in ("urgent", "normal")
    ]
    persisted = await store._db.fetchone(
        f"SELECT seq,content,meta_json,send_index FROM {store.t.messages} "
        "WHERE session_id=? AND role='user'", (root.session_id,))
    assert persisted["seq"] == row.seq and persisted["send_index"] == row.send_index
    assert persisted["content"] == row.content
    assert json.loads(persisted["meta_json"]) == row.meta


async def test_capability_is_decided_before_owner_in_a_concurrent_claim(registry_store):
    """An incapable node may hold the state lock but never installs the owner."""
    store, connect = registry_store
    other = await connect(schema="verify")
    port, capable_port = RuntimeInputStore(store), RuntimeInputStore(other)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    entered, release = asyncio.Event(), asyncio.Event()

    def gated_can_execute(descriptor):
        entered.set()
        return False

    async with _hold_next_transaction(store, held=entered, release=release):
        incapable = asyncio.create_task(port.claim(
            root, node_id="incapable", boot_id="i", can_execute=gated_can_execute))
        await asyncio.wait_for(entered.wait(), 5)
        # The capable node starts while the incapable one holds the state lock.
        capable = asyncio.create_task(capable_port.claim(
            root, node_id="capable", boot_id="c", can_execute=lambda descriptor: True))
        await asyncio.sleep(0)
        release.set()
        first, second = await asyncio.wait_for(asyncio.gather(incapable, capable), 5)
    assert first is None and second is not None
    assert second.owner_node_id == "capable"
    state = await store._db.fetchone(
        f"SELECT execution_state,send_index,owner_node_id,owner_epoch "
        f"FROM {store.t.sessions} WHERE session_id=?", (root.session_id,))
    assert state["execution_state"] == "running"
    assert state["send_index"] == second.send_index
    assert state["owner_node_id"] == "capable" and state["owner_epoch"] == second.owner_epoch
    messages = await port.messages(root, actor=ACTOR)
    assert len(messages.items) == 1 and messages.items[0].send_index == second.send_index


async def test_claim_rollback_does_not_report_a_batch(registry_store, monkeypatch):
    """An uncommitted claim is not mistaken for the committed-but-ACK-lost case."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "go", message_id="m1", actor=ACTOR)

    async def broken_insert(*args, **kwargs):
        raise RuntimeError("insert failed before commit")

    monkeypatch.setattr(port, "_insert_message", broken_insert)
    with pytest.raises(StoreError):
        await port.claim(root, node_id="a", boot_id="a")
    state = await store._db.fetchone(
        f"SELECT execution_state,owner_node_id FROM {store.t.sessions} "
        "WHERE session_id=?", (root.session_id,))
    assert state == {"execution_state": "queued", "owner_node_id": None}
    pending = await store._db.fetchone(
        f"SELECT status FROM {store.t.follow_up_queue} WHERE session_id=? AND item_id='m1'",
        (root.session_id,))
    assert pending == {"status": "pending"}
    assert (await port.messages(root, actor=ACTOR)).items == ()
    monkeypatch.undo()
    assert (await port.claim(root, node_id="b", boot_id="b")).send_index == 1


async def test_follow_up_claim_returns_seq_and_advances_the_live_cursor(registry_store):
    """A follow-up reuses the batch read and keeps the next append's cursor current."""
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port)
    await port.accept(root, "first", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    await port.accept(root, "second", message_id="m2", actor=ACTOR)
    async with store.send_read_scope(root.session_id) as holding:
        holding.next_seq = batch.message_seq + 1
        refs, seq, merged, version = await port.claim_follow_up(batch)
        assert holding.next_seq == seq + 1 and holding.change_snapshot is None
    assert [ref.message_id for ref in refs] == ["m2"]
    assert seq == batch.message_seq + 1
    assert version == 2
    assert merged["items"][0]["message_id"] == "m2"
    persisted = await store._db.fetchone(
        f"SELECT seq,content,send_index FROM {store.t.messages} "
        "WHERE session_id=? AND role='user' AND seq=?", (root.session_id, seq))
    assert persisted["seq"] == seq and persisted["send_index"] == batch.send_index
    assert json.loads(persisted["content"]) == merged


async def test_final_only_completes_consumed_and_never_claims_at_final_boundary(registry_store):
    store, connect = registry_store
    port = RuntimeInputStore(store)
    other = RuntimeInputStore(await connect(schema="verify"))
    root = await target(port)
    for message_id in ("one", "two"):
        await port.accept(root, message_id, message_id=message_id, actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    entered, release, accept_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    transaction = other.store._db.transaction

    @asynccontextmanager
    async def signaled_transaction():
        accept_started.set()
        async with transaction() as tx:
            yield tx

    other.store._db.transaction = signaled_transaction
    async with _hold_next_transaction(store, held=entered, release=release):
        finishing = asyncio.create_task(port.final(
            batch, consumed=(batch.input_refs[0],), final_text="first result"
        ))
        await asyncio.wait_for(entered.wait(), 5)
        accepting = asyncio.create_task(other.accept(root, "late", message_id="late", actor=ACTOR))
        await asyncio.wait_for(accept_started.wait(), 5)
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


async def test_final_merges_the_appended_assistant_meta_without_rereading_it(registry_store):
    """The live run's last assistant meta is passed to final, not read back from the row."""
    from power_loop.agent.sink import SQLiteSink

    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port, "result-meta")
    await port.accept(root, "go", message_id="m1", actor=ACTOR)
    batch = await port.claim(root, node_id="a", boot_id="a")
    sink = SQLiteSink(store, root.session_id)
    await sink.on_message_appended(
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "because",
            "meta": {"custom": "kept"},
            "_round_usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "send_index": batch.send_index,
        },
        round_index=0,
    )
    assert sink.last_assistant_seq is not None
    stored = await store._db.fetchone(
        f"SELECT meta_json FROM {store.t.messages} WHERE session_id=? AND seq=?",
        (root.session_id, sink.last_assistant_seq),
    )
    # The in-memory fact must be exactly the row meta (including reasoning and round usage).
    assert json.loads(stored["meta_json"]) == sink.last_assistant_meta
    await port.final(
        batch, consumed=batch.input_refs, final_text="answer",
        usage={"total_tokens": 5}, structured_output={"ok": True},
        result_seq=sink.last_assistant_seq, assistant_meta=sink.last_assistant_meta,
    )
    row = await store._db.fetchone(
        f"SELECT meta_json FROM {store.t.messages} WHERE session_id=? AND seq=?",
        (root.session_id, sink.last_assistant_seq),
    )
    meta = json.loads(row["meta_json"])
    assert meta["reasoning_content"] == "because"
    assert meta["custom"] == "kept"
    assert meta["_round_usage"]["total_tokens"] == 5
    assert meta["usage"] == {"total_tokens": 5}
    assert meta["structured_output"] == {"ok": True}
    assert meta["error"] is None and meta["error_code"] is None
    assert meta["result_delivery_error"] is None


def _stored_bytes(value):
    if value is None:
        return 0
    return len(value.encode() if isinstance(value, str) else bytes(value))


async def test_accept_read_aggregates_pending_and_waiting_once(registry_store):
    store, _ = registry_store
    port = RuntimeInputStore(store)
    root = await target(port, session_id="aggregate-root")
    other = await target(port, session_id="aggregate-other")
    await port.accept(root, "汉字正文", message_id="a", actor=ACTOR)
    await port.accept(root, "second body", message_id="b", actor=ACTOR)
    # A cancelled row is removed from the pending capacity counters.
    c_receipt = await port.accept(root, "finished", message_id="c", actor=ACTOR)
    await port.cancel(c_receipt.input_ref, actor=ACTOR)
    # Another session's pending input must not leak into this session's counters.
    await port.accept(other, "other session", message_id="other", actor=ACTOR)
    rows = await store._db.fetchall(
        f"SELECT content_json,meta_json FROM {store.t.follow_up_queue} "
        "WHERE session_id=? AND status='pending'",
        (root.session_id,),
    )
    expected_count = len(rows)
    expected_bytes = sum(_stored_bytes(r["content_json"]) + _stored_bytes(r["meta_json"])
                         for r in rows)
    current = await store._db.fetchone(
        f"SELECT pending_count,pending_bytes FROM {store.t.sessions} WHERE session_id=?",
        (root.session_id,),
    )
    assert int(current["pending_count"]) == expected_count == 2
    assert int(current["pending_bytes"]) == expected_bytes
    assert expected_bytes > len("汉字正文".encode())  # multibyte body counted in UTF-8 bytes
    with pytest.raises(RuntimeRequestError) as missing:
        await port.accept(
            type(root)(store.store_id, "missing-session"), "body", message_id="absent", actor=ACTOR,
        )
    assert missing.value.code == "not_found"
