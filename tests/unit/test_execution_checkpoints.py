"""The real store plus memory reducer, including failure and growth contracts."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from power_loop.runtime.execution_journal import ExecutionJournal
from power_loop.runtime.execution_state import (
    CheckpointPolicy,
    ExecutionIdentity,
    ExecutionState,
)
from power_loop.runtime.jobs import JobError

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class CountedDatabase:
    def __init__(self, db):
        self.db = db
        self.dialect = db.dialect
        self.sql = self.transactions = 0
        self.statements = []
        self.read_rows = self.read_bytes = self.parameter_bytes = 0

    def statement(self, sql, params):
        self.sql += 1
        self.statements.append(sql)
        self.parameter_bytes += len(json.dumps(params, default=str, ensure_ascii=False).encode())

    def result(self, rows):
        self.read_rows += len(rows)
        self.read_bytes += len(json.dumps(rows, default=str, ensure_ascii=False).encode())

    def __getattr__(self, name):
        return getattr(self.db, name)

    @asynccontextmanager
    async def transaction(self):
        self.transactions += 1
        async with self.db.transaction() as tx:
            yield CountedReader(self, tx)

    async def fetchone(self, sql, params=()):
        self.statement(sql, params)
        result = await self.db.fetchone(sql, params)
        self.result([] if result is None else [result])
        return result

    async def fetchall(self, sql, params=()):
        self.statement(sql, params)
        result = await self.db.fetchall(sql, params)
        self.result(result)
        return result

    async def execute(self, sql, params=()):
        self.statement(sql, params)
        return await self.db.execute(sql, params)


class CountedReader:
    def __init__(self, counter, tx):
        self.counter, self.tx = counter, tx

    async def execute(self, sql, params=()):
        self.counter.statement(sql, params)
        return await self.tx.execute(sql, params)

    async def fetchone(self, sql, params=()):
        self.counter.statement(sql, params)
        result = await self.tx.fetchone(sql, params)
        self.counter.result([] if result is None else [result])
        return result

    async def fetchall(self, sql, params=()):
        self.counter.statement(sql, params)
        result = await self.tx.fetchall(sql, params)
        self.counter.result(result)
        return result


async def case(registry_store, *, actions=8):
    store, connect = registry_store
    identity = ExecutionIdentity(store.store_id, "control-job", 1, "executor", "boot", "tenant", "host", "session")
    journal = ExecutionJournal(store)
    journal.db = counted = CountedDatabase(store._db)
    journal.transaction = counted.transaction
    now = [1000]
    state = ExecutionState(identity, journal, now_ms=lambda: now[0],
                           policy=CheckpointPolicy(actions=actions))
    await journal.create(identity, state.summary())
    return store, connect, state, journal, counted, now


def complete(state, number, *, kind="model", tokens=7):
    key = "attempt-" + str(number)
    state.begin_attempt(key, kind, {"request_hash": str(number)})
    state.dispatch(key)
    state.observe(key, {"status": "returned", "usage": {"total_tokens": tokens}})


async def test_normal_control_steps_have_no_database_operations(registry_store):
    _, _, state, journal, counted, _ = await case(registry_store)
    baseline = counted.sql, counted.transactions
    for number in range(7):
        complete(state, number, kind="model" if number % 2 == 0 else "tool")
        assert not await state.checkpoint()
    assert (counted.sql, counted.transactions) == baseline
    await state.close()
    assert (counted.sql - baseline[0], counted.transactions - baseline[1]) == (1, 1)
    saved = await journal.inspect(state.identity)
    assert saved["summary"]["completed"] == 7
    assert saved["summary"]["closed"]


async def test_empty_inbox_has_no_write_transaction_and_later_input_is_claimed(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    original = store._db
    store._db = counted = CountedDatabase(original)
    try:
        for _ in range(5):
            assert await store.inbox_claim(sid) == []
        print("empty inbox claim transactions:", counted.transactions)
        await store.inbox_put(sid, [{"item_id": "later", "content": "new instruction"}])
        [claimed] = await store.inbox_claim(sid)
        assert claimed["item_id"] == "later"
        assert await store.inbox_claim(sid) == []
    finally:
        store._db = original


async def test_closing_transaction_and_concurrent_checkpoint_do_not_invert_locks(registry_store, monkeypatch):
    store, _, state, journal, _, _ = await case(registry_store)
    complete(state, 1)
    entering, release = asyncio.Event(), asyncio.Event()
    transaction = journal.transaction

    @asynccontextmanager
    async def delayed_transaction():
        entering.set()
        await release.wait()
        async with transaction() as tx:
            yield tx

    monkeypatch.setattr(journal, "transaction", delayed_transaction)
    flushing = None
    hooks = []
    try:
        async with store._db.transaction() as tx:
            flushing = asyncio.create_task(state.checkpoint(force=True))
            await asyncio.wait_for(entering.wait(), 2)
            await asyncio.wait_for(state.close(transaction=tx,
                defer=lambda commit, rollback: hooks.append(commit)), 2)
        for hook in hooks:
            hook()
    finally:
        release.set()
        if flushing is not None:
            await asyncio.wait_for(asyncio.gather(flushing, return_exceptions=True), 3)
    assert flushing.result() is False
    saved = await journal.inspect(state.identity)
    assert saved["summary"]["closed"] and saved["summary"]["completed"] == 1
    assert saved["batch_no"] == state.cursor == 1
    assert not state._dirty and state._pending is None


@pytest.mark.parametrize("n", [1, 7, 15, 31, 63])
async def test_database_growth_tracks_checkpoints_not_internal_calls(registry_store, n):
    _, _, state, _, counted, _ = await case(registry_store)
    for number in range(n):
        complete(state, number)
        await state.checkpoint()
    await state.close()
    batches = n // 8 + 1
    assert counted.transactions == 1 + batches
    assert counted.sql == 1 + batches


async def test_checkpoint_lost_ack_retries_exact_batch(registry_store):
    _, _, state, journal, _, _ = await case(registry_store)
    complete(state, 1)
    original = journal.transaction
    failed = False

    @asynccontextmanager
    async def lose_ack():
        nonlocal failed
        async with original() as tx:
            yield tx
        if not failed:
            failed = True
            raise OSError("commit acknowledged after connection loss")

    journal.transaction = lose_ack
    with pytest.raises(OSError):
        await state.checkpoint(force=True)
    assert (await journal.inspect(state.identity))["batch_no"] == 1
    with pytest.raises(JobError, match="checkpoint_pending"):
        state.begin_attempt("new", "model", {})
    with pytest.raises(JobError, match="execution_checkpoint_conflict"):
        await state.checkpoint(force=True)
    assert (await journal.inspect(state.identity))["batch_no"] == 1


async def test_late_observation_survives_stop_and_failed_checkpoint(registry_store):
    _, _, state, journal, _, _ = await case(registry_store)
    state.begin_attempt("original", "model", {})
    state.dispatch("original")
    original = journal.append

    async def fail(*args, **kwargs):
        raise OSError("offline")

    journal.append = fail
    with pytest.raises(OSError):
        await state.checkpoint(force=True)
    state.cancel_requested = True
    state.observe("original", {"status": "returned", "cost": "0.17"})
    with pytest.raises(JobError, match="cancel_requested"):
        state.begin_attempt("forbidden", "model", {})
    journal.append = original
    await state.close()
    saved = await journal.inspect(state.identity)
    assert saved["summary"]["completed"] == 1
    assert state.get("attempt", "original")["observation"]["cost"] == "0.17"
    assert saved["batch_no"] == 2






async def test_reopen_of_open_prefix_cannot_grant_replay_or_reset_quota(registry_store):
    _, connect, state, _, _, _ = await case(registry_store)
    complete(state, 1)
    await state.checkpoint(force=True)
    state.begin_attempt("lost-tail", "model", {})
    state.dispatch("lost-tail")
    other = ExecutionJournal(await connect())
    saved = await other.inspect(state.identity)
    assert not saved["summary"]["closed"]
    with pytest.raises(JobError, match="execution_checkpoint_conflict"):
        await other.create(state.identity, saved["summary"])


async def test_same_identity_concurrent_batch_conflict_is_atomic(registry_store):
    _, connect, state, first, _, _ = await case(registry_store)
    second = ExecutionJournal(await connect())
    rows = [[{"kind": "fact", "key": "one", "value": {"value": n}}] for n in (1, 2)]
    results = await asyncio.gather(
        first.append(state.identity, 1, rows[0], state.summary()),
        second.append(state.identity, 1, rows[1], state.summary()),
        return_exceptions=True,
    )
    assert sum(r is None for r in results) == 1
    assert sum(isinstance(r, JobError) for r in results) == 1


async def test_owner_and_store_bindings_cannot_be_replaced(registry_store):
    _, _, state, journal, _, _ = await case(registry_store)
    with pytest.raises(JobError, match="checkpoint_conflict"):
        await journal.create(replace(state.identity, boot_id="new-boot"), state.summary())
    with pytest.raises(JobError, match="checkpoint_conflict"):
        await journal.create(replace(state.identity, store_id="other-store"), state.summary())


async def test_local_lease_expiry_blocks_admission_but_not_original_observation(registry_store):
    _, _, state, _, _, now = await case(registry_store)
    state.begin_attempt("original", "model", {})
    state.dispatch("original")
    from power_loop import SessionLeaseLost
    from power_loop.runtime.session_ownership import SessionOwnership
    state.ownership = SessionOwnership(('store', 'session'), 'owner', 0, lambda: None)
    with pytest.raises(SessionLeaseLost):
        state.begin_attempt("expired", "tool", {})
    assert state.observe("original", {"status": "returned", "usage": {"total_tokens": 13}})
    await state.close()







async def test_real_model_attempt_api_advances_in_memory(registry_store):
    from power_loop.runtime import model_attempts
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.task_runtime import TaskRuntime

    store, _ = registry_store
    original_db = store._db
    store._db = counted = CountedDatabase(original_db)
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="tenant", principal_id="host", executor_id="executor", boot_id="boot")
    sid = await store.create_session()
    actor = runtime.actor(sid)
    root = await manager.register_job(actor, kind="send", session_id=sid, system_key="control-test",
                                      request={"send_index": 1, "model_attempt_protocol": 1})
    binding = await runtime.claim(actor, root)
    await model_attempts.enable(binding)
    counted.sql = counted.transactions = 0
    try:
        for n in range(4):
            attempt = await model_attempts.prepare(binding, nonce=str(n), round_index=n, retry_index=1,
                                                   request_hash="a" * 64, model="fixture", max_tokens=100)
            await model_attempts.dispatch(attempt)
            await model_attempts.observe(attempt, usage={"total_tokens": 10}, estimated=False)
        assert counted.sql == 0
        print("empty inbox claim transactions:", counted.transactions)
        await manager.close_model_execution(actor, binding.lease, evidence={"exit": "completed"})
        from power_loop.runtime.control_runtime import current
        control = current(binding)
        saved = await control.journal.inspect(control.identity)
        assert saved['summary']['models'] == 4 and saved['summary']['closed']
    finally:
        store._db = original_db
