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
    LocalAllocation,
)
from power_loop.runtime.jobs import JobError
from power_loop.runtime.store.task_schema import migrate_task_schema

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
    now = [1000]
    state = ExecutionState(identity, journal, lease_until_ms=31000, now_ms=lambda: now[0],
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
    assert (counted.sql - baseline[0], counted.transactions - baseline[1]) == (2, 1)
    saved = await journal.recover(state.identity)
    assert saved["summary"]["completed"] == 7
    assert not saved["needs_reconciliation"]
    assert len([key for key in saved["records"] if key[0] == "attempt"]) == 7


async def test_empty_inbox_has_no_write_transaction_and_later_input_is_claimed(registry_store):
    store, _ = registry_store
    sid = await store.create_session()
    original = store._db
    store._db = counted = CountedDatabase(original)
    try:
        for _ in range(5):
            assert await store.inbox_claim(sid) == []
        assert counted.transactions == 0
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
    transaction = journal.db.transaction

    @asynccontextmanager
    async def delayed_transaction():
        entering.set()
        await release.wait()
        async with transaction() as tx:
            yield tx

    monkeypatch.setattr(journal.db, "transaction", delayed_transaction)
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
    saved = await journal.recover(state.identity)
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
    assert counted.sql == 1 + 2 * batches


async def test_checkpoint_lost_ack_retries_exact_batch(registry_store):
    _, _, state, journal, _, _ = await case(registry_store)
    complete(state, 1)
    original = journal.db.transaction
    failed = False

    @asynccontextmanager
    async def lose_ack():
        nonlocal failed
        async with original() as tx:
            yield tx
        if not failed:
            failed = True
            raise OSError("commit acknowledged after connection loss")

    journal.db.transaction = lose_ack
    with pytest.raises(OSError):
        await state.checkpoint(force=True)
    assert len(await journal.db.fetchall(f"SELECT batch_no FROM {journal.batches}")) == 1
    with pytest.raises(JobError, match="checkpoint_pending"):
        state.begin_attempt("new", "model", {})
    await state.checkpoint(force=True)
    assert state.cursor == 1
    rows = await journal.db.fetchall(f"SELECT batch_no FROM {journal.batches}")
    assert len(rows) == 1


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
    saved = await journal.recover(state.identity)
    assert saved["records"][("attempt", "original")]["observation"]["cost"] == "0.17"
    assert ("attempt", "forbidden") not in saved["records"]
    assert saved["batch_no"] == 2


async def test_corrupt_derived_summary_rebuilds_from_originals(registry_store):
    store, _, state, journal, _, _ = await case(registry_store)
    complete(state, 1)
    await state.close()
    await store._db.execute(f"UPDATE {journal.heads} SET summary_json='corrupt',summary_hash='bad'")
    with pytest.raises(JobError, match="checkpoint_corrupt"):
        await journal.inspect(state.identity)
    repaired = await journal.recover(state.identity, repair=True)
    assert repaired["summary"]["completed"] == 1
    assert (await journal.inspect(state.identity))["summary"] == repaired["summary"]


@pytest.mark.parametrize("damage", ["payload", "tail"])
async def test_original_damage_never_becomes_empty_success(registry_store, damage):
    store, _, state, journal, _, _ = await case(registry_store)
    complete(state, 1)
    await state.close()
    if damage == "payload":
        await store._db.execute(f"UPDATE {journal.batches} SET payload_json='{{}}'")
    else:
        await store._db.execute(f"DELETE FROM {journal.batches}")
    with pytest.raises(JobError, match="journal_corrupt|journal_incomplete"):
        await journal.recover(state.identity, repair=True)


async def test_reopen_of_open_prefix_cannot_grant_replay_or_reset_quota(registry_store):
    _, connect, state, _, _, _ = await case(registry_store)
    complete(state, 1)
    await state.checkpoint(force=True)
    state.begin_attempt("lost-tail", "model", {})
    state.dispatch("lost-tail")
    other = ExecutionJournal(await connect())
    saved = await other.recover(state.identity)
    assert saved["needs_reconciliation"]
    assert ("attempt", "lost-tail") not in saved["records"]
    with pytest.raises(JobError, match="requires_reconciliation"):
        await other.create(state.identity, saved["summary"])


async def test_same_identity_concurrent_batch_conflict_is_atomic(registry_store):
    _, connect, state, first, _, _ = await case(registry_store)
    second = ExecutionJournal(await connect())
    rows = [[{"kind": "fact", "key": "one", "value": {"value": n}}] for n in (1, 2)]
    results = await asyncio.gather(
        first.append(state.identity, 1, state.chain_hash, rows[0], state.summary()),
        second.append(state.identity, 1, state.chain_hash, rows[1], state.summary()),
        return_exceptions=True,
    )
    assert sum(isinstance(r, str) for r in results) == 1
    assert sum(isinstance(r, JobError) for r in results) == 1


async def test_owner_and_store_bindings_cannot_be_replaced(registry_store):
    _, _, state, journal, _, _ = await case(registry_store)
    with pytest.raises(JobError, match="checkpoint_corrupt"):
        await journal.create(replace(state.identity, boot_id="new-boot"), state.summary())
    with pytest.raises(JobError, match="store_identity_mismatch"):
        await journal.create(replace(state.identity, store_id="other-store"), state.summary())


async def test_local_lease_expiry_blocks_admission_but_not_original_observation(registry_store):
    _, _, state, _, _, now = await case(registry_store)
    state.begin_attempt("original", "model", {})
    state.dispatch("original")
    now[0] = 32000
    with pytest.raises(JobError, match="lease_expired"):
        state.begin_attempt("expired", "tool", {})
    assert state.observe("original", {"status": "returned", "usage": {"total_tokens": 13}})
    await state.close()


async def test_allocation_never_refunds_unknown_or_invents_capacity():
    allowance = LocalAllocation("authority-grant", calls=2, tokens=100, max_inflight=2)
    allowance.reserve("unknown", 40)
    allowance.reserve("known", 50)
    allowance.observe("known", 12)
    assert allowance.held_tokens == 40
    assert allowance.used_calls == 2
    with pytest.raises(JobError, match="max_calls_exhausted"):
        allowance.reserve("extra", 1)
    allowance.observe("known", 12)
    assert allowance.used_tokens == 12
    with pytest.raises(JobError, match="usage_conflict"):
        allowance.observe("known", 13)


async def test_protocol21_requires_explicit_migration(registry_store):
    store, connect = registry_store
    await store._db.execute(f"UPDATE {store.table_prefix}task_store SET protocol_version=21 WHERE id=1")
    from power_loop.runtime.store.schema import StoreSchemaError
    with pytest.raises(StoreSchemaError):
        await connect()
    await migrate_task_schema(store._db, store.table_prefix, maintenance=True)
    reopened = await connect()
    assert reopened.store_id == store.store_id


async def test_empty_committed_prefix_is_not_permission_for_a_replacement_owner(registry_store):
    store, _connect, state, journal, _counted, _now = await case(registry_store)
    # The retained creator can retry its own lost admission ACK before dispatch.
    await journal.create(state.identity, state.summary())
    state.begin_attempt("unflushed", "model", {"request_hash": "private-original"})
    state.dispatch("unflushed")
    replacement = ExecutionJournal(store)
    with pytest.raises(JobError, match="requires_reconciliation"):
        await replacement.create(state.identity, (await journal.inspect(state.identity))["summary"])


async def test_recovery_retires_expired_owner_without_replaying_or_refunding(registry_store):
    from power_loop.runtime import model_attempts
    from power_loop.runtime.control_runtime import current
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.task_runtime import TaskRuntime
    store, _ = registry_store
    now = [1000]
    manager = JobManager(store, now_ms=lambda: now[0])
    runtime = TaskRuntime(manager, tenant_id="repair", principal_id="host", executor_id="executor", boot_id="boot", lease_ms=1000)
    sid = await store.create_session()
    actor = runtime.actor(sid)
    registered = await manager.register_job(actor, kind="send", session_id=sid, system_key="repair",
                                            request={"model_attempt_protocol": 1, "send_index": 1})
    binding = await runtime.claim(actor, registered)
    await model_attempts.enable(binding)
    attempt = await model_attempts.prepare(binding, nonce="unflushed", round_index=1, retry_index=1,
                                           request_hash="a" * 64, model="model", max_tokens=8)
    await model_attempts.dispatch(attempt)
    with pytest.raises(JobError, match="owner_still_live"):
        await runtime.repair_execution_checkpoint(sid, binding.lease.job_id, generation=1, retire_expired=True)
    now[0] += 2000
    result = await runtime.repair_execution_checkpoint(sid, binding.lease.job_id, generation=1, retire_expired=True)
    assert result["owner_retired"] and result["needs_reconciliation"] and result["summary"]["unknown_suffix"]
    assert not result["replay_allowed"] and not result["capacity_released"]
    assert (await manager.get_job(actor, binding.lease.job_id)).resource_state == "unknown"
    with pytest.raises(JobError):
        current(binding).require_admission()
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='model_attempt_finished'")


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
        assert counted.transactions == 0
        await manager.close_model_execution(actor, binding.lease, evidence={"exit": "completed"})
        rows = await manager.db.fetchall(
            f"SELECT event_id FROM {manager.events} WHERE event_type='model_attempt_finished' AND job_id=?",
            (binding.lease.job_id,))
        assert len(rows) == 4
    finally:
        store._db = original_db
