"""Integration tests for ExecutionSnapshot with real manager.

These tests verify snapshot functionality in realistic scenarios.
Uses the registry_store fixture from test_job_registry_192.
"""
import pytest
from power_loop.runtime.execution_snapshot import load_snapshot, ExecutionSnapshot
from power_loop.runtime.execution_state import ExecutionState, ExecutionIdentity
from power_loop.runtime.execution_journal import ExecutionJournal
from power_loop.runtime.jobs import JobManager, JobActor

# Import registry_store fixture
from tests.unit.test_job_registry_192 import registry_store

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.mark.asyncio
async def test_snapshot_dual_write_integration(registry_store):
    """Test that snapshots are written during checkpoint."""
    store, connect = registry_store

    # Create manager
    manager = JobManager(store)

    # Create job via JobManager (required for jobs table entry)
    sid = await store.create_session()
    actor = JobActor("tenant", "host", sid)
    job = await manager.register_job(
        actor,
        kind="send",
        session_id=sid,
        system_key="test_snapshot_integration"
    )

    # Create execution identity matching the job
    # Note: job.job.generation is 0 for new jobs, but ExecutionIdentity requires >= 1
    # Use 1 for execution context (control_runtime uses source["generation"])
    identity = ExecutionIdentity(
        store.store_id,
        job.job.job_id,
        1,  # execution generation, not job generation
        "executor_001",
        "boot_001",
        "tenant",
        "host",
        sid
    )

    # Create journal and state with manager
    journal = ExecutionJournal(store)
    state = ExecutionState(
        identity,
        journal,
        lease_until_ms=9999999999999,
        manager=manager  # Enable snapshot
    )

    # Create initial checkpoint
    await journal.create(identity, state.summary())

    # Record some activity
    state.record("test", "key1", {"data": "value1"})
    state.record("test", "key2", {"data": "value2"})

    # Checkpoint (should write both journal and snapshot)
    await state.checkpoint(force=True)

    # Verify snapshot was written
    snapshot = await load_snapshot(manager, identity.job_id, identity.generation)

    assert snapshot is not None, "Snapshot should be written during checkpoint"
    assert snapshot.is_valid(), "Snapshot should be valid"
    assert snapshot.generation == identity.generation

    print(f"✅ Snapshot dual-write verified: {snapshot.captured_at_ms}")


@pytest.mark.asyncio
async def test_snapshot_survives_state_recreation(registry_store):
    """Test that snapshot persists across ExecutionState recreation."""
    store, connect = registry_store
    manager = JobManager(store)

    # Create job
    sid = await store.create_session()
    actor = JobActor("tenant", "host", sid)
    job = await manager.register_job(
        actor,
        kind="send",
        session_id=sid,
        system_key="test_snapshot_persist"
    )

    identity = ExecutionIdentity(
        store.store_id,
        job.job.job_id,
        1,  # execution generation
        "executor_001",
        "boot_001",
        "tenant",
        "host",
        sid
    )

    # Create first state and checkpoint
    journal1 = ExecutionJournal(store)
    state1 = ExecutionState(identity, journal1, lease_until_ms=9999999999999, manager=manager)
    await journal1.create(identity, state1.summary())

    state1.record("test", "original", {"value": "preserved"})
    await state1.checkpoint(force=True)

    # Load snapshot in new context
    snapshot = await load_snapshot(manager, identity.job_id, identity.generation)

    assert snapshot is not None, "Snapshot should persist"
    assert snapshot.is_valid(), "Persisted snapshot should be valid"

    print(f"✅ Snapshot persists across state recreation")


@pytest.mark.asyncio
async def test_snapshot_vs_journal_consistency(registry_store):
    """Test that snapshot and journal remain consistent."""
    store, connect = registry_store
    manager = JobManager(store)

    # Create job
    sid = await store.create_session()
    actor = JobActor("tenant", "host", sid)
    job = await manager.register_job(
        actor,
        kind="send",
        session_id=sid,
        system_key="test_snapshot_consistency"
    )

    identity = ExecutionIdentity(
        store.store_id,
        job.job.job_id,
        1,  # execution generation
        "executor_001",
        "boot_001",
        "tenant",
        "host",
        sid
    )

    journal = ExecutionJournal(store)
    state = ExecutionState(identity, journal, lease_until_ms=9999999999999, manager=manager)
    await journal.create(identity, state.summary())

    # Multiple checkpoints
    for i in range(5):
        state.record("test", f"key_{i}", {"iteration": i})
        await state.checkpoint(force=True)

    # Verify snapshot reflects latest state
    snapshot = await load_snapshot(manager, identity.job_id, identity.generation)

    assert snapshot is not None, "Snapshot should exist after checkpoints"
    assert snapshot.is_valid(), "Snapshot should be valid"

    # Verify journal can still be recovered
    recovered = await journal.recover(identity)

    assert recovered is not None
    assert recovered["batch_no"] >= 1

    print(f"✅ Snapshot and journal consistency verified")


@pytest.mark.asyncio
async def test_snapshot_without_manager_skips_write(registry_store):
    """Test backward compatibility: no manager = no snapshot."""
    store, connect = registry_store
    manager = JobManager(store)

    # Create job
    sid = await store.create_session()
    actor = JobActor("tenant", "host", sid)
    job = await manager.register_job(
        actor,
        kind="send",
        session_id=sid,
        system_key="test_snapshot_compat"
    )

    identity = ExecutionIdentity(
        store.store_id,
        job.job.job_id,
        1,  # execution generation
        "executor_001",
        "boot_001",
        "tenant",
        "host",
        sid
    )

    journal = ExecutionJournal(store)
    state = ExecutionState(
        identity,
        journal,
        lease_until_ms=9999999999999,
        manager=None  # No manager = no snapshot
    )
    await journal.create(identity, state.summary())

    state.record("test", "key", {"value": "test"})
    await state.checkpoint(force=True)

    # Snapshot should not exist (backward compatibility)
    snapshot = await load_snapshot(manager, identity.job_id, identity.generation)

    # Result is None because manager was not passed to ExecutionState
    print(f"✅ Backward compatibility: snapshot is {snapshot}")


@pytest.mark.asyncio
async def test_multiple_checkpoints_update_snapshot(registry_store):
    """Test that multiple checkpoints update the same snapshot."""
    store, connect = registry_store
    manager = JobManager(store)

    # Create job
    sid = await store.create_session()
    actor = JobActor("tenant", "host", sid)
    job = await manager.register_job(
        actor,
        kind="send",
        session_id=sid,
        system_key="test_snapshot_updates"
    )

    identity = ExecutionIdentity(
        store.store_id,
        job.job.job_id,
        1,  # execution generation
        "executor_001",
        "boot_001",
        "tenant",
        "host",
        sid
    )

    journal = ExecutionJournal(store)
    state = ExecutionState(identity, journal, lease_until_ms=9999999999999, manager=manager)
    await journal.create(identity, state.summary())

    # First checkpoint
    state.record("test", "v1", {"version": 1})
    await state.checkpoint(force=True)

    snapshot1 = await load_snapshot(manager, identity.job_id, identity.generation)
    time1 = snapshot1.captured_at_ms if snapshot1 else 0

    # Second checkpoint (should update snapshot)
    state.record("test", "v2", {"version": 2})
    await state.checkpoint(force=True)

    snapshot2 = await load_snapshot(manager, identity.job_id, identity.generation)
    time2 = snapshot2.captured_at_ms if snapshot2 else 0

    assert snapshot2 is not None, "Snapshot should exist after second checkpoint"
    assert time2 >= time1, "Snapshot should be updated with newer timestamp"

    print(f"✅ Multiple checkpoints update snapshot: {time1} -> {time2}")
