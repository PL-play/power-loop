"""Performance benchmarks for ExecutionSnapshot vs ExecutionJournal.

These benchmarks measure and compare:
1. Inspection speed: snapshot vs journal recovery
2. Snapshot size growth with different workloads
3. Checkpoint overhead with/without snapshot
4. Memory usage patterns
"""
import asyncio
import time
import json
import pytest
from power_loop.runtime.execution_snapshot import load_snapshot, ExecutionSnapshot
from power_loop.runtime.execution_snapshot_inspect import inspect_from_snapshot
from power_loop.runtime.execution_state import ExecutionState, ExecutionIdentity
from power_loop.runtime.execution_journal import ExecutionJournal
from power_loop.runtime.execution_recovery import inspect as inspect_from_journal
from power_loop.runtime.jobs import JobManager, JobActor

# Import registry_store fixture
from tests.unit.test_job_registry_192 import registry_store

pytestmark = [pytest.mark.bench, pytest.mark.asyncio]


async def create_test_state(store, manager, job_id="perf_test_job", with_snapshot=True):
    """Create ExecutionState for performance testing."""
    identity = ExecutionIdentity(
        store.store_id,
        job_id,
        1,
        "perf_executor",
        "perf_boot",
        "perf_tenant",
        "perf_principal",
        "perf_session"
    )

    journal = ExecutionJournal(store)
    state = ExecutionState(
        identity,
        journal,
        lease_until_ms=9999999999999,
        manager=manager if with_snapshot else None
    )

    await journal.create(identity, state.summary())
    return state, identity


@pytest.mark.asyncio
async def test_benchmark_snapshot_vs_journal_inspection(registry_store):
    """Benchmark: snapshot inspection vs journal recovery speed."""
    store, connect = registry_store
    manager = JobManager(store)
    actor = JobActor("tenant", "host", await store.create_session())

    # Create state and simulate workload
    state, identity = await create_test_state(store, manager)

    # Simulate workload: 50 operations, 10 checkpoints
    for batch in range(10):
        for op in range(5):
            state.record("test", f"batch_{batch}_op_{op}", {"data": f"value_{op}"})
        await state.checkpoint(force=True)

    print(f"\n=== Performance Benchmark ===")
    print(f"Workload: 50 operations, 10 checkpoints")

    # Benchmark 1: Snapshot inspection
    snapshot_start = time.time()
    for _ in range(10):
        result = await inspect_from_snapshot(manager, actor, identity.job_id, generation=identity.generation)
    snapshot_time = (time.time() - snapshot_start) / 10

    # Benchmark 2: Journal inspection
    journal_start = time.time()
    for _ in range(10):
        result = await inspect_from_journal(manager, actor, identity.job_id, generation=identity.generation, repair=False)
    journal_time = (time.time() - journal_start) / 10

    # Calculate speedup
    if journal_time > 0:
        speedup = ((journal_time - snapshot_time) / journal_time) * 100
    else:
        speedup = 0

    print(f"\nSnapshot inspection: {snapshot_time*1000:.2f}ms")
    print(f"Journal inspection:  {journal_time*1000:.2f}ms")
    print(f"Speedup:             {speedup:.1f}%")

    # Verify correctness
    assert snapshot_time <= journal_time * 2, "Snapshot should not be slower than 2x journal"
    print(f"\n✅ Performance benchmark completed")


@pytest.mark.asyncio
async def test_snapshot_size_growth(registry_store):
    """Measure snapshot size growth with different workloads."""
    store, connect = registry_store
    manager = JobManager(store)

    state, identity = await create_test_state(store, manager)

    sizes = []

    print(f"\n=== Snapshot Size Growth ===")

    for i in range(10):
        # Add tool calls
        for j in range(i + 1):
            state.record("attempt", f"tool_{i}_{j}", {
                "tool_call_id": f"call_{i}_{j}",
                "tool_name": "test_tool",
                "status": "running"
            })

        await state.checkpoint(force=True)

        snapshot = state.to_snapshot()
        snapshot_json = json.dumps(snapshot.to_dict())
        size = len(snapshot_json.encode('utf-8'))
        sizes.append(size)

        print(f"Checkpoint {i+1}: {len(snapshot.pending_tool_calls):2d} tool calls, {size:5d} bytes")

    print(f"\nAverage size: {sum(sizes)/len(sizes):.0f} bytes")
    print(f"Max size:     {max(sizes)} bytes")
    print(f"Growth rate:  {(sizes[-1]-sizes[0])/len(sizes):.1f} bytes/checkpoint")

    # Verify size is reasonable
    assert max(sizes) < 50000, "Snapshot should stay under 50KB"
    print(f"\n✅ Snapshot size within limits")


@pytest.mark.asyncio
async def test_checkpoint_overhead_comparison(registry_store):
    """Measure checkpoint overhead with and without snapshot."""
    store, connect = registry_store
    manager = JobManager(store)

    print(f"\n=== Checkpoint Overhead Comparison ===")

    # Test 1: Without snapshot
    state_no_snap, identity_no_snap = await create_test_state(
        store, manager, job_id="perf_no_snap", with_snapshot=False
    )

    start = time.time()
    for i in range(20):
        state_no_snap.record("test", f"op_{i}", {"data": i})
        await state_no_snap.checkpoint(force=True)
    time_no_snap = time.time() - start

    # Test 2: With snapshot
    state_with_snap, identity_with_snap = await create_test_state(
        store, manager, job_id="perf_with_snap", with_snapshot=True
    )

    start = time.time()
    for i in range(20):
        state_with_snap.record("test", f"op_{i}", {"data": i})
        await state_with_snap.checkpoint(force=True)
    time_with_snap = time.time() - start

    overhead = ((time_with_snap - time_no_snap) / time_no_snap) * 100 if time_no_snap > 0 else 0

    print(f"Without snapshot: {time_no_snap*1000:.2f}ms (20 checkpoints)")
    print(f"With snapshot:    {time_with_snap*1000:.2f}ms (20 checkpoints)")
    print(f"Overhead:         {overhead:.1f}%")

    # Verify overhead is acceptable
    assert overhead < 50, "Snapshot overhead should be less than 50%"
    print(f"\n✅ Checkpoint overhead acceptable")


@pytest.mark.asyncio
async def test_snapshot_consistency_over_time(registry_store):
    """Verify snapshot consistency across many checkpoints."""
    store, connect = registry_store
    manager = JobManager(store)

    state, identity = await create_test_state(store, manager)

    print(f"\n=== Snapshot Consistency Test ===")

    snapshots = []

    for i in range(20):
        state.record("test", f"op_{i}", {"value": i})
        await state.checkpoint(force=True)

        snapshot = state.to_snapshot()
        snapshots.append(snapshot)

    # Verify all snapshots are valid
    invalid_count = sum(1 for s in snapshots if not s.is_valid())

    # Verify timestamps are monotonic
    non_monotonic = sum(1 for i in range(1, len(snapshots))
                       if snapshots[i].captured_at_ms < snapshots[i-1].captured_at_ms)

    print(f"Total snapshots:      {len(snapshots)}")
    print(f"Invalid snapshots:    {invalid_count}")
    print(f"Non-monotonic times:  {non_monotonic}")

    assert invalid_count == 0, "All snapshots should be valid"
    assert non_monotonic == 0, "Timestamps should be monotonic"

    print(f"\n✅ Snapshot consistency verified")


@pytest.mark.asyncio
async def test_large_workload_performance(registry_store):
    """Benchmark with large workload (100 checkpoints)."""
    store, connect = registry_store
    manager = JobManager(store)
    actor = JobActor("tenant", "host", await store.create_session())

    state, identity = await create_test_state(store, manager, job_id="large_workload")

    print(f"\n=== Large Workload Performance ===")

    # Simulate large workload
    start = time.time()
    for i in range(100):
        state.record("test", f"op_{i}", {"iteration": i})
        await state.checkpoint(force=True)
    checkpoint_time = time.time() - start

    print(f"100 checkpoints: {checkpoint_time:.2f}s ({checkpoint_time/100*1000:.2f}ms avg)")

    # Test inspection speed
    snapshot_start = time.time()
    result_snapshot = await inspect_from_snapshot(manager, actor, identity.job_id, generation=identity.generation)
    snapshot_inspect_time = time.time() - snapshot_start

    journal_start = time.time()
    result_journal = await inspect_from_journal(manager, actor, identity.job_id, generation=identity.generation, repair=False)
    journal_inspect_time = time.time() - journal_start

    speedup = ((journal_inspect_time - snapshot_inspect_time) / journal_inspect_time) * 100 if journal_inspect_time > 0 else 0

    print(f"\nInspection after 100 checkpoints:")
    print(f"  Snapshot: {snapshot_inspect_time*1000:.2f}ms")
    print(f"  Journal:  {journal_inspect_time*1000:.2f}ms")
    print(f"  Speedup:  {speedup:.1f}%")

    print(f"\n✅ Large workload completed")
