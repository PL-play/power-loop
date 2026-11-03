"""Snapshot-based execution inspection for faster debugging.

This provides a lightweight alternative to journal.recover() for inspection
and debugging purposes. It reads the snapshot from jobs.result_json instead
of replaying all batches from execution_batches.

Use this for:
- Admin dashboards showing task state
- Debugging tools examining execution progress
- Quick health checks without full journal replay

Do NOT use this for:
- Normal execution flow (tasks don't resume, they create new generations)
- Financial reconciliation (journal is authoritative)
- State mutations (inspection is read-only)
"""
from __future__ import annotations

import json

from .execution_snapshot import ExecutionSnapshot, load_snapshot, rebuild_from_events
from .execution_state import ExecutionIdentity
from .jobs import JobError


async def inspect_from_snapshot(manager, actor, job_id, *, generation):
    """Inspect execution state from snapshot (faster than journal replay).

    Returns execution summary for inspection/debugging. Falls back to event
    reconstruction if snapshot is missing or corrupted.

    This is read-only inspection - does not grant execution authority or
    perform any mutations.
    """
    from .managed_completion import _host_only
    _host_only()

    if type(generation) is not int or generation < 1:
        raise JobError("invalid_execution_recovery")

    # Authorize access
    await manager._authorize(manager.db, actor, job_id)

    # Try snapshot first (fast path)
    snapshot = await load_snapshot(manager, job_id, generation)

    if snapshot is not None and snapshot.is_valid():
        # Fast path: snapshot exists and is valid
        return {
            "job_id": job_id,
            "generation": generation,
            "source": "snapshot",
            "round_index": snapshot.round_index,
            "send_index": snapshot.send_index,
            "total_rounds": snapshot.total_rounds,
            "total_tokens": snapshot.total_tokens,
            "pending_tool_calls": snapshot.pending_tool_calls,
            "cancel_requested": snapshot.cancel_requested,
            "needs_reconciliation": snapshot.needs_reconciliation,
            "resource_state": snapshot.resource_state,
            "captured_at_ms": snapshot.captured_at_ms,
            "execution_authority": False,
            "replay_allowed": False,
        }

    # Fallback: reconstruct from events (slower but always works)
    snapshot = await rebuild_from_events(manager, job_id, generation)

    return {
        "job_id": job_id,
        "generation": generation,
        "source": "events_reconstructed",
        "round_index": snapshot.round_index,
        "send_index": snapshot.send_index,
        "total_rounds": snapshot.total_rounds,
        "total_tokens": snapshot.total_tokens,
        "pending_tool_calls": snapshot.pending_tool_calls,
        "cancel_requested": snapshot.cancel_requested,
        "needs_reconciliation": snapshot.needs_reconciliation,
        "resource_state": snapshot.resource_state,
        "captured_at_ms": snapshot.captured_at_ms,
        "execution_authority": False,
        "replay_allowed": False,
        "warning": "Snapshot missing or corrupted, reconstructed from events",
    }


async def compare_snapshot_with_journal(manager, actor, job_id, *, generation):
    """Compare snapshot with journal for validation (debugging tool).

    Returns both snapshot and journal views for comparison. Use this to
    verify snapshot accuracy during development/testing.
    """
    from .managed_completion import _host_only
    from .execution_recovery import inspect as inspect_from_journal

    _host_only()

    # Get both views
    snapshot_view = await inspect_from_snapshot(manager, actor, job_id, generation=generation)
    journal_view = await inspect_from_journal(manager, actor, job_id, generation=generation, repair=False, retire_expired=False)

    # Compare key fields
    differences = []

    if snapshot_view.get("round_index") != journal_view["summary"].get("round_index"):
        differences.append(f"round_index: snapshot={snapshot_view.get('round_index')}, journal={journal_view['summary'].get('round_index')}")

    if snapshot_view.get("needs_reconciliation") != journal_view.get("needs_reconciliation"):
        differences.append(f"needs_reconciliation: snapshot={snapshot_view.get('needs_reconciliation')}, journal={journal_view.get('needs_reconciliation')}")

    return {
        "job_id": job_id,
        "generation": generation,
        "snapshot_view": snapshot_view,
        "journal_view": {
            "batch_no": journal_view["batch_no"],
            "needs_reconciliation": journal_view["needs_reconciliation"],
            "summary": journal_view["summary"],
        },
        "differences": differences,
        "match": len(differences) == 0,
    }
