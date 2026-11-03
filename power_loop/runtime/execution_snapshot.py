"""Lightweight state snapshots for task recovery without execution journals.

This module provides snapshot-based recovery as an alternative to the heavy
execution_checkpoints/batches mechanism. Snapshots are stored in jobs.result_json
and enable fast recovery for normal scenarios, with event-based rebuild as fallback.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .jobs import JobError


@dataclass
class ExecutionSnapshot:
    """Lightweight snapshot of execution state for recovery."""

    snapshot_version: int = 1
    snapshot_type: str = "intermediate"  # or "final"
    captured_at_ms: int = 0

    # Core execution state
    round_index: int = 0
    send_index: int = 0
    last_model_call_at: int | None = None
    total_rounds: int = 0
    total_tokens: int = 0

    # Pending operations
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    user_input_expected: bool = False
    pause_reason: str | None = None

    # Budget tracking
    budget_policy_key: str | None = None
    budget_used_calls: int = 0
    budget_used_tokens: int = 0
    budget_remaining_calls: int | None = None
    budget_remaining_tokens: int | None = None
    budget_last_checkpoint_at: int | None = None

    # Execution flags
    cancel_requested: bool = False
    needs_reconciliation: bool = False
    resource_state: str = "running"

    # Metadata
    executor_id: str | None = None
    boot_id: str | None = None
    generation: int = 1
    checkpoint_reason: str = "periodic"

    def __post_init__(self):
        if self.snapshot_version != 1:
            raise JobError("unsupported_snapshot_version")
        if self.snapshot_type not in ("intermediate", "final"):
            raise JobError("invalid_snapshot_type")
        if self.checkpoint_reason not in ("periodic", "forced", "error", "pause", "complete", "reconstructed"):
            raise JobError("invalid_checkpoint_reason")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionSnapshot:
        """Create snapshot from dictionary."""
        if not isinstance(data, dict):
            raise JobError("invalid_snapshot_data")

        # Handle nested structure (old format compatibility)
        if "execution" in data:
            # Old nested format, flatten it
            snapshot_data = {
                "snapshot_version": data.get("snapshot_version", 1),
                "snapshot_type": data.get("snapshot_type", "intermediate"),
                "captured_at_ms": data.get("captured_at_ms", 0),
                "round_index": data["execution"].get("round_index", 0),
                "send_index": data["execution"].get("send_index", 0),
                "last_model_call_at": data["execution"].get("last_model_call_at"),
                "total_rounds": data["execution"].get("total_rounds", 0),
                "total_tokens": data["execution"].get("total_tokens", 0),
                "pending_tool_calls": data.get("pending", {}).get("tool_calls", []),
                "user_input_expected": data.get("pending", {}).get("user_input_expected", False),
                "pause_reason": data.get("pending", {}).get("pause_reason"),
                "budget_policy_key": data.get("budget", {}).get("policy_key"),
                "budget_used_calls": data.get("budget", {}).get("used_calls", 0),
                "budget_used_tokens": data.get("budget", {}).get("used_tokens", 0),
                "budget_remaining_calls": data.get("budget", {}).get("remaining_calls"),
                "budget_remaining_tokens": data.get("budget", {}).get("remaining_tokens"),
                "budget_last_checkpoint_at": data.get("budget", {}).get("last_checkpoint_at"),
                "cancel_requested": data.get("flags", {}).get("cancel_requested", False),
                "needs_reconciliation": data.get("flags", {}).get("needs_reconciliation", False),
                "resource_state": data.get("flags", {}).get("resource_state", "running"),
                "executor_id": data.get("metadata", {}).get("executor_id"),
                "boot_id": data.get("metadata", {}).get("boot_id"),
                "generation": data.get("metadata", {}).get("generation", 1),
                "checkpoint_reason": data.get("metadata", {}).get("checkpoint_reason", "periodic"),
            }
            return cls(**snapshot_data)

        # Flat format (current)
        return cls(**data)

    def is_valid(self) -> bool:
        """Check if snapshot is valid for recovery."""
        if self.snapshot_version != 1:
            return False
        if self.captured_at_ms <= 0:
            return False
        if self.generation < 1:
            return False
        return True

    def needs_repair(self) -> bool:
        """Check if snapshot needs reconciliation."""
        return self.needs_reconciliation or self.resource_state == "unknown"


async def load_snapshot(manager, job_id: str, generation: int) -> ExecutionSnapshot | None:
    """Load execution snapshot from jobs.result_json.

    Returns None if no snapshot exists or if the snapshot is invalid.
    """
    row = await manager.db.fetchone(
        f"SELECT result_json FROM {manager.jobs} WHERE job_id=? AND generation=?",
        (job_id, generation)
    )

    if row is None or row["result_json"] is None:
        return None

    try:
        data = json.loads(row["result_json"])
        snapshot = ExecutionSnapshot.from_dict(data)

        if not snapshot.is_valid():
            return None

        return snapshot
    except (json.JSONDecodeError, KeyError, TypeError, JobError):
        # Snapshot is corrupted or in unexpected format
        return None


async def update_snapshot(manager, tx, job_id: str, generation: int, snapshot: ExecutionSnapshot) -> None:
    """Update execution snapshot in jobs.result_json.

    Uses optimistic locking via version field to prevent conflicts.
    """
    snapshot.captured_at_ms = manager.now()
    snapshot_json = json.dumps(snapshot.to_dict(), allow_nan=False)

    affected = await tx.execute(
        f"UPDATE {manager.jobs} SET result_json=?, updated_at=?, version=version+1 "
        f"WHERE job_id=? AND generation=?",
        (snapshot_json, manager.now(), job_id, generation)
    )

    if affected != 1:
        raise JobError("snapshot_update_failed")


async def clear_snapshot(manager, tx, job_id: str, generation: int) -> None:
    """Clear execution snapshot (typically after task completion).

    This allows final result to be stored in result_json without snapshot data.
    """
    await tx.execute(
        f"UPDATE {manager.jobs} SET result_json=NULL, updated_at=?, version=version+1 "
        f"WHERE job_id=? AND generation=?",
        (manager.now(), job_id, generation)
    )


async def rebuild_from_events(manager, job_id: str, generation: int) -> ExecutionSnapshot:
    """Rebuild execution snapshot from job_events when snapshot is missing or corrupted.

    This is a fallback recovery path that reconstructs minimal state from events.
    Follows the design philosophy: allow intermediate results to be lost, but ensure
    the task is visible and can continue.
    """
    # Read key lifecycle events
    events = await manager.db.fetchall(
        f"SELECT event_type, payload_json, created_at FROM {manager.events} "
        f"WHERE job_id=? ORDER BY created_at ASC",
        (job_id,)
    )

    # Initialize with defaults
    snapshot = ExecutionSnapshot(
        snapshot_type="intermediate",
        generation=generation,
        checkpoint_reason="reconstructed",
        needs_reconciliation=True,
        resource_state="unknown"
    )

    # Reconstruct state from events
    for event in events:
        event_type = event["event_type"]
        try:
            payload = json.loads(event["payload_json"]) if event["payload_json"] else {}
        except json.JSONDecodeError:
            continue

        if event_type == "task_started":
            snapshot.captured_at_ms = event["created_at"]
        elif event_type == "model_call_started":
            snapshot.round_index = payload.get("round_index", snapshot.round_index)
            snapshot.last_model_call_at = event["created_at"]
        elif event_type == "tool_call_started":
            # Add to pending if not already completed
            tool_call_id = payload.get("tool_call_id")
            if tool_call_id and not any(t.get("tool_call_id") == tool_call_id for t in snapshot.pending_tool_calls):
                snapshot.pending_tool_calls.append({
                    "tool_call_id": tool_call_id,
                    "tool_name": payload.get("tool_name"),
                    "status": "running",
                    "dispatched_at": event["created_at"]
                })
        elif event_type == "tool_call_completed":
            # Remove from pending
            tool_call_id = payload.get("tool_call_id")
            snapshot.pending_tool_calls = [
                t for t in snapshot.pending_tool_calls
                if t.get("tool_call_id") != tool_call_id
            ]
        elif event_type == "task_paused":
            snapshot.pause_reason = payload.get("reason", "manual")
            snapshot.user_input_expected = payload.get("user_input_expected", False)
        elif event_type == "task_cancelled":
            snapshot.cancel_requested = True

    return snapshot
