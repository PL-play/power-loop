"""Task schema limits must permit a real workflow to reach and finish its models."""
from __future__ import annotations

import pytest

from power_loop.runtime import completion_windows

from .test_job_registry_192 import registry_store
from .test_workflow_task_runtime_192 import managed_workflow, settle

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_workflow_initialization_fits_event_keys_and_reaches_core_assertions(managed_workflow):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    model.hold.clear()
    assert (await loop.send("initialize and complete the original workflow", sid)).status == "completed"
    await settle(managed_workflow)
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 1
    workflow = await manager.get_job(runtime.actor(sid), state["handles"][0].job_id)
    assert workflow.outcome == "succeeded"
    keys = await manager.db.fetchall(f"SELECT event_id,event_type,job_id FROM {manager.events}")
    assert keys and all(0 < len(row["event_id"]) <= 64 for row in keys), keys
    windows = [row for row in keys if row["event_type"] == "completion_window"]
    assert len(windows) == 2
    assert all(row["event_id"] == completion_windows.event_id(manager, row["job_id"]) for row in windows)
    assert (await store.get_state(sid)).pending is None
