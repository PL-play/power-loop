"""Callback coordinates identify an immutable result, never today's job state."""
from __future__ import annotations

import pytest

from power_loop.runtime.jobs import JobError
from power_loop.runtime.result_revisions import result_id
from power_loop.workflow import WorkflowCompletion
from power_loop.workflow.runner import _run_on_complete

from .test_workflow_task_runtime_192 import managed_workflow, registry_store, settle

__all__ = ["managed_workflow", "registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_legacy_callback_retains_optional_empty_revision_coordinates():
    seen = []
    async def callback(value):
        seen.append(value)
    assert await _run_on_complete(callback, "parent", "run", "completed", "note")
    [value] = seen
    assert value == WorkflowCompletion("parent", "run", "completed", "note")
    assert value.task_generation is None and value.result_event_id is None


@pytest.mark.parametrize("generation", [0, 1, 2])
async def test_explicit_native_coordinates_are_passed_without_reconstruction(generation):
    seen = []
    async def callback(value):
        seen.append(value)
    event = result_id("job_original", generation)
    assert await _run_on_complete(callback, "parent", "run", "failed", "note",
        task_generation=generation, result_event_id=event)
    [value] = seen
    assert value.task_generation == generation and value.result_event_id == event


@pytest.mark.parametrize("unstarted", [False, True])
async def test_actual_native_workflow_callback_carries_committed_result_revision(managed_workflow, monkeypatch, unstarted):
    store, manager, runtime, loop, model, sid, state, _ = managed_workflow
    seen = []
    async def callback(value):
        seen.append(value)
        row = await manager.db.fetchone(f"SELECT event_type,job_id FROM {manager.events} WHERE event_id=?",
            (value.result_event_id,))
        assert row is not None and row["event_type"] == "result"
    state["on_complete"] = callback
    if unstarted:
        claim = runtime.claim
        async def fail_claim(actor, registration):
            if registration.job.kind == "workflow":
                raise JobError("test_executor_unavailable")
            return await claim(actor, registration)
        monkeypatch.setattr(runtime, "claim", fail_claim)
    await loop.send("Run original workflow", sid)
    await settle(managed_workflow)
    [handle], [value] = state["handles"], seen
    assert value.parent_session_id == sid and value.run_id == handle.run_id
    assert value.task_generation == (0 if unstarted else 1)
    assert value.result_event_id == result_id(handle.job_id, value.task_generation)
    assert (await manager.get_job(runtime.actor(sid), handle.job_id)).generation == value.task_generation
