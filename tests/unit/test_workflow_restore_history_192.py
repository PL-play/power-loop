"""Recovery must retain real child/input evidence after driver cleanup."""
from __future__ import annotations

import asyncio
import json

import pytest

from power_loop.runtime.jobs import JobError
from power_loop.workflow import runner

from .test_job_registry_192 import registry_store
from .test_node_nudge_runtime_192 import client, target
from .test_workflow_restore_plan_192 import finished, preview
from .test_workflow_task_runtime_192 import leaf, managed_workflow, settle

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("fault", ["none", "unhandled_child", "child_unknown", "child_membership_missing"])
async def test_plan_checks_original_nested_child_after_all_model_and_driver_sessions_end(managed_workflow, fault):
    _, manager, runtime, _, model, sid, state, _ = managed_workflow
    model.spawn_from_build = True
    state["spec"]["root"]["steps"][0] = leaf("BUILD", tools=["spawn_agent"])
    await finished(managed_workflow)
    [child] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='subagent'")
    assert child["result_status"] == "handled" and child["resource_state"] == "stopped"
    before = model.counts.copy()
    if fault == "unhandled_child":
        await manager.db.execute(f"UPDATE {manager.events} SET delivery_status='pending_delivery' WHERE event_id=?", ("result_"+child["job_id"],))
    elif fault == "child_unknown":
        await manager.db.execute(f"UPDATE {manager.jobs} SET resource_state='unknown' WHERE job_id=?", (child["job_id"],))
    elif fault == "child_membership_missing":
        await manager.db.execute(f"DELETE FROM {manager.events} WHERE job_id=? AND event_type='execution_membership_seal'", (child["job_id"],))
    if fault == "none":
        plan = await preview(managed_workflow)
        build = await manager.get_job(runtime.actor(sid), plan["nodes"]["BUILD"]["lease"]["job_id"])
        assert build.job_id == child["control_owner_job_id"]
        assert model.counts["AUDIT"] == 1
    else:
        with pytest.raises(JobError):
            await preview(managed_workflow)
    assert model.counts == before


@pytest.mark.parametrize("fault", ["none", "unhandled_nudge", "nudge_target_changed", "nudge_descriptor_changed"])
async def test_plan_verifies_real_handled_nudge_with_original_detached_descriptor(managed_workflow, monkeypatch, fault):
    store, manager, _, loop, model, parent, state, _ = managed_workflow
    state["on_complete"] = None
    c = await client(managed_workflow, monkeypatch)
    await loop.send("start workflow", parent)
    await asyncio.wait_for(model.entered["BUILD"].wait(), 3)
    [(row, data)] = await target(c)
    c.sid, c.job_id = data["session_id"], row["job_id"]
    c.run_id = state["handles"][0].run_id
    await c.issue()
    assert len(c.replies) == 1
    await settle(managed_workflow)
    await loop.flush_follow_ups(parent)
    assert (await store.get_session(c.sid)).parent_session_id is None
    [event] = await manager.db.fetchall(f"SELECT * FROM {manager.events} WHERE action='nudge'")
    assert event["delivery_status"] == "handled"
    if fault == "unhandled_nudge":
        await manager.db.execute(f"UPDATE {manager.events} SET delivery_status='pending_delivery' WHERE event_id=?", (event["event_id"],))
    elif fault in ("nudge_target_changed", "nudge_descriptor_changed"):
        body = json.loads(event["payload_json"])
        if fault == "nudge_target_changed":
            body["request"]["target"]["generation"] = True
        else:
            body["request"]["node"]["route"] = ["forged"]
        await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(body), event["event_id"]))
    before = model.counts.copy()
    if fault == "none":
        plan = await preview(managed_workflow)
        assert plan["nodes"]["BUILD"]["binding"]["session_id"] == c.sid
        assert c.run_id not in runner._ENGINES
    else:
        with pytest.raises(JobError):
            await preview(managed_workflow)
    assert model.counts == before


@pytest.mark.parametrize("owner", ["workflow", "node"])
async def test_unclaimed_child_missing_from_sealed_members_cannot_be_hidden_by_preview(managed_workflow, owner):
    _, manager, _, _, model, _, _, _ = managed_workflow
    await finished(managed_workflow)
    plan = await preview(managed_workflow)
    owner_id = plan["workflow"]["job_id"] if owner == "workflow" else plan["nodes"]["BUILD"]["lease"]["job_id"]
    original = await manager._row(manager.db, plan["nodes"]["BUILD"]["lease"]["job_id"])
    # Corrupt-store fault: a late unclaimed registration has no member record.
    # The production admission fence never authorizes this post-close insert.
    copied = {**original, "job_id": "queued_hidden_test", "control_owner_job_id": owner_id,
        "generation": 0, "status": "queued", "resource_state": "not_started", "result_status": "none",
        "result_json": None, "outcome": None, "executor_id": None, "boot_id": None}
    cols = tuple(copied)
    await manager.db.execute(f"INSERT INTO {manager.jobs} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})", tuple(copied[c] for c in cols))
    before = model.counts.copy()
    with pytest.raises(JobError, match="workflow_restore_execution_unsettled"):
        await preview(managed_workflow)
    assert model.counts == before
