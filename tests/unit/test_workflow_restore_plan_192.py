"""A recovery preview freezes old evidence without authorizing any execution."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace

import pytest

from power_loop.runtime import workflow_restore as restore
from power_loop.runtime.jobs import JobError, JobManager, _hash, _json
from power_loop.workflow.spec import BranchNode, ForeachNode, LoopNode, WorkflowSpec

from .test_job_registry_192 import registry_store
from .test_workflow_task_runtime_192 import leaf, managed_workflow, settle

__all__ = ["registry_store", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def finished(fixture, *, consume=True):
    _, _, _, loop, _, sid, state, _ = fixture
    state["on_complete"] = None
    assert (await loop.send("launch", sid)).status == "completed"
    await settle(fixture)
    if consume:
        assert (await loop.flush_follow_ups(sid)).status == "completed"
    return state["handles"][0]


async def preview(fixture, *, target="BUILD", generation=1, message="compute 20+23", manager=None):
    _, m, runtime, _, _, sid, state, _ = fixture
    return await (manager or m).get_workflow_restore_plan(runtime.actor(sid), state["handles"][0].job_id,
        generation=generation, node_id=target, message=message)


@pytest.mark.parametrize("target,invalidated,replay", [("BUILD", ["BUILD", "REVIEW"], []), ("REVIEW", ["REVIEW"], ["BUILD"])])
async def test_complete_original_evidence_and_dependency_plan_survive_deleted_driver_without_dispatch(managed_workflow, target, invalidated, replay):
    store, manager, runtime, loop, model, sid, _, connect = managed_workflow
    handle = await finished(managed_workflow)
    actor = runtime.actor(sid)
    before = await manager.get_job(actor, handle.job_id)
    assert await store.get_session(before.session_id) is None
    events = await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id")
    counts = model.counts.copy()
    data = await preview(managed_workflow, target=target)
    assert data["target"] == target and data["dependency"]["invalidated"] == invalidated and data["replay"] == replay
    assert data["workflow"]["generation"] == 1 and data["creation_operation_id"] == handle.operation_id
    assert data["result"]["result_event_id"] == "result_"+handle.job_id
    assert set(data["nodes"]) == {"BUILD", "REVIEW"}
    for nid, node in data["nodes"].items():
        session = await store.get_session(node["binding"]["session_id"])
        assert session.parent_session_id is None
        assert node["binding"]["node_id"] == nid and node["binding_hash"] == _hash(node["binding"])
        assert node["contract_hash"] == _hash(node["contract"]) and node["result"]["status"] == "completed"
    other = JobManager(await connect())
    assert await preview(managed_workflow, target=target, manager=other) == data
    assert await manager.get_job(actor, handle.job_id) == before and model.counts == counts
    assert await manager.db.fetchall(f"SELECT * FROM {manager.events} ORDER BY event_id") == events
    # Even a valid plan is not a dispatch/controller credential.
    from power_loop.workflow.resume import resume_detached
    with pytest.raises(JobError, match="task_adapter_unavailable:workflow_resume"):
        await resume_detached(loop, sid, handle.run_id, continuation=(target, "new", 2), force=True)
    assert model.counts == counts


async def test_parallel_independent_branch_is_preserved_and_sequence_tail_invalidated(managed_workflow):
    _, _, _, _, _, _, state, _ = managed_workflow
    state["spec"]["root"] = {"type": "sequence", "steps": [
        {"type": "parallel", "branches": [leaf("BUILD"), leaf("AUDIT")]}, leaf("REVIEW", inputs_from=["BUILD"])]}
    await finished(managed_workflow)
    data = await preview(managed_workflow)
    assert data["dependency"]["invalidated"] == ["BUILD", "REVIEW"] and data["replay"] == ["AUDIT"]


@pytest.mark.parametrize("fault", ["unhandled", "generation", "target", "empty_message", "nul_message", "bool_generation"])
async def test_invalid_restore_intent_is_rejected_before_any_new_generation(managed_workflow, fault):
    _, manager, runtime, _, model, sid, _, _ = managed_workflow
    handle = await finished(managed_workflow, consume=fault != "unhandled")
    args = {}
    if fault == "generation":
        args["generation"] = 2
    elif fault == "target":
        args["target"] = "UNKNOWN"
    elif fault == "empty_message":
        args["message"] = " "
    elif fault == "nul_message":
        args["message"] = "bad\x00input"
    elif fault == "bool_generation":
        args["generation"] = True
    before = model.counts.copy()
    with pytest.raises(JobError):
        await preview(managed_workflow, **args)
    assert model.counts == before and (await manager.get_job(runtime.actor(sid), handle.job_id)).generation == 1


@pytest.mark.parametrize("fault", ["contract_missing", "contract_widened", "tools_widened", "metadata_widened", "owner_changed", "binding_deleted", "session_parent", "pending_tool", "pending_input", "node_result", "node_unknown", "membership", "controller", "profile_missing"])
async def test_corrupt_unsettled_or_changed_physical_evidence_cannot_be_accepted(managed_workflow, fault):
    store, manager, runtime, _, model, sid, _, _ = managed_workflow
    handle = await finished(managed_workflow)
    data = await preview(managed_workflow)
    node = data["nodes"]["BUILD"]
    node_id, node_sid = node["lease"]["job_id"], node["binding"]["session_id"]
    if fault == "contract_missing":
        await manager.db.execute(f"DELETE FROM {store.t.session_runtime_state} WHERE session_id=? AND state_key=?", (node_sid, "workflow_execution_contract"))
    elif fault in ("contract_widened", "tools_widened", "metadata_widened"):
        contract = copy.deepcopy(node["contract"])
        if fault == "contract_widened":
            contract["spec"]["system_prompt"] = "different task"
        elif fault == "tools_widened":
            contract["tools"] = ["extra_privileged_tool"]
        else:
            contract["spec"]["metadata"]["privileged"] = True
        await store.set_runtime_state(node_sid, "workflow_execution_contract", contract)
    elif fault == "binding_deleted":
        await manager.db.execute(f"DELETE FROM {manager.events} WHERE event_id=?", (manager._node_session_event_id(node_id, node_sid, 1),))
    elif fault == "session_parent":
        await manager.db.execute(f"UPDATE {store.t.sessions} SET parent_session_id=? WHERE session_id=?", (sid, node_sid))
    elif fault == "pending_tool":
        await manager.db.execute(f"UPDATE {store.t.session_state} SET pending_json=? WHERE session_id=?", ('{"unfinished":true}', node_sid))
    elif fault == "pending_input":
        await store.inbox_put(node_sid, [{"content": "unfinished", "kind": "relay", "mode": "steer", "item_id": "test:unfinished"}])
    elif fault == "node_result":
        await manager.db.execute(f"UPDATE {manager.jobs} SET result_json=? WHERE job_id=?", ('{"status":"completed"}', node_id))
    elif fault == "node_unknown":
        await manager.db.execute(f"UPDATE {manager.jobs} SET resource_state='unknown' WHERE job_id=?", (node_id,))
    elif fault == "owner_changed":
        await manager.db.execute(f"UPDATE {manager.jobs} SET control_owner_generation=2 WHERE job_id=?", (node_id,))
    elif fault == "membership":
        await manager.db.execute(f"DELETE FROM {manager.events} WHERE job_id=? AND event_type='execution_membership_seal'", (node_id,))
    elif fault == "controller":
        await manager.db.execute(f"DELETE FROM {manager.events} WHERE job_id=? AND event_type='workflow_controller'", (handle.job_id,))
    elif fault == "profile_missing":
        event = await manager.db.fetchone(f"SELECT payload_json FROM {manager.events} WHERE event_id=?", (handle.operation_id,))
        body = json.loads(event["payload_json"])
        body["request"].pop("execution_profile")
        await manager.db.execute(f"UPDATE {manager.events} SET payload_json=? WHERE event_id=?", (_json(body), handle.operation_id))
    before = model.counts.copy()
    with pytest.raises(JobError):
        await preview(managed_workflow)
    assert model.counts == before and (await manager.get_job(runtime.actor(sid), handle.job_id)).generation == 1


@pytest.mark.parametrize("shape", ["foreach", "loop", "branch", "fallback", "file_io", "file_reference"])
async def test_first_restore_adapter_refuses_ambiguous_or_external_layouts(shape):
    s = WorkflowSpec.from_json({"name": "shape", "root": leaf("BUILD")})
    if shape == "foreach":
        root = ForeachNode(type="foreach", as_var="item", body=s.root, items=("x",))
    elif shape == "loop":
        root = LoopNode(type="loop", id="rounds", body=s.root, until={"ref": "BUILD.text", "op": "truthy"})
    elif shape == "branch":
        root = BranchNode(type="branch", on="BUILD.text", cases={"x": s.root})
    elif shape == "fallback":
        root = replace(s.root, fallback=replace(s.root, id="FALLBACK"))
    elif shape == "file_io":
        root = replace(s.root, output_file="out.md")
    else:
        root = replace(s.root, input="@@FILEREF:external.md@@")
    with pytest.raises(JobError, match="workflow_restore_scope_unsupported"):
        restore.selection(replace(s, root=root), "BUILD")


async def test_historical_read_of_detached_node_still_cannot_authorize_current_input_or_bind(managed_workflow):
    _, manager, runtime, _, _, sid, _, _ = managed_workflow
    await finished(managed_workflow)
    data = await preview(managed_workflow)
    node = data["nodes"]["BUILD"]
    row = asdict(await manager.get_job(runtime.actor(sid), node["lease"]["job_id"]))
    from power_loop.runtime.node_inputs import descriptor
    with pytest.raises(JobError, match="node_input_binding_mismatch"):
        await descriptor(manager, manager.db, row, node["binding"]["session_id"])
    assert await descriptor(manager, manager.db, row, node["binding"]["session_id"], historical=True) == node["binding"]
