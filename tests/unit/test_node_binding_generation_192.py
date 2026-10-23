"""Generation-isolated evidence; synthetic generation bumps never dispatch work."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from power_loop.runtime import node_inputs
from power_loop.runtime.jobs import JobActor, JobError, JobManager, _hash

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime
from .test_workflow_task_runtime_192 import managed_workflow, settle

__all__ = ["registry_store", "registry_runtime", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def physical_node(registry_runtime):
    store, manager, actor, _, root, connect, _ = registry_runtime
    wf = await manager.register_job(actor, kind="workflow", parent=root, independent=True, system_key="wf")
    wp = await manager.claim_job(actor, wf.job.job_id, executor_id="executor", boot_id="boot", expected_version=1)
    driver = await store.create_session(metadata={"kind": "wf_driver", "execution_role": "workflow",
        "task_owner_job_id": wf.job.job_id, "workflow_run_id": wf.operation.operation_id[3:]})
    await manager.bind_workflow_driver(actor, wp, driver)
    controller = JobActor(actor.tenant_id, actor.principal_id, driver)
    route = [{"foreach": "items", "index": 0}]
    node = await manager.register_job(controller, kind="workflow_node", parent=wp, system_key="node",
        request={"node": {"id": "BUILD", "retry": {"max_attempts": 2}}, "route": route})
    proof = await manager.claim_job(controller, node.job.job_id, executor_id="executor", boot_id="boot", expected_version=1)
    sid = await store.create_session(parent_session_id=driver, metadata={"execution_role": "workflow_node",
        "task_owner_job_id": node.job.job_id, "workflow_run_id": wf.operation.operation_id[3:],
        "workflow_node_id": "BUILD", "task_node_route": route, "attempt": 1})
    await manager.bind_node_session(controller, proof, sid)
    binding = SimpleNamespace(actor=controller, lease=proof)
    runtime = SimpleNamespace(manager=manager)
    row = asdict(await manager.get_job(controller, node.job.job_id))
    data = await node_inputs.descriptor(manager, manager.db, row, sid)
    return SimpleNamespace(store=store, manager=manager, actor=actor, controller=controller, node=node,
        proof=proof, sid=sid, row=row, data=data, binding=binding, runtime=runtime, connect=connect, wf=wf)


async def evidence(c):
    return await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")


async def install_generation(c, generation, *, bound=False):
    """Storage-only fixture for the future admission port, never a resume API."""
    async with c.manager._write() as tx:
        await tx.execute(f"UPDATE {c.manager.jobs} SET generation=? WHERE job_id=?", (generation, c.node.job.job_id))
        if bound:
            await tx.execute(f"INSERT INTO {c.manager.events} "
                "(event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
                (c.manager._node_session_event_id(c.node.job.job_id, c.sid, generation), c.node.job.job_id,
                 "node_session", c.row["version"], json.dumps({**c.data, "generation": generation}), "not_applicable", c.manager.now()))
    c.row["generation"] = generation
    c.binding.lease = replace(c.proof, generation=generation)


async def test_first_generation_ids_payloads_and_idempotency_remain_compatible(physical_node):
    c = physical_node
    digest = _hash([c.store.store_id, c.node.job.job_id, c.sid])[:32]
    assert c.manager._node_session_event_id(c.node.job.job_id, c.sid) == "session_" + digest
    assert c.manager._node_session_event_id(c.node.job.job_id, c.sid, 1) == "session_" + digest
    assert node_inputs.closed_id(c.manager, c.node.job.job_id, c.sid) == "input_closed_" + digest
    assert "generation" not in c.data
    before = await evidence(c)
    await c.manager.bind_node_session(c.controller, c.proof, c.sid)
    assert await evidence(c) == before
    await node_inputs.close(c.runtime, c.binding, c.sid)
    before = await evidence(c)
    await node_inputs.close(c.runtime, c.binding, c.sid)
    assert await evidence(c) == before
    reopened = JobManager(await c.connect(), now_ms=c.manager.now)
    assert await node_inputs.descriptor(reopened, reopened.db, c.row, c.sid) == c.data
    assert await node_inputs.is_closed(reopened, reopened.db, c.row, c.sid)
    marker = next(e for e in before if e["event_type"] == "node_input_closed")
    assert json.loads(marker["payload_json"]) == {"session_id": c.sid}


async def test_ids_separate_generations_stores_nodes_and_physical_sessions(physical_node):
    c = physical_node
    for function in (c.manager._node_session_event_id, lambda job, sid, generation: node_inputs.closed_id(c.manager, job, sid, generation)):
        ids = {function(job, sid, generation) for job in (c.node.job.job_id, "other-node")
               for sid in (c.sid, "other-sid") for generation in (1, 2, 3, 2**63-1)}
        assert len(ids) == 16
    other = SimpleNamespace(store=SimpleNamespace(store_id="other-store"))
    assert node_inputs.closed_id(other, c.node.job.job_id, c.sid) != node_inputs.closed_id(c.manager, c.node.job.job_id, c.sid)


@pytest.mark.parametrize("generation", [None, True, False, 0, -1, 1.0, "2", 2**63])
async def test_invalid_generation_cannot_alias_an_existing_binding_or_closure(physical_node, generation):
    c = physical_node
    for function in (c.manager._node_session_event_id, lambda job, sid, generation: node_inputs.closed_id(c.manager, job, sid, generation)):
        with pytest.raises(JobError, match="invalid_execution_generation"):
            function(c.node.job.job_id, c.sid, generation)


@pytest.mark.parametrize("fault", ["wrong_job", "wrong_type", "invalid_json", "list", "null", "wrong_sid", "extra_generation", "bool_attempt"])
async def test_corrupt_binding_never_authorizes_send_or_successful_duplicate(physical_node, fault):
    c = physical_node
    changes = {"wrong_job": ("job_id", "other-job"), "wrong_type": ("event_type", "operation"),
        "invalid_json": ("payload_json", "{"), "list": ("payload_json", "[]"), "null": ("payload_json", "null"),
        "wrong_sid": ("payload_json", json.dumps({**c.data, "session_id": "another-sid"})),
        "extra_generation": ("payload_json", json.dumps({**c.data, "generation": 1})),
        "bool_attempt": ("payload_json", json.dumps({**c.data, "attempt": True}))}
    column, value = changes[fault]
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET {column}=? WHERE event_id=?",
        (value, c.manager._node_session_event_id(c.node.job.job_id, c.sid)))
    before = await evidence(c)
    version = (await c.manager.get_job(c.controller, c.node.job.job_id)).version
    with pytest.raises(JobError, match="node_input_binding_mismatch"):
        await c.manager.bind_node_session(c.controller, c.proof, c.sid)
    with pytest.raises(JobError, match="node_input_binding_mismatch|node_session_admission_required"):
        await c.manager.register_job(c.controller, kind="send", parent=c.proof, session_id=c.sid, system_key="blocked-send")
    assert await evidence(c) == before
    assert (await c.manager.get_job(c.controller, c.node.job.job_id)).version == version


@pytest.mark.parametrize("field,value", [("workflow_run_id", "other-run"), ("workflow_node_id", "OTHER"),
    ("task_node_route", [{"foreach": "items", "index": 1}]), ("task_node_route", [{"foreach": "items", "index": False}]),
    ("attempt", 3), ("attempt", True)])
async def test_matching_metadata_and_event_still_cannot_rewrite_frozen_route(physical_node, field, value):
    c = physical_node
    await c.store.merge_session_metadata(c.sid, {field: value})
    payload = {**c.data, {"workflow_run_id": "workflow_run_id", "workflow_node_id": "node_id",
        "task_node_route": "route", "attempt": "attempt"}[field]: value}
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
        (json.dumps(payload), c.manager._node_session_event_id(c.node.job.job_id, c.sid)))
    with pytest.raises(JobError, match="node_input_binding_mismatch"):
        await node_inputs.descriptor(c.manager, c.manager.db, c.row, c.sid)
    with pytest.raises(JobError, match="node_route_mismatch"):
        await c.manager.bind_node_session(c.controller, c.proof, c.sid)
    with pytest.raises(JobError, match="node_input_binding_mismatch"):
        await c.manager.register_job(c.controller, kind="send", parent=c.proof, session_id=c.sid, system_key="blocked-send")


@pytest.mark.parametrize("frozen_request", [None, [], {"node": None}, {"node": {"retry": []}},
    {"node": {"id": "BUILD", "retry": {"max_attempts": True}}}, {"node": {"id": "BUILD", "retry": {"max_attempts": "2"}}}])
async def test_malformed_frozen_contract_is_a_controlled_rejection(physical_node, frozen_request):
    c = physical_node
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
        (json.dumps({"request": frozen_request}), c.node.operation.operation_id))
    with pytest.raises(JobError, match="node_route_mismatch"):
        await c.manager.bind_node_session(c.controller, c.proof, c.sid)
    with pytest.raises(JobError, match="node_input_binding_mismatch"):
        await node_inputs.descriptor(c.manager, c.manager.db, c.row, c.sid)


@pytest.mark.parametrize("missing", ["workflow", "node"])
async def test_missing_create_operation_cannot_be_replaced_by_matching_mutable_metadata(physical_node, missing):
    c = physical_node
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?",
        (c.wf.operation.operation_id if missing == "workflow" else c.node.operation.operation_id,))
    with pytest.raises(JobError, match="node_input_binding_mismatch"):
        await node_inputs.descriptor(c.manager, c.manager.db, c.row, c.sid)


@pytest.mark.parametrize("raw", ["{", "[]", "null"])
@pytest.mark.parametrize("location", ["session", "admission"])
async def test_malformed_json_at_either_binding_source_fails_without_internal_type_errors(physical_node, location, raw):
    c = physical_node
    if location == "session":
        await c.manager.db.execute(f"UPDATE {c.store.t.sessions} SET metadata_json=? WHERE session_id=?", (raw, c.sid))
    else:
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (raw, c.node.operation.operation_id))
    before = await evidence(c)
    with pytest.raises(JobError, match="node_input_binding_mismatch"):
        await c.manager.bind_node_session(c.controller, c.proof, c.sid)
    with pytest.raises(JobError, match="node_input_binding_mismatch"):
        await node_inputs.descriptor(c.manager, c.manager.db, c.row, c.sid)
    assert await evidence(c) == before


@pytest.mark.parametrize("fault", ["wrong_job", "wrong_type", "invalid_json", "list", "wrong_sid", "extra_generation"])
async def test_corrupt_closure_is_not_a_successful_duplicate_and_is_not_overwritten(physical_node, fault):
    c = physical_node
    await node_inputs.close(c.runtime, c.binding, c.sid)
    column, value = {"wrong_job": ("job_id", "other-job"), "wrong_type": ("event_type", "operation"),
        "invalid_json": ("payload_json", "{"), "list": ("payload_json", "[]"),
        "wrong_sid": ("payload_json", json.dumps({"session_id": "other-sid"})),
        "extra_generation": ("payload_json", json.dumps({"session_id": c.sid, "generation": 1}))}[fault]
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET {column}=? WHERE event_id=?",
        (value, node_inputs.closed_id(c.manager, c.node.job.job_id, c.sid)))
    before = await evidence(c)
    version = (await c.manager.get_job(c.controller, c.node.job.job_id)).version
    for check in (node_inputs.is_closed(c.manager, c.manager.db, c.row, c.sid), node_inputs.close(c.runtime, c.binding, c.sid)):
        with pytest.raises(JobError, match="node_input_closure_mismatch"):
            await check
    assert await evidence(c) == before
    assert (await c.manager.get_job(c.controller, c.node.job.job_id)).version == version


async def test_old_generation_binding_does_not_authorize_new_send_and_rebind_stays_closed(physical_node):
    c = physical_node
    await node_inputs.close(c.runtime, c.binding, c.sid)
    before = await evidence(c)
    await install_generation(c, 2)
    with pytest.raises(JobError, match="node_input_binding_mismatch"):
        await node_inputs.descriptor(c.manager, c.manager.db, c.row, c.sid)
    with pytest.raises(JobError, match="node_session_admission_required"):
        await c.manager.register_job(c.controller, kind="send", parent=c.binding.lease, session_id=c.sid, system_key="unadmitted-resume")
    with pytest.raises(JobError, match="task_adapter_unavailable:workflow_node_resume"):
        await c.manager.bind_node_session(c.controller, c.binding.lease, c.sid)
    assert not await node_inputs.is_closed(c.manager, c.manager.db, c.row, c.sid)
    assert await evidence(c) == before


async def test_storage_only_new_generation_requires_own_binding_and_retains_both_closures(physical_node):
    c = physical_node
    await node_inputs.close(c.runtime, c.binding, c.sid)
    old = await evidence(c)
    await install_generation(c, 2, bound=True)
    assert await node_inputs.descriptor(c.manager, c.manager.db, c.row, c.sid) == {**c.data, "generation": 2}
    assert not await node_inputs.is_closed(c.manager, c.manager.db, c.row, c.sid)
    with pytest.raises(JobError, match="stale_execution"):
        await node_inputs.close(c.runtime, SimpleNamespace(actor=c.controller, lease=c.proof), c.sid)
    await node_inputs.close(c.runtime, c.binding, c.sid)
    after = await evidence(c)
    assert all(row in after for row in old)
    assert len(after) == len(old) + 2
    assert await node_inputs.is_closed(c.manager, c.manager.db, c.row, c.sid)
    reopened = JobManager(await c.connect(), now_ms=c.manager.now)
    assert await node_inputs.is_closed(reopened, reopened.db, c.row, c.sid)
    assert await node_inputs.is_closed(reopened, reopened.db, {**c.row, "generation": 1}, c.sid)
    new_closed = next(row for row in after if row["event_id"] == node_inputs.closed_id(c.manager, c.node.job.job_id, c.sid, 2))
    assert json.loads(new_closed["payload_json"]) == {"session_id": c.sid, "generation": 2}


@pytest.mark.parametrize("generation", [None, 1, True, 2.0, "2"])
async def test_new_generation_does_not_accept_missing_or_coerced_payload_generation(physical_node, generation):
    c = physical_node
    await install_generation(c, 2, bound=True)
    payload = dict(c.data)
    if generation is not None:
        payload["generation"] = generation
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
        (json.dumps(payload), c.manager._node_session_event_id(c.node.job.job_id, c.sid, 2)))
    with pytest.raises(JobError, match="node_input_binding_mismatch"):
        await node_inputs.descriptor(c.manager, c.manager.db, c.row, c.sid)


@pytest.mark.parametrize("generation", [None, 1, True, 2.0, "2"])
async def test_new_closure_requires_exact_generation_in_its_own_record(physical_node, generation):
    c = physical_node
    await install_generation(c, 2, bound=True)
    await node_inputs.close(c.runtime, c.binding, c.sid)
    payload = {"session_id": c.sid}
    if generation is not None:
        payload["generation"] = generation
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
        (json.dumps(payload), node_inputs.closed_id(c.manager, c.node.job.job_id, c.sid, 2)))
    with pytest.raises(JobError, match="node_input_closure_mismatch"):
        await node_inputs.close(c.runtime, c.binding, c.sid)


@pytest.mark.parametrize("order", ["close_first", "generation_first"])
async def test_control_lock_serializes_generation_change_against_old_closure(physical_node, order):
    c = physical_node
    entered, release = asyncio.Event(), asyncio.Event()
    async def bump():
        async with c.manager._write() as tx:
            entered.set()
            await release.wait()
            await tx.execute(f"UPDATE {c.manager.jobs} SET generation=2 WHERE job_id=?", (c.node.job.job_id,))
    if order == "close_first":
        await node_inputs.close(c.runtime, c.binding, c.sid)
        release.set()
        await bump()
    else:
        task = asyncio.create_task(bump())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            closing = asyncio.create_task(node_inputs.close(c.runtime, c.binding, c.sid))
            release.set()
            await task
            with pytest.raises(JobError, match="stale_execution"):
                await closing
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    assert not await node_inputs.is_closed(c.manager, c.manager.db, {**c.row, "generation": 2}, c.sid)
    assert await node_inputs.is_closed(c.manager, c.manager.db, c.row, c.sid) == (order == "close_first")


@pytest.mark.parametrize("generation", [None, 1, True, 2.0, "2"])
async def test_old_input_descriptor_cannot_be_processed_as_new_generation(physical_node, generation):
    c = physical_node
    await install_generation(c, 2, bound=True)
    actor = JobActor(c.actor.tenant_id, c.actor.principal_id, c.sid)
    send = {"control_owner_job_id": c.node.job.job_id}
    old_data = dict(c.data)
    if generation is not None:
        old_data["generation"] = generation
    old = {"job_id": c.node.job.job_id, "payload_json": json.dumps({"request": {"node": old_data}})}
    with pytest.raises(JobError, match="processing_input_mismatch"):
        await node_inputs.input_target(c.manager, c.manager.db, actor, send, old)
    current = {**old, "payload_json": json.dumps({"request": {"node": {**c.data, "generation": 2}}})}
    assert (await node_inputs.input_target(c.manager, c.manager.db, actor, send, current))["generation"] == 2


async def test_corrupt_binding_blocks_actual_workflow_send_before_provider_and_downstream(managed_workflow, monkeypatch):
    _, manager, runtime, loop, model, parent, state, _ = managed_workflow
    model.hold.clear()
    bind = manager.bind_node_session
    async def corrupted(actor, lease, sid):
        await bind(actor, lease, sid)
        await manager.db.execute(f"UPDATE {manager.events} SET payload_json='[]' WHERE event_id=?",
            (manager._node_session_event_id(lease.job_id, sid, lease.generation),))
    monkeypatch.setattr(manager, "bind_node_session", corrupted)
    await loop.send("start workflow", parent)
    await settle(managed_workflow)
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 0
    job = await manager.get_job(runtime.actor(parent), state["handles"][0].job_id)
    assert job.outcome == "failed" and "binding" in str(job.result)
    assert not await manager.db.fetchall(f"SELECT job_id FROM {manager.jobs} WHERE kind='send' AND created_by_session_id<>?", (parent,))
