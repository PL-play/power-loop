"""Frozen driver authority, exact generation boundaries, and zero bad dispatch."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from power_loop.runtime import workflow_controller as controller
from power_loop.runtime.jobs import JobError, JobLease, JobManager, _hash

from .test_job_registry_192 import registry_store
from .test_job_registry_192 import runtime as registry_runtime
from .test_node_binding_generation_192 import physical_node
from .test_workflow_task_runtime_192 import managed_workflow, settle

__all__ = ["registry_store", "registry_runtime", "physical_node", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def unbound(registry_runtime):
    store, manager, actor, _, parent, connect, now = registry_runtime
    wf = await manager.register_job(actor, kind="workflow", parent=parent, independent=True, system_key="wf",
        request={"spec": {"name": "immutable-admission"}})
    lease = await manager.claim_job(actor, wf.job.job_id, executor_id="engine", boot_id="boot", expected_version=1)
    meta = {"kind": "wf_driver", "execution_role": "workflow", "task_owner_job_id": lease.job_id,
        "workflow_run_id": wf.operation.operation_id[3:], "task_workflow_generation": 1}
    sid = await store.create_session(metadata=meta)
    return SimpleNamespace(store=store, manager=manager, actor=actor, wf=wf, proof=lease, sid=sid,
        meta=meta, connect=connect, now=now, parent=parent)


async def rows(c):
    return await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")


async def test_bind_is_atomic_idempotent_immutable_and_reopenable(unbound):
    c = unbound
    first = await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    frozen = await c.manager.get_workflow_controller(c.actor, c.proof.job_id, 1)
    assert frozen["workflow"] == asdict(c.proof) and frozen["session_id"] == c.sid
    assert frozen["creation_operation_id"] == c.wf.operation.operation_id and len(frozen["creation_hash"]) == 64
    before = await rows(c)
    assert await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid) == first
    assert await rows(c) == before and first.version == 3
    reopened = JobManager(await c.connect(), now_ms=c.manager.now)
    assert await reopened.get_workflow_controller(c.actor, c.proof.job_id, 1) == frozen
    assert await reopened.bind_workflow_driver(c.actor, c.proof, c.sid) == first


async def test_closed_original_main_does_not_prevent_independent_workflow_driver_bind(unbound):
    c = unbound
    await c.manager.renew_execution(c.actor, c.proof, lease_ms=60000)
    await c.manager.close_model_execution(c.actor, c.parent)
    c.now[0] += 31000
    assert (await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)).session_id == c.sid


@pytest.mark.parametrize("generation", [None, True, False, 0, -1, 1.0, "1", 2**63])
async def test_invalid_generation_cannot_alias_a_controller(unbound, generation):
    c = unbound
    with pytest.raises(JobError, match="invalid_execution_generation"):
        controller.event_id(c.manager, c.proof.job_id, generation)
    with pytest.raises(JobError, match="invalid_execution_generation"):
        await c.manager.get_workflow_controller(c.actor, c.proof.job_id, generation)


async def test_controller_ids_partition_store_job_and_generation(unbound):
    c = unbound
    values = {controller.event_id(c.manager, job, g) for job in (c.proof.job_id, "other") for g in (1, 2, 3, 2**63-1)}
    assert len(values) == 8 and all(len(value) <= 64 for value in values)
    other = SimpleNamespace(store=SimpleNamespace(store_id="other"))
    assert controller.event_id(other, c.proof.job_id, 1) not in values


@pytest.mark.parametrize("field,value", [("kind", "other"), ("execution_role", "main"),
    ("task_owner_job_id", "other"), ("workflow_run_id", "other"), ("task_workflow_generation", True),
    ("task_workflow_generation", 1.0), ("task_workflow_generation", "1"), ("task_workflow_generation", 2)])
@pytest.mark.parametrize("repeated", [False, True])
async def test_driver_metadata_is_checked_on_first_and_repeated_binding(unbound, field, value, repeated):
    c = unbound
    if repeated:
        await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    await c.store.merge_session_metadata(c.sid, {field: value})
    before, saved = await rows(c), await c.manager.get_job(c.actor, c.proof.job_id)
    with pytest.raises(JobError, match="ownership|workflow_run_mismatch|workflow_controller_mismatch"):
        await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    assert await rows(c) == before and await c.manager.get_job(c.actor, c.proof.job_id) == saved


@pytest.mark.parametrize("fault", ["missing", "json", "array", "nonfinite", "job", "type", "status", "hash", "generation", "sid", "creation"])
async def test_corrupt_binding_cannot_be_healed_by_repeated_bind_or_used_to_create_nodes(unbound, fault):
    c = unbound
    await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    key = controller.event_id(c.manager, c.proof.job_id, 1)
    if fault == "missing":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (key,))
    elif fault in ("json", "array", "nonfinite"):
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?",
            ({"json": "{bad", "array": "[]", "nonfinite": '{"version":NaN}'}[fault], key))
    elif fault in ("job", "type", "status", "hash"):
        col, value = {"job": ("job_id", c.parent.job_id), "type": ("event_type", "node_session"),
            "status": ("delivery_status", "handled"), "hash": ("request_hash", "wrong")}[fault]
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET {col}=? WHERE event_id=?", (value, key))
    elif fault == "creation":
        [op] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (c.wf.operation.operation_id,))
        payload = json.loads(op["payload_json"])
        payload["request"] = {"spec": {"name": "changed-after-binding"}}
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(payload), c.wf.operation.operation_id))
    else:
        data = await c.manager.get_workflow_controller(c.actor, c.proof.job_id, 1)
        if fault == "generation":
            data["workflow"]["generation"] = 2
        else:
            data["session_id"] = "forged-sid"
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?",
            (json.dumps(data), _hash(data), key))
    before = await rows(c)
    with pytest.raises(JobError, match="workflow_controller"):
        await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    with pytest.raises(JobError, match="workflow_controller"):
        await c.manager.register_job(replace(c.actor, session_id=c.sid), kind="workflow_node", parent=c.proof, system_key="bad-node")
    assert await rows(c) == before
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='workflow_node'")


@pytest.mark.parametrize("fault", ["pointer", "parent", "deleted", "metadata_json", "scope_missing", "scope_proof", "creation_type", "creation_missing"])
async def test_mutable_or_missing_controller_sources_fail_closed(unbound, fault):
    c = unbound
    await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    if fault == "pointer":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET session_id=NULL WHERE job_id=?", (c.proof.job_id,))
    elif fault == "parent":
        await c.manager.db.execute(f"UPDATE {c.store.t.sessions} SET parent_session_id=? WHERE session_id=?", (c.actor.session_id, c.sid))
    elif fault == "deleted":
        await c.store.close_session(c.sid, cascade=False)
    elif fault == "metadata_json":
        await c.manager.db.execute(f"UPDATE {c.store.t.sessions} SET metadata_json='[]' WHERE session_id=?", (c.sid,))
    elif fault.startswith("scope_"):
        from power_loop.runtime.execution_membership import scope_id
        key = scope_id(c.manager, c.proof.job_id, 1)
        if fault == "scope_missing":
            await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (key,))
        else:
            [row] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (key,))
            payload = json.loads(row["payload_json"])
            payload["lease"]["boot_id"] = "different"
            await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(payload), key))
    elif fault == "creation_type":
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET event_type='node_session' WHERE event_id=?", (c.wf.operation.operation_id,))
    else:
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (c.wf.operation.operation_id,))
    before = await rows(c)
    caller = c.actor if fault == "pointer" else replace(c.actor, session_id=c.sid)
    with pytest.raises(JobError):
        await c.manager.register_job(caller, kind="workflow_node", parent=c.proof, system_key="forbidden-node")
    assert await rows(c) == before


@pytest.mark.parametrize("point", ["update", "event", "cancel"])
async def test_partial_driver_binding_rolls_back_and_retry_is_exactly_once(unbound, monkeypatch, point):
    c = unbound
    from contextlib import asynccontextmanager
    write = c.manager._write
    @asynccontextmanager
    async def broken():
        async with write() as tx:
            execute = tx.execute
            async def fail(sql, params=()):
                value = await execute(sql, params)
                if (point == "update" and sql.startswith("UPDATE "+c.manager.jobs) or
                        point in ("event", "cancel") and sql.startswith("INSERT") and "workflow_controller" in params):
                    if point == "cancel":
                        raise asyncio.CancelledError()
                    raise RuntimeError("controller transaction fault")
                return value
            with monkeypatch.context() as patch:
                patch.setattr(tx, "execute", fail)
                yield tx
    saved = await c.manager.get_job(c.actor, c.proof.job_id)
    before = await rows(c)
    with monkeypatch.context() as patch:
        patch.setattr(c.manager, "_write", broken)
        with pytest.raises(asyncio.CancelledError if point == "cancel" else RuntimeError):
            await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    assert await c.manager.get_job(c.actor, c.proof.job_id) == saved and await rows(c) == before
    await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    assert (await c.manager.get_workflow_controller(c.actor, c.proof.job_id, 1))["session_id"] == c.sid


async def test_two_managers_compete_for_one_controller_and_cannot_replace_winner(unbound):
    c = unbound
    other = JobManager(await c.connect(), now_ms=c.manager.now)
    sid2 = await c.store.create_session(metadata=c.meta)
    results = await asyncio.gather(c.manager.bind_workflow_driver(c.actor, c.proof, c.sid),
        other.bind_workflow_driver(c.actor, c.proof, sid2), return_exceptions=True)
    assert sum(isinstance(r, JobError) and r.code == "session_already_bound" for r in results) == 1
    data = await other.get_workflow_controller(c.actor, c.proof.job_id, 1)
    assert data["session_id"] in (c.sid, sid2)
    assert len([r for r in await rows(c) if r["event_type"] == "workflow_controller"]) == 1


@pytest.mark.parametrize("field", ["tenant_id", "principal_id", "session_id"])
async def test_historical_controller_read_remains_authorized(unbound, field):
    c = unbound
    await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    wrong = replace(c.actor, **{field: await c.store.create_session()})
    with pytest.raises(JobError, match="access_denied"):
        await c.manager.get_workflow_controller(wrong, c.proof.job_id, 1)


async def test_driver_deletion_keeps_history_and_actual_late_exit_but_prevents_dispatch(physical_node):
    c = physical_node
    _, m, actor = c.store, c.manager, c.controller
    wf_proof = (await m.get_workflow_controller(c.actor, c.wf.job.job_id, 1))["workflow"]
    saved = await m.get_workflow_controller(c.actor, c.wf.job.job_id, 1)
    reg = await m.register_job(actor, kind="send", parent=c.proof, session_id=c.sid, system_key="actual-send")
    proof = await m.claim_job(replace(actor, session_id=c.sid), reg.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
    await c.store.close_session(c.controller.session_id, cascade=False)
    assert await m.get_workflow_controller(c.actor, c.wf.job.job_id, 1) == saved
    with pytest.raises(JobError, match="ownership"):
        await m.check_execution(replace(actor, session_id=c.sid), proof)
    await m.close_model_execution(replace(actor, session_id=c.sid), proof, evidence={"usage": {"total_tokens": 17}})
    await m.record_execution_result(c.actor, c.proof, outcome="failed", result={"text": "actual exit"}, resource_state="stopped")
    assert (await m.get_result_revision(c.actor, c.node.job.job_id, 1))["payload"]["result"]["text"] == "actual exit"
    assert await m.execution_usage(c.actor, wf_proof["job_id"], generation=1) == {"total_tokens": 17}


@pytest.mark.parametrize("port", ["node_claim", "node_bind", "send_create", "send_claim", "dispatch"])
async def test_each_dispatch_boundary_rechecks_the_same_controller(physical_node, port):
    c = physical_node
    m = c.manager
    reg = proof = None
    if port == "node_claim":
        reg = await m.register_job(c.controller, kind="workflow_node", parent=JobLease(
            **(await m.get_workflow_controller(c.actor, c.wf.job.job_id, 1))["workflow"]), system_key="later-node")
    elif port in ("send_claim", "dispatch"):
        reg = await m.register_job(c.controller, kind="send", parent=c.proof, session_id=c.sid, system_key="queued-send")
        if port == "dispatch":
            proof = await m.claim_job(replace(c.actor, session_id=c.sid), reg.job.job_id, executor_id="e", boot_id="b", expected_version=1)
    await c.store.merge_session_metadata(c.controller.session_id, {"task_workflow_generation": 2})
    before = await m.db.fetchall(f"SELECT * FROM {m.events} ORDER BY event_id")
    with pytest.raises(JobError, match="workflow_controller_mismatch"):
        if port == "node_claim":
            await m.claim_job(c.controller, reg.job.job_id, executor_id="e", boot_id="b", expected_version=1)
        elif port == "node_bind":
            await m.bind_node_session(c.controller, c.proof, c.sid)
        elif port == "send_create":
            await m.register_job(c.controller, kind="send", parent=c.proof, session_id=c.sid, system_key="forbidden-send")
        elif port == "send_claim":
            await m.claim_job(replace(c.actor, session_id=c.sid), reg.job.job_id, executor_id="e", boot_id="b", expected_version=1)
        else:
            await m.check_execution(replace(c.actor, session_id=c.sid), proof)
    assert await m.db.fetchall(f"SELECT * FROM {m.events} ORDER BY event_id") == before


@pytest.mark.parametrize("stage", ["driver_bind", "node_claim", "send_claim"])
async def test_actual_workflow_never_calls_provider_after_controller_corruption(managed_workflow, monkeypatch, stage):
    _, manager, runtime, loop, model, parent, state, _ = managed_workflow
    model.hold.clear()
    driver = []
    bind, claim = manager.bind_workflow_driver, manager.claim_job
    async def bound(actor, lease, sid):
        result = await bind(actor, lease, sid)
        driver[:] = [sid]
        if stage == "driver_bind":
            await manager.store.merge_session_metadata(sid, {"task_workflow_generation": True})
        return result
    async def claimed(actor, job_id, **kwargs):
        result = await claim(actor, job_id, **kwargs)
        job = await manager.get_job(actor, job_id)
        if driver and ((stage == "node_claim" and job.kind == "workflow_node") or
                (stage == "send_claim" and job.kind == "send" and job.execution_role == "workflow_node")):
            await manager.store.merge_session_metadata(driver[0], {"task_workflow_generation": True})
        return result
    monkeypatch.setattr(manager, "bind_workflow_driver", bound)
    monkeypatch.setattr(manager, "claim_job", claimed)
    await loop.send("run", parent)
    await settle(managed_workflow)
    assert model.counts["BUILD"] == model.counts["REVIEW"] == 0
    job = await manager.get_job(runtime.actor(parent), state["handles"][0].job_id)
    assert job.outcome == "failed" and job.result["status"] == "failed"


@pytest.mark.parametrize("sid", [None, "", 1, "x"*65])
async def test_first_binding_requires_an_actual_session_identifier(unbound, sid):
    c = unbound
    before = await rows(c)
    with pytest.raises(JobError, match="session_ownership_mismatch"):
        await c.manager.bind_workflow_driver(c.actor, c.proof, sid)
    assert await rows(c) == before


@pytest.mark.parametrize("fault", ["kind", "role", "subject", "proof", "orphan"])
async def test_first_binding_requires_exact_claim_snapshot_and_no_orphan_controller(unbound, fault):
    c = unbound
    from power_loop.runtime.execution_membership import scope_id
    key = scope_id(c.manager, c.proof.job_id, 1)
    if fault == "orphan":
        await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET session_id=NULL WHERE job_id=?", (c.proof.job_id,))
    else:
        [row] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (key,))
        data = json.loads(row["payload_json"])
        if fault == "kind":
            data["kind"] = "send"
        elif fault == "role":
            data["execution_role"] = "main"
        elif fault == "subject":
            data["tenant_id"] = "other"
        else:
            data["lease"]["boot_id"] = "other"
        await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), key))
    before = await rows(c)
    with pytest.raises(JobError, match="workflow_controller_mismatch"):
        await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    assert await rows(c) == before and (await c.manager.get_job(c.actor, c.proof.job_id)).session_id is None


async def test_new_workflow_generation_cannot_borrow_old_controller_or_create_permission(unbound):
    c = unbound
    await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    old = await c.manager.get_workflow_controller(c.actor, c.proof.job_id, 1)
    from power_loop.runtime.execution_membership import start
    # Storage-only fixture; no runtime resume, model, or new execution license.
    async with c.manager._write() as tx:
        await tx.execute(f"UPDATE {c.manager.jobs} SET generation=2 WHERE job_id=?", (c.proof.job_id,))
        row = await c.manager._row(tx, c.proof.job_id)
        await start(c.manager, tx, row)
    proof = replace(c.proof, generation=2)
    before = await rows(c)
    with pytest.raises(JobError, match="task_adapter_unavailable:workflow_resume"):
        await c.manager.bind_workflow_driver(c.actor, proof, c.sid)
    assert await rows(c) == before
    assert await c.manager.get_workflow_controller(c.actor, c.proof.job_id, 1) == old
    with pytest.raises(JobError, match="workflow_controller_missing"):
        await c.manager.get_workflow_controller(c.actor, c.proof.job_id, 2)
    # Even a coherent synthetic generation-2 history record is not admission.
    async with c.manager._write() as tx:
        row = await c.manager._row(tx, c.proof.job_id)
        with pytest.raises(JobError, match="workflow_restore_execution_mismatch"):
            await controller.expected(c.manager, tx, row, 2, c.sid)
        data = {**old, "workflow": asdict(proof)}
        await tx.execute(f"INSERT INTO {c.manager.events} (event_id,job_id,event_type,job_version,request_hash,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (controller.event_id(c.manager, c.proof.job_id, 2), c.proof.job_id, "workflow_controller", row["version"], _hash(data), json.dumps(data), "not_applicable", c.manager.now()))
    with pytest.raises(JobError, match="workflow_restore_execution_mismatch"):
        await c.manager.get_workflow_controller(c.actor, c.proof.job_id, 2)
    with pytest.raises(JobError, match="workflow_restore_execution_mismatch"):
        await c.manager.register_job(replace(c.actor, session_id=c.sid), kind="workflow_node", parent=proof, system_key="forbidden-resume-node")


@pytest.mark.parametrize("fault", ["parent", "role", "creator", "owner_proof", "anchor_owner", "anchor_lease"])
async def test_node_authority_checks_original_controller_and_own_claim(physical_node, fault):
    c = physical_node
    m = c.manager
    row = await m._row(m.db, c.node.job.job_id)
    if fault.startswith("anchor_"):
        from power_loop.runtime.execution_membership import scope_id
        key = scope_id(m, row["job_id"], 1)
        [event] = await m.db.fetchall(f"SELECT payload_json FROM {m.events} WHERE event_id=?", (key,))
        data = json.loads(event["payload_json"])
        data["owner" if fault == "anchor_owner" else "lease"]["boot_id"] = "another"
        await m.db.execute(f"UPDATE {m.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), key))
    else:
        col, value = {"parent": ("parent_job_id", c.actor.session_id), "role": ("execution_role", "main"),
            "creator": ("created_by_session_id", c.actor.session_id), "owner_proof": ("control_owner_boot_id", "other")}[fault]
        row = {**row, col: value}
    with pytest.raises(JobError, match="workflow_controller_mismatch"):
        await controller.for_node(m, m.db, row)


@pytest.mark.parametrize("port", ["send_create", "send_claim", "dispatch"])
async def test_nested_required_subagent_dispatch_also_checks_original_workflow(physical_node, port):
    c = physical_node
    m = c.manager
    sub = await m.register_job(c.controller, kind="subagent", parent=c.proof, system_key="required-audit")
    sid = await c.store.create_session(parent_session_id=c.controller.session_id, metadata={"execution_role": "subagent"})
    bound = await m.bind_session(c.controller, sub.job.job_id, sid)
    actor = replace(c.actor, session_id=sid)
    proof = await m.claim_job(actor, bound.job_id, executor_id="audit", boot_id="boot", expected_version=bound.version)
    if port != "send_create":
        reg = await m.register_job(actor, kind="send", parent=proof, session_id=sid, system_key="audit-send")
        if port == "dispatch":
            send = await m.claim_job(actor, reg.job.job_id, executor_id="audit-send", boot_id="boot", expected_version=1)
    await c.store.merge_session_metadata(c.controller.session_id, {"task_workflow_generation": 2})
    with pytest.raises(JobError, match="workflow_controller_mismatch"):
        if port == "send_create":
            await m.register_job(actor, kind="send", parent=proof, session_id=sid, system_key="audit-send")
        elif port == "send_claim":
            await m.claim_job(actor, reg.job.job_id, executor_id="audit-send", boot_id="boot", expected_version=1)
        else:
            await m.check_execution(actor, send)


async def test_logical_workflow_without_driver_has_no_physical_dispatch_authority(unbound):
    c = unbound
    await c.manager.check_execution(c.actor, c.proof)
    node = await c.manager.register_job(c.actor, kind="workflow_node", parent=c.proof, system_key="logical")
    proof = await c.manager.claim_job(c.actor, node.job.job_id, executor_id="logical", boot_id="boot", expected_version=1)
    await c.manager.check_execution(c.actor, proof)
    sid = await c.store.create_session(parent_session_id=c.actor.session_id, metadata={"execution_role": "workflow_node", "task_owner_job_id": proof.job_id})
    with pytest.raises(JobError, match="workflow_controller_missing"):
        await c.manager.bind_node_session(c.actor, proof, sid)


async def test_original_main_cannot_impersonate_driver_to_claim_a_physical_node(physical_node):
    c = physical_node
    wp = JobLease(**(await c.manager.get_workflow_controller(c.actor, c.wf.job.job_id, 1))["workflow"])
    reg = await c.manager.register_job(c.controller, kind="workflow_node", parent=wp, system_key="pending")
    with pytest.raises(JobError, match="session_ownership_mismatch"):
        await c.manager.claim_job(c.actor, reg.job.job_id, executor_id="wrong", boot_id="boot", expected_version=1)
    assert (await c.manager.get_job(c.actor, reg.job.job_id)).generation == 0


async def test_same_driver_cannot_be_adopted_by_another_workflow(unbound):
    c = unbound
    await c.manager.bind_workflow_driver(c.actor, c.proof, c.sid)
    other = await c.manager.register_job(c.actor, kind="workflow", parent=c.parent, independent=True, system_key="other-wf")
    proof = await c.manager.claim_job(c.actor, other.job.job_id, executor_id="other", boot_id="boot", expected_version=1)
    await c.store.merge_session_metadata(c.sid, {"task_owner_job_id": proof.job_id, "workflow_run_id": other.operation.operation_id[3:]})
    before = await rows(c)
    with pytest.raises(JobError, match="workflow_controller_already_bound"):
        await c.manager.bind_workflow_driver(c.actor, proof, c.sid)
    assert await rows(c) == before and (await c.manager.get_job(c.actor, proof.job_id)).session_id is None


async def test_unbound_logical_node_cannot_delegate_around_physical_driver_requirement(unbound):
    c = unbound
    m = c.manager
    node = await m.register_job(c.actor, kind="workflow_node", parent=c.proof, system_key="logical-node")
    proof = await m.claim_job(c.actor, node.job.job_id, executor_id="logical", boot_id="boot", expected_version=1)
    sub = await m.register_job(c.actor, kind="subagent", parent=proof, system_key="physical-child")
    sid = await c.store.create_session(parent_session_id=c.actor.session_id, metadata={"execution_role": "subagent"})
    bound = await m.bind_session(c.actor, sub.job.job_id, sid)
    before = await rows(c)
    with pytest.raises(JobError, match="workflow_controller_missing"):
        await m.claim_job(replace(c.actor, session_id=sid), bound.job_id, executor_id="child", boot_id="boot", expected_version=bound.version)
    assert await rows(c) == before and (await m.get_job(c.actor, bound.job_id)).generation == 0
