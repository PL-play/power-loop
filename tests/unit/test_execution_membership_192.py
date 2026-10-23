"""Historical execution ownership is independent of mutable live job control."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from power_loop.runtime import execution_membership as membership
from power_loop.runtime.jobs import JobError, JobManager

from .test_job_registry_192 import claimed, node_tree, registry_store, runtime

__all__ = ["registry_store", "runtime"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def rewrite(m, event_id, change):
    row = await m.db.fetchone(f"SELECT payload_json FROM {m.events} WHERE event_id=?", (event_id,))
    data = json.loads(row["payload_json"])
    change(data)
    await m.db.execute(f"UPDATE {m.events} SET payload_json=? WHERE event_id=?", (json.dumps(data), event_id))


async def test_frozen_edges_survive_live_rebinding_and_reopen_without_double_rollup(runtime):
    _, m, actor, root, rp, connect, _ = runtime
    wf, wp, node, np, _, sp = await node_tree(runtime)
    # A required subagent of a workflow node is nested below its model send.
    _, ap = await claimed(m, actor, "subagent", sp, "audit")
    _, asp = await claimed(m, actor, "send", ap, "audit-send")
    await m.close_model_execution(actor, asp, evidence={"usage": {"total_tokens": 7, "calls": 1}})
    await m.close_model_execution(actor, sp, evidence={"usage": {"total_tokens": 11, "calls": 1}})
    for proof in (ap, np, wp):
        await m.record_execution_result(actor, proof, outcome="succeeded", resource_state="stopped",
            result={"usage": {"total_tokens": 999999}, "text": "rollups must not be added"})
    await m.close_model_execution(actor, rp, evidence={"usage": {"total_tokens": 3, "calls": 1}})
    original = await m.get_execution_membership(actor, wf.job.job_id, 1)
    assert original["sealed"] and original["scope"]["owner"] is None
    assert [e["payload"]["member"] for e in original["members"]] == [asdict(np)]
    # Independent workflow is supervised by the original send but not controlled by it.
    assert await m.execution_usage(actor, root.job.job_id, generation=1) == {"total_tokens": 3, "calls": 1}
    assert await m.execution_usage(actor, root.job.job_id) == {"total_tokens": 21, "calls": 3}
    assert await m.execution_usage(actor, wf.job.job_id, generation=1) == {"total_tokens": 18, "calls": 2}
    revision = await m.get_result_revision(actor, node.job.job_id, 1)
    # Emulate a future controller rebind only in storage, never grant dispatch permission.
    await m.db.execute(f"UPDATE {m.jobs} SET control_owner_generation=99,control_owner_executor_id='future',"
        "executor_id='future',boot_id='future',generation=88 WHERE job_id=?", (node.job.job_id,))
    reopened = JobManager(await connect())
    assert await reopened.get_execution_membership(actor, wf.job.job_id, 1) == original
    assert await reopened.execution_usage(actor, wf.job.job_id, generation=1) == {"total_tokens": 18, "calls": 2}
    assert await reopened.execution_usage(actor, node.job.job_id, generation=1) == {"total_tokens": 18, "calls": 2}
    assert await reopened.get_result_revision(actor, node.job.job_id, 1) == revision


async def test_seal_is_not_supervision_completion_and_late_provider_exit_keeps_original_owner(runtime):
    _, m, actor, _, _, _, _ = runtime
    wf, wp, _, np, _, sp = await node_tree(runtime)
    await m.record_execution_result(actor, np, outcome="failed", result={"text": "interrupted"}, resource_state="unknown")
    await m.record_execution_result(actor, wp, outcome="failed", result={"text": "interrupted"}, resource_state="unknown")
    revision = await m.get_result_revision(actor, wf.job.job_id, 1)
    before = await m.get_execution_membership(actor, wf.job.job_id, 1)
    assert before["sealed"] and await m.execution_usage(actor, wf.job.job_id, generation=1) == {}
    await m.request_cancel(actor, wf.job.job_id, system_key="late-stop", reason="stop outstanding provider")
    # Actual exit evidence remains admissible even when the scope has failed/cancelled.
    await m.close_model_execution(actor, sp, evidence={"usage": {"total_tokens": 17}})
    assert await m.execution_usage(actor, wf.job.job_id, generation=1) == {"total_tokens": 17}
    assert await m.get_execution_membership(actor, wf.job.job_id, 1) == before
    assert await m.get_result_revision(actor, wf.job.job_id, 1) == revision
    assert (await m.get_job(actor, wf.job.job_id)).supervision_status == "open"


async def test_model_close_and_result_replays_do_not_duplicate_members_or_rewrite_usage(runtime):
    _, m, actor, _, _, _, _ = runtime
    wf, wp, _, np, _, sp = await node_tree(runtime)
    await m.close_model_execution(actor, sp, evidence={"usage": {"total_tokens": 4}})
    await m.close_model_execution(actor, sp, evidence={"usage": {"total_tokens": 400}})
    for _ in range(2):
        await m.record_execution_result(actor, np, outcome="succeeded", result={"text": "done"}, resource_state="stopped")
        await m.record_execution_result(actor, wp, outcome="succeeded", result={"text": "done"}, resource_state="stopped")
    assert await m.execution_usage(actor, wf.job.job_id, generation=1) == {"total_tokens": 4}
    saved = await m.get_execution_membership(actor, wf.job.job_id, 1)
    assert saved["sealed"] and len(saved["members"]) == 1


async def test_storage_only_owner_generations_partition_the_same_logical_node(runtime):
    _, m, actor, _, _, _, _ = runtime
    wf, wp, node, np, _, sp = await node_tree(runtime)
    await m.close_model_execution(actor, sp, evidence={"usage": {"total_tokens": 13}})
    await m.record_execution_result(actor, wp, outcome="failed", result={}, resource_state="unknown")
    # Storage-level fixtures for the future protocol. This does not enable a
    # finished workflow, mutate live leases, call a model or authorize dispatch.
    async with m._write() as tx:
        w2 = {**await m._row(tx, wp.job_id), "generation": 2, "executor_id": "future-owner"}
        n2 = {**await m._row(tx, np.job_id), "generation": 2, "control_owner_generation": 2,
            "control_owner_executor_id": "future-owner"}
        s2 = {**await m._row(tx, sp.job_id), "job_id": "history-only-send-2", "control_owner_generation": 2}
        for row in (w2, n2, s2):
            await membership.start(m, tx, row)
        await membership._insert(m, tx, "model_closed_"+s2["job_id"], s2["job_id"], "model_closed", 3,
            {"generation": 1, "evidence": {"usage": {"total_tokens": 17}}, "resource_state": "stopped"})
        for row in (s2, n2, w2):
            await membership.seal(m, tx, row)
    assert await m.execution_usage(actor, wf.job.job_id, generation=1) == {"total_tokens": 13}
    assert await m.execution_usage(actor, wf.job.job_id, generation=2) == {"total_tokens": 17}
    assert await m.execution_usage(actor, node.job.job_id, generation=1) == {"total_tokens": 13}
    assert await m.execution_usage(actor, node.job.job_id, generation=2) == {"total_tokens": 17}
    assert (await m.get_job(actor, wf.job.job_id)).generation == 1


@pytest.mark.parametrize("generation", [True, False, 0, -1, 1.0, "1", None, 2**63])
async def test_membership_rejects_invalid_generation(runtime, generation):
    _, m, actor, root, _, _, _ = runtime
    with pytest.raises(JobError, match="invalid_result_generation"):
        await m.get_execution_membership(actor, root.job.job_id, generation)
    if generation is not None:
        with pytest.raises(JobError, match="invalid_result_generation"):
            await m.execution_usage(actor, root.job.job_id, generation=generation)


@pytest.mark.parametrize("field", ["tenant_id", "principal_id", "session_id"])
async def test_membership_history_does_not_bypass_authorization(runtime, field):
    store, m, actor, root, _, _, _ = runtime
    stranger = await store.create_session()
    wrong = replace(actor, **{field: stranger})
    for method in (m.get_execution_membership, m.execution_usage):
        with pytest.raises(JobError, match="access_denied"):
            await method(wrong, root.job.job_id, generation=1)


@pytest.mark.parametrize("fault", ["scope_missing", "scope_json", "scope_array", "scope_version", "scope_tenant",
    "scope_generation", "scope_lease", "scope_kind", "scope_role", "scope_creator", "scope_event_type",
    "scope_event_job", "scope_status", "edge_missing", "edge_version", "edge_owner", "edge_member",
    "edge_lease", "edge_extra", "seal_missing", "seal_hash", "model_generation", "model_evidence", "model_usage",
    "scope_parent", "scope_session", "scope_nonfinite"])
async def test_corrupt_history_is_an_error_never_zero_usage(runtime, fault):
    _, m, actor, _, _, _, _ = runtime
    wf, wp, node, _, _, sp = await node_tree(runtime)
    await m.close_model_execution(actor, sp, evidence={"usage": {"total_tokens": 13}})
    await m.record_execution_result(actor, wp, outcome="failed", result={}, resource_state="unknown")
    scope = membership.scope_id(m, wf.job.job_id, 1)
    edge = membership.member_id(m, node.job.job_id, 1)
    seal = membership.seal_id(m, wf.job.job_id, 1)
    model = "model_closed_" + sp.job_id
    if fault.endswith("_missing"):
        key = {"scope_missing": scope, "edge_missing": edge, "seal_missing": seal}[fault]
        await m.db.execute(f"DELETE FROM {m.events} WHERE event_id=?", (key,))
    elif fault in ("scope_json", "scope_array", "scope_nonfinite"):
        await m.db.execute(f"UPDATE {m.events} SET payload_json=? WHERE event_id=?", (
            {"scope_json": "{broken", "scope_array": "[]", "scope_nonfinite": '{"version":NaN}'}[fault], scope))
    elif fault in ("scope_event_type", "scope_event_job", "scope_status"):
        column, value = {"scope_event_type": ("event_type", "execution_member"),
            "scope_event_job": ("job_id", node.job.job_id), "scope_status": ("delivery_status", "handled")}[fault]
        await m.db.execute(f"UPDATE {m.events} SET {column}=? WHERE event_id=?", (value, scope))
    else:
        key, path, value = {
            "scope_version": (scope, ["version"], True), "scope_tenant": (scope, ["tenant_id"], "foreign"),
            "scope_generation": (scope, ["lease", "generation"], 2), "scope_lease": (scope, ["lease", "boot_id"], ""),
            "scope_kind": (scope, ["kind"], "other"), "scope_role": (scope, ["execution_role"], "other"),
            "scope_creator": (scope, ["created_by_session_id"], None),
            "scope_parent": (scope, ["parent_job_id"], []), "scope_session": (scope, ["session_id"], ""),
            "edge_version": (edge, ["version"], 3), "edge_owner": (edge, ["owner", "executor_id"], "other"),
            "edge_member": (edge, ["member", "generation"], 2), "edge_lease": (edge, ["member", "generation"], False),
            "edge_extra": (edge, ["unexpected"], 1), "seal_hash": (seal, ["members_hash"], "wrong"),
            "model_generation": (model, ["generation"], 99), "model_evidence": (model, ["evidence"], []),
            "model_usage": (model, ["evidence", "usage"], [])}[fault]
        def damage(data):
            for part in path[:-1]:
                data = data[part]
            data[path[-1]] = value
        await rewrite(m, key, damage)
    with pytest.raises(JobError, match="execution_membership"):
        await m.execution_usage(actor, wf.job.job_id, generation=1)


@pytest.mark.parametrize("point", ["execution_scope", "execution_member"])
async def test_claim_membership_failure_rolls_back_lease_and_all_edges(runtime, monkeypatch, point):
    _, m, actor, _, parent, _, _ = runtime
    child = await m.register_job(actor, kind="subagent", parent=parent, system_key="pending-child")
    original = membership._insert
    async def fail(*args, **kwargs):
        await original(*args, **kwargs)
        if args[4] == point:
            raise RuntimeError("injected membership write failure")
    monkeypatch.setattr(membership, "_insert", fail)
    with pytest.raises(RuntimeError, match="injected"):
        await m.claim_job(actor, child.job.job_id, executor_id="e", boot_id="b", expected_version=1)
    saved = await m.get_job(actor, child.job.job_id)
    assert saved.status == "queued" and saved.generation == 0 and saved.resource_state == "not_started" and saved.version == 1
    assert not await m.db.fetchone(f"SELECT event_id FROM {m.events} WHERE event_id IN (?,?)", (
        membership.scope_id(m, saved.job_id, 1), membership.member_id(m, saved.job_id, 1)))


@pytest.mark.parametrize("close", ["model", "scope"])
async def test_seal_failure_rolls_back_exit_and_retry_saves_once(runtime, monkeypatch, close):
    _, m, actor, root, rp, _, _ = runtime
    if close == "scope":
        root, rp = await claimed(m, actor, "subagent", rp, "scope")
    previous = await m.get_job(actor, rp.job_id)
    original = membership._insert
    async def fail(*args, **kwargs):
        await original(*args, **kwargs)
        if args[4] == "execution_membership_seal":
            raise RuntimeError("injected seal write failure")
    monkeypatch.setattr(membership, "_insert", fail)
    async def finish():
        if close == "model":
            return await m.close_model_execution(actor, rp, evidence={"usage": {"total_tokens": 9}})
        return await m.record_execution_result(actor, rp, outcome="succeeded", result={}, resource_state="stopped")
    with pytest.raises(RuntimeError, match="injected"):
        await finish()
    assert await m.get_job(actor, rp.job_id) == previous
    assert not (await m.get_execution_membership(actor, rp.job_id, 1))["sealed"]
    monkeypatch.setattr(membership, "_insert", original)
    await finish()
    assert (await m.get_execution_membership(actor, rp.job_id, 1))["sealed"]


async def test_legacy_exit_is_preserved_without_inventing_historical_membership(runtime):
    _, m, actor, root, rp, _, _ = runtime
    await m.db.execute(f"DELETE FROM {m.events} WHERE event_id=?", (membership.scope_id(m, root.job.job_id, 1),))
    await m.close_model_execution(actor, rp, evidence={"usage": {"total_tokens": 23}})
    assert await m.execution_usage(actor, root.job.job_id) == {"total_tokens": 23}
    with pytest.raises(JobError, match="execution_membership_missing"):
        await m.execution_usage(actor, root.job.job_id, generation=1)
    with pytest.raises(JobError, match="execution_membership_missing"):
        async with m._write() as tx:
            await membership.require_current(m, tx, await m._row(tx, root.job.job_id))


@pytest.mark.parametrize("fault", ["owner_missing", "owner_wrong_boot", "owner_foreign", "owner_sealed", "scope_conflict"])
async def test_child_claim_requires_original_unsealed_owner_and_atomic_anchor(runtime, fault):
    _, m, actor, root, rp, _, _ = runtime
    child = await m.register_job(actor, kind="subagent", parent=rp, system_key="queued-child")
    key = membership.scope_id(m, root.job.job_id, 1)
    if fault == "owner_missing":
        await m.db.execute(f"DELETE FROM {m.events} WHERE event_id=?", (key,))
    elif fault == "owner_sealed":
        async with m._write() as tx:
            await membership.seal(m, tx, await m._row(tx, root.job.job_id))
    elif fault == "scope_conflict":
        row = await m.db.fetchone(f"SELECT payload_json FROM {m.events} WHERE event_id=?", (key,))
        await m.db.execute(f"INSERT INTO {m.events} (event_id,job_id,event_type,job_version,payload_json,delivery_status,created_at) VALUES (?,?,?,?,?,?,?)",
            (membership.scope_id(m, child.job.job_id, 1), child.job.job_id, "execution_scope", 1, row["payload_json"], "not_applicable", m.now()))
    else:
        await rewrite(m, key, lambda d: d["lease"].update(boot_id="foreign") if fault == "owner_wrong_boot" else d.update(tenant_id="foreign"))
    with pytest.raises(JobError, match="execution_membership"):
        await m.claim_job(actor, child.job.job_id, executor_id="e", boot_id="b", expected_version=1)
    assert (await m.get_job(actor, child.job.job_id)).generation == 0


@pytest.mark.parametrize("fault", ["unsealed", "current_mismatch", "child_missing", "child_owner", "child_proof", "child_tenant"])
async def test_new_generation_requires_complete_original_execution_evidence(runtime, fault):
    _, m, actor, _, _, _, _ = runtime
    wf, wp, node, np, _, sp = await node_tree(runtime)
    if fault != "unsealed":
        await m.record_execution_result(actor, wp, outcome="failed", result={}, resource_state="unknown")
    if fault == "current_mismatch":
        await m.db.execute(f"UPDATE {m.jobs} SET boot_id='another-boot' WHERE job_id=?", (wp.job_id,))
    elif fault == "child_missing":
        await m.db.execute(f"DELETE FROM {m.events} WHERE event_id=?", (membership.scope_id(m, np.job_id, 1),))
    elif fault.startswith("child_"):
        def change(d):
            if fault == "child_owner":
                d["owner"]["boot_id"] = "other"
            elif fault == "child_proof":
                d["lease"]["boot_id"] = "other"
            else:
                d["tenant_id"] = "foreign"
        await rewrite(m, membership.scope_id(m, node.job.job_id, 1), change)
    code = {"unsealed": "execution_membership_unsealed", "current_mismatch": "execution_membership_mismatch",
        "child_missing": "execution_membership_missing", "child_owner": "invalid_execution_membership",
        "child_proof": "execution_membership_corrupt", "child_tenant": "execution_membership_mismatch"}[fault]
    with pytest.raises(JobError, match=code):
        async with m._write() as tx:
            await membership.require_current(m, tx, await m._row(tx, wf.job.job_id))


@pytest.mark.parametrize("fault", ["cycle", "depth"])
async def test_corrupt_membership_graph_has_bounded_traversal(runtime, fault):
    _, m, actor, root, rp, _, _ = runtime
    data = (await m.get_execution_membership(actor, root.job.job_id, 1))["scope"]
    # Persist a deliberately impossible graph; do not grant leases or run tasks.
    async with m._write() as tx:
        previous = asdict(rp)
        for index in range(65 if fault == "depth" else 1):
            proof = {**asdict(rp), "job_id": "corrupt-"+str(index)}
            await membership._insert(m, tx, membership.scope_id(m, proof["job_id"], 1), proof["job_id"], "execution_scope", 1,
                {**data, "lease": proof, "owner": previous})
            await membership._insert(m, tx, membership.member_id(m, proof["job_id"], 1), previous["job_id"], "execution_member", 1,
                {"version": 1, "owner": previous, "member": proof})
            previous = proof
        if fault == "cycle":
            await membership._insert(m, tx, membership.member_id(m, rp.job_id, 1), previous["job_id"], "execution_member", 1,
                {"version": 1, "owner": previous, "member": asdict(rp)})
    with pytest.raises(JobError, match="job_depth_exceeded" if fault == "depth" else "invalid_execution_membership"):
        await m.execution_usage(actor, root.job.job_id, generation=1)
