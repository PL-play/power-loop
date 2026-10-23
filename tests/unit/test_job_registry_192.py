"""B1 acceptance on SQLite and explicitly configured isolated server test DBs."""
from __future__ import annotations

import asyncio
import json
import os
import secrets
from dataclasses import replace
from urllib.parse import urlparse

import pytest

from power_loop.runtime.jobs import JobActor, JobError, JobLease, JobManager
from power_loop.runtime.store.factory import open_store
from power_loop.runtime.store.schema import _STORE_TABLES
from power_loop.runtime.store.task_schema import TASK_TABLES, migrate_task_schema

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture(params=["sqlite", "postgres", "mysql"])
async def registry_store(request, tmp_path):
    backend = request.param
    dsn = str(tmp_path / "registry.db") if backend == "sqlite" else os.environ.get(
        "POWER_LOOP_TASK_PG_DSN" if backend == "postgres" else "POWER_LOOP_TASK_MYSQL_DSN")
    if not dsn:
        pytest.skip(f"isolated {backend} test DSN not configured")
    if backend != "sqlite" and not urlparse(dsn).path.endswith("_test"):
        pytest.fail("server registry tests refuse a database without _test suffix")
    prefix = "j192_" + secrets.token_hex(4) + "_"
    stores, prefixes = [], set()

    async def connect(*, schema="tasks", suffix=""):
        active_prefix = prefix + suffix
        s = await open_store(dsn, table_prefix=active_prefix, schema=schema)
        stores.append(s)
        prefixes.add(active_prefix)
        return s

    store = await connect()
    try:
        yield store, connect
    finally:
        if backend != "sqlite":
            for active_prefix in prefixes:
                for name in (*TASK_TABLES, *_STORE_TABLES, "schema_migrations"):
                    await store._db.execute(f"DROP TABLE IF EXISTS {active_prefix}{name}")
        for s in reversed(stores):
            await s.close()


@pytest.fixture
async def runtime(registry_store):
    store, connect = registry_store
    now = [1_000_000]
    manager = JobManager(store, now_ms=lambda: now[0])
    sid = await store.create_session()
    actor = JobActor("tenant", "host", sid)
    root = await manager.register_job(actor, kind="send", session_id=sid, system_key="durable-root-key")
    lease = await manager.claim_job(actor, root.job.job_id, executor_id="executor", boot_id="boot",
                                    expected_version=1)
    return store, manager, actor, root, lease, connect, now


async def call(store, actor, call_id="call-1"):
    seq = await store.append_message(actor.session_id, role="assistant", tool_calls=[{
        "id": call_id, "type": "function", "function": {"name": "task", "arguments": "{}"}}])
    return await store.message_ref(actor.session_id, seq)


async def child(runtime, *, key="child", kind="subagent", independent=False):
    _, manager, actor, _, lease, _, _ = runtime
    return await manager.register_job(actor, kind=kind, parent=lease,
                                      system_key=key, independent=independent)


async def claimed(m, actor, kind, parent, key, **kwargs):
    reg = await m.register_job(actor, kind=kind, parent=parent, system_key=key, **kwargs)
    proof = await m.claim_job(actor, reg.job.job_id, executor_id=key, boot_id="boot", expected_version=1)
    return reg, proof


async def node_tree(runtime):
    _, m, actor, _, root_proof, _, _ = runtime
    workflow, wp = await claimed(m, actor, "workflow", root_proof, "workflow", independent=True)
    node, np = await claimed(m, actor, "workflow_node", wp, "node")
    send, sp = await claimed(m, actor, "send", np, "node-send")
    return workflow, wp, node, np, send, sp


async def test_session_binding_is_direct_child_role_checked_immutable_and_survives_caller_close(runtime):
    store, m, actor, _, caller, _, _ = runtime
    reg = await child(runtime, independent=True)
    wrong = await store.create_session(parent_session_id=actor.session_id, metadata={"execution_role": "main"})
    with pytest.raises(JobError, match="ownership"):
        await m.bind_session(actor, reg.job.job_id, wrong)
    sid = await store.create_session(parent_session_id=actor.session_id, metadata={"execution_role": "subagent"})
    other = await store.create_session(parent_session_id=actor.session_id, metadata={"execution_role": "subagent"})
    await m.close_model_execution(actor, caller)
    bound = await m.bind_session(actor, reg.job.job_id, sid)
    assert bound.session_id == sid and bound.version == 2
    assert (await m.bind_session(actor, reg.job.job_id, sid)) == bound
    with pytest.raises(JobError, match="already_bound"):
        await m.bind_session(actor, reg.job.job_id, other)
    proof = await m.claim_job(actor, bound.job_id, executor_id="executor", boot_id="boot", expected_version=2)
    child_actor = JobActor(actor.tenant_id, actor.principal_id, sid)
    sub_send = await m.register_job(child_actor, kind="send", session_id=sid, parent=proof, system_key="child-send")
    assert sub_send.job.execution_role == "subagent"


@pytest.mark.parametrize("barrier", ["expired", "cancelled"])
async def test_renewal_cannot_erase_barriers_but_actual_model_exit_evidence_is_saved(runtime, barrier):
    _, m, actor, root, proof, _, now = runtime
    await m.renew_execution(actor, proof, lease_ms=1000)
    if barrier == "expired":
        now[0] += 1001
    else:
        await m.request_cancel(actor, root.job.job_id, system_key="stop", reason="stop")
    with pytest.raises(JobError, match="expired|cancel"):
        await m.renew_execution(actor, proof)
    closed = await m.close_model_execution(actor, proof, evidence={"usage": {"total_tokens": 17}})
    assert closed.status == "model_closed" and closed.resource_state == "stopped"
    assert (await m.close_model_execution(actor, proof)).version == closed.version
    event = await m.db.fetchone(f"SELECT payload_json FROM {m.events} WHERE event_id=?", ("model_closed_" + root.job.job_id,))
    assert json.loads(event["payload_json"])["evidence"]["usage"]["total_tokens"] == 17
    with pytest.raises(JobError, match="cancel|closed|not_running"):
        await m.check_execution(actor, proof)


@pytest.mark.parametrize("barrier", ["expired", "cancelled", "ancestor_cancelled"])
async def test_result_evidence_keeps_usage_after_stop_or_expiry_without_claiming_final_success(runtime, barrier):
    _, m, actor, root, caller, _, now = runtime
    reg = await child(runtime, independent=True)
    proof = await m.claim_job(actor, reg.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
    if barrier == "expired":
        now[0] += 30001
    else:
        await m.request_cancel(actor, root.job.job_id if barrier == "ancestor_cancelled" else reg.job.job_id,
                               system_key="stop", reason="stop")
    payload = {"status": "completed", "usage": {"total_tokens": 17}, "final_text": "actual output"}
    event = await m.record_execution_result(actor, proof, outcome="succeeded", result=payload, resource_state="stopped")
    assert await m.record_execution_result(actor, proof, outcome="succeeded", result=payload, resource_state="stopped") == event
    result = await m.get_job(actor, reg.job.job_id)
    assert result.status == "settling" and result.supervision_status == "open" and result.result["usage"]["total_tokens"] == 17
    assert result.outcome == ("succeeded" if barrier == "expired" else "cancelled")
    if barrier != "expired":
        assert result.result["execution_status"] == "completed" and result.result["status"] == "cancelled"
    with pytest.raises(JobError, match="cancel|closed|not_running"):
        await m.register_job(actor, kind="background_tool", parent=proof, system_key="after-result-must-not-run")


async def test_actual_receipt_can_be_reconciled_after_caller_closes_without_reopening_execution(runtime):
    store, m, actor, _, caller, _, _ = runtime
    origin = await call(store, actor)
    reg = await m.register_job(actor, kind="subagent", parent=caller, call_ref=origin,
                               tool_call_id="call-1", independent=True)
    assert (await m.operation_for_call(actor, origin, "call-1")).operation_id == reg.operation.operation_id
    seq = await store.append_message(actor.session_id, role="tool", tool_call_id="call-1", content="accepted")
    receipt = await store.message_ref(actor.session_id, seq)
    await m.close_model_execution(actor, caller)
    reconciled = await m.reconcile_receipt(actor, reg.operation.operation_id, receipt)
    assert reconciled.receipt_ref == receipt.to_dict()
    with pytest.raises(JobError, match="closed|not_running"):
        await m.bind_receipt(actor, reg.operation.operation_id, receipt, caller=caller)
    other = await store.create_session()
    with pytest.raises(JobError, match="denied"):
        await m.reconcile_receipt(JobActor(actor.tenant_id, actor.principal_id, other), reg.operation.operation_id, receipt)


async def test_async_workflow_outlives_closed_creator_and_survives_reopen(runtime):
    store, m, actor, root, caller, connect, now = runtime
    reg = await child(runtime, key="async-workflow", kind="workflow", independent=True)
    await m.close_model_execution(actor, caller)
    now[0] += 31_000  # Original model execution lease has expired.
    recovered = JobManager(await connect(), now_ms=m.now)
    proof = await recovered.claim_job(actor, reg.job.job_id, executor_id="engine", boot_id="boot", expected_version=1)
    node = await recovered.register_job(actor, kind="workflow_node", parent=proof, system_key="new-node")
    saved = await recovered.get_job(actor, reg.job.job_id)
    assert saved.parent_job_id == root.job.job_id and not saved.blocks_completion
    assert saved.control_owner_job_id is None and node.job.control_owner_job_id == saved.job_id
    assert (await recovered.get_job(actor, root.job.job_id)).status == "model_closed"
    with pytest.raises(JobError, match="closed|not_running"):
        await recovered.register_job(actor, kind="workflow", parent=caller, system_key="stale-caller", independent=True)
    assert (await store._db.fetchone(f"SELECT COUNT(*) AS n FROM {m.jobs}"))["n"] == 3


async def test_node_subagent_depends_on_node_scope_instead_of_creator_model_send(runtime):
    store, m, actor, _, caller, connect, now = runtime
    _, wp, node, np, send, sp = await node_tree(runtime)
    reg = await m.register_job(actor, kind="subagent", parent=sp, system_key="audit")
    assert reg.job.control_owner_job_id == node.job.job_id
    assert reg.job.completion_scope_job_id == node.job.job_id and reg.job.blocks_completion
    await m.close_model_execution(actor, sp)
    await m.close_model_execution(actor, caller)
    await store._db.execute(f"UPDATE {m.jobs} SET lease_until_ms=?,generation=generation+1 WHERE job_id=?",
                            (now[0]-1, send.job.job_id))
    recovered = JobManager(await connect(), now_ms=m.now)
    ap = await recovered.claim_job(actor, reg.job.job_id, executor_id="audit", boot_id="boot", expected_version=1)
    bg = await recovered.register_job(actor, kind="background_tool", parent=ap, system_key="ordinary-tool")
    assert bg.job.execution_role == "subagent" and bg.job.control_owner_job_id == reg.job.job_id
    assert wp.job_id != np.job_id


@pytest.mark.parametrize("case", ["expired", "new-generation", "cancelled"])
async def test_actual_node_control_owner_still_fences_subagent(runtime, case):
    store, m, actor, _, _, _, now = runtime
    _, _, node, np, _, sp = await node_tree(runtime)
    reg, ap = await claimed(m, actor, "subagent", sp, "audit")
    await m.close_model_execution(actor, sp)
    if case == "expired":
        await store._db.execute(f"UPDATE {m.jobs} SET lease_until_ms=? WHERE job_id=?", (now[0]-1, np.job_id))
    elif case == "new-generation":
        await store._db.execute(f"UPDATE {m.jobs} SET generation=generation+1 WHERE job_id=?", (np.job_id,))
    else:
        await m.request_cancel(actor, node.job.job_id, system_key="cancel-owner", reason="stop")
    with pytest.raises(JobError, match="expired|stale|cancel"):
        await m.register_job(actor, kind="background_tool", parent=ap, system_key="must-not-run")
    assert (await m.get_job(actor, reg.job.job_id)).control_owner_job_id == node.job.job_id


async def test_closed_creator_cancel_still_stops_detached_work_after_reopen(runtime):
    _, m, actor, root, caller, connect, _ = runtime
    reg = await child(runtime, kind="workflow", independent=True)
    await m.close_model_execution(actor, caller)
    recovered = JobManager(await connect(), now_ms=m.now)
    await recovered.request_cancel(actor, root.job.job_id, system_key="stop-tree", reason="user stop")
    with pytest.raises(JobError, match="cancel"):
        await recovered.claim_job(actor, reg.job.job_id, executor_id="late", boot_id="boot", expected_version=1)


@pytest.mark.parametrize("kind", ["workflow", "subagent"])
@pytest.mark.parametrize("through_send", [False, True])
async def test_subagent_cannot_delegate_recursively_even_through_another_send(runtime, kind, through_send):
    _, m, actor, _, parent, _, _ = runtime
    _, ap = await claimed(m, actor, "subagent", parent, "child")
    if through_send:
        _, ap = await claimed(m, actor, "send", ap, "child-send")
    with pytest.raises(JobError, match="role_forbids"):
        await m.register_job(actor, kind=kind, parent=ap, system_key="recursive", independent=True)


async def test_workflow_node_can_spawn_child_but_cannot_spawn_workflow(runtime):
    _, m, actor, _, _, _, _ = runtime
    _, _, _, _, _, sp = await node_tree(runtime)
    audit = await m.register_job(actor, kind="subagent", parent=sp, system_key="allowed-audit")
    assert audit.job.execution_role == "subagent"
    with pytest.raises(JobError, match="role_forbids"):
        await m.register_job(actor, kind="workflow", parent=sp, system_key="nested-workflow")
    with pytest.raises(JobError, match="required_child"):
        await m.register_job(actor, kind="subagent", parent=sp, system_key="escape-node", independent=True)


async def test_close_model_execution_is_idempotent_and_is_not_tree_completion(runtime):
    _, m, actor, root, proof, _, _ = runtime
    await child(runtime, kind="workflow", independent=True)
    await m.close_model_execution(actor, proof)
    await m.close_model_execution(actor, proof)
    closed = await m.get_job(actor, root.job.job_id)
    assert closed.status == "model_closed" and closed.resource_state == "stopped"
    assert closed.supervision_status == "open" and closed.outcome is None
    with pytest.raises(JobError, match="closed|not_running"):
        await m.register_job(actor, kind="background_tool", parent=proof, system_key="old-send")


async def test_only_model_send_can_use_model_close_and_stale_proof_is_rejected(runtime):
    _, m, actor, _, proof, _, _ = runtime
    _, wp = await claimed(m, actor, "workflow", proof, "workflow", independent=True)
    with pytest.raises(JobError, match="not_model_send"):
        await m.close_model_execution(actor, wp)
    with pytest.raises(JobError, match="stale"):
        await m.close_model_execution(actor, replace(proof, generation=0))


async def test_registration_is_durable_and_execution_result_is_not_a_success_gate(runtime):
    store, manager, actor, root, lease, connect, _ = runtime
    reg = await child(runtime)
    assert reg.job.parent_job_id == root.job.job_id and reg.job.root_job_id == root.job.job_id
    assert reg.job.completion_scope_job_id == root.job.job_id and reg.job.blocks_completion
    proof = await manager.claim_job(actor, reg.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
    event = await manager.record_result(actor, proof, outcome="succeeded", result={"answer": 42})
    assert event == await manager.record_result(actor, proof, outcome="succeeded", result={"answer": 42})
    recovered = JobManager(await connect(), now_ms=manager.now)
    saved = await recovered.get_job(actor, reg.job.job_id)
    assert saved.status == "settling" and saved.outcome == "succeeded" and saved.result == {"answer": 42}
    assert saved.supervision_status == "open" and saved.result_status == "pending_delivery"
    assert await recovered.get_operation(actor, reg.operation.operation_id) == reg.operation
    with pytest.raises(JobError, match="result_conflict"):
        await manager.record_result(actor, proof, outcome="succeeded", result={"answer": 43})


async def test_concurrent_replayed_call_creates_only_one_job_and_operation(runtime):
    store, m, actor, _, lease, connect, _ = runtime
    ref = await call(store, actor)
    other = JobManager(await connect(), now_ms=m.now)
    results = await asyncio.gather(*(manager.register_job(actor, kind="subagent", parent=lease,
        call_ref=ref, tool_call_id="call-1", request={"task": "audit"}) for manager in [m, other] * 4))
    assert len({r.job.job_id for r in results}) == 1 and sum(not r.replayed for r in results) == 1
    assert len({r.operation.operation_id for r in results}) == 1
    assert (await store._db.fetchone(f"SELECT COUNT(*) AS n FROM {m.jobs}"))["n"] == 2
    with pytest.raises(JobError, match="operation_conflict"):
        await m.register_job(actor, kind="subagent", parent=lease,
            call_ref=ref, tool_call_id="call-1", request={"task": "different task"})
    with pytest.raises(JobError, match="operation_conflict"):
        await m.register_job(actor, kind="subagent", parent=lease,
            call_ref=replace(ref, id=ref.id + 99), tool_call_id="call-1", request={"task": "audit"})


async def test_claim_has_one_winner_across_connections(runtime):
    _, m, actor, _, _, connect, _ = runtime
    reg = await child(runtime)
    other = JobManager(await connect(), now_ms=m.now)
    results = await asyncio.gather(*(manager.claim_job(actor, reg.job.job_id, executor_id=f"e{i}",
        boot_id="boot", expected_version=1) for i, manager in enumerate((m, other))), return_exceptions=True)
    assert len([r for r in results if isinstance(r, JobLease)]) == 1
    assert len([r for r in results if isinstance(r, JobError) and r.code == "job_claim_conflict"]) == 1
    assert (await m.get_job(actor, reg.job.job_id)).generation == 1


async def test_call_and_receipt_are_bound_by_actual_message_ids(runtime):
    store, m, actor, _, lease, _, _ = runtime
    ref = await call(store, actor)
    reg = await m.register_job(actor, kind="background_tool", parent=lease, call_ref=ref, tool_call_id="call-1")
    seq = await store.append_message(actor.session_id, role="tool", tool_call_id="call-1", content="accepted")
    receipt = await store.message_ref(actor.session_id, seq)
    op = await m.bind_receipt(actor, reg.operation.operation_id, receipt, caller=lease)
    assert await m.bind_receipt(actor, op.operation_id, receipt, caller=lease) == op
    assert op.call_ref["id"] == str(ref.id) and op.receipt_ref["id"] == str(receipt.id)
    assert op.call_ref["id"] != op.receipt_ref["id"]


async def test_tool_call_id_reuse_does_not_merge_different_messages_or_steering(runtime):
    store, m, actor, _, lease, _, _ = runtime
    reg = await child(runtime)
    target = await m.claim_job(actor, reg.job.job_id, executor_id="target", boot_id="boot", expected_version=1)
    ops = []
    for number in (1, 2):
        ref = await call(store, actor, "reused")
        op = await m.register_operation(actor, target, caller=lease, action="steer", call_ref=ref,
            tool_call_id="reused", request={"message": str(number)}, result={"accepted": True})
        seq = await store.append_message(actor.session_id, role="tool", tool_call_id="reused", content="accepted")
        ops.append(await m.bind_receipt(actor, op.operation_id,
                                        await store.message_ref(actor.session_id, seq), caller=lease))
    assert ops[0].operation_id != ops[1].operation_id and ops[0].call_ref != ops[1].call_ref
    assert ops[0].job_id == ops[1].job_id == reg.job.job_id


@pytest.mark.parametrize("case", ["other-session", "wrong-tool-call", "wrong-role", "later-reused-call"])
async def test_receipt_rejects_wrong_origin_without_overwriting_operation(runtime, case):
    store, m, actor, _, lease, _, _ = runtime
    ref = await call(store, actor)
    reg = await m.register_job(actor, kind="subagent", parent=lease, call_ref=ref, tool_call_id="call-1")
    sid, role, tool_call = actor.session_id, "tool", "call-1"
    if case == "other-session":
        sid = await store.create_session()
    elif case == "wrong-tool-call":
        tool_call = "foreign-call"
    elif case == "wrong-role":
        role = "user"
    else:
        await call(store, actor)
    seq = await store.append_message(sid, role=role, tool_call_id=tool_call, content="receipt")
    with pytest.raises(JobError):
        await m.bind_receipt(actor, reg.operation.operation_id, await store.message_ref(sid, seq), caller=lease)
    assert (await m.get_operation(actor, reg.operation.operation_id)).receipt_ref is None


async def test_scope_boundaries_and_independent_edges_remain_distinct_from_tree(runtime):
    _, m, actor, root, _, _, _ = runtime
    outer = await child(runtime)
    proof = await m.claim_job(actor, outer.job.job_id, executor_id="outer", boot_id="boot", expected_version=1)
    inner = await m.register_job(actor, kind="background_tool", parent=proof, system_key="inner")
    assert inner.job.root_job_id == root.job.job_id
    assert inner.job.completion_scope_job_id == outer.job.job_id
    service = await child(runtime, key="service", kind="background_tool", independent=True)
    assert not service.job.blocks_completion and service.job.completion_scope_job_id == service.job.job_id
    service_proof = await m.claim_job(actor, service.job.job_id, executor_id="service", boot_id="boot", expected_version=1)
    follower = await m.register_job(actor, kind="background_tool", parent=service_proof, system_key="follower")
    assert follower.job.completion_scope_job_id == service.job.job_id and follower.job.blocks_completion


@pytest.mark.parametrize("field", ["tenant_id", "principal_id", "session_id"])
async def test_foreign_or_sibling_actor_cannot_read_or_cancel(runtime, field):
    _, m, actor, root, _, _, _ = runtime
    foreign = replace(actor, **{field: "foreign"})
    with pytest.raises(JobError, match="access_denied"):
        await m.get_job(foreign, root.job.job_id)
    with pytest.raises(JobError, match="access_denied"):
        await m.request_cancel(foreign, root.job.job_id, system_key="stop", reason="foreign")


async def test_cancelled_ancestor_blocks_new_admission_and_queued_child_claim(runtime):
    _, m, actor, root, lease, _, _ = runtime
    reg = await child(runtime)
    stop = await m.request_cancel(actor, root.job.job_id, system_key="stop-1", reason="user stopped")
    assert stop.result["status"] == "cancelling" and stop.result["resource_state"] == "running"
    assert await m.request_cancel(actor, root.job.job_id, system_key="stop-1", reason="user stopped") == stop
    with pytest.raises(JobError, match="cancel_requested"):
        await m.register_job(actor, kind="subagent", parent=lease, system_key="new-after-stop")
    with pytest.raises(JobError, match="cancel_requested"):
        await m.claim_job(actor, reg.job.job_id, executor_id="late", boot_id="boot", expected_version=1)
    # Receipt replay returns the old admission, never starts replacement work.
    assert (await m.register_job(actor, kind="subagent", parent=lease, system_key="child")).replayed


async def test_queued_cancel_is_persistent_and_cannot_later_start(runtime):
    _, m, actor, _, _, connect, _ = runtime
    reg = await child(runtime)
    await m.request_cancel(actor, reg.job.job_id, system_key="queued-stop", reason="stop before execution")
    recovered = JobManager(await connect(), now_ms=m.now)
    saved = await recovered.get_job(actor, reg.job.job_id)
    assert saved.status == "cancelled" and saved.resource_state == "not_started"
    with pytest.raises(JobError, match="cancel_requested"):
        await recovered.claim_job(actor, reg.job.job_id, executor_id="late", boot_id="boot", expected_version=2)


@pytest.mark.parametrize("case", ["own-generation", "parent-generation", "own-expiry", "parent-expiry"])
async def test_stale_execution_cannot_create_new_descendant(runtime, case):
    store, m, actor, root, root_proof, _, now = runtime
    reg = await child(runtime)
    proof = await m.claim_job(actor, reg.job.job_id, executor_id="child", boot_id="boot", expected_version=1)
    if case == "own-generation":
        proof = replace(proof, generation=proof.generation - 1)
    elif case == "parent-generation":
        await store._db.execute(f"UPDATE {m.jobs} SET generation=generation+1 WHERE job_id=?", (root.job.job_id,))
    elif case == "own-expiry":
        await store._db.execute(f"UPDATE {m.jobs} SET lease_until_ms=? WHERE job_id=?", (now[0] - 1, proof.job_id))
    else:
        await store._db.execute(f"UPDATE {m.jobs} SET lease_until_ms=? WHERE job_id=?", (now[0] - 1, root_proof.job_id))
    with pytest.raises(JobError, match="stale|expired"):
        await m.register_job(actor, kind="background_tool", parent=proof, system_key="must-not-start")


async def test_registration_and_operation_rollback_together(runtime, monkeypatch):
    store, m, actor, _, lease, _, _ = runtime
    async def broken(*args, **kwargs):
        raise RuntimeError("injected operation persistence failure")
    monkeypatch.setattr(m, "_insert_operation", broken)
    with pytest.raises(RuntimeError, match="persistence failure"):
        await m.register_job(actor, kind="subagent", parent=lease, system_key="atomic-child")
    assert (await store._db.fetchone(f"SELECT COUNT(*) AS n FROM {m.jobs}"))["n"] == 1


async def test_control_survives_session_message_deletion(runtime):
    store, m, actor, root, _, _, _ = runtime
    await child(runtime)
    await store.close_session(actor.session_id)
    assert (await m.get_job(actor, root.job.job_id)).session_id == actor.session_id
    stop = await m.request_cancel(actor, root.job.job_id, system_key="after-delete", reason="retained control")
    assert stop.result["cancel_requested"]


async def test_explicit_migration_runs_on_each_real_backend(registry_store):
    _, connect = registry_store
    legacy = await connect(schema="auto_create", suffix="old_")
    sid = await legacy.create_session()
    await legacy.append_message(sid, role="assistant", content="preserved", tool_calls=[{"id": "old-call"}])
    await migrate_task_schema(legacy._db, legacy.table_prefix, maintenance=True)
    migrated = await connect(schema="compatible", suffix="old_")
    ref = await migrated.message_ref(sid, 1)
    assert ref.id is not None and ref.store_id == migrated.store_id
    assert (await migrated.message_by_id(sid, ref.id)).tool_calls == [{"id": "old-call"}]
    assert await migrated.append_message(sid, role="user", content="next") == 2


async def test_unknown_resource_keeps_supervision_and_blocks_more_work(runtime):
    _, m, actor, _, _, _, _ = runtime
    reg = await child(runtime)
    proof = await m.claim_job(actor, reg.job.job_id, executor_id="worker", boot_id="boot", expected_version=1)
    await m.record_result(actor, proof, outcome="interrupted", result={"reason": "remote request unknown"})
    row = await m.get_job(actor, reg.job.job_id)
    assert row.outcome == "interrupted" and row.resource_state == "unknown" and row.supervision_status == "open"
    with pytest.raises(JobError, match="unresolved_prior_attempt"):
        await m.register_job(actor, kind="background_tool", parent=proof, system_key="do-not-repeat")
    stop = await m.request_cancel(actor, reg.job.job_id, system_key="stop-unknown", reason="query/cancel needed")
    assert stop.result["status"] == "cancelling" and stop.result["resource_state"] == "unknown"


async def test_stale_caller_cannot_attach_receipt_or_steer_with_valid_target_lease(runtime):
    store, m, actor, _, lease, _, _ = runtime
    ref = await call(store, actor)
    reg = await m.register_job(actor, kind="subagent", parent=lease, call_ref=ref, tool_call_id="call-1")
    target = await m.claim_job(actor, reg.job.job_id, executor_id="target", boot_id="boot", expected_version=1)
    seq = await store.append_message(actor.session_id, role="tool", tool_call_id="call-1", content="accepted")
    receipt = await store.message_ref(actor.session_id, seq)
    stale = replace(lease, generation=lease.generation - 1)
    with pytest.raises(JobError, match="stale_execution"):
        await m.bind_receipt(actor, reg.operation.operation_id, receipt, caller=stale)
    with pytest.raises(JobError, match="stale_execution"):
        await m.register_operation(actor, target, caller=stale, action="steer", call_ref=ref,
            tool_call_id="call-1", request={"message": "late"}, result={"accepted": True})


async def test_model_origin_must_reference_an_actual_assistant_tool_call(runtime):
    store, m, actor, _, lease, _, _ = runtime
    ref = await call(store, actor)
    with pytest.raises(JobError, match="tool_call_not_found"):
        await m.register_job(actor, kind="subagent", parent=lease, call_ref=ref, tool_call_id="not-in-row")
    with pytest.raises(JobError, match="message_ref_mismatch"):
        await m.register_job(actor, kind="subagent", parent=lease, call_ref=replace(ref, id=ref.id + 99), tool_call_id="call-1")
    with pytest.raises(JobError, match="persistent_system_key_required"):
        await m.register_job(actor, kind="subagent", parent=lease)


async def test_migration_failure_after_raw_table_change_rolls_back_everything(registry_store, monkeypatch):
    from power_loop.runtime.store import task_schema

    store, connect = registry_store
    legacy = await connect(schema="auto_create", suffix="fail_")
    sid = await legacy.create_session()
    await legacy.append_message(sid, role="user", content="preserve on rollback")

    async def broken(*args):
        raise RuntimeError("injected late DDL failure")

    monkeypatch.setattr(task_schema, "_run_task_ddl", broken)
    with pytest.raises(RuntimeError, match="late DDL failure"):
        await migrate_task_schema(legacy._db, legacy.table_prefix, maintenance=True)
    assert (await legacy._db.fetchone(f"SELECT version FROM {legacy.table_prefix}schema_migrations"))["version"] == 9
    assert (await legacy.load_all_messages(sid))[0].content == "preserve on rollback"
