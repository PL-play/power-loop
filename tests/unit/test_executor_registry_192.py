"""Real store boot admission, atomic drain, original binding and ACK boundaries."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.runtime import executor_registry as registry
from power_loop.runtime.jobs import JobActor, JobError, JobManager, _hash
from power_loop.runtime.store.schema import StoreSchemaError
from power_loop.runtime.store.task_schema import migrate_task_schema, verify_task_schema

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def boot(registry_store):
    store, connect = registry_store
    now = [1_000_000]
    manager = JobManager(store, now_ms=lambda: now[0])
    sid = await store.create_session()
    actor = JobActor("boot-tenant", "host", sid)
    identity = registry.ExecutorIdentity(actor.tenant_id, actor.principal_id, "executor", "boot-one")
    return SimpleNamespace(store=store, manager=manager, now=now, actor=actor, identity=identity, connect=connect)


async def start(c, identity=None):
    identity = identity or c.identity
    await registry.register(c.manager, identity, contract_hash="a"*64, startup_lease_ms=1000)
    return await registry.activate(c.manager, identity, expected_version=1, system_key="startup", lease_ms=1000)


async def root(c, *, identity=None, key="root"):
    identity = identity or c.identity
    reg = await c.manager.register_job(c.actor, kind="send", session_id=c.actor.session_id, system_key=key)
    lease = await c.manager.claim_job(c.actor, reg.job.job_id, executor_id=identity.executor_id,
        boot_id=identity.boot_id, expected_version=reg.job.version, lease_ms=300000)
    return reg, lease


async def test_register_starting_is_not_dispatch_authority_and_activation_binds_original_scope(boot):
    c = boot
    first = await registry.register(c.manager, c.identity, contract_hash="a"*64, startup_lease_ms=1000)
    pending = await c.manager.register_job(c.actor, kind="send", session_id=c.actor.session_id, system_key="starting")
    with pytest.raises(JobError, match="executor_not_accepting"):
        await c.manager.claim_job(c.actor, pending.job.job_id, executor_id="executor", boot_id="boot-one", expected_version=1)
    assert first["executor"]["state"] == "starting" and not first["execution_authority"]
    accepted = await registry.activate(c.manager, c.identity, expected_version=1, system_key="startup", lease_ms=1000)
    _, lease = await root(c)
    scope = await c.manager.get_execution_membership(c.actor, lease.job_id, 1)
    assert scope["scope"]["version"] == 2
    assert scope["scope"]["executor_binding"]["executor_key"] == accepted["executor"]["executor_key"]
    assert not accepted["physical_stop_confirmed"]
    await c.manager.check_execution(c.actor, lease)
    other = JobManager(await c.connect(), now_ms=lambda: c.now[0])
    assert (await registry.get(other, c.identity))["admission_open"]
    assert await other.get_execution_membership(c.actor, lease.job_id, 1) == scope


async def test_drain_atomically_freezes_claim_and_records_active_and_queued_stop_scope(boot):
    c = boot
    await start(c)
    reg, lease = await root(c)
    child = await c.manager.register_job(c.actor, kind="subagent", parent=lease, system_key="unclaimed", independent=True)
    stopped = await registry.drain(c.manager, c.identity, system_key="shutdown", reason="host shutdown")
    assert stopped["executor"]["state"] == "draining" and not stopped["physical_stop_confirmed"]
    assert {j["job_id"] for j in stopped["effects"]["executions"]} == {reg.job.job_id, child.job.job_id}
    assert len(stopped["effects"]["stop_operations"]) == 2 and stopped["effects"]["scope_incomplete"] is False
    for job_id in (reg.job.job_id, child.job.job_id):
        job = await c.manager.get_job(c.actor, job_id)
        assert job.cancel_requested_at is not None and job.supervision_status == "open"
    assert (await c.manager.get_job(c.actor, reg.job.job_id)).resource_state == "running"
    assert (await c.manager.get_job(c.actor, child.job.job_id)).resource_state == "not_started"
    with pytest.raises(JobError, match="cancel_requested|executor_not_accepting"):
        await c.manager.check_execution(c.actor, lease)
    assert (await c.manager.get_execution_membership(c.actor, reg.job.job_id, 1))["scope"]["executor_binding"]
    # Historical reads are not conditioned on the boot still accepting work.
    await c.manager.close_model_execution(c.actor, lease, evidence={"exit": "cancelled"}, resource_state="unknown")
    assert (await c.manager.get_job(c.actor, reg.job.job_id)).resource_state == "unknown"


async def test_drain_does_not_stop_other_boot_independent_roots_or_destroy_terminal_proofs(boot):
    c = boot
    await start(c)
    other = replace(c.identity, executor_id="other-executor", boot_id="other-boot")
    await start(c, other)
    _, first = await root(c)
    second, lease2 = await root(c, identity=other, key="independent")
    await registry.drain(c.manager, c.identity, system_key="shutdown", reason="stop only this boot")
    await c.manager.check_execution(c.actor, lease2)
    assert (await c.manager.get_job(c.actor, second.job.job_id)).cancel_requested_at is None
    assert first.executor_id != lease2.executor_id


@pytest.mark.parametrize("action", ["register", "activate", "heartbeat", "drain"])
async def test_lost_ack_replays_exact_original_decision_without_new_effects(boot, monkeypatch, action):
    c = boot
    if action != "register":
        await registry.register(c.manager, c.identity, contract_hash="a"*64, startup_lease_ms=1000)
    if action in ("heartbeat", "drain"):
        await registry.activate(c.manager, c.identity, expected_version=1, system_key="startup", lease_ms=1000)
    if action == "drain":
        await root(c)
    write = c.manager._write
    lost = []
    @asynccontextmanager
    async def lose_ack():
        async with write() as tx:
            yield tx
        if not lost:
            lost.append(True)
            raise RuntimeError("test: original commit ACK lost")
    fn = getattr(registry, action)
    args = {"register": {"contract_hash": "a"*64, "startup_lease_ms": 1000},
        "activate": {"expected_version": 1, "system_key": "startup", "lease_ms": 1000},
        "heartbeat": {"expected_version": 2, "system_key": "beat", "lease_ms": 1000},
        "drain": {"system_key": "stop", "reason": "shutdown"}}[action]
    with monkeypatch.context() as patch:
        patch.setattr(c.manager, "_write", lose_ack)
        with pytest.raises(RuntimeError, match="ACK lost"):
            await fn(c.manager, c.identity, **args)
    before = await registry.get(c.manager, c.identity)
    c.now[0] += 10
    replay = await fn(c.manager, c.identity, **args)
    assert replay["replayed"] and replay["executor"]["version"] == before["version"]
    assert replay["executor"]["lease_until_ms"] == before["lease_until_ms"]
    assert len(await c.manager.db.fetchall(f"SELECT * FROM {c.manager.prefix}task_executor_events")) == before["version"]


async def test_replayed_old_activation_or_heartbeat_cannot_revive_drain(boot):
    c = boot
    accepted = await start(c)
    beat = await registry.heartbeat(c.manager, c.identity, expected_version=2, system_key="beat", lease_ms=1000)
    await registry.drain(c.manager, c.identity, system_key="shutdown", reason="shutdown")
    assert (await registry.activate(c.manager, c.identity, expected_version=1, system_key="startup", lease_ms=1000))["executor"] == accepted["executor"]
    assert (await registry.heartbeat(c.manager, c.identity, expected_version=2, system_key="beat", lease_ms=1000))["executor"] == beat["executor"]
    assert (await registry.get(c.manager, c.identity))["state"] == "draining"
    with pytest.raises(JobError, match="executor_not_accepting"):
        await registry.heartbeat(c.manager, c.identity, expected_version=4, system_key="new-beat")


async def test_expired_boot_cannot_renew_or_dispatch_while_job_lease_is_still_live(boot):
    c = boot
    await start(c)
    _, lease = await root(c)
    c.now[0] += 1001
    with pytest.raises(JobError, match="executor_lease_expired"):
        await registry.heartbeat(c.manager, c.identity, expected_version=2, system_key="late")
    with pytest.raises(JobError, match="executor_lease_expired"):
        await c.manager.check_execution(c.actor, lease)
    assert not (await registry.get(c.manager, c.identity))["admission_open"]


async def test_expired_registered_boot_cannot_commit_success_but_negative_evidence_is_saved(boot):
    c = boot
    await start(c)
    _, parent = await root(c)
    child = await c.manager.register_job(c.actor, kind="subagent", parent=parent, system_key="child", independent=True)
    lease = await c.manager.claim_job(c.actor, child.job.job_id, executor_id="executor", boot_id="boot-one", expected_version=1)
    c.now[0] += 1001
    raw = {"status": "completed", "final_text": "known late output", "usage": {"total_tokens": 17}}
    with pytest.raises(JobError, match="executor_lease_expired"):
        await c.manager.record_execution_result(c.actor, lease, outcome="succeeded", result=raw, resource_state="stopped")
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='result'", (child.job.job_id,))
    await c.manager.record_execution_result(c.actor, lease, outcome="failed", result={**raw, "status": "failed", "execution_status": "completed"}, resource_state="stopped")
    saved = await c.manager.get_job(c.actor, child.job.job_id)
    assert saved.outcome == "failed" and saved.result["final_text"] == "known late output"
    assert saved.result["usage"]["total_tokens"] == 17 and saved.supervision_status == "open"


async def test_new_boot_cannot_bypass_live_admission_or_unknown_old_provider(boot):
    c = boot
    await start(c)
    _, lease = await root(c)
    other = replace(c.identity, boot_id="boot-two")
    await registry.register(c.manager, other, contract_hash="a"*64, startup_lease_ms=300000)
    with pytest.raises(JobError, match="executor_prior_boot_open"):
        await registry.activate(c.manager, other, expected_version=1, system_key="startup")
    await c.manager.close_model_execution(c.actor, lease, evidence={"exit": "interrupted"}, resource_state="unknown")
    c.now[0] += 1001
    with pytest.raises(JobError, match="executor_prior_execution_unsettled"):
        await registry.activate(c.manager, other, expected_version=1, system_key="startup")
    assert (await registry.get(c.manager, other))["state"] == "starting"
    assert (await c.manager.get_job(c.actor, lease.job_id)).resource_state == "unknown"


async def test_concurrent_start_and_drain_cannot_leave_a_new_claim_open(boot):
    c = boot
    await start(c)
    reg = await c.manager.register_job(c.actor, kind="send", session_id=c.actor.session_id, system_key="race")
    m2 = JobManager(await c.connect(), now_ms=lambda: c.now[0])
    results = await asyncio.gather(registry.drain(c.manager, c.identity, system_key="stop", reason="race"),
        m2.claim_job(c.actor, reg.job.job_id, executor_id="executor", boot_id="boot-one", expected_version=1), return_exceptions=True)
    assert isinstance(results[0], dict)
    if not isinstance(results[1], JobError):
        with pytest.raises(JobError, match="cancel_requested|executor_not_accepting"):
            await c.manager.check_execution(c.actor, results[1])
    assert (await registry.get(c.manager, c.identity))["state"] == "draining"


@pytest.mark.parametrize("fault", ["registry_missing", "binding_removed", "registration_event_missing", "control_row_changed"])
async def test_lost_or_corrupt_original_control_cannot_downgrade_registered_job(boot, fault):
    c = boot
    accepted = await start(c)
    _, lease = await root(c)
    key = accepted["executor"]["executor_key"]
    if fault == "registry_missing":
        await c.manager.db.execute(f"DELETE FROM {c.manager.prefix}task_executors WHERE executor_key=?", (key,))
    elif fault == "binding_removed":
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET executor_key=NULL WHERE job_id=?", (lease.job_id,))
    elif fault == "registration_event_missing":
        await c.manager.db.execute(f"DELETE FROM {c.manager.prefix}task_executor_events WHERE event_id=?", (registry._event_id(c.manager, key, 1),))
    else:
        await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_executors SET lease_until_ms=lease_until_ms+1 WHERE executor_key=?", (key,))
    with pytest.raises(JobError, match="executor_(registration_required|binding_missing|evidence_missing|evidence_corrupt)"):
        await c.manager.check_execution(c.actor, lease)


async def test_register_does_not_retroactively_adopt_old_unregistered_execution(boot):
    c = boot
    _, lease = await root(c)
    scope = await c.manager.get_execution_membership(c.actor, lease.job_id, 1)
    assert scope["scope"]["version"] == 1
    with pytest.raises(JobError, match="executor_history_requires_reconciliation"):
        await registry.register(c.manager, c.identity, contract_hash="a"*64)
    await c.manager.check_execution(c.actor, lease)


async def test_oversized_drain_still_persists_admission_barrier_and_reports_pending_scope(boot, monkeypatch):
    c = boot
    await start(c)
    await root(c)
    monkeypatch.setattr(registry, "_LIMIT", 0)
    stopped = await registry.drain(c.manager, c.identity, system_key="stop", reason="large scope")
    assert stopped["effects"]["scope_incomplete"] and not stopped["physical_stop_confirmed"]
    assert not (await registry.get(c.manager, c.identity))["admission_open"]


@pytest.mark.parametrize("failure", [False, True])
async def test_explicit_protocol_two_migration_preserves_identity_messages_and_jobs(boot, monkeypatch, failure):
    c = boot
    reg, _ = await root(c)
    seq = await c.store.append_message(c.actor.session_id, role="user", content="retained raw")
    ref = await c.store.message_ref(c.actor.session_id, seq)
    store_id = c.store.store_id
    for table in ("task_executors", "task_executor_events"):
        await c.manager.db.execute(f"DROP TABLE {c.manager.prefix}{table}")
    suffix = f" ON {c.manager.jobs}" if c.manager.db.dialect.name == "mysql" else ""
    await c.manager.db.execute(f"DROP INDEX {c.manager.prefix}idx_jobs_executor{suffix}")
    await c.manager.db.execute(f"ALTER TABLE {c.manager.jobs} DROP COLUMN executor_key")
    await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_store SET protocol_version=2 WHERE id=1")
    with pytest.raises(StoreSchemaError, match="incomplete|unsupported"):
        await verify_task_schema(c.manager.db, c.manager.prefix)
    with pytest.raises(StoreSchemaError, match="maintenance_required"):
        await migrate_task_schema(c.manager.db, c.manager.prefix)
    if failure:
        from power_loop.runtime.store import task_schema
        original = task_schema._run_task_ddl
        async def broken(*args):
            await original(*args)
            raise RuntimeError("test: migration DDL failure")
        with monkeypatch.context() as patch:
            patch.setattr(task_schema, "_run_task_ddl", broken)
            with pytest.raises(RuntimeError, match="DDL failure"):
                await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
        row = await c.manager.db.fetchone(f"SELECT protocol_version FROM {c.manager.prefix}task_store WHERE id=1")
        assert row["protocol_version"] == 2
    await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
    await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
    await verify_task_schema(c.manager.db, c.manager.prefix)
    assert c.store.store_id == store_id and await c.store.message_ref(c.actor.session_id, seq) == ref
    assert (await c.manager.get_job(c.actor, reg.job.job_id)).generation == 1


@pytest.mark.parametrize("field,value", [("executor_id", ""), ("boot_id", "bad\nboot"), ("executor_id", "x"*65),
    ("tenant_id", "x"*256), ("principal_id", False)])
async def test_host_identity_is_bounded_and_typed(field, value):
    args = dict(tenant_id="tenant", principal_id="host", executor_id="executor", boot_id="boot")
    with pytest.raises(JobError, match="invalid_executor_identity"):
        registry.ExecutorIdentity(**(args | {field: value}))


@pytest.mark.parametrize("field,value", [("contract_hash", "invalid"), ("contract_hash", "G"*64),
    ("startup_lease_ms", True), ("startup_lease_ms", 0), ("startup_lease_ms", 300001)])
async def test_invalid_registration_does_not_create_control_rows(boot, field, value):
    c = boot
    args = dict(contract_hash="a"*64, startup_lease_ms=1000) | {field: value}
    with pytest.raises(JobError, match="invalid_executor_registration"):
        await registry.register(c.manager, c.identity, **args)
    assert not await c.manager.db.fetchall(f"SELECT * FROM {c.manager.prefix}task_executors")


async def test_immutable_registration_and_operation_requests_cannot_be_reinterpreted(boot):
    c = boot
    await start(c)
    with pytest.raises(JobError, match="executor_operation_conflict"):
        await registry.register(c.manager, c.identity, contract_hash="b"*64, startup_lease_ms=1000)
    await registry.heartbeat(c.manager, c.identity, expected_version=2, system_key="beat", lease_ms=1000)
    with pytest.raises(JobError, match="executor_operation_conflict"):
        await registry.heartbeat(c.manager, c.identity, expected_version=2, system_key="beat", lease_ms=2000)
    with pytest.raises(JobError, match="executor_registration_required"):
        await registry.get(c.manager, replace(c.identity, tenant_id="another-tenant"))


async def test_registration_and_heartbeat_are_serialized_across_two_store_connections(boot):
    c = boot
    other = JobManager(await c.connect(), now_ms=lambda: c.now[0])
    replies = await asyncio.gather(*(registry.register(m, c.identity, contract_hash="a"*64, startup_lease_ms=1000) for m in (c.manager, other)))
    assert sorted(r["replayed"] for r in replies) == [False, True]
    assert replies[0]["event_id"] == replies[1]["event_id"]
    await registry.activate(c.manager, c.identity, expected_version=1, system_key="start", lease_ms=1000)
    replies = await asyncio.gather(*(registry.heartbeat(m, c.identity, expected_version=2, system_key=key, lease_ms=1000)
        for m, key in ((c.manager, "a"), (other, "b"))), return_exceptions=True)
    assert sum(isinstance(r, dict) for r in replies) == 1
    assert sum(isinstance(r, JobError) and r.code == "executor_heartbeat_conflict" for r in replies) == 1
    assert (await registry.get(c.manager, c.identity))["version"] == 3


async def test_drain_storage_failure_rolls_back_stop_intents_and_boot_view_together(boot, monkeypatch):
    c = boot
    await start(c)
    _, lease = await root(c)
    original = registry._save
    async def fail(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("test: executor control storage failure")
    with monkeypatch.context() as patch:
        patch.setattr(registry, "_save", fail)
        with pytest.raises(RuntimeError, match="storage failure"):
            await registry.drain(c.manager, c.identity, system_key="stop", reason="shutdown")
    assert (await registry.get(c.manager, c.identity))["state"] == "accepting"
    assert (await c.manager.get_job(c.actor, lease.job_id)).cancel_requested_at is None
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE action='stop'")


async def test_deleted_control_row_cannot_allow_a_new_root_to_downgrade(boot):
    c = boot
    await start(c)
    await root(c)
    await c.manager.db.execute(f"DELETE FROM {c.manager.prefix}task_executors")
    pending = await c.manager.register_job(c.actor, kind="send", session_id=c.actor.session_id, system_key="new-root")
    with pytest.raises(JobError, match="executor_registration_required"):
        await c.manager.claim_job(c.actor, pending.job.job_id, executor_id="executor", boot_id="boot-one", expected_version=1)
    assert (await c.manager.get_job(c.actor, pending.job.job_id)).generation == 0


async def test_registered_parent_cannot_dispatch_an_unregistered_child_executor(boot):
    c = boot
    await start(c)
    _, lease = await root(c)
    pending = await c.manager.register_job(c.actor, kind="subagent", parent=lease, system_key="child", independent=True)
    with pytest.raises(JobError, match="executor_registration_required"):
        await c.manager.claim_job(c.actor, pending.job.job_id, executor_id="unregistered", boot_id="new-boot", expected_version=1)


async def test_mutable_stopped_cache_cannot_hide_old_unconfirmed_model_request(boot):
    from power_loop.runtime import model_attempts
    from power_loop.runtime.task_runtime import Execution, TaskRuntime
    c = boot
    await start(c)
    _, lease = await root(c)
    runtime = TaskRuntime(c.manager, tenant_id=c.identity.tenant_id, principal_id=c.identity.principal_id,
        executor_id=c.identity.executor_id, boot_id=c.identity.boot_id)
    binding = Execution(runtime, c.actor, lease)
    await model_attempts.enable(binding)
    attempt = await model_attempts.prepare(binding, nonce="old-request", round_index=0, retry_index=1,
        request_hash="a"*64, model="test", max_tokens=100)
    await model_attempts.dispatch(attempt)
    # Deliberately contradictory cache/exit; the immutable dispatch remains.
    await c.manager.close_model_execution(c.actor, lease, evidence={"exit": "completed"}, resource_state="stopped")
    c.now[0] += 1001
    newer = replace(c.identity, boot_id="new-boot")
    await registry.register(c.manager, newer, contract_hash="a"*64)
    with pytest.raises(JobError, match="executor_prior_execution_unsettled"):
        await registry.activate(c.manager, newer, expected_version=1, system_key="startup")


@pytest.mark.parametrize("fault", ["not_object", "boolean_version", "unknown_action", "bad_effect_snapshot", "bad_stop_reference"])
async def test_self_hashed_but_invalid_control_event_is_rejected(boot, fault):
    c = boot
    await start(c)
    await root(c)
    receipt = await registry.drain(c.manager, c.identity, system_key="stop", reason="shutdown")
    table = c.manager.prefix + "task_executor_events"
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {table} WHERE event_id=?", (receipt["event_id"],))
    data = json.loads(row["payload_json"])
    if fault == "not_object":
        payload, request_hash, action = "[]", row["request_hash"], row["action"]
    else:
        if fault == "boolean_version":
            data["version"] = True
        elif fault == "unknown_action":
            data["request"]["action"] = "grant-anything"
        elif fault == "bad_effect_snapshot":
            data["effects"]["executions"][0]["generation"] = True
        else:
            data["effects"]["stop_operations"][0] = False
        data["evidence_hash"] = _hash({k: v for k, v in data.items() if k != "evidence_hash"})
        payload, request_hash, action = json.dumps(data), _hash(data["request"]), data["request"]["action"]
    await c.manager.db.execute(f"UPDATE {table} SET payload_json=?,request_hash=?,action=? WHERE event_id=?",
        (payload, request_hash, action, receipt["event_id"]))
    with pytest.raises(JobError, match="executor_evidence_corrupt"):
        await registry.get(c.manager, c.identity)


@pytest.mark.parametrize("action,args", [("activate", {"expected_version": True}), ("activate", {"lease_ms": 0}),
    ("heartbeat", {"expected_version": False}), ("heartbeat", {"lease_ms": 300001}),
    ("drain", {"reason": ""}), ("drain", {"reason": "bad\x00reason"}), ("drain", {"system_key": "bad\nkey"})])
async def test_invalid_control_arguments_cannot_change_boot_admission(boot, action, args):
    c = boot
    await start(c)
    before = await registry.get(c.manager, c.identity)
    defaults = {"activate": {"expected_version": 1, "lease_ms": 1000, "system_key": "bad"},
        "heartbeat": {"expected_version": 2, "lease_ms": 1000, "system_key": "bad"},
        "drain": {"system_key": "bad", "reason": "shutdown"}}[action]
    with pytest.raises(JobError, match="invalid_executor_"):
        await getattr(registry, action)(c.manager, c.identity, **(defaults | args))
    assert await registry.get(c.manager, c.identity) == before
