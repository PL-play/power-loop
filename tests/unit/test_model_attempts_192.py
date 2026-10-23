"""Actual model-attempt persistence, fencing, uncertainty and ACK boundaries."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop import CancellationToken, LLMRequest
from power_loop.contracts.errors import LLMNonRetryable
from power_loop.runtime import model_attempts as attempts
from power_loop.runtime.jobs import JobError, JobManager, _hash
from power_loop.runtime.model_attempt_query import list_attempts
from power_loop.runtime.retry import LLMRetryPolicy, with_retry
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store
from .test_jobs_tool_192 import invoke
from .test_task_runtime_192 import managed
from .test_workflow_task_runtime_192 import leaf, managed_workflow, records, settle

__all__ = ["registry_store", "managed", "managed_workflow"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def attempt_case(registry_store):
    store, connect = registry_store
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id="attempt-test", principal_id="host", executor_id="executor", boot_id="boot")
    sid = await store.create_session()
    actor = runtime.actor(sid)
    registration = await manager.register_job(actor, kind="send", session_id=sid, system_key="root",
        request={"send_index": 1, "model_attempt_protocol": 1})
    binding = await runtime.claim(actor, registration)
    await attempts.enable(binding)
    return SimpleNamespace(store=store, manager=manager, runtime=runtime, sid=sid, actor=actor,
        registration=registration, binding=binding, connect=connect)


async def prepare(c, **kwargs):
    values = dict(nonce="call-one", round_index=0, retry_index=1, request_hash="a"*64, model="test-model", max_tokens=100)
    return await attempts.prepare(c.binding, **(values | kwargs))


async def completed(c, attempt=None):
    attempt = attempt or await prepare(c)
    await attempts.dispatch(attempt)
    await attempts.finish(attempt, status="returned", usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}, estimated=False)
    return attempt


async def closed(c):
    await c.manager.close_model_execution(c.actor, c.binding.lease, evidence={"exit": "completed", "usage": {"total_tokens": 10}})
    return await c.manager.finalize_supervision(c.actor, c.binding.lease.job_id, generation=1)


async def test_physical_calls_same_round_have_distinct_persistent_identities_and_do_not_duplicate_usage(attempt_case):
    c = attempt_case
    first = await completed(c)
    second = await completed(c, await prepare(c, nonce="another-call-same-coordinate"))
    assert first.attempt_id != second.attempt_id
    assert first.prepared["round_index"] == second.prepared["round_index"] == 0
    proof = await closed(c)
    reopened = JobManager(await c.connect())
    assert await reopened.finalize_supervision(c.actor, c.binding.lease.job_id, generation=1) == proof
    # Observations are not added again to the original canonical send ledger.
    assert (await c.manager.execution_usage(c.actor, c.binding.lease.job_id))["total_tokens"] == 10
    events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='model_attempt_prepared'")
    assert len(events) == 2


async def test_duplicate_prepare_is_recall_but_dispatch_cannot_run_twice(attempt_case):
    c = attempt_case
    first = await prepare(c)
    assert await prepare(c) == first
    await attempts.dispatch(first)
    with pytest.raises(JobError, match="dispatch_already_admitted"):
        await attempts.dispatch(first)
    with pytest.raises(JobError, match="unresolved_prior_attempt"):
        await prepare(c, nonce="must-not-overlap")
    with pytest.raises(JobError, match="model_attempt_conflict"):
        await prepare(c, model="changed-original-request")


@pytest.mark.parametrize("fault", ["cancel", "expired", "old-generation"])
async def test_dispatch_is_fenced_after_prepare_and_cannot_claim_remote_start(attempt_case, fault):
    c = attempt_case
    attempt = await prepare(c)
    if fault == "cancel":
        await c.manager.request_cancel(c.actor, c.binding.lease.job_id, system_key="stop", reason="before dispatch")
    else:
        column, value = ("lease_until_ms", 0) if fault == "expired" else ("generation", 2)
        await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET {column}=? WHERE job_id=?", (value, c.binding.lease.job_id))
    with pytest.raises(JobError, match="cancel_requested|lease_expired|stale_execution"):
        await attempts.dispatch(attempt)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_dispatched'")


@pytest.mark.parametrize("status", ["returned", "unknown"])
async def test_actual_late_observation_survives_stop_and_expiry_without_dispatch_authority(attempt_case, status):
    c = attempt_case
    attempt = await prepare(c)
    await attempts.dispatch(attempt)
    await c.manager.request_cancel(c.actor, c.binding.lease.job_id, system_key="stop", reason="after dispatch")
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET lease_until_ms=0 WHERE job_id=?", (c.binding.lease.job_id,))
    args = dict(status=status, usage={"total_tokens": 17}, estimated=status == "unknown", error_type="TimeoutError" if status == "unknown" else None)
    await attempts.finish(attempt, **args)
    await attempts.finish(attempt, **args)
    with pytest.raises(JobError, match="model_attempt_conflict"):
        await attempts.finish(attempt, **(args | {"usage": {"total_tokens": 18}}))
    with pytest.raises(JobError, match="cancel_requested|lease_expired"):
        await attempts.dispatch(attempt)
    [event] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='model_attempt_finished'")
    data = json.loads(event["payload_json"])
    assert data["status"] == status and data["usage"] == {"total_tokens": 17}
    assert (await c.manager.get_job(c.actor, c.binding.lease.job_id)).outcome is None


@pytest.mark.parametrize("phase", ["prepared", "dispatched", "finished"])
async def test_hash_or_evidence_loss_blocks_supervision_even_if_mutable_view_says_stopped(attempt_case, phase):
    c = attempt_case
    attempt = await completed(c)
    key = attempts.event_id(c.manager, phase, attempt.attempt_id)
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET request_hash='corrupt' WHERE event_id=?", (key,))
    with pytest.raises(JobError, match="model_attempt_evidence_corrupt"):
        await closed(c)
    assert (await c.manager.get_job(c.actor, c.binding.lease.job_id)).supervision_status == "open"


async def test_tracking_deleted_is_not_legacy_or_zero_calls(attempt_case):
    c = attempt_case
    await completed(c)
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='model_attempt_tracking'")
    with pytest.raises(JobError, match="model_attempt_evidence_missing"):
        await closed(c)


@pytest.mark.parametrize("ending", ["prepared", "dispatched", "unknown"])
async def test_only_proven_unstarted_attempt_is_quiescent_without_provider_response(attempt_case, ending):
    c = attempt_case
    attempt = await prepare(c)
    if ending != "prepared":
        await attempts.dispatch(attempt)
    if ending == "unknown":
        await attempts.finish(attempt, status="unknown", usage={"total_tokens": 11}, estimated=True, error_type="TimeoutError")
    if ending == "prepared":
        await closed(c)
        with pytest.raises(JobError, match="intake_closed"):
            await attempts.dispatch(attempt)
    else:
        with pytest.raises(JobError, match="model_attempt_unsettled"):
            await closed(c)


@pytest.mark.parametrize("field,value", [("round_index", True), ("round_index", -1), ("retry_index", 0),
    ("retry_index", True), ("nonce", ""), ("request_hash", "not-a-hash"), ("max_tokens", True), ("max_tokens", -1)])
async def test_invalid_admission_is_rejected_without_any_dispatch(attempt_case, field, value):
    c = attempt_case
    with pytest.raises(JobError, match="invalid_model_attempt"):
        await prepare(c, **{field: value})
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_prepared'")


@pytest.mark.parametrize("phase", ["tracking", "prepared", "dispatched", "finished"])
async def test_lost_committed_ack_reuses_original_identity_and_calls_provider_once(managed, monkeypatch, phase):
    _, manager, _, loop, model, sid, _ = managed
    model.root_calls = 10  # Ordinary one-call main send, no delegation.
    model.release.set()
    write = manager._write
    lost = []
    @asynccontextmanager
    async def lose_ack():
        hit = False
        async with write() as tx:
            execute = tx.execute
            async def record(sql, params=()):
                nonlocal hit
                if sql.startswith("INSERT") and "model_attempt_"+phase in params:
                    hit = True
                return await execute(sql, params)
            with monkeypatch.context() as patch:
                patch.setattr(tx, "execute", record)
                yield tx
        if hit and not lost:
            lost.append(True)
            raise RuntimeError("test: commit completed, ACK lost")
    monkeypatch.setattr(manager, "_write", lose_ack)
    result = await loop.send("one physical request only", sid)
    assert result.status == "completed" and model.root_calls == 11 and lost == [True]
    assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='model_attempt_prepared'")) == 1
    [row] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
    assert row["supervision_status"] == "closed" and row["resource_state"] == "stopped"


@pytest.mark.parametrize("phase", ["prepared", "dispatched"])
async def test_pre_request_storage_failure_never_calls_provider(managed, monkeypatch, phase):
    _, manager, _, loop, model, sid, _ = managed
    model.root_calls = 10
    fn = "prepare" if phase == "prepared" else "dispatch"
    async def fail(*args, **kwargs):
        raise RuntimeError("test: before physical request")
    monkeypatch.setattr(attempts, fn, fail)
    result = await loop.send("do not dispatch without durable admission", sid)
    assert result.status == "degraded" and model.root_calls == 10
    assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='model_attempt_finished'")


async def test_return_observation_write_failure_preserves_actual_usage_and_pending_control(managed, monkeypatch):
    _, manager, _, loop, model, sid, _ = managed
    model.root_calls = 10
    async def fail(*args, **kwargs):
        raise RuntimeError("test: returned observation unavailable")
    monkeypatch.setattr(attempts, "finish", fail)
    result = await loop.send("one known paid response", sid)
    assert model.root_calls == 11 and result.usage["total_tokens"] == 10
    [row] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
    assert row["resource_state"] == "unknown" and row["supervision_status"] == "open"
    assert (await manager.execution_usage(loop.task_runtime.actor(sid), row["job_id"]))["total_tokens"] == 10


@pytest.mark.parametrize("policy", ["retry", "fallback", "continuation"])
async def test_unknown_provider_call_cannot_restart_business_in_another_leaf(managed_workflow, monkeypatch, policy):
    _, manager, _, loop, model, sid, state, _ = managed_workflow
    model.hold = set()
    node = leaf("BUILD")
    if policy == "retry":
        node["retry"] = {"max_attempts": 3, "on": ["failed"]}
    elif policy == "fallback":
        node["fallback"] = leaf("FALLBACK")
    else:
        node["continuation"] = {"max_continuations": 2, "extra_rounds": 1, "gate": "always"}
    state["spec"]["root"]["steps"][0] = node
    complete = model.complete
    calls = []
    async def unknown(request, **kwargs):
        if "ROLE-BUILD" in (request.system_prompt or ""):
            calls.append(True)
            raise TimeoutError("test: provider may still be processing")
        return await complete(request, **kwargs)
    monkeypatch.setattr(model, "complete", unknown)
    await loop.send("run", sid)
    await settle(managed_workflow)
    assert calls == [True] and model.counts["REVIEW"] == model.counts["FALLBACK"] == 0
    wf = next(r for r in await records(managed_workflow) if r.kind == "workflow")
    assert wf.outcome == "failed" and wf.supervision_status == "open" and wf.resource_state == "unknown"
    assert len(await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE event_type='node_session'")) == 1


async def test_request_contents_are_hashed_and_never_persisted(attempt_case):
    c = attempt_case
    from power_loop.runtime.task_runtime import _SEND
    token = _SEND.set(c.binding)
    try:
        request = LLMRequest(messages=[{"role": "user", "content": "PRIVATE-REQUEST-TEXT"}], max_tokens=100,
            system_prompt="PRIVATE-SYSTEM-PROMPT", extra={"api_key": "PRIVATE-CREDENTIAL"})
        attempt = await attempts.start_call(SimpleNamespace(session_id=c.sid), request, round_index=0, retry_index=1, model="test")
        assert attempt is not None and len(attempt.prepared["request_hash"]) == 64
    finally:
        _SEND.reset(token)
    events = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events}")
    assert all("PRIVATE-" not in row["payload_json"] for row in events)


async def test_wrong_original_actor_cannot_append_late_evidence(attempt_case):
    c = attempt_case
    attempt = await prepare(c)
    await attempts.dispatch(attempt)
    bad = replace(c.binding, actor=replace(c.actor, session_id="unrelated"))
    with pytest.raises(JobError, match="access_denied|owner_mismatch"):
        await attempts.finish(replace(attempt, binding=bad), status="returned", usage={"total_tokens": 10}, estimated=False)


async def test_snapshot_and_normal_usage_are_separate_from_billing(attempt_case):
    c = attempt_case
    attempt = await prepare(c)
    await attempts.dispatch(attempt)
    args = dict(status="returned", usage={"total_tokens": 10, "prompt_image_tokens": 4}, estimated=False)
    await attempts.finish(attempt, **args)
    [row] = await c.manager.db.fetchall(f"SELECT payload_json,request_hash FROM {c.manager.events} WHERE event_type='model_attempt_finished'")
    data = json.loads(row["payload_json"])
    assert _hash(data) == row["request_hash"] and data["usage"]["prompt_image_tokens"] == 4
    assert not any(key in data for key in ("reservation_id", "settled", "billed_amount"))


async def test_jobs_attempts_query_is_bounded_authorized_and_never_exposes_control_values(attempt_case):
    c = attempt_case
    for n in range(3):
        await completed(c, await prepare(c, nonce=f"call-{n}"))
    loop = SimpleNamespace(task_runtime=c.runtime)
    before = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")
    first = await invoke(loop, c.sid, action="get", job_id=c.binding.lease.job_id, view="attempts", limit=1)
    assert first["view"] == "attempts" and first["truncated"] and first["next_cursor"]
    ids, page = set(), first
    while True:
        assert len(page["items"]) == 1
        item = page["items"][0]
        assert item["attempt_id"] not in ids and item["status"] == "returned" and item["response_received"]
        assert item["observed_usage"]["total_tokens"] == 10 and item["execution_authority"] is False
        ids.add(item["attempt_id"])
        if not page["next_cursor"]:
            break
        page = await list_attempts(c.manager, c.actor, c.binding.lease.job_id, cursor=page["next_cursor"], limit=1)
    assert len(ids) == 3 and first["billing_status"] == "unavailable"
    assert all(word not in json.dumps(first) for word in ("nonce", "executor_id", "boot_id", "call-0"))
    assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id") == before
    with pytest.raises(JobError, match="access_denied"):
        await list_attempts(c.manager, replace(c.actor, session_id="foreign"), c.binding.lease.job_id)


@pytest.mark.parametrize("state", ["prepared", "dispatched", "unknown", "returned"])
async def test_attempt_view_reports_observation_state_without_inventing_a_response(attempt_case, state):
    c = attempt_case
    attempt = await prepare(c)
    if state != "prepared":
        await attempts.dispatch(attempt)
    if state in ("unknown", "returned"):
        await attempts.finish(attempt, status=state, usage={"total_tokens": 10}, estimated=state == "unknown")
    result = await list_attempts(c.manager, c.actor, c.binding.lease.job_id)
    [item] = result["items"]
    assert item["status"] == ("dispatched_unconfirmed" if state == "dispatched" else state)
    assert item["response_received"] == (state == "returned")


async def test_query_rechecks_snapshot_and_refuses_orphaned_evidence(attempt_case):
    c = attempt_case
    first = await completed(c)
    await completed(c, await prepare(c, nonce="second"))
    page = await list_attempts(c.manager, c.actor, c.binding.lease.job_id, limit=1)
    await completed(c, await prepare(c, nonce="third"))
    with pytest.raises(JobError, match="model_attempt_snapshot_changed"):
        await list_attempts(c.manager, c.actor, c.binding.lease.job_id, cursor=page["next_cursor"])
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (attempts.event_id(c.manager, "prepared", first.attempt_id),))
    with pytest.raises(JobError, match="model_attempt_evidence_corrupt"):
        await list_attempts(c.manager, c.actor, c.binding.lease.job_id)


@pytest.mark.parametrize("field,value", [("limit", True), ("limit", 0), ("limit", 101), ("max_chars", 1000), ("cursor", "bad")])
async def test_attempt_query_rejects_invalid_bounds_and_cursor(attempt_case, field, value):
    c = attempt_case
    with pytest.raises(JobError, match="invalid_.*(query|cursor)"):
        await list_attempts(c.manager, c.actor, c.binding.lease.job_id, **{field: value})


@pytest.mark.parametrize("field,value", [("estimated", "yes"), ("status", "success"), ("usage", {"total_tokens": -1}),
    ("usage", {"total_tokens": True}), ("usage", []), ("scope", []), ("attempt_id", "wrong")])
async def test_self_hashed_invalid_exit_is_rejected_not_only_hash_mismatch(attempt_case, field, value):
    c = attempt_case
    attempt = await completed(c)
    key = attempts.event_id(c.manager, "finished", attempt.attempt_id)
    [row] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_id=?", (key,))
    payload = json.loads(row["payload_json"]) | {field: value}
    await c.manager.db.execute(f"UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?", (json.dumps(payload), _hash(payload), key))
    with pytest.raises(JobError, match="model_attempt_evidence_corrupt"):
        await list_attempts(c.manager, c.actor, c.binding.lease.job_id)


async def test_terminal_nested_admission_error_is_not_retried():
    calls = []
    async def refused():
        calls.append(True)
        raise LLMNonRetryable(attempts=1, last_error=JobError("unresolved_prior_attempt"))
    with pytest.raises(LLMNonRetryable):
        await with_retry(refused, policy=LLMRetryPolicy(max_attempts=5, backoff_initial=0), token=CancellationToken.never())
    assert calls == [True]


async def test_managed_provider_timeout_uses_one_actual_request_even_with_retry_policy(managed, monkeypatch):
    _, manager, _, loop, model, sid, _ = managed
    loop.config.retry_policy = LLMRetryPolicy(max_attempts=5, backoff_initial=0)
    calls = []
    async def timed_out(request, **kwargs):
        calls.append(True)
        raise TimeoutError("transport lost; supplier outcome unknown")
    monkeypatch.setattr(model, "complete", timed_out)
    result = await loop.send("one request only", sid)
    assert result.status == "degraded" and calls == [True]
    [row] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
    assert row["resource_state"] == "unknown" and row["supervision_status"] == "open"
    page = await list_attempts(manager, loop.task_runtime.actor(sid), row["job_id"])
    assert page["items"][0]["status"] == "unknown" and page["items"][0]["estimated"]


@pytest.mark.parametrize("phase", ["prepared", "dispatched"])
async def test_deleted_attempt_parent_cannot_disappear_from_closed_proof(attempt_case, phase):
    c = attempt_case
    attempt = await completed(c)
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (attempts.event_id(c.manager, phase, attempt.attempt_id),))
    with pytest.raises(JobError, match="model_attempt_evidence_corrupt"):
        await closed(c)
    assert (await c.manager.get_job(c.actor, c.binding.lease.job_id)).supervision_status == "open"


async def test_dispatch_rechecks_original_tracking_before_any_provider_call(attempt_case):
    c = attempt_case
    attempt = await prepare(c)
    await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_type='model_attempt_tracking'")
    with pytest.raises(JobError, match="model_attempt_evidence_missing"):
        await attempts.dispatch(attempt)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_dispatched'")


async def test_provider_cost_and_response_id_are_observations_with_unknown_currency(attempt_case):
    c = attempt_case
    attempt = await prepare(c)
    await attempts.dispatch(attempt)
    await attempts.finish(attempt, status="returned", usage={"total_tokens": 10}, estimated=False,
        cost_amount="0.001250", response_id="completion-original-123")
    [item] = (await list_attempts(c.manager, c.actor, c.binding.lease.job_id))["items"]
    assert item["provider_reported_cost"] == "0.001250" and item["provider_cost_currency"] is None
    assert item["provider_response_id"] == "completion-original-123" and item["provider_query_supported"] is False
    assert "billed_amount" not in item


@pytest.mark.parametrize("cost", ["NaN", "Infinity", "-1", "invalid", 0.001, True])
async def test_invalid_cost_is_not_normalized_to_zero(attempt_case, cost):
    c = attempt_case
    attempt = await prepare(c)
    await attempts.dispatch(attempt)
    with pytest.raises(JobError, match="invalid_model_attempt_observation"):
        await attempts.finish(attempt, status="returned", usage={"total_tokens": 10}, estimated=False, cost_amount=cost)
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='model_attempt_finished'")


async def test_attempt_query_char_budget_is_whole_json_not_per_item(attempt_case):
    c = attempt_case
    for n in range(4):
        await completed(c, await prepare(c, nonce=f"item-{n}"))
    page = await list_attempts(c.manager, c.actor, c.binding.lease.job_id, max_chars=2000, limit=4)
    assert len(json.dumps(page, ensure_ascii=False, separators=(",", ":"))) <= 2000
    assert page["truncated"] and page["next_cursor"] and len(page["items"]) < 4


async def test_fresh_main_control_send_can_run_without_erasing_old_unknown_risk(managed, monkeypatch):
    _, manager, runtime, loop, model, sid, _ = managed
    model.root_calls = 10
    original = model.complete
    async def unknown(request, **kwargs):
        raise TimeoutError("test: old supplier request has no receipt")
    monkeypatch.setattr(model, "complete", unknown)
    assert (await loop.send("old request", sid)).status == "degraded"
    [old] = await manager.db.fetchall(f"SELECT * FROM {manager.jobs}")
    assert old["resource_state"] == "unknown"
    monkeypatch.setattr(model, "complete", original)
    assert (await loop.send("new authorized control turn", sid)).status == "completed"
    assert model.root_calls == 11
    current = await manager.get_job(runtime.actor(sid), old["job_id"])
    assert current.resource_state == "unknown" and current.supervision_status == "open"
    page = await list_attempts(manager, runtime.actor(sid), old["job_id"])
    assert page["items"][0]["status"] == "unknown"


async def test_late_original_observation_does_not_modify_superseding_generation(attempt_case):
    c = attempt_case
    attempt = await prepare(c)
    await attempts.dispatch(attempt)
    await c.manager.db.execute(f"UPDATE {c.manager.jobs} SET generation=2,boot_id='new-boot' WHERE job_id=?", (c.binding.lease.job_id,))
    before = await c.manager._row(c.manager.db, c.binding.lease.job_id)
    await attempts.finish(attempt, status="returned", usage={"total_tokens": 10}, estimated=False)
    assert await c.manager._row(c.manager.db, c.binding.lease.job_id) == before
    [event] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='model_attempt_finished'")
    assert json.loads(event["payload_json"])["scope"]["generation"] == 1
    with pytest.raises(JobError, match="stale_execution"):
        await attempts.dispatch(attempt)
