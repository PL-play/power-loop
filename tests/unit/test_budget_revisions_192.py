"""Trusted late snapshots use the original attempt, counters and durable ACK."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.budget_contracts import (
    BudgetAck,
    BudgetConfig,
    BudgetSubmission,
    BudgetUsageEvidence,
    LocalBudgetScope,
)
from power_loop.runtime.jobs import JobError, JobManager, _hash
from power_loop.runtime.task_runtime import TaskRuntime

from .test_budget_reconcile_192 import reconcile, reopened
from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host, counters, new_send, observed, start

__all__ = ["registry_store"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class RevisionHost(Host):
    def __init__(self):
        super().__init__()
        self.revisions, self.prepared = {}, []
        self.lost_revision = None

    async def prepare_usage_revision(self, context, grant, previous_submission, previous_ack, evidence):
        assert context["attempt_id"] == grant.attempt_id == previous_ack.attempt_id
        assert previous_submission.request_hash == previous_ack.request_hash
        revision = previous_submission.revision + 1
        self.prepared.append((revision, evidence.fingerprint))
        return BudgetSubmission(context["attempt_id"], revision,
            {"evidence_hash": evidence.fingerprint, "usage": evidence.usage, "details": evidence.details,
             "previous_hash": previous_ack.request_hash})

    async def apply_usage(self, submission):
        self.apply_calls += 1
        key = (submission.attempt_id, submission.revision)
        ack = BudgetAck(submission.attempt_id, submission.revision, submission.request_hash)
        assert self.revisions.setdefault(key, ack) == ack
        if self.lost_revision == submission.revision:
            raise OSError("committed usage revision ACK lost")
        return ack


@pytest.fixture
async def late_case(registry_store, request):
    store, connect = registry_store
    host = RevisionHost()
    scopes = (LocalBudgetScope("shared", max_calls=4, max_tokens=1000),)
    if getattr(request, "param", False):
        scopes += (LocalBudgetScope("root", max_calls=4, max_tokens=1000, binding="root"),
                   LocalBudgetScope("owner", max_calls=4, max_tokens=1000, binding="owner"))
    config = BudgetConfig("host-budget-v1", scopes, host,
                          max_pending=1)
    runtime = TaskRuntime(JobManager(store), tenant_id="budget-tenant", principal_id="host",
        executor_id="executor", boot_id="boot", boot_contract_hash="a" * 64, budget_config=config)
    c = SimpleNamespace(store=store, connect=connect, manager=runtime.manager, host=host, config=config, runtime=runtime)
    await new_send(c)
    try:
        yield c
    finally:
        await runtime.shutdown(timeout_s=1)


def evidence(tokens=12, *, key="late-1", quality="actual", correction_of=None):
    return BudgetUsageEvidence(key, "original-provider", key, quality,
        {} if quality == "unknown" else {"total_tokens": tokens},
        correction_of=correction_of, correction_reason="Supplier correction" if correction_of else None)


async def stage(c, a, value, *, runtime=None, expected=1):
    return await (runtime or c.runtime).stage_budget_usage_revision(c.sid,
        job_id=c.binding.lease.job_id, attempt_id=a.attempt_id, expected_revision=expected, evidence=value)


async def rows(c):
    return await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} ORDER BY event_id")


async def test_unknown_original_after_cancel_and_disabled_policy_corrects_tokens_not_resource_exit(late_case):
    c = late_case
    a = await start(c)
    original = await observed(c, a, error=OSError("unknown provider completion"))
    await c.manager.request_cancel(c.actor, c.binding.lease.job_id, system_key="stop", reason="test")
    await c.manager.db.execute(f"UPDATE {c.manager.prefix}task_budget_policies SET enabled=0 WHERE policy_key=?",
                             (original["policy_key"],))
    before_job = await c.manager.get_job(c.actor, c.binding.lease.job_id)
    runtime = await reopened(c)
    value = evidence(12)
    receipt = await stage(c, a, value, runtime=runtime)
    assert receipt["revision"] == 2 and not receipt["execution_authority"]
    assert await stage(c, a, value, runtime=runtime) == receipt
    result = await reconcile(runtime, c, a)
    assert result["revision"] == 2 and result["usage_acknowledged"]
    assert await stage(c, a, value, runtime=runtime) == receipt
    assert await counters(c) == dict(used_calls=1, held_calls=0, used_tokens=12, held_tokens=0, inflight=1)
    assert await c.manager.get_job(c.actor, c.binding.lease.job_id) == before_job
    assert runtime._lifecycle._start_task is None and c.host.reserve_calls == 1 and not c.host.released
    saved = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert saved["observation"] == original["observation"] and not saved["pending"]


@pytest.mark.parametrize("tokens", [5, 15, 105])
async def test_actual_correction_changes_all_original_scope_counters_once(late_case, tokens):
    c = late_case
    a = await start(c)
    original = await observed(c, a, tokens=10)
    before = await counters(c)
    with pytest.raises(JobError, match="budget_actual_correction_required"):
        await stage(c, a, evidence(tokens))
    assert await counters(c) == before
    value = evidence(tokens, correction_of=_hash(original["observation"]))
    one, two = await reopened(c), await reopened(c)
    results = await asyncio.gather(stage(c, a, value, runtime=one), stage(c, a, value, runtime=two))
    assert results[0] == results[1]
    await asyncio.gather(reconcile(one, c, a), reconcile(two, c, a))
    assert (await counters(c))["used_tokens"] == tokens and (await counters(c))["used_calls"] == 1
    assert len(c.host.revisions) == 2 and c.host.reserve_calls == 1
    [policy] = await c.manager.db.fetchall(f"SELECT enabled FROM {c.manager.prefix}task_budget_policies")
    assert policy["enabled"] == (0 if tokens > 100 else 1)


async def test_pending_conflicting_and_stale_revisions_are_not_accepted_or_charged(late_case):
    c = late_case
    a = await start(c)
    original = await observed(c, a)
    await stage(c, a, evidence(correction_of=_hash(original["observation"])))
    before = await rows(c)
    with pytest.raises(JobError, match="budget_revision_pending"):
        await stage(c, a, evidence(14, key="other"))
    with pytest.raises(JobError, match="budget_revision_evidence_conflict"):
        await stage(c, a, evidence(14))
    assert await rows(c) == before
    with pytest.raises(JobError, match="budget_reconciliation_required"):
        await start(c)
    await reconcile(c.runtime, c, a)
    with pytest.raises(JobError, match="budget_revision_changed"):
        await stage(c, a, evidence(14, key="other"))
    with pytest.raises(JobError, match="budget_correction_source_mismatch"):
        await stage(c, a, evidence(14, key="other", correction_of="a" * 64), expected=2)


async def test_staging_rollback_and_new_runtime_recovery_never_repeat_local_delta(late_case, monkeypatch):
    c = late_case
    a = await start(c)
    await observed(c, a, error=OSError())
    before, counts = await rows(c), await counters(c)
    save = budgets._save
    async def fail(*args, **kwargs):
        await save(*args, **kwargs)
        raise OSError("staging transaction rollback")
    with monkeypatch.context() as patch:
        patch.setattr(budgets, "_save", fail)
        with pytest.raises(OSError, match="staging transaction"):
            await stage(c, a, evidence())
    assert await rows(c) == before and await counters(c) == counts
    await stage(c, a, evidence())
    c.host.lost_revision = 2
    with pytest.raises(OSError, match="ACK lost"):
        await reconcile(c.runtime, c, a)
    staged = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert staged["pending"] and staged["submission"]["revision"] == 2
    c.host.lost_revision = None
    await reconcile(await reopened(c), c, a)
    saved = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert saved["submission"] == staged["submission"] and not saved["pending"]
    assert len(c.host.revisions) == 2 and (await counters(c))["used_tokens"] == 12


async def test_late_old_ack_cannot_clear_new_pending_revision(late_case):
    c = late_case
    c.host.lost_revision = 1
    a = await start(c)
    await observed(c, a, error=OSError())
    c.host.lost_revision = None
    entered, release = asyncio.Event(), asyncio.Event()
    apply = c.host.apply_usage
    async def held(submission):
        ack = await apply(submission)
        if not entered.is_set():
            entered.set()
            await release.wait()
        return ack
    c.host.apply_usage = held
    old_runtime, fresh = await reopened(c), await reopened(c)
    task = asyncio.create_task(reconcile(old_runtime, c, a))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await reconcile(fresh, c, a)
        await stage(c, a, evidence(), runtime=fresh)
        release.set()
        with pytest.raises(JobError, match="budget_submission_conflict"):
            await task
        assert (await budgets.get_budget(c.manager, c.actor, a.attempt_id))["pending"]
        await reconcile(fresh, c, a)
        assert (await counters(c))["used_tokens"] == 12 and len(c.host.revisions) == 2
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("fault", ["quality", "missing", "provider", "noncallable", "closed", "scope", "revision"])
async def test_invalid_evidence_context_never_stages_or_calls_a_provider(late_case, fault):
    c = late_case
    a = await start(c)
    await observed(c, a)
    runtime = await reopened(c)
    value = evidence(10)
    if fault == "quality":
        value = evidence(10, quality="estimated")
    if fault == "missing":
        value = replace(value, usage={"prompt_tokens": 10})
    if fault == "provider":
        runtime = await reopened(c, config=replace(c.config, provider=Host()))
    if fault == "noncallable":
        c.host.prepare_usage_revision = 1
    if fault == "closed":
        await runtime.shutdown(timeout_s=1)
    before = await rows(c)
    calls = c.host.apply_calls
    with pytest.raises(JobError):
        await runtime.stage_budget_usage_revision("foreign" if fault == "scope" else c.sid,
            job_id=c.binding.lease.job_id, attempt_id=a.attempt_id,
            expected_revision=True if fault == "revision" else 1, evidence=value)
    assert await rows(c) == before and c.host.apply_calls == calls


async def test_late_preparation_cannot_overwrite_a_later_staged_revision(late_case):
    c = late_case
    a = await start(c)
    await observed(c, a, error=OSError())
    first = evidence(12)
    await stage(c, a, first)
    entered, release = asyncio.Event(), asyncio.Event()
    prepare = c.host.prepare_usage_revision
    async def held(*args):
        value = await prepare(*args)
        if not entered.is_set():
            entered.set()
            await release.wait()
        return value
    c.host.prepare_usage_revision = held
    old, fresh = await reopened(c), await reopened(c)
    task = asyncio.create_task(reconcile(old, c, a))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await reconcile(fresh, c, a)
        await stage(c, a, evidence(5, key="next", correction_of=first.fingerprint), runtime=fresh, expected=2)
        release.set()
        with pytest.raises(JobError, match="budget_revision_changed"):
            await task
        pending = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
        assert pending["pending"] and pending["submission"] is None and pending["usage_revision"]["revision"] == 3
        await reconcile(fresh, c, a)
        assert (await counters(c))["used_tokens"] == 5
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("port", ["prepare_usage_revision", "apply_usage"])
async def test_cancelled_waiter_retains_real_revision_port_and_reuses_its_result(late_case, port):
    c = late_case
    a = await start(c)
    await observed(c, a, error=OSError())
    await stage(c, a, evidence())
    entered, release = asyncio.Event(), asyncio.Event()
    original = getattr(c.host, port)
    calls = []
    async def held(*args):
        calls.append(args)
        entered.set()
        await release.wait()
        return await original(*args)
    setattr(c.host, port, held)
    runtime = await reopened(c)
    task = asyncio.create_task(reconcile(runtime, c, a))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(runtime.budget_control_tasks()) == 1
        assert (await budgets.get_budget(c.manager, c.actor, a.attempt_id))["pending"]
        release.set()
        await reconcile(runtime, c, a)
        assert len(calls) == 1 and not runtime.budget_control_tasks()
        assert (await counters(c))["used_tokens"] == 12
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.shutdown(timeout_s=1)


@pytest.mark.parametrize("late_case", [True], indirect=True)
async def test_successive_corrections_update_each_original_scope_without_counting_calls_again(late_case):
    import json
    c = late_case
    a = await start(c)
    original = await observed(c, a)
    first = evidence(12, correction_of=_hash(original["observation"]))
    receipt = await stage(c, a, first)
    await reconcile(c.runtime, c, a)
    second = evidence(5, key="correction-2", correction_of=first.fingerprint)
    await stage(c, a, second, expected=2)
    await reconcile(c.runtime, c, a)
    assert await stage(c, a, first) == receipt
    scopes = await c.manager.db.fetchall(f"SELECT counters_json FROM {c.manager.prefix}task_budget_scopes")
    assert len(scopes) == 3
    assert all(json.loads(s["counters_json"]) == dict(
        used_calls=1, held_calls=0, used_tokens=5, held_tokens=0, inflight=0) for s in scopes)
    assert len(c.host.revisions) == 3 and c.host.reserve_calls == 1


@pytest.mark.parametrize("fault", ["missing_marker", "frozen_missing_marker", "bad_submission", "wrong_ack"])
async def test_revision_failures_retain_pending_without_new_reservation(late_case, fault):
    c = late_case
    a = await start(c)
    await observed(c, a, error=OSError())
    receipt = await stage(c, a, evidence())
    if fault == "frozen_missing_marker":
        c.host.lost_revision = 2
        with pytest.raises(OSError):
            await reconcile(c.runtime, c, a)
        c.host.lost_revision = None
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (receipt["event_id"],))
    if fault == "missing_marker":
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE event_id=?", (receipt["event_id"],))
    if fault == "bad_submission":
        prepare = c.host.prepare_usage_revision
        async def wrong(*args):
            value = await prepare(*args)
            return replace(value, revision=1)
        c.host.prepare_usage_revision = wrong
    if fault == "wrong_ack":
        apply = c.host.apply_usage
        async def wrong(submission):
            return replace(await apply(submission), request_hash="0" * 64)
        c.host.apply_usage = wrong
    calls = c.host.apply_calls
    with pytest.raises(JobError):
        await reconcile(c.runtime, c, a)
    state = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert state["pending"] and state["ack"] is None
    assert c.host.reserve_calls == 1 and not c.host.released
    if fault in ("missing_marker", "frozen_missing_marker"):
        assert c.host.apply_calls == calls


@pytest.mark.parametrize("changes", [
    {"token_quality": "invented"}, {"usage": {"total_tokens": True}}, {"usage": {"total_tokens": -1}},
    {"usage": {"total_tokens": 2**63}}, {"usage": {}}, {"usage": {"invented": 3}},
    {"token_quality": "unknown"}, {"correction_of": "a" * 64}, {"correction_reason": "No reference"},
    {"source": ""}, {"source_record_id": " leading"}, {"details": {"api_key": "forbidden"}},
])
async def test_evidence_rejects_unbounded_or_ambiguous_data(changes):
    with pytest.raises(JobError):
        replace(evidence(), **changes)
