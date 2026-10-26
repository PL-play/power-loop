"""Independent financial evidence cannot manufacture an original model exit."""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.budget_contracts import BudgetConfig, BudgetSubmission, LocalBudgetScope
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.task_runtime import TaskRuntime

from .test_budget_reconcile_192 import reconcile, reopened
from .test_budget_revisions_192 import RevisionHost, evidence, rows, stage
from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import new_send, observed, start

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class InitialHost(RevisionHost):
    async def prepare_initial_usage(self, context, grant, evidence, *, started_at_ms):
        assert context['attempt_id'] == grant.attempt_id
        self.prepared.append((1, evidence.fingerprint))
        return BudgetSubmission(grant.attempt_id, 1, {
            'evidence_hash': evidence.fingerprint, 'usage': evidence.usage,
            'started_at_ms': started_at_ms, 'finished_at_ms': None,
            'outcome': 'unknown', 'details': evidence.details,
        })


@pytest.fixture
async def initial_case(registry_store):
    store, connect = registry_store
    host = InitialHost()
    scopes = tuple(LocalBudgetScope(k, max_calls=2, max_tokens=200, max_inflight=1, binding=k)
        for k in ('authority', 'root', 'owner'))
    config = BudgetConfig('host-budget-v1', scopes, host, max_pending=1)
    runtime = TaskRuntime(JobManager(store), tenant_id='budget-tenant', principal_id='host',
        executor_id='executor', boot_id='boot', boot_contract_hash='a' * 64, budget_config=config)
    c = SimpleNamespace(store=store, connect=connect, manager=runtime.manager, host=host, config=config, runtime=runtime)
    await new_send(c)
    try:
        yield c
    finally:
        await runtime.shutdown(timeout_s=1)


async def counts(c):
    return [json.loads(r['counters_json']) for r in await c.manager.db.fetchall(
        f'SELECT counters_json FROM {c.manager.prefix}task_budget_scopes ORDER BY scope_key')]


async def assert_counts(c, *, tokens, inflight=1, held=0):
    values = await counts(c)
    assert len(values) == 3
    assert all(v == dict(used_calls=1, held_calls=0, used_tokens=tokens, held_tokens=held, inflight=inflight)
        for v in values)


@pytest.mark.parametrize('quality', ['actual', 'estimated', 'unknown'])
async def test_initial_evidence_ack_counts_once_without_model_exit_or_completion(initial_case, quality):
    c = initial_case
    a = await start(c)
    original = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    await c.manager.request_cancel(c.actor, c.binding.lease.job_id, system_key='stop', reason='test')
    before_job = await c.manager.get_job(c.actor, c.binding.lease.job_id)
    value = evidence(12, quality=quality)
    control = await reopened(c)
    receipt = await stage(c, a, value, expected=0, runtime=control)
    assert receipt['revision'] == 1 and not receipt['execution_authority']
    assert c.host.apply_calls == 0
    assert await stage(c, a, value, expected=0, runtime=control) == receipt
    await reconcile(control, c, a)
    assert await stage(c, a, value, expected=0, runtime=control) == receipt
    saved = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert saved['observation'] is None and 'finished_at_ms' not in saved
    assert saved['context'] == original['context'] and saved['grant'] == original['grant']
    assert not saved['pending'] and saved['ack']['revision'] == 1
    assert saved['submission']['request']['started_at_ms'] == original['started_at_ms']
    assert await attempts._read(c.manager, c.manager.db, 'finished', a.attempt_id, required=False) is None
    assert await c.manager.get_job(c.actor, c.binding.lease.job_id) == before_job
    async with c.manager._write() as tx:
        row = await c.manager._authorize(tx, c.actor, c.binding.lease.job_id)
        with pytest.raises(JobError):
            await attempts.require_quiescent(c.manager, tx, row)
    assert control._lifecycle._start_task is None and c.host.reserve_calls == 1 and not c.host.released
    await assert_counts(c, tokens=12 if quality == 'actual' else 0, held=0 if quality == 'actual' else 100)


async def test_lost_initial_ack_and_next_correction_preserve_original_source(initial_case):
    c = initial_case
    a = await start(c)
    value = evidence(12)
    receipt = await stage(c, a, value, expected=0)
    c.host.lost_revision = 1
    with pytest.raises(OSError, match='ACK lost'):
        await reconcile(c.runtime, c, a)
    pending = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    c.host.lost_revision = None
    control = await reopened(c)
    await reconcile(control, c, a)
    assert (await budgets.get_budget(c.manager, c.actor, a.attempt_id))['submission'] == pending['submission']
    await stage(c, a, evidence(9, key='corrected', correction_of=value.fingerprint), runtime=control)
    await reconcile(control, c, a)
    saved = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert saved['initial_usage_evidence']['event_id'] == receipt['event_id']
    assert saved['usage_revision']['revision'] == 2 and saved['observation'] is None
    assert len(c.host.revisions) == 2 and c.host.reserve_calls == 1
    await assert_counts(c, tokens=9)


@pytest.mark.parametrize('returned', [True, False])
@pytest.mark.parametrize('ack_pending', [True, False])
async def test_late_original_observation_changes_only_execution_not_financial_facts(initial_case, returned, ack_pending):
    c = initial_case
    a = await start(c)
    await stage(c, a, evidence(12), expected=0)
    if not ack_pending:
        await reconcile(c.runtime, c, a)
    saved = await observed(c, a, tokens=7, error=None if returned else OSError('unknown'))
    assert saved['observation']['status'] == ('returned' if returned else 'unknown')
    assert saved['observation']['usage']['total_tokens'] == 7
    assert saved['usage_revision']['evidence']['usage']['total_tokens'] == 12
    assert saved['submission']['request']['usage']['total_tokens'] == 12
    await reconcile(await reopened(c), c, a)
    await assert_counts(c, tokens=12, inflight=0 if returned else 1)
    assert len(c.host.revisions) == 1 and c.host.reserve_calls == 1


async def test_observation_projection_crash_recovers_without_reapplying_money(initial_case, monkeypatch):
    c = initial_case
    a = await start(c)
    await stage(c, a, evidence(12), expected=0)
    await reconcile(c.runtime, c, a)
    original = budgets.observe
    async def broken(*args):
        raise OSError('execution projection failed after authentic observation commit')
    monkeypatch.setattr(budgets, 'observe', broken)
    await observed(c, a, tokens=7)
    assert (await budgets.get_budget(c.manager, c.actor, a.attempt_id))['observation'] is None
    monkeypatch.setattr(budgets, 'observe', original)
    one, two = await reopened(c), await reopened(c)
    await asyncio.gather(reconcile(one,c,a), reconcile(two,c,a))
    await assert_counts(c, tokens=12, inflight=0)
    assert len(c.host.revisions) == 1


async def test_two_controls_stage_one_initial_snapshot_and_refuse_conflicting_one(initial_case):
    c = initial_case
    a = await start(c)
    one, two = await reopened(c), await reopened(c)
    receipts = await asyncio.gather(*(stage(c, a, evidence(), expected=0, runtime=r) for r in (one,two)))
    assert receipts[0] == receipts[1]
    before = await rows(c)
    with pytest.raises(JobError, match='initial_evidence_not_ready'):
        await stage(c, a, evidence(13, key='conflict'), expected=0)
    assert await rows(c) == before
    await assert_counts(c, tokens=12)


@pytest.mark.parametrize('fault', ['prepared', 'dispatched', 'tracking', 'provider', 'correction', 'context', 'closed'])
async def test_invalid_initial_source_never_mutates_counters_or_calls_host(initial_case, fault):
    c = initial_case
    a = await start(c)
    value = evidence()
    control = await reopened(c)
    if fault in ('prepared','dispatched','tracking'):
        identity = a.prepared['scope'] if fault == 'tracking' else a.attempt_id
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?',
            (attempts.event_id(c.manager, fault, identity),))
    elif fault == 'provider':
        c.host.prepare_initial_usage = None
    elif fault == 'correction':
        value = replace(value, correction_of='a' * 64, correction_reason='no previous evidence')
    elif fault == 'closed':
        await control.shutdown(timeout_s=1)
    before, counters = await rows(c), await counts(c)
    with pytest.raises(JobError):
        await control.stage_budget_usage_revision('foreign' if fault == 'context' else c.sid,
            job_id=c.binding.lease.job_id, attempt_id=a.attempt_id, expected_revision=0, evidence=value)
    assert await rows(c) == before and await counts(c) == counters
    assert c.host.apply_calls == 0


@pytest.mark.parametrize('frozen', [True, False])
async def test_missing_initial_marker_refuses_even_frozen_submission(initial_case, frozen):
    c = initial_case
    a = await start(c)
    receipt = await stage(c, a, evidence(), expected=0)
    if frozen:
        c.host.lost_revision = 1
        with pytest.raises(OSError):
            await reconcile(c.runtime, c, a)
    await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (receipt['event_id'],))
    calls = c.host.apply_calls
    with pytest.raises(JobError):
        await reconcile(await reopened(c), c, a)
    assert c.host.apply_calls == calls


async def test_stage_failure_rolls_back_initial_projection_and_source(initial_case, monkeypatch):
    c = initial_case
    a = await start(c)
    before, counters = await rows(c), await counts(c)
    save = budgets._save
    async def rollback(*args, **kwargs):
        await save(*args, **kwargs)
        raise OSError('state commit failed')
    monkeypatch.setattr(budgets, '_save', rollback)
    with pytest.raises(OSError):
        await stage(c, a, evidence(), expected=0)
    assert await rows(c) == before and await counts(c) == counters


async def test_existing_finished_observation_requires_original_delivery_not_replacement(initial_case):
    c = initial_case
    a = await start(c)
    await observed(c, a)
    before, counters = await rows(c), await counts(c)
    with pytest.raises(JobError, match='initial_evidence_not_ready'):
        await stage(c, a, evidence(), expected=0)
    assert await rows(c) == before and await counts(c) == counters


async def test_over_bound_initial_usage_closes_policy_but_not_execution(initial_case):
    c = initial_case
    a = await start(c)
    await stage(c, a, evidence(101), expected=0)
    await reconcile(c.runtime, c, a)
    [policy] = await c.manager.db.fetchall(f'SELECT enabled FROM {c.manager.prefix}task_budget_policies')
    assert policy['enabled'] == 0
    await assert_counts(c, tokens=101)


@pytest.mark.parametrize('fault', ['submission', 'ack'])
async def test_wrong_initial_host_receipt_retains_pending_obligation(initial_case, fault):
    c = initial_case
    a = await start(c)
    await stage(c, a, evidence(), expected=0)
    if fault == 'submission':
        original = c.host.prepare_initial_usage
        async def bad(*args, **kwargs):
            return replace(await original(*args, **kwargs), revision=2)
        c.host.prepare_initial_usage = bad
    else:
        original = c.host.apply_usage
        async def bad(submission):
            return replace(await original(submission), revision=2)
        c.host.apply_usage = bad
    with pytest.raises(JobError):
        await reconcile(c.runtime, c, a)
    saved = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert saved['pending'] and saved['ack'] is None and saved['observation'] is None
    await assert_counts(c, tokens=12)


async def test_cancellation_keeps_actual_initial_prepare_port_alive(initial_case):
    c = initial_case
    a = await start(c)
    await stage(c, a, evidence(), expected=0)
    entered, release = asyncio.Event(), asyncio.Event()
    prepare = c.host.prepare_initial_usage
    async def held(*args, **kwargs):
        entered.set()
        await release.wait()
        return await prepare(*args, **kwargs)
    c.host.prepare_initial_usage = held
    control = await reopened(c)
    waiter = asyncio.create_task(reconcile(control, c, a))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert control.budget_control_tasks()
        assert (await budgets.get_budget(c.manager, c.actor, a.attempt_id))['pending']
    finally:
        release.set()
        await asyncio.gather(*control.budget_control_tasks(), return_exceptions=True)
        await control.shutdown(timeout_s=2)
    await reconcile(await reopened(c), c, a)
    await assert_counts(c, tokens=12)


async def test_original_return_and_initial_evidence_race_never_double_counts(initial_case):
    c = initial_case
    a = await start(c)
    results = await asyncio.gather(stage(c,a,evidence(),expected=0), observed(c,a,tokens=7), return_exceptions=True)
    assert not isinstance(results[1], BaseException)
    accepted = not isinstance(results[0], BaseException)
    if not accepted:
        assert isinstance(results[0], JobError)
    await reconcile(await reopened(c),c,a)
    await assert_counts(c, tokens=12 if accepted else 7, inflight=0)
    assert c.host.reserve_calls == 1 and len(c.host.revisions) == 1


async def test_late_exit_projection_rollback_retains_inflight_until_original_retry(initial_case, monkeypatch):
    c = initial_case
    a = await start(c)
    await stage(c,a,evidence(),expected=0)
    await reconcile(c.runtime,c,a)
    save = budgets._save
    async def rollback(manager, tx, state, job, **kwargs):
        await save(manager,tx,state,job,**kwargs)
        if state['observation'] is not None:
            raise OSError('rollback execution projection')
    monkeypatch.setattr(budgets,'_save',rollback)
    await observed(c,a,tokens=7)
    await assert_counts(c,tokens=12,inflight=1)
    assert (await budgets.get_budget(c.manager,c.actor,a.attempt_id))['observation'] is None
    monkeypatch.setattr(budgets,'_save',save)
    await reconcile(await reopened(c),c,a)
    await assert_counts(c,tokens=12,inflight=0)
