"""New control hosts seal original unpaid attempts without old execution rights."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from power_loop import LLMRequest
from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.jobs import JobError

from .test_budget_reconcile_192 import reopened
from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case, counters, start
from .test_unstarted_budget_attempts_192 import prepared

__all__ = ['registry_store', 'budget_case']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def release(runtime, c, attempt):
    return await runtime.release_unstarted_budget(c.sid,
        job_id=attempt.binding.lease.job_id, attempt_id=attempt.attempt_id)


@pytest.mark.parametrize('stage', ['host_reserving', 'host_reserved'])
async def test_new_control_closes_original_reservation_without_starting_boot_or_rewriting_job(budget_case, monkeypatch, stage):
    c = budget_case
    a = await prepared(c)
    lookup = c.host.lookup
    async def unavailable(*args):
        raise OSError('host acknowledgement unavailable')
    if stage == 'host_reserving':
        monkeypatch.setattr(c.host, 'lookup', unavailable)
        with pytest.raises(OSError):
            await budgets.reserve(a, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
        monkeypatch.setattr(c.host, 'lookup', lookup)
    else:
        await budgets.reserve(a, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    before = await c.manager.get_job(c.actor, a.binding.lease.job_id)
    messages = await c.store.load_all_messages(c.sid)
    assert (await budgets.get_budget(c.manager, c.actor, a.attempt_id))['phase'] == stage
    runtime = await reopened(c)
    try:
        result = await release(runtime, c, a)
        assert result['released'] and not result['execution_authority']
        assert result == await release(runtime, c, a)
        assert runtime._lifecycle._start_task is None
        assert await c.manager.get_job(c.actor, a.binding.lease.job_id) == before
        assert await c.store.load_all_messages(c.sid) == messages
        assert await counters(c) == dict(used_calls=0, held_calls=0, used_tokens=0, held_tokens=0, inflight=0)
        assert c.host.reserve_calls == len(c.host.released) == 1 and not c.host.applied
        with pytest.raises(JobError, match='budget_dispatch_not_authorized'):
            await attempts.dispatch(a)
    finally:
        await runtime.shutdown(timeout_s=1)
        await runtime.manager.store.close()


@pytest.mark.parametrize('fault', ['lookup', 'release', 'release_ack'])
async def test_failed_host_release_retains_local_risk_until_original_retry(budget_case, monkeypatch, fault):
    c = budget_case
    a = await prepared(c)
    # Leave a durable intent with no confirmed grant: reserve committed but lookup failed.
    lookup, host_release = c.host.lookup, c.host.release
    async def lost_lookup(*args):
        raise OSError('original lookup unavailable')
    monkeypatch.setattr(c.host, 'lookup', lost_lookup)
    with pytest.raises(OSError):
        await budgets.reserve(a, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    if fault != 'lookup':
        monkeypatch.setattr(c.host, 'lookup', lookup)
        async def lost_release(submission):
            if fault == 'release_ack':
                await host_release(submission)
            raise OSError('original release unavailable')
        monkeypatch.setattr(c.host, 'release', lost_release)
    runtime = await reopened(c)
    try:
        before = await counters(c)
        with pytest.raises(OSError):
            await release(runtime, c, a)
        state = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
        assert state['phase'] == 'dispatch_closed' and state['pending']
        assert await counters(c) == before
        with pytest.raises(JobError, match='budget_dispatch_not_authorized'):
            await attempts.dispatch(a)
        monkeypatch.setattr(c.host, 'lookup', lookup)
        monkeypatch.setattr(c.host, 'release', host_release)
        await release(runtime, c, a)
        assert await counters(c) == dict(used_calls=0, held_calls=0, used_tokens=0, held_tokens=0, inflight=0)
        assert len(c.host.released) == 1 and c.host.reserve_calls == 1
    finally:
        await runtime.shutdown(timeout_s=1)
        await runtime.manager.store.close()


async def test_dispatched_attempt_never_refunds_from_a_new_host(budget_case):
    c = budget_case
    a = await start(c)
    runtime = await reopened(c)
    try:
        before = await counters(c)
        with pytest.raises(JobError, match='budget_release_not_permitted'):
            await release(runtime, c, a)
        assert await counters(c) == before and not c.host.released
    finally:
        await runtime.shutdown(timeout_s=1)
        await runtime.manager.store.close()


@pytest.mark.parametrize('fault', ['foreign_session', 'provider_changed', 'missing_preparation'])
async def test_original_authority_and_preparation_are_required_before_any_release(budget_case, fault):
    c = budget_case
    a = await prepared(c)
    await budgets.reserve(a, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    runtime = await reopened(c, config=replace(c.config, provider_key='different') if fault == 'provider_changed' else c.config)
    sid = await c.store.create_session() if fault == 'foreign_session' else c.sid
    if fault == 'missing_preparation':
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?',
            (attempts.event_id(c.manager, 'prepared', a.attempt_id),))
    try:
        before = await counters(c)
        with pytest.raises(JobError):
            await runtime.release_unstarted_budget(sid, job_id=a.binding.lease.job_id, attempt_id=a.attempt_id)
        assert await counters(c) == before and not c.host.released
        assert runtime._lifecycle._start_task is None
    finally:
        await runtime.shutdown(timeout_s=1)
        await runtime.manager.store.close()


async def test_dispatch_and_mechanical_release_have_one_atomic_winner(budget_case):
    c = budget_case
    a = await prepared(c)
    await budgets.reserve(a, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    runtime = await reopened(c)
    try:
        dispatched, released = await asyncio.gather(attempts.dispatch(a), release(runtime, c, a), return_exceptions=True)
        state = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
        if state['phase'] == 'released':
            assert isinstance(dispatched, JobError) and not isinstance(released, BaseException)
            assert len(c.host.released) == 1
        else:
            assert state['phase'] == 'dispatch_authorized'
            assert not isinstance(dispatched, BaseException) and isinstance(released, JobError)
            assert not c.host.released and (await counters(c))['held_calls'] == 1
    finally:
        await runtime.shutdown(timeout_s=1)
        await runtime.manager.store.close()


async def test_scanner_cannot_close_live_attempt_but_explicit_cancel_allows_recovery(budget_case):
    c = budget_case
    a = await prepared(c)
    await budgets.reserve(a, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    runtime = await reopened(c)
    try:
        before = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
        held = await counters(c)
        with pytest.raises(JobError, match='budget_release_requires_cancellation'):
            await runtime.release_unstarted_budget(c.sid, job_id=a.binding.lease.job_id,
                attempt_id=a.attempt_id, require_cancelled=True)
        assert await budgets.get_budget(c.manager, c.actor, a.attempt_id) == before
        assert await counters(c) == held and not c.host.released
        await c.manager.request_cancel(c.actor, a.binding.lease.job_id, system_key='stop', reason='test')
        stopped = await c.manager.get_job(c.actor, a.binding.lease.job_id)
        result = await runtime.release_unstarted_budget(c.sid, job_id=a.binding.lease.job_id,
            attempt_id=a.attempt_id, require_cancelled=True)
        assert result['released'] and not result['execution_authority']
        assert await c.manager.get_job(c.actor, a.binding.lease.job_id) == stopped
        assert (await counters(c))['held_calls'] == 0
    finally:
        await runtime.shutdown(timeout_s=1)
        await runtime.manager.store.close()


async def test_scanner_retries_original_dispatch_closure_without_inventing_cancellation(budget_case, monkeypatch):
    c = budget_case
    a = await prepared(c)
    await budgets.reserve(a, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    runtime = await reopened(c)
    original = c.host.release
    async def lost(*args):
        raise OSError('host unavailable')
    monkeypatch.setattr(c.host, 'release', lost)
    try:
        with pytest.raises(OSError):
            await release(runtime, c, a)
        assert (await c.manager.get_job(c.actor, a.binding.lease.job_id)).cancel_requested_at is None
        monkeypatch.setattr(c.host, 'release', original)
        result = await runtime.release_unstarted_budget(c.sid, job_id=a.binding.lease.job_id,
            attempt_id=a.attempt_id, require_cancelled=True)
        assert result['released'] and (await counters(c))['held_calls'] == 0
        assert len(c.host.released) == 1
    finally:
        await runtime.shutdown(timeout_s=1)
        await runtime.manager.store.close()


async def test_two_fresh_controllers_release_same_original_risk_once(budget_case):
    c = budget_case
    a = await prepared(c)
    await budgets.reserve(a, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    first, second = await reopened(c), await reopened(c)
    try:
        results = await asyncio.gather(release(first, c, a), release(second, c, a), return_exceptions=True)
        assert any(not isinstance(r, BaseException) for r in results)
        for result in results:
            if isinstance(result, BaseException):
                assert isinstance(result, JobError) and result.code == 'budget_submission_conflict'
        # A conflicting in-flight preparation must re-read the canonical submission.
        assert await release(first, c, a) == await release(second, c, a)
        assert (await counters(c))['held_calls'] == 0 and len(c.host.released) == 1
        assert not c.host.applied
    finally:
        for runtime in (first, second):
            await runtime.shutdown(timeout_s=1)
            await runtime.manager.store.close()


async def test_cancelled_ancestor_authorizes_child_release_without_mutating_child(budget_case):
    c = budget_case
    root = c.binding
    child = await c.manager.register_job(c.actor, kind='send', parent=root.lease,
        session_id=c.sid, system_key='nested-send', request={'model_attempt_protocol': 1})
    c.binding = await c.runtime.claim(c.actor, child)
    await attempts.enable(c.binding)
    a = await prepared(c)
    await budgets.reserve(a, LLMRequest(messages=[], max_tokens=20), service=c.host.service)
    await c.manager.request_cancel(c.actor, root.lease.job_id, system_key='root-stop', reason='test')
    before = await c.manager.get_job(c.actor, child.job.job_id)
    assert before.cancel_requested_at is None
    runtime = await reopened(c)
    try:
        result = await runtime.release_unstarted_budget(c.sid, job_id=child.job.job_id,
            attempt_id=a.attempt_id, require_cancelled=True)
        assert result['released'] and not result['execution_authority']
        assert await c.manager.get_job(c.actor, child.job.job_id) == before
        assert (await counters(c))['held_calls'] == 0 and len(c.host.released) == 1
    finally:
        await runtime.shutdown(timeout_s=1)
        await runtime.manager.store.close()
