"""A positive SDK pre-admission checkpoint, never absence-based non-execution."""

import pytest

from power_loop.runtime.jobs import JobError

from ._tool_plan_faults_192 import fail_plan_once
from .test_job_registry_192 import registry_store
from .test_managed_effects_192 import scenario

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('stage', ['first_registration', 'seal', 'commit', 'commit_ack'])
async def test_original_prepare_failure_records_abort_and_retains_unprocessed_pending(registry_store, stage):
    async def configure(loop, state):
        fail_plan_once(loop.task_runtime.manager, stage)
    c = await scenario(registry_store, call_count=2, budgeted=True, configure=configure,
                       expected_error='test_plan_' + stage)
    assert c.result.status == 'failed'
    assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
    assert c.model.calls == c.host.reserve_calls == 1
    # Original assistant calls remain queryable; no action can be replayed by
    # interpreting the absence of an obsolete admission checkpoint.
    raw = await c.store.load_all_messages(c.sid)
    calls = [call for row in raw for call in (row.tool_calls or [])]
    assert {call['id'] for call in calls} == {'original-call', 'original-call-2'}
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='managed_effect_dispatch'")





async def test_abort_checkpoint_cannot_be_reused_from_host_or_dispatch_or_claim(registry_store, monkeypatch):
    from power_loop.runtime import tool_call_abort, tool_call_plans
    from power_loop.runtime.invocation import invocation_context
    actual = tool_call_abort._record
    captured, rejected = [], []
    async def observe(attempt):
        value = await actual(attempt)
        captured.append(attempt)
        runtime = attempt.execution.runtime
        manager, actor = runtime.manager, attempt.execution.actor
        jobs = await manager.db.fetchall(f"SELECT * FROM {manager.jobs} WHERE kind='background_tool'")
        assert len(jobs) == 2
        for job in jobs:
            with pytest.raises(JobError, match='tool_call_batch_aborted'):
                await manager.claim_job(actor, job['job_id'], executor_id='forbidden', boot_id='boot', expected_version=1)
        entry = value['entries'][0]
        with invocation_context(actor.session_id, attempt.ref.seq, entry['call_id']):
            with pytest.raises(JobError, match='tool_call_plan_original_requires_reconciliation'):
                await tool_call_plans.dispatch(attempt.pipeline, entry['tool_name'], {'text': 'hello'}, attempt.pipeline.tool_registry.get(entry['tool_name']))
        rejected.append(True)
        return value
    monkeypatch.setattr(tool_call_abort, '_record', observe)
    async def configure(loop, state):
        fail_plan_once(loop.task_runtime.manager, 'commit_ack')
    c = await scenario(registry_store, call_count=2, configure=configure, expected_error='test_plan_commit_ack')
    assert rejected == [True]
    with pytest.raises(JobError, match='original_stack_required'):
        await actual(captured[0])
    assert c.state.prepared == c.state.dispatched == 0


async def test_exception_after_actual_admission_cannot_be_reclassified_as_not_admitted(registry_store, monkeypatch):
    from power_loop.runtime import tool_call_plans
    actual = tool_call_plans._prepare_children
    admitted = []
    async def after_grant(pipeline, attempt):
        await actual(pipeline, attempt)
        assert pipeline._native_tool_batch_bindings is not None
        admitted.append(True)
        raise OSError('test_after_actual_admission')
    monkeypatch.setattr(tool_call_plans, '_prepare_children', after_grant)
    c = await scenario(registry_store, expected_error='test_after_actual_admission')
    assert admitted == [True]
    assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='tool_call_abort'")
    assert c.state.dispatched == 0 and c.model.calls == 1






async def test_positive_checkpoint_reader_does_not_grant_another_principal_access(registry_store):
    from dataclasses import replace

    from power_loop.runtime.tool_call_abort import read_abort
    async def configure(loop, state):
        fail_plan_once(loop.task_runtime.manager, 'seal')
    c = await scenario(registry_store, configure=configure, expected_error='test_plan_seal')
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    pending = (await c.store.get_state(c.sid)).pending
    with pytest.raises(JobError):
        await read_abort(c.manager, replace(c.actor, principal_id='another'), source['job_id'], assistant_seq=pending['assistant_seq'])
    assert c.state.dispatched == 0 and c.model.calls == 1
