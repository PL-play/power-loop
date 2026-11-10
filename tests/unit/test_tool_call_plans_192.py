"""A whole native batch is registered before its first finite callback starts."""

import json

import pytest

from power_loop.runtime.jobs import JobError
from power_loop.runtime.pending_calls import retire_pending_runtime
from power_loop.runtime.pricing import Free
from power_loop.runtime.tool_call_plans import reconcile_pending

from .test_job_registry_192 import registry_store
from .test_managed_delivery_192 import consume_queued
from .test_managed_delivery_192 import controller as _controller
from .test_managed_effects_192 import scenario

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def controller(context):
    from power_loop.runtime.representation import ProjectedRepresentation
    loop, model = await _controller(context)
    loop.config.representation = ProjectedRepresentation()
    from power_loop import HookPoint
    async def original_tool_rows_only(ctx):
        originals = [row for row in await context.store.load_all_messages(context.sid) if row.role == 'tool']
        for message in ctx.messages:
            if message.get('role') == 'tool':
                assert any(row.tool_call_id == message.get('tool_call_id') and row.content == message.get('content')
                           for row in originals), 'a missing original tool reply cannot be fabricated in the provider request'
    loop.hooks.register(HookPoint.LLM_BEFORE, original_tool_rows_only, name='test.plan.no-fabricated-receipt')
    return loop, model


async def test_two_managed_calls_normally_dispatch_exact_original_once_each(registry_store):
    c = await scenario(registry_store, call_count=2)
    assert c.result.status == 'completed' and c.model.calls == 2
    assert c.state.prepared == c.state.dispatched == 2 and c.state.lookups == 0
    jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert len(jobs) == 2 and all(job['generation'] == 1 and job['result_status'] == 'handled' for job in jobs)
    plans = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='tool_call_plan'")
    assert len(plans) == 1
    assert [e['call_id'] for e in json.loads(plans[0]['payload_json'])['entries']] == ['original-call', 'original-call-2']


async def test_lost_first_raw_reply_retains_unstarted_sibling_without_replay(registry_store):
    c = await scenario(registry_store, budgeted=True, raw_receipt_failure=True, call_count=2)
    jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool' ORDER BY generation DESC")
    assert [job['generation'] for job in jobs] == [1, 0]
    assert c.state.prepared == c.state.dispatched == c.model.calls == 1 and c.state.lookups == 0
    source_id = jobs[0]['parent_job_id']
    pending = (await c.store.get_state(c.sid)).pending
    assert pending['tool_call_ids'] == ['original-call', 'original-call-2']
    raw = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
    loop, reviewer = await controller(c)
    try:
        staged = await reconcile_pending(c.runtime, c.actor, source_id)
        assert await reconcile_pending(c.runtime, c.actor, source_id) == staged
        assert len(staged['notices']) == 2 and not staged['execution_authority'] and not staged['acknowledgement_created']
        assert reviewer.calls == 0 and (await c.store.get_state(c.sid)).pending == pending
        negative = await c.manager.get_job(c.actor, jobs[1]['job_id'])
        assert negative.generation == 0 and negative.result_status == 'suppressed' and negative.supervision_status == 'closed'
        await retire_pending_runtime(c.runtime, c.actor, source_id)
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor, source_id, generation=1)
        result = await consume_queued(loop)
        assert result.status == 'completed' and reviewer.calls == 1
        assert (await c.manager.get_job(c.actor, jobs[0]['job_id'])).result_status == 'handled'
        negative = await c.manager.get_job(c.actor, jobs[1]['job_id'])
        assert negative.generation == 0 and negative.result_status == 'suppressed' and negative.supervision_status == 'closed'
        assert (await c.manager.get_job(c.actor, source_id)).supervision_status == 'closed'
        assert c.state.prepared == c.state.dispatched == c.model.calls == 1 and c.state.lookups == 0
        for message in raw:
            assert await c.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (message['id'],)) == message
        assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")
    finally:
        await loop.aclose()


@pytest.mark.parametrize('change', ['same_contract_handler', 'same_handler_registration'])
async def test_batch_cannot_dispatch_replacement_registered_after_plan_commit(registry_store, change):
    from power_loop import HookPoint
    from power_loop.runtime.managed_effects import FunctionManagedTool, ManagedToolResult

    replacements = []
    async def configure(loop, state):
        original = loop.registry.get('post_once')
        adapter = original.handler.adapter
        async def replacement(context, reference):
            replacements.append(context.tool_call_id)
            return ManagedToolResult('Replacement executed', {'receipt': 'replacement'})
        alternate = FunctionManagedTool(adapter.contract, adapter.on_prepare, replacement, adapter.on_recover)
        rebound = alternate.bind(loop.task_runtime, original.definition)
        async def replace_after_plan(ctx):
            if ctx.tool_name == 'post_once':
                loop.registry.register(original.definition,
                    rebound if change == 'same_contract_handler' else original.handler,
                    overwrite=True)
        loop.hooks.register(HookPoint.TOOL_BEFORE, replace_after_plan, name='test.plan.actual-registration')
    c = await scenario(registry_store, configure=configure)
    assert replacements == []
    assert c.state.prepared == c.state.dispatched == 0
    [intent] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert intent['generation'] == 0 and intent['resource_state'] == 'stopped'
    assert intent['result_status'] == 'suppressed' and intent['supervision_status'] == 'closed'
    [raw] = await c.manager.db.fetchall(f"SELECT content FROM {c.store.t.messages} WHERE role='tool'")
    assert 'tool_call_plan_binding_changed' in raw['content']


async def test_every_managed_intent_and_seal_exist_before_first_prepare(registry_store):
    observations = []
    async def inspect(context, arguments):
        manager = context.runtime.manager
        rows = await manager.db.fetchall(f"SELECT generation FROM {manager.jobs} WHERE kind='background_tool' ORDER BY generation")
        plans = await manager.db.fetchall(f"SELECT event_type FROM {manager.events} WHERE event_type IN ('tool_call_plan','tool_call_plan_seal')")
        observations.append(([row['generation'] for row in rows], {p['event_type'] for p in plans}))
    c = await scenario(registry_store, call_count=3, inspect_prepare=inspect)
    assert c.result.status == 'completed' and c.state.prepared == c.state.dispatched == 3
    assert [value[0] for value in observations] == [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
    assert all(value[1] == {'tool_call_plan'} for value in observations)


@pytest.mark.parametrize('stage', ['first_registration', 'second_registration', 'plan', 'commit', 'commit_ack'])
async def test_whole_plan_transaction_never_dispatches_partial_or_unacknowledged_intentions(registry_store, stage):
    from ._tool_plan_faults_192 import fail_plan_once
    failures = []
    async def configure(loop, state):
        failures.append(fail_plan_once(loop.task_runtime.manager, stage))
    c = await scenario(registry_store, call_count=2, budgeted=True, configure=configure, expected_error='test_plan_' + stage)
    assert failures[0].fired and c.model.calls == 1
    assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
    jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    plans = await c.manager.db.fetchall(f"SELECT event_type FROM {c.manager.events} WHERE event_type IN ('tool_call_plan','tool_call_plan_seal')")
    assert len(jobs) == (2 if stage == 'commit_ack' else 0)
    assert len(plans) == (1 if stage == 'commit_ack' else 0)
    assert all(job['generation'] == 0 and job['resource_state'] == 'not_started' for job in jobs)
    assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")
    pending = (await c.store.get_state(c.sid)).pending
    assert pending['tool_call_ids'] == ['original-call', 'original-call-2']
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    # A failed/unknown batch remains queryable. Fault recovery does not
    # reconstruct missing plan/exit proofs or replay its callbacks.
    assert (await c.manager.get_job(c.actor, source['job_id'])).status == 'model_closed'
    assert c.result.status in {'failed', 'interrupted'}
    assert c.state.dispatched == 0



@pytest.mark.parametrize('registered_plain', [True, False])
async def test_mixed_unsupported_intent_cannot_be_silently_dropped_or_partly_resolved(registry_store, registered_plain):
    from power_loop import ToolDefinition
    plain_calls = []
    async def configure(loop, state):
        if registered_plain:
            loop.registry.register(ToolDefinition('inspect_note', '', input_schema={'type': 'object', 'properties': {'text': {'type': 'string'}}}, pricing=Free("test_fixture")),
                                        lambda **args: plain_calls.append(args))
    c = await scenario(registry_store, budgeted=True, raw_receipt_failure=True, configure=configure,
                       call_names=['post_once', 'inspect_note', 'post_once'])
    [plan] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE event_type='tool_call_plan'")
    entries = json.loads(plan['payload_json'])['entries']
    assert [e['call_id'] for e in entries] == ['original-call', 'original-call-2', 'original-call-3']
    assert entries[1]['recovery'] is entries[1]['job_id'] is None
    assert (entries[1]['registration'] is not None) == registered_plain
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    jobs = await c.manager.db.fetchall(f"SELECT generation FROM {c.manager.jobs} WHERE kind='background_tool' ORDER BY generation")
    assert [r['generation'] for r in jobs] == [0, 1]
    pending = (await c.store.get_state(c.sid)).pending
    loop, reviewer = await controller(c)
    try:
        with pytest.raises(JobError, match='pending_call_adapter_required'):
            await reconcile_pending(c.runtime, c.actor, source['job_id'])
        assert (await c.store.get_state(c.sid)).pending == pending
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type IN ('managed_effect_unstarted','managed_effect_delivery','pending_call_disposition')")
        assert reviewer.calls == 0 and plain_calls == []
        assert c.state.prepared == c.state.dispatched == c.model.calls == 1 and c.state.lookups == 0
    finally:
        await loop.aclose()




@pytest.mark.parametrize('part', ['too_many_calls', 'oversized_arguments'])
async def test_plan_limits_refuse_all_callbacks_and_keep_original_responsibility(registry_store, part):
    count, arguments = (65, 'hello') if part == 'too_many_calls' else (2, 'x' * 140000)
    expected = 'tool_call_plan_invalid_call_set' if part == 'too_many_calls' else 'tool_call_plan_raw_unavailable'
    c = await scenario(registry_store, call_count=count, argument_text=arguments, expected_error=expected)
    assert c.result.status == 'failed' and c.model.calls == 1
    assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
    pending = (await c.store.get_state(c.sid)).pending
    assert len(pending['tool_call_ids']) == count
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    assert source['resource_state'] == 'unknown' and source['supervision_status'] == 'open'
