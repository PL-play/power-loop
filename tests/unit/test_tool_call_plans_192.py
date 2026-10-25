"""A whole native batch is registered before its first finite callback starts."""
import json

import pytest

from power_loop.runtime.jobs import JobError
from power_loop.runtime.pending_calls import retire_pending
from power_loop.runtime.tool_call_plans import reconcile_pending

from .test_job_registry_192 import registry_store
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
        staged = await reconcile_pending(loop, c.actor, source_id)
        assert await reconcile_pending(loop, c.actor, source_id) == staged
        assert len(staged['notices']) == 2 and not staged['execution_authority'] and not staged['acknowledgement_created']
        assert reviewer.calls == 0 and (await c.store.get_state(c.sid)).pending == pending
        negative = await c.manager.get_job(c.actor, jobs[1]['job_id'])
        assert negative.generation == 0 and negative.result_status == 'suppressed' and negative.supervision_status == 'closed'
        await retire_pending(loop, c.actor, source_id)
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor, source_id, generation=1)
        result = await loop.send('Read both original outcomes; acknowledge without executing either action.', c.sid)
        assert result.status == 'completed' and reviewer.calls == 1
        assert (await c.manager.get_job(c.actor, jobs[0]['job_id'])).result_status == 'handled'
        negative = await c.manager.get_job(c.actor, jobs[1]['job_id'])
        assert negative.generation == 0 and negative.result_status == 'suppressed' and negative.supervision_status == 'closed'
        assert (await c.manager.get_job(c.actor, source_id)).supervision_status == 'closed'
        assert c.state.prepared == c.state.dispatched == c.model.calls == 1 and c.state.lookups == 0
        assert c.host.reserve_calls == 2
        for message in raw:
            assert await c.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (message['id'],)) == message
        assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('change', ['same_contract_handler', 'same_handler_registration'])
async def test_batch_cannot_dispatch_replacement_registered_after_plan_commit(registry_store, change):
    from power_loop import HookPoint
    from power_loop.runtime.managed_effects import FunctionManagedTool, ManagedToolResult

    replacements = []
    async def configure(loop, state):
        original = loop.tool_registry.get('post_once')
        adapter = original.handler.adapter
        async def replacement(context, reference):
            replacements.append(context.tool_call_id)
            return ManagedToolResult('Replacement executed', {'receipt': 'replacement'})
        alternate = FunctionManagedTool(adapter.contract, adapter.on_prepare, replacement, adapter.on_recover)
        rebound = alternate.bind(loop.task_runtime, original.definition)
        async def replace_after_plan(ctx):
            if ctx.tool_name == 'post_once':
                loop.tool_registry.register(original.definition,
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
    assert all(value[1] == {'tool_call_plan', 'tool_call_plan_seal'} for value in observations)


@pytest.mark.parametrize('stage', ['first_registration', 'second_registration', 'plan', 'seal', 'commit', 'commit_ack'])
async def test_whole_plan_transaction_never_dispatches_partial_or_unacknowledged_intentions(registry_store, stage):
    from ._tool_plan_faults_192 import fail_plan_once
    failures = []
    async def configure(loop, state):
        failures.append(fail_plan_once(loop.task_runtime.manager, stage))
    c = await scenario(registry_store, call_count=2, budgeted=True, configure=configure, expected_error='test_plan_' + stage)
    assert failures[0].fired and c.model.calls == c.host.reserve_calls == 1
    assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
    jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    plans = await c.manager.db.fetchall(f"SELECT event_type FROM {c.manager.events} WHERE event_type IN ('tool_call_plan','tool_call_plan_seal')")
    assert len(jobs) == (2 if stage == 'commit_ack' else 0)
    assert len(plans) == (2 if stage == 'commit_ack' else 0)
    assert all(job['generation'] == 0 and job['resource_state'] == 'not_started' for job in jobs)
    assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")
    pending = (await c.store.get_state(c.sid)).pending
    assert pending['tool_call_ids'] == ['original-call', 'original-call-2']
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    loop, reviewer = await controller(c)
    try:
        staged = await reconcile_pending(loop, c.actor, source['job_id'])
        assert staged['mode'] == 'retired_for_abort_review'
        assert len(staged['notices']) == 1 and reviewer.calls == 0
        assert (await c.store.get_state(c.sid)).pending is None
        await retire_pending(loop, c.actor, source['job_id'])
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor, source['job_id'], generation=1)
        result = await loop.send('Review both original never-dispatched outcomes without performing either action.', c.sid)
        assert result.status == 'completed' and reviewer.calls == 1 and c.host.reserve_calls == 2
        assert (await c.manager.get_job(c.actor, source['job_id'])).supervision_status == 'closed'
        assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
        assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('part', ['missing_plan', 'missing_seal', 'plan_hash', 'scope_boolean', 'snapshot_hash'])
async def test_recovery_rechecks_plan_and_original_exit_snapshot(registry_store, part):
    from power_loop.runtime.jobs import _hash, _json
    from power_loop.runtime.pending_calls import SNAPSHOT
    c = await scenario(registry_store, call_count=2, budgeted=True, raw_receipt_failure=True)
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    [plan] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type='tool_call_plan'")
    if part.startswith('missing_'):
        key = plan['event_id'] + ('_seal' if part == 'missing_seal' else '')
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (key,))
    elif part == 'snapshot_hash':
        key = 'model_closed_' + source['job_id']
        event = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', (key,))
        data = json.loads(event['payload_json'])
        data['evidence'][SNAPSHOT]['call_plan']['payload_hash'] = 'f' * 64
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?', (_json(data), key))
    else:
        data = json.loads(plan['payload_json'])
        if part == 'scope_boolean':
            data['scope']['generation'] = True
        else:
            data['entries'][0]['arguments_hash'] = 'f' * 64
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?', (_json(data), _hash(data), plan['event_id']))
    pending = (await c.store.get_state(c.sid)).pending
    loop, reviewer = await controller(c)
    try:
        with pytest.raises(JobError):
            await reconcile_pending(loop, c.actor, source['job_id'])
        assert (await c.store.get_state(c.sid)).pending == pending
        assert reviewer.calls == 0 and c.state.prepared == c.state.dispatched == 1
        assert c.state.lookups == 0
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('registered_plain', [True, False])
async def test_mixed_unsupported_intent_cannot_be_silently_dropped_or_partly_resolved(registry_store, registered_plain):
    from power_loop import ToolDefinition
    plain_calls = []
    async def configure(loop, state):
        if registered_plain:
            loop.tool_registry.register(ToolDefinition('inspect_note', '', input_schema={'type': 'object', 'properties': {'text': {'type': 'string'}}}),
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
            await reconcile_pending(loop, c.actor, source['job_id'])
        assert (await c.store.get_state(c.sid)).pending == pending
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type IN ('managed_effect_unstarted','managed_effect_delivery','pending_call_disposition')")
        assert reviewer.calls == 0 and plain_calls == []
        assert c.state.prepared == c.state.dispatched == c.model.calls == 1 and c.state.lookups == 0
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def test_modified_coalesced_notice_cannot_acknowledge_hidden_negative_or_close_source(registry_store):
    from power_loop import HookPoint
    from power_loop.runtime.managed_delivery import render
    c = await scenario(registry_store, budgeted=True, raw_receipt_failure=True, call_count=2)
    jobs = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool' ORDER BY generation DESC")
    loop, reviewer = await controller(c)
    try:
        await reconcile_pending(loop, c.actor, jobs[0]['parent_job_id'])
        await retire_pending(loop, c.actor, jobs[0]['parent_job_id'])
        [event] = await c.manager.db.fetchall(f"SELECT payload_json FROM {c.manager.events} WHERE job_id=? AND event_type='managed_effect_delivery'", (jobs[1]['job_id'],))
        hidden = render(json.loads(event['payload_json'])['public'])
        async def hide(ctx):
            for message in ctx.messages:
                if isinstance(message.get('content'), str):
                    message['content'] = message['content'].replace(hidden, '[negative outcome deliberately hidden]')
        loop.hooks.register(HookPoint.LLM_BEFORE, hide, name='test.plan.hide-negative')
        result = await loop.send('Read the supplied original outcomes, no actions.', c.sid)
        assert result.status == 'completed' and reviewer.calls == 1
        assert (await c.manager.get_job(c.actor, jobs[0]['job_id'])).result_status == 'pending_delivery'
        negative = await c.manager.get_job(c.actor, jobs[1]['job_id'])
        assert negative.result_status == 'suppressed' and negative.supervision_status == 'closed'
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='processing_decision'", (jobs[1]['job_id'],))
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor, jobs[0]['parent_job_id'], generation=1)
        assert c.state.dispatched == c.model.calls == 1
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('part', ['too_many_calls', 'oversized_pending', 'oversized_plan'])
async def test_plan_limits_refuse_all_callbacks_and_keep_original_responsibility(registry_store, part):
    from dataclasses import replace
    count, arguments, expected = 2, 'hello', 'tool_call_plan_too_large'
    async def configure(loop, state):
        if part == 'oversized_plan':
            original = loop.tool_registry.get('post_once')
            definition = replace(original.definition, description='large schema description ' * 10000)
            loop.tool_registry.register(definition, original.handler.adapter.bind(loop.task_runtime, definition), overwrite=True)
    if part == 'too_many_calls':
        count, expected = 65, 'tool_call_plan_invalid_call_set'
    elif part == 'oversized_pending':
        arguments, expected = 'x' * 40000, 'pending_call_evidence_too_large'
    c = await scenario(registry_store, call_count=count, argument_text=arguments, configure=configure, expected_error=expected)
    assert c.result.status == 'interrupted' and c.model.calls == 1
    assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
    pending = (await c.store.get_state(c.sid)).pending
    assert len(pending['tool_call_ids']) == count
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='background_tool'")
    assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    assert source['resource_state'] == 'unknown' and source['supervision_status'] == 'open'


@pytest.mark.parametrize('mode', ['control_alias', 'plain_same_name', 'mixed'])
async def test_expired_batch_only_preserves_actual_control_contract_without_new_work(registry_store, mode):
    from dataclasses import replace

    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

    reads = []
    async def configure(loop, state):
        manager = loop.task_runtime.manager
        definition = replace(loop.tool_registry.get('post_once').definition, name='read_original')
        async def read(text):
            reads.append(text)
            return 'Original status read'
        handler = read if mode == 'plain_same_name' else FunctionRuntimeTool(
            read, ToolRuntimeContract('example.original-read.v1'),
            policy=lambda arguments: ToolDispatchPolicy(native_admitted=True, control_only=True))
        loop.tool_registry.register(definition, handler)
        complete = loop.llm.complete
        now = manager.now
        async def expire_after_actual_response(request, **kwargs):
            response = await complete(request, **kwargs)
            manager.now = lambda: now() + 600000
            return response
        loop.llm.complete = expire_after_actual_response
    c = await scenario(registry_store, configure=configure,
                       call_names=['read_original', 'post_once'] if mode == 'mixed' else ['read_original'],
                       expected_error='execution_lease_expired')
    assert c.model.calls == 1 and c.state.prepared == c.state.dispatched == c.state.lookups == 0
    assert reads == (['hello'] if mode == 'control_alias' else [])
    raw = await c.manager.db.fetchall(f"SELECT content FROM {c.store.t.messages} WHERE role='tool'")
    assert [r['content'] for r in raw] == (['Original status read'] if mode == 'control_alias' else [])
    assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind='background_tool'")
    if mode != 'control_alias':
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='tool_call_dispatch_started'")
