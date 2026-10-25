"""Original-main handling of a positive aborted batch, without replacement work."""
import pytest

from power_loop.runtime.jobs import JobError
from power_loop.runtime.pending_calls import retire_pending
from power_loop.runtime.tool_call_plans import reconcile_pending

from ._tool_plan_faults_192 import fail_plan_once
from .test_job_registry_192 import registry_store
from .test_managed_effects_192 import scenario
from .test_tool_call_plans_192 import controller

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('stage', ['first_registration', 'seal', 'commit', 'commit_ack'])
async def test_aborted_batch_is_retired_for_actual_original_main_review_without_replay(registry_store, stage):
    async def configure(loop, state):
        fail_plan_once(loop.task_runtime.manager, stage)
    c = await scenario(registry_store, budgeted=True, call_count=2, configure=configure, expected_error='test_plan_'+stage)
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    pending = (await c.store.get_state(c.sid)).pending
    raw = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
    loop, model = await controller(c)
    try:
        staged = await reconcile_pending(loop, c.actor, source['job_id'])
        assert staged['mode'] == 'retired_for_abort_review' and len(staged['notices']) == 1
        assert not staged['execution_authority'] and not staged['acknowledgement_created'] and not staged['billing_settled']
        assert await reconcile_pending(loop, c.actor, source['job_id']) == staged
        disposition = await retire_pending(loop, c.actor, source['job_id'])
        assert disposition['disposition'] == staged['disposition']
        assert (await c.store.get_state(c.sid)).pending is None and model.calls == 0
        current = await c.manager.get_job(c.actor, source['job_id'])
        assert current.status == 'model_closed' and current.result_status == 'pending_delivery'
        assert current.result['aborted_call_batch']['original_call']['seq'] == str(pending['assistant_seq'])
        assert current.outcome == 'failed' and current.supervision_status == 'open'
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor, source['job_id'], generation=1)
        result = await loop.send('Read the original aborted batch outcomes and acknowledge without performing any action.', c.sid)
        assert result.status == 'completed' and model.calls == 1
        current = await c.manager.get_job(c.actor, source['job_id'])
        assert current.status == 'model_closed' and current.result_status == 'handled' and current.supervision_status == 'closed'
        assert c.model.calls == 1 and c.host.reserve_calls == 2
        assert c.state.prepared == c.state.dispatched == c.state.lookups == 0
        for row in raw:
            assert await c.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (row['id'],)) == row
        assert not await c.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE role='tool'")
        children = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
        assert len(children) == (2 if stage == 'commit_ack' else 0)
        assert all(row['generation'] == 0 and row['result_status'] == 'suppressed' and row['supervision_status'] == 'closed' for row in children)
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def pending_batch(registry_store, *, stage='seal', **kwargs):
    async def configure(loop, state):
        fail_plan_once(loop.task_runtime.manager, stage)
    c = await scenario(registry_store, budgeted=True, call_count=2, configure=configure, expected_error='test_plan_'+stage, **kwargs)
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    return c, source


@pytest.mark.parametrize('stage', ['child', 'result', 'notice', 'inbox', 'disposition', 'pending', 'commit', 'commit_ack'])
async def test_batch_result_notice_children_and_pending_are_atomic_and_recallable(registry_store, stage):
    from ._abort_outcome_faults_192 import fail_outcome_once
    c, source = await pending_batch(registry_store, stage='commit_ack')
    pending = (await c.store.get_state(c.sid)).pending
    before = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
    loop, model = await controller(c)
    fault = fail_outcome_once(c.manager, source['job_id'], stage)
    try:
        with pytest.raises(OSError, match='^test_outcome_'+stage+'$'):
            await reconcile_pending(loop, c.actor, source['job_id'])
        assert fault.fired and model.calls == 0
        if stage != 'commit_ack':
            assert (await c.store.get_state(c.sid)).pending == pending
            assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id') == before
            assert not await c.manager.db.fetchall(f'SELECT id FROM {c.store.t.follow_up_queue}')
        else:
            assert (await c.store.get_state(c.sid)).pending is None
        first = await reconcile_pending(loop, c.actor, source['job_id'])
        assert await reconcile_pending(loop, c.actor, source['job_id']) == first
        assert len(await c.manager.db.fetchall(f'SELECT id FROM {c.store.t.follow_up_queue}')) == 1
        assert c.model.calls == c.host.reserve_calls == 1 and c.state.dispatched == 0
        result = await loop.send('Acknowledge the original negative batch without repeating any action.', c.sid)
        assert result.status == 'completed' and model.calls == 1
        assert (await c.manager.get_job(c.actor, source['job_id'])).supervision_status == 'closed'
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('failure', ['hidden', 'model_error', 'budget'])
async def test_inbox_or_failed_review_never_acknowledges_source(registry_store, failure):
    from .test_managed_delivery_192 import controller as failing_controller
    c, source = await pending_batch(registry_store, max_calls=1 if failure == 'budget' else 2)
    loop, model = await failing_controller(c, sees=failure != 'hidden', fail=failure == 'model_error')
    from power_loop.runtime.representation import ProjectedRepresentation
    loop.config.representation = ProjectedRepresentation()
    try:
        await reconcile_pending(loop, c.actor, source['job_id'])
        result = await loop.send('Read the negative outcome.', c.sid)
        assert result.status == 'completed' if failure == 'hidden' else result.status != 'completed'
        assert (await c.manager.get_job(c.actor, source['job_id'])).result_status == 'pending_delivery'
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor, source['job_id'], generation=1)
        assert c.state.prepared == c.state.dispatched == 0
        assert model.calls == (0 if failure == 'budget' else 1)
        assert c.host.reserve_calls == (1 if failure == 'budget' else 2)
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('attack', ['unoffered_effect', 'control_starts_resource'])
async def test_actual_review_rejects_new_effects_even_with_a_control_schema(registry_store, attack):
    import json

    from power_loop import LLMResponse, LLMTokenUsage, ToolDefinition
    from power_loop.runtime.task_runtime import _SEND
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract
    c, source = await pending_batch(registry_store)
    loop, model = await controller(c)
    callbacks = []
    async def control(**args):
        callbacks.append(args)
        execution = _SEND.get()
        await c.manager.register_job(execution.actor, kind='background_tool', parent=execution.lease,
            system_key='forbidden-resource', title='Forbidden review resource')
        raise AssertionError('new resource cannot be admitted')
    definition = ToolDefinition('inspect_original', '', input_schema={'type': 'object'})
    loop.tool_registry.register(definition, FunctionRuntimeTool(control, ToolRuntimeContract('example.original-control'),
        policy=lambda args: ToolDispatchPolicy(True, True), reviewer=lambda spec: spec))
    original = c.registry.get('post_once')
    loop.tool_registry.register(original.definition, original.handler)
    async def complete(request, **kwargs):
        model.calls += 1
        assert 'ORIGINAL OUTCOME REVIEW' in request.system_prompt
        assert [t['function']['name'] for t in request.tools] == ['inspect_original']
        return LLMResponse(raw_text='', token_usage=LLMTokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            tool_calls=[{'id': 'forbidden-replay', 'type': 'function', 'function': {
                'name': 'post_once' if attack == 'unoffered_effect' else 'inspect_original',
                'arguments': json.dumps({'text':'do not execute'} if attack == 'unoffered_effect' else {})}}])
    model.complete = complete
    try:
        await reconcile_pending(loop, c.actor, source['job_id'])
        try:
            result = await loop.send('Review without new work.', c.sid)
        except JobError as exc:
            assert attack == 'unoffered_effect' and exc.code == 'outcome_review_requires_control_only'
        else:
            assert attack == 'control_starts_resource' and result.status != 'completed'
        assert model.calls == 1 and c.host.reserve_calls == 2
        assert len(callbacks) == (1 if attack == 'control_starts_resource' else 0)
        assert c.state.prepared == c.state.dispatched == 0
        assert not await c.manager.db.fetchall(f"SELECT job_id FROM {c.manager.jobs} WHERE kind<>'send'")
        assert (await c.manager.get_job(c.actor, source['job_id'])).result_status == 'pending_delivery'
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('part', ['notice', 'inbox', 'result', 'disposition', 'both_notice_and_disposition', 'newer_pending', 'roletool'])
async def test_corrupt_or_newer_original_responsibility_cannot_be_cleared_or_acknowledged(registry_store, part):
    from power_loop.runtime.abort_outcomes import pending_reviews
    from power_loop.runtime.jobs import _json
    c, source = await pending_batch(registry_store)
    loop, model = await controller(c)
    try:
        if part in ('newer_pending', 'roletool'):
            if part == 'newer_pending':
                await c.manager.db.execute(f'UPDATE {c.store.t.session_state} SET pending_json=? WHERE session_id=?',
                    (_json({'newer-responsibility': True}), c.sid))
            else:
                await c.store.append_message(c.sid, role='tool', tool_call_id='original-call', content='contradictory actual reply', send_index=(await c.store.get_message(c.sid, (await c.store.get_state(c.sid)).pending['assistant_seq'])).send_index)
            pending = (await c.store.get_state(c.sid)).pending
            with pytest.raises(JobError):
                await reconcile_pending(loop, c.actor, source['job_id'])
            assert (await c.store.get_state(c.sid)).pending == pending
            assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='source_abort_delivery'")
            return
        await reconcile_pending(loop, c.actor, source['job_id'])
        eventtypes = {'notice': ['source_abort_delivery'], 'result': ['result'],
            'disposition': ['pending_call_disposition'], 'both_notice_and_disposition': ['source_abort_delivery','pending_call_disposition']}
        if part == 'inbox':
            await c.manager.db.execute(f'DELETE FROM {c.store.t.follow_up_queue}')
        else:
            for kind in eventtypes[part]:
                await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE job_id=? AND event_type=?', (source['job_id'],kind))
        with pytest.raises(JobError):
            await reconcile_pending(loop, c.actor, source['job_id'])
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor, source['job_id'], generation=1)
        async with c.manager._write() as tx:
            # A corrupt known outcome may raise, or remain restrictive; never
            # silently become an empty review set that authorizes new work.
            try:
                reviews = await pending_reviews(c.manager, tx, c.actor)
            except JobError:
                pass
            else:
                assert reviews
        assert model.calls == 0 and c.state.dispatched == 0
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def test_retained_protocol_thirteen_per_item_notices_keep_their_original_route(registry_store):
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.managed_delivery import enqueue
    from power_loop.runtime.managed_effects import terminate_unstarted
    from power_loop.runtime.store.schema import StoreSchemaError
    from power_loop.runtime.store.task_schema import migrate_task_schema, verify_task_schema
    c, source = await pending_batch(registry_store, stage='commit_ack')
    loop, model = await controller(c)
    try:
        children = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
        for child in children:
            await terminate_unstarted(c.manager, c.actor, child['job_id'])
            await enqueue(loop, c.actor, child['job_id'])
        old_disposition = await retire_pending(loop, c.actor, source['job_id'])
        events = await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id')
        raw = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
        await c.manager.db.execute(f'UPDATE {c.manager.prefix}task_store SET protocol_version=13 WHERE id=1')
        with pytest.raises(StoreSchemaError):
            await verify_task_schema(c.manager.db, c.manager.prefix)
        with pytest.raises(StoreSchemaError, match='maintenance_required'):
            await migrate_task_schema(c.manager.db, c.manager.prefix)
        await migrate_task_schema(c.manager.db, c.manager.prefix, maintenance=True)
        assert await c.manager.db.fetchall(f'SELECT * FROM {c.manager.events} ORDER BY event_id') == events
        assert await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq') == raw
        assert JobManager(c.store).store.store_id == c.manager.store.store_id
        staged = await reconcile_pending(loop, c.actor, source['job_id'])
        assert len(staged['notices']) == 2 and staged.get('mode') != 'retired_for_abort_review'
        assert await retire_pending(loop, c.actor, source['job_id']) == old_disposition
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='source_abort_delivery'")
        result = await loop.send('Read both original negative outcomes.', c.sid)
        assert result.status == 'completed' and model.calls == 1
        assert (await c.manager.get_job(c.actor, source['job_id'])).supervision_status == 'closed'
        for child in children:
            saved = await c.manager.get_job(c.actor, child['job_id'])
            assert saved.result_status == 'suppressed' and saved.supervision_status == 'closed'
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def test_reused_provider_call_id_does_not_attach_an_earlier_negative_to_a_later_batch(registry_store):
    from power_loop import LLMResponse, LLMTokenUsage
    from power_loop.runtime.managed_delivery import negative_review_view
    async def configure(loop, state):
        complete = loop.llm.complete
        async def reused(request, **kwargs):
            if loop.llm.calls == 1:
                loop.llm.calls += 1
                fail_plan_once(loop.task_runtime.manager, 'seal')
                return LLMResponse(raw_text='', token_usage=LLMTokenUsage(prompt_tokens=5,completion_tokens=5,total_tokens=10),
                    tool_calls=[{'id':'original-call','type':'function','function':{'name':'post_once','arguments':'{"text":"second"}'}}])
            return await complete(request, **kwargs)
        loop.llm.complete = reused
    c = await scenario(registry_store, budgeted=True, max_calls=3, claim_failure=True, configure=configure, expected_error='test_plan_seal')
    [source] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='send'")
    [child] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    loop, model = await controller(c)
    async def state():
        async with c.manager._write() as tx:
            return await negative_review_view(c.manager,tx,await c.manager._row(tx,child['job_id']))
    try:
        assert (await state())['state'] == 'not_required'
        staged = await reconcile_pending(loop,c.actor,source['job_id'])
        assert staged['mode'] == 'retired_for_abort_review' and (await state())['state'] == 'not_required'
        result = await loop.send('Read the original negative batch.',c.sid)
        assert result.status == 'completed' and model.calls == 1 and c.model.calls == 2
        assert (await state())['state'] == 'not_required'
        assert (await c.manager.get_job(c.actor,source['job_id'])).supervision_status == 'closed'
        assert c.state.prepared == c.state.dispatched == 0
    finally:
        await loop.aclose(drain_timeout_s=.1)


async def test_mechanical_concurrent_recovery_preserves_original_financial_attempts_and_newer_pending(registry_store):
    import asyncio

    from power_loop.runtime.jobs import _json
    c, source = await pending_batch(registry_store,stage='commit_ack')
    loop, model = await controller(c)
    tables = ['task_budget_attempts','task_budget_scopes','task_budget_policies']
    before = {name:await c.manager.db.fetchall(f'SELECT * FROM {c.manager.prefix}{name}') for name in tables}
    model_events = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type LIKE 'model_%' ORDER BY event_id")
    try:
        results = await asyncio.gather(*(reconcile_pending(loop,c.actor,source['job_id']) for _ in range(3)))
        assert results[0] == results[1] == results[2]
        assert {name:await c.manager.db.fetchall(f'SELECT * FROM {c.manager.prefix}{name}') for name in tables} == before
        assert await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE event_type LIKE 'model_%' ORDER BY event_id") == model_events
        assert model.calls == 0 and c.host.reserve_calls == c.host.apply_calls == 1
        newer = {'newer-responsibility':True}
        await c.manager.db.execute(f'UPDATE {c.store.t.session_state} SET pending_json=? WHERE session_id=?',(_json(newer),c.sid))
        assert await reconcile_pending(loop,c.actor,source['job_id']) == results[0]
        assert (await c.store.get_state(c.sid)).pending == newer
        assert len(await c.manager.db.fetchall(f'SELECT id FROM {c.store.t.follow_up_queue}')) == 1
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('when', ['before', 'after'])
async def test_cancelled_original_batch_retains_its_actual_model_exit_and_negative_result(registry_store, when):
    c, source = await pending_batch(registry_store,stage='commit_ack')
    loop, model = await controller(c)
    exit_before = await c.manager.db.fetchone(f'SELECT * FROM {c.manager.events} WHERE event_id=?',('model_closed_'+source['job_id'],))
    try:
        if when == 'after':
            await reconcile_pending(loop,c.actor,source['job_id'])
        await c.manager.request_cancel(c.actor,source['job_id'],system_key='cancel-aborted-source',reason='Stop original work')
        staged = await reconcile_pending(loop,c.actor,source['job_id'])
        assert staged['mode'] == 'retired_for_abort_review' and model.calls == 0
        current = await c.manager.get_job(c.actor,source['job_id'])
        assert current.status == 'model_closed' and current.cancel_requested_at is not None
        assert current.result_status == 'pending_delivery' and current.supervision_status == 'open'
        assert current.outcome == ('cancelled' if when == 'before' else 'failed')
        assert await c.manager.db.fetchone(f'SELECT * FROM {c.manager.events} WHERE event_id=?',('model_closed_'+source['job_id'],)) == exit_before
        assert c.host.reserve_calls == 1 and c.state.dispatched == c.state.prepared == 0
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor,source['job_id'],generation=1)
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('role', ['subagent', 'workflow_node'])
async def test_original_main_outcome_path_cannot_promote_a_child_role(registry_store, role):
    c, source = await pending_batch(registry_store)
    loop, model = await controller(c)
    try:
        pending = (await c.store.get_state(c.sid)).pending
        await c.store.merge_session_metadata(c.sid, {'execution_role':role})
        with pytest.raises(JobError):
            await reconcile_pending(loop,c.actor,source['job_id'])
        assert (await c.store.get_state(c.sid)).pending == pending and model.calls == 0
        assert not await c.manager.db.fetchall(f'SELECT id FROM {c.store.t.follow_up_queue}')
    finally:
        await loop.aclose(drain_timeout_s=.1)


@pytest.mark.parametrize('attempt', ['false_proof', 'success', 'unknown_resource'])
async def test_closed_model_publication_requires_an_actual_negative_stopped_proof(registry_store, attempt):
    from power_loop.runtime.jobs import JobLease
    from power_loop.runtime.result_proof import ResultPublicationProof
    from power_loop.runtime.workflow_controller import proof
    c, source = await pending_batch(registry_store)
    class HostProof(ResultPublicationProof):
        async def validate(self, manager, transaction, execution_row, requested_result):
            return attempt != 'false_proof'
    async with c.manager._write() as tx:
        with pytest.raises(JobError):
            await c.manager._record_result_tx(tx,c.actor,JobLease(**proof(source)),
                outcome='succeeded' if attempt=='success' else 'failed',result={'status':'failed'},
                resource_state='unknown' if attempt=='unknown_resource' else 'stopped',
                evidence_only=True,publication_proof=HostProof())
    assert (await c.manager.get_job(c.actor,source['job_id'])).result_status == 'none'
    assert (await c.store.get_state(c.sid)).pending is not None


async def test_unrelated_large_send_result_is_not_an_aborted_batch_review(registry_store):
    from power_loop.runtime.abort_outcomes import candidates, pending_reviews
    from power_loop.runtime.jobs import JobActor, JobManager
    store, _ = registry_store
    sid = await store.create_session()
    manager = JobManager(store)
    actor = JobActor('example-tenant', 'host', sid)
    root = await manager.register_job(actor, kind='send', session_id=sid, system_key='ordinary-source')
    lease = await manager.claim_job(actor, root.job.job_id, executor_id='original', boot_id='boot', expected_version=1)
    await manager.record_result(actor, lease, outcome='failed', result={'ordinary_host_payload':'A'*300000}, resource_state='stopped')
    before = await manager.db.fetchall(f'SELECT * FROM {manager.events} ORDER BY event_id')
    async with manager._write() as tx:
        assert await pending_reviews(manager, tx, actor) == []
        assert await candidates(manager, tx, actor) == []
    assert await manager.db.fetchall(f'SELECT * FROM {manager.events} ORDER BY event_id') == before
