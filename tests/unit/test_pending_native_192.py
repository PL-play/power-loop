"""Actual model-exit snapshots distinguish an unstarted native subset."""
import json

import pytest

from power_loop.runtime.controller_recovery import reconcile as recover_owner
from power_loop.runtime.jobs import JobError, _hash, _json
from power_loop.runtime.pending_calls import recover_runtime
from power_loop.runtime.pending_native import FIELD, identity

from ._owned_pending_case_192 import owned_pending_case
from .test_controller_handoff_192 import new_main
from .test_job_registry_192 import registry_store
from .test_owned_pending_review_192 import cleanup

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('kind', ['subagent', 'workflow_node'])
@pytest.mark.parametrize('name', ['pass_turn', 'renamed_native_operation'])
async def test_original_main_reviews_only_unstarted_subset_and_separate_completed_effect(registry_store, kind, name):
    c = await owned_pending_case(registry_store, kind, native_tail=name)
    loop = runtime = None
    try:
        sid = c.state.source_binding.actor.session_id
        original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY session_id,seq')
        effect = await c.manager.get_job(c.runtime.actor(c.state.root), c.effect['job_id'])
        usage = await c.manager.execution_usage(c.state.owner_binding.actor, c.state.owner_binding.lease.job_id, generation=1)
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        result = await recover_runtime(runtime, actor, c.state.source)
        assert result['state'] == 'queued_for_review' and len(result['notices']) == 1
        assert not result['acknowledgement_created']
        source = await runtime.manager.get_job(actor, c.state.source)
        public = source.result['aborted_call_batch']
        assert public['version'] == 2 and public['outcome_scope'] == 'original_undispatched_calls'
        assert public['calls'] == [{'tool_call_id': 'original-native', 'tool_name': name,
            'arguments_hash': _hash({}), 'resolution': 'not_dispatched'}]
        assert not public['execution_authority'] and not public['billing_settled']
        assert (await c.store.get_state(sid)).pending is None
        snapshot = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?',
            (identity(c.manager, {'job_id': c.state.source, 'generation': 1}),))
        dispatches = json.loads(snapshot['payload_json'])['dispatches']
        assert dispatches[0]['started'] and dispatches[1] == {'call_id': 'original-native', 'started': None}
        await loop.send('Handle both original outcomes without repeating any action.', c.state.root, tools=[])
        source = await runtime.manager.get_job(actor, c.state.source)
        assert source.supervision_status == 'closed' and source.result_status == 'handled'
        completed = await runtime.manager.get_job(actor, c.effect['job_id'])
        assert completed.result == effect.result and completed.result_status == 'handled'
        assert completed.supervision_status == 'closed'
        await recover_owner(runtime, actor, c.state.owner_binding.lease.job_id)
        await loop.send('Handle the original controller result.', c.state.root, tools=[])
        owner = await runtime.manager.get_job(actor, c.state.owner_binding.lease.job_id)
        assert owner.supervision_status == 'closed' and owner.outcome == 'interrupted'
        assert await runtime.manager.execution_usage(actor, owner.job_id, generation=1) == usage
        for row in original:
            assert await runtime.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (row['id'],)) == row
        assert not await runtime.manager.db.fetchall(f"SELECT id FROM {c.store.t.messages} WHERE session_id=? AND role='tool'", (sid,))
        assert c.state.native_calls == c.state.lookups == c.state.counts['AFTER'] == 0
        assert c.state.executed == c.state.prepared == len(c.state.effects) == 1
    finally:
        await cleanup(c, loop, runtime)


async def test_started_native_without_original_return_checkpoint_stays_unknown(registry_store):
    c = await owned_pending_case(registry_store, 'subagent', native_tail='another_operation', native_first=True)
    loop = runtime = None
    try:
        sid = c.state.source_binding.actor.session_id
        pending = (await c.store.get_state(sid)).pending
        assert c.state.native_calls == 1 and c.state.executed == c.state.prepared == 0
        await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='native_callback_returned'", (c.state.source,))
        loop, runtime = await new_main(c, registry_store)
        result = await recover_runtime(runtime, runtime.actor(c.state.root), c.state.source)
        assert result['state'] == 'unconfirmed' and result['error'] == 'pending_native_return_required'
        assert (await c.store.get_state(sid)).pending == pending
        source = await runtime.manager.get_job(runtime.actor(c.state.root), c.state.source)
        assert source.result_status == 'none' and source.supervision_status == 'open'
        assert c.state.native_calls == 1 and not c.state.effects
    finally:
        await cleanup(c, loop, runtime)


@pytest.mark.parametrize('fault', ['missing_snapshot', 'changed_snapshot', 'missing_start', 'legacy_no_snapshot'])
async def test_changed_or_absent_original_dispatch_proof_cannot_create_negative_result(registry_store, fault):
    c = await owned_pending_case(registry_store, 'workflow_node', native_tail='renamed_operation')
    loop = runtime = None
    try:
        sid = c.state.source_binding.actor.session_id
        pending = (await c.store.get_state(sid)).pending
        key = identity(c.manager, {'job_id': c.state.source, 'generation': 1})
        if fault == 'missing_snapshot':
            await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (key,))
        elif fault == 'missing_start':
            await c.manager.db.execute(f"DELETE FROM {c.manager.events} WHERE job_id=? AND event_type='tool_call_dispatch_started'", (c.state.source,))
        elif fault == 'changed_snapshot':
            row = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', (key,))
            data = json.loads(row['payload_json'])
            data['dispatches'][0]['started'] = None
            await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?', (_json(data), _hash(data), key))
        else:
            key = 'model_closed_' + c.state.source
            row = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', (key,))
            data = json.loads(row['payload_json'])
            del data['evidence'][FIELD]
            await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?', (_json(data), _hash(data), key))
        loop, runtime = await new_main(c, registry_store)
        try:
            result = await recover_runtime(runtime, runtime.actor(c.state.root), c.state.source)
        except JobError:
            pass
        else:
            assert result['state'] == 'unconfirmed'
        assert (await c.store.get_state(sid)).pending == pending
        source = await runtime.manager.get_job(runtime.actor(c.state.root), c.state.source)
        assert source.result_status == 'none' and source.supervision_status == 'open'
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type IN ('pending_call_disposition','source_abort_delivery','pending_owner_transfer')", (c.state.source,))
        assert c.state.native_calls == c.state.lookups == 0
        assert c.state.executed == 1
    finally:
        await cleanup(c, loop, runtime)


async def test_closed_original_source_cannot_dispatch_even_a_control_call(registry_store):
    from types import SimpleNamespace

    from power_loop.runtime.invocation import invocation_context
    from power_loop.runtime.task_runtime import _SEND
    from power_loop.runtime.tool_call_plans import dispatch

    c = await owned_pending_case(registry_store, 'subagent', native_tail='renamed_control')
    try:
        binding = c.state.source_binding
        pending = (await c.store.get_state(binding.actor.session_id)).pending
        token = _SEND.set(binding)
        try:
            with invocation_context(binding.actor.session_id, pending['assistant_seq'], 'original-native'):
                with pytest.raises(JobError, match='tool_call_source_model_closed'):
                    await dispatch(SimpleNamespace(session_id=binding.actor.session_id), 'renamed_control', {},
                                   c.loop.tool_registry.get('renamed_control'))
        finally:
            _SEND.reset(token)
        assert c.state.native_calls == 0
    finally:
        await cleanup(c)


async def test_native_result_notice_and_disposition_roll_back_as_one_transaction(registry_store, monkeypatch):
    from power_loop.runtime import pending_calls
    from power_loop.runtime.abort_outcomes import review_view

    c = await owned_pending_case(registry_store, 'workflow_node', native_tail='ordinary_operation')
    loop = runtime = None
    try:
        sid = c.state.source_binding.actor.session_id
        pending = (await c.store.get_state(sid)).pending
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        expected = pending_calls._expected
        async def fail(*args):
            raise OSError('actual native disposition transaction failed')
        monkeypatch.setattr(pending_calls, '_expected', fail)
        with pytest.raises(OSError, match='disposition transaction failed'):
            await recover_runtime(runtime, actor, c.state.source)
        assert (await c.store.get_state(sid)).pending == pending
        source = await runtime.manager.get_job(actor, c.state.source)
        assert source.result_status == 'none'
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type IN ('pending_call_disposition','source_abort_delivery','pending_owner_transfer')", (c.state.source,))
        monkeypatch.setattr(pending_calls, '_expected', expected)
        assert (await recover_runtime(runtime, actor, c.state.source))['state'] == 'queued_for_review'
        assert (await recover_runtime(runtime, actor, c.state.source))['state'] == 'disposed'
        async def view():
            async with runtime.manager._write() as tx:
                row = await runtime.manager._row(tx, c.state.source)
                return await review_view(runtime.manager, tx, row)
        before = await view()
        assert before['state'] == 'pending' and before['review_scope'] == 'original_undispatched_calls'
        assert before['notice_event_id'].startswith('pc_native_note_') and len(before['notice_event_id']) <= 64
        await loop.send('Handle both original outcomes without replay.', c.state.root, tools=[])
        after = await view()
        assert after['state'] == 'handled' and after['decision_event_id'] and after['response_ref']
        assert c.state.native_calls == c.state.lookups == 0 and c.state.executed == 1
    finally:
        await cleanup(c, loop, runtime)


async def test_unstarted_managed_sibling_keeps_its_separate_review_view(registry_store):
    from power_loop.runtime.managed_delivery import negative_review_view

    c = await owned_pending_case(registry_store, 'subagent', effect_calls=2, native_tail='native_sibling')
    loop = runtime = None
    try:
        loop, runtime = await new_main(c, registry_store)
        actor = runtime.actor(c.state.root)
        result = await recover_runtime(runtime, actor, c.state.source)
        assert result['state'] == 'queued_for_review' and len(result['notices']) == 2
        [negative] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool' AND generation=0")
        async def view():
            async with runtime.manager._write() as tx:
                row = await runtime.manager._row(tx, negative['job_id'])
                return await negative_review_view(runtime.manager, tx, row)
        before = await view()
        assert before['state'] == 'pending' and before['notice_event_id']
        await loop.send('Handle the completed and two separate unstarted original outcomes.', c.state.root, tools=[])
        assert (await view())['state'] == 'handled'
        source = await runtime.manager.get_job(actor, c.state.source)
        assert source.supervision_status == 'closed' and source.result_status == 'handled'
        assert c.state.executed == c.state.prepared == 1 and c.state.native_calls == 0
    finally:
        await cleanup(c, loop, runtime)


async def test_original_main_source_can_review_its_own_mixed_batch(registry_store):
    from power_loop import ToolDefinition
    from power_loop.runtime.tool_contracts import FunctionRuntimeTool, ToolDispatchPolicy, ToolRuntimeContract

    from .test_managed_effects_192 import scenario
    from .test_tool_call_plans_192 import controller

    async def configure(loop, state):
        state.native_calls = 0
        async def native(text):
            state.native_calls += 1
            return text
        loop.tool_registry.register(ToolDefinition('arbitrary_operation', 'Native operation', input_schema={'type':'object'}),
            FunctionRuntimeTool(native, ToolRuntimeContract('example.renamed-operation'), policy=lambda args:ToolDispatchPolicy(True)))
    c = await scenario(registry_store, budgeted=True, raw_receipt_failure=True,
        call_names=['post_once', 'arbitrary_operation'], configure=configure)
    loop, _ = await controller(c)
    try:
        [effect] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
        source_id = effect['parent_job_id']
        result = await recover_runtime(loop.task_runtime, c.actor, source_id)
        assert result['state'] == 'queued_for_review'
        await loop.send('Handle both original outcomes without replay.', c.sid)
        source = await c.manager.get_job(c.actor, source_id)
        assert source.result_status == 'handled' and source.supervision_status == 'closed'
        assert (await c.store.get_state(c.sid)).pending is None
        assert c.state.native_calls == c.state.lookups == 0
        assert c.state.prepared == c.state.dispatched == 1
    finally:
        await loop.aclose(drain_timeout_s=.1)
