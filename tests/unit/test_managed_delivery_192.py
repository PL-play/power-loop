"""Actual main input/response, not queue acceptance, handles recovered outcomes."""

import pytest

from power_loop import AgentLoopConfig, HookPoint, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.runtime.jobs import JobError
from power_loop.runtime.managed_completion import flush_local
from power_loop.runtime.managed_delivery import enqueue, identity
from power_loop.runtime.managed_effects import reconcile
from power_loop.tools.registry import ToolRegistry

from .test_job_registry_192 import registry_store
from .test_managed_completion_192 import failure_case
from .test_managed_effects_192 import scenario

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Controller(LLMService):
    def __init__(self, *, sees=True, fail=False):
        self.calls, self.sees, self.fail = 0, sees, fail

    async def complete(self, request, **kwargs):
        self.calls += 1
        seen = any('Original tool outcome:' in str(m.get('content')) for m in request.messages)
        assert seen == self.sees
        if self.fail:
            raise RuntimeError('test_main_review_failed')
        return LLMResponse(raw_text='ACKNOWLEDGED', token_usage=LLMTokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10))

    def stream(self, request):
        raise AssertionError('complete transport selected')

    async def close(self):
        pass


async def controller(c, *, sees=True, fail=False):
    model = Controller(sees=sees, fail=fail)
    c.host.service = model
    loop = StatefulAgentLoop(llm=model, store=c.store, task_runtime=c.runtime, tool_registry=ToolRegistry(),
        config=AgentLoopConfig(model='test-model', max_tokens=20, max_rounds=2, delegation_policy='single_level'))
    if not sees:
        async def hide(ctx):
            ctx.messages = [m for m in ctx.messages if 'Original tool outcome:' not in str(m.get('content'))]
        loop.hooks.register(HookPoint.LLM_BEFORE, hide, name='test.hide-notice')
    return loop, model


@pytest.mark.parametrize('unknown', [False, True])
async def test_main_actually_consumes_the_known_outcome_and_original_error_is_unchanged(registry_store, unknown):
    if unknown:
        c = await scenario(registry_store, budgeted=True, pending=True)
        [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
        c.state.resolved = True
        await reconcile(c.runtime, c.actor, row['job_id'], c.adapter)
    else:
        c, row = await failure_case(registry_store)
        await flush_local(c.runtime, c.actor, row['job_id'])
    original = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
    loop, model = await controller(c)
    try:
        first = await enqueue(loop, c.actor, row['job_id'])
        again = await enqueue(loop, c.actor, row['job_id'])
        assert first == again and not first['acknowledgement_created'] and model.calls == 0
        assert (await c.manager.get_job(c.actor, row['job_id'])).result_status == 'pending_delivery'
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
        result = await loop.send('Review the original outcome and acknowledge it without repeating any action.', c.sid)
        assert result.status == 'completed' and model.calls == 1
        current = await c.manager.get_job(c.actor, row['job_id'])
        assert current.result_status == 'handled'
        await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
        assert (await c.manager.get_job(c.actor, row['job_id'])).supervision_status == 'closed'
        for raw in original:
            assert await c.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (raw['id'],)) == raw
        assert c.state.dispatched == c.state.prepared == c.model.calls == model.calls == 1
        assert c.host.reserve_calls == 2 and c.state.lookups == (1 if unknown else 0)
        inbox = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.follow_up_queue} WHERE item_id=?', (identity(c.manager, row),))
        assert len(inbox) == 1 and inbox[0]['status'] == 'delivered'
    finally:
        await loop.aclose(drain_timeout_s=0.1)


@pytest.mark.parametrize('failure', ['hidden', 'model_error'])
async def test_queued_or_transcribed_but_unprocessed_outcome_keeps_scope_open(registry_store, failure):
    c, row = await failure_case(registry_store)
    await flush_local(c.runtime, c.actor, row['job_id'])
    loop, model = await controller(c, sees=failure != 'hidden', fail=failure == 'model_error')
    try:
        await enqueue(loop, c.actor, row['job_id'])
        result = await loop.send('Review the pending notice.', c.sid)
        assert result.status != 'completed' if failure == 'model_error' else result.status == 'completed'
        assert (await c.manager.get_job(c.actor, row['job_id'])).result_status != 'handled'
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
        assert c.state.dispatched == c.model.calls == 1
    finally:
        await loop.aclose(drain_timeout_s=0.1)


async def test_unknown_business_outcome_is_not_published_as_a_known_resolution(registry_store):
    c = await scenario(registry_store, budgeted=True, pending=True)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    loop, model = await controller(c)
    try:
        with pytest.raises(JobError):
            await enqueue(loop, c.actor, row['job_id'])
        assert model.calls == 0 and c.state.lookups == 0
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='managed_effect_delivery'")
    finally:
        await loop.aclose(drain_timeout_s=0.1)


@pytest.fixture
async def pending_review(registry_store):
    c, row = await failure_case(registry_store)
    await flush_local(c.runtime, c.actor, row['job_id'])
    loop, model = await controller(c)
    await enqueue(loop, c.actor, row['job_id'])
    try:
        yield c, row, loop, model
    finally:
        await loop.aclose(drain_timeout_s=0.1)


@pytest.mark.parametrize('part', ['notice_hash', 'notice_lease', 'root_role', 'root_session',
                                  'inbox_missing', 'inbox_content', 'inbox_void', 'canonical_result'])
async def test_untrusted_or_missing_delivery_evidence_cannot_be_queued_again(pending_review, part):
    import json

    from power_loop.runtime.jobs import _hash, _json

    c, row, loop, model = pending_review
    key = identity(c.manager, row)
    if part in ('notice_hash', 'notice_lease'):
        notice = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', (key,))
        value = json.loads(notice['payload_json'])
        if part == 'notice_lease':
            value['lease']['generation'] = 2
        else:
            value['message_hash'] = 'f' * 64
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?', (_json(value), _hash(value), key))
    elif part == 'root_role':
        await c.manager.db.execute(f'UPDATE {c.store.t.sessions} SET metadata_json=? WHERE session_id=?', (_json({'execution_role': 'subagent'}), c.sid))
    elif part == 'root_session':
        other = await c.store.create_session()
        await c.manager.db.execute(f'UPDATE {c.manager.jobs} SET session_id=?,created_by_session_id=? WHERE job_id=?', (other, other, row['root_job_id']))
    elif part == 'inbox_missing':
        await c.manager.db.execute(f'DELETE FROM {c.store.t.follow_up_queue} WHERE item_id=?', (key,))
    elif part in ('inbox_content', 'inbox_void'):
        column, value = ('content', 'invented output') if part == 'inbox_content' else ('status', 'void')
        await c.manager.db.execute(f'UPDATE {c.store.t.follow_up_queue} SET {column}=? WHERE item_id=?', (value, key))
    else:
        from power_loop.runtime.result_revisions import result_id
        saved = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', (result_id(row['job_id'], 1),))
        value = json.loads(saved['payload_json'])
        value['result']['display'] = 'invented canonical result'
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?', (_json(value), result_id(row['job_id'], 1)))
        await c.manager.db.execute(f'UPDATE {c.manager.jobs} SET result_json=? WHERE job_id=?', (_json(value['result']), row['job_id']))
    with pytest.raises(JobError):
        await enqueue(loop, c.actor, row['job_id'])
    assert model.calls == 0 and c.state.dispatched == c.state.prepared == c.model.calls == 1


@pytest.mark.parametrize('part', ['delivered_message', 'decision', 'snapshot', 'model_closed', 'response'])
async def test_closed_resolution_revalidates_actual_handling_evidence(pending_review, part):
    import json

    from power_loop.runtime.jobs import _json

    c, row, loop, model = pending_review
    await loop.send('Read and handle the original result without repeating work.', c.sid)
    await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
    [decision] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.events} WHERE job_id=? AND event_type='processing_decision'", (row['job_id'],))
    value = json.loads(decision['payload_json'])
    if part == 'delivered_message':
        await c.manager.db.execute(f'DELETE FROM {c.store.t.messages} WHERE session_id=? AND seq=?', (c.sid, value['event']['delivered_seq']))
    elif part == 'decision':
        value['event']['managed_delivery']['payload_hash'] = 'f' * 64
        await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=? WHERE event_id=?', (_json(value), decision['event_id']))
    elif part == 'snapshot':
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', (value['snapshot_id'],))
    elif part == 'response':
        await c.manager.db.execute(f'DELETE FROM {c.store.t.messages} WHERE id=?', (int(value['response_ref']['id']),))
    else:
        await c.manager.db.execute(f'DELETE FROM {c.manager.events} WHERE event_id=?', ('model_closed_' + value['handler_job_id'],))
    with pytest.raises(JobError):
        await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
    assert c.state.dispatched == c.model.calls == model.calls == 1


async def test_duplicate_queue_after_handling_does_not_create_another_notification_or_model(pending_review):
    c, row, loop, model = pending_review
    await loop.send('Handle this original result once.', c.sid)
    await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
    before = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
    receipt = await enqueue(loop, c.actor, row['job_id'])
    assert receipt['inbox_status'] == 'delivered'
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq') == before
    assert model.calls == c.state.dispatched == 1 and c.host.reserve_calls == 2


async def test_foreign_session_and_child_context_cannot_redirect_the_original_main(pending_review):
    from power_loop.runtime.jobs import JobActor
    from power_loop.runtime.task_runtime import _SEND

    c, row, loop, model = pending_review
    outsider = await c.store.create_session()
    with pytest.raises(JobError):
        await enqueue(loop, JobActor(c.actor.tenant_id, c.actor.principal_id, outsider), row['job_id'])
    token = _SEND.set(object())
    try:
        with pytest.raises(JobError, match='requires_host'):
            await enqueue(loop, c.actor, row['job_id'])
    finally:
        _SEND.reset(token)
    assert model.calls == 0


@pytest.mark.parametrize('stage', ['inbox', 'notice', 'projection', 'commit_ack'])
async def test_queue_and_receipt_publish_atomically_or_recall_the_original_commit(registry_store, stage):
    from contextlib import asynccontextmanager

    c, row = await failure_case(registry_store)
    await flush_local(c.runtime, c.actor, row['job_id'])
    loop, model = await controller(c)
    original_write, fired = c.manager._write, []

    class Transaction:
        def __init__(self, target, flags):
            self.target, self.flags = target, flags

        def __getattr__(self, key):
            return getattr(self.target, key)

        async def execute(self, statement, params=()):
            is_inbox = statement.startswith('INSERT') and c.store.t.follow_up_queue in statement
            is_notice = statement.startswith(f'INSERT INTO {c.manager.events} ') and len(params) > 2 and params[2] == 'managed_effect_delivery'
            is_projection = "SET result_status='pending_delivery'" in statement
            if not fired and ((stage == 'inbox' and is_inbox) or (stage in ('notice', 'commit_ack') and is_notice)
                              or (stage == 'projection' and is_projection)):
                self.flags['target'] = True
                if stage != 'commit_ack':
                    fired.append(stage)
                    raise OSError('test_managed_delivery_' + stage)
            return await self.target.execute(statement, params)

    @asynccontextmanager
    async def write():
        flags = {'target': False}
        async with original_write() as tx:
            yield Transaction(tx, flags)
        if flags['target'] and not fired:
            fired.append(stage)
            raise OSError('test_managed_delivery_' + stage)

    c.manager._write = write
    try:
        with pytest.raises(OSError, match='test_managed_delivery_'):
            await enqueue(loop, c.actor, row['job_id'])
        rows = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.follow_up_queue} WHERE item_id=?', (identity(c.manager, row),))
        notices = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='managed_effect_delivery'", (row['job_id'],))
        assert len(rows) == len(notices) == (1 if stage == 'commit_ack' else 0)
        assert await enqueue(loop, c.actor, row['job_id']) == await enqueue(loop, c.actor, row['job_id'])
        assert model.calls == 0 and c.state.dispatched == c.model.calls == 1
        assert (await c.manager.get_job(c.actor, row['job_id'])).result_status == 'pending_delivery'
    finally:
        await loop.aclose(drain_timeout_s=0.1)


async def test_local_hint_failure_does_not_lose_or_repeat_a_durable_outcome(pending_review, monkeypatch):
    c, row, loop, model = pending_review
    monkeypatch.setattr(loop, '_note_committed_inbox', lambda sid: (_ for _ in ()).throw(RuntimeError('test_hint_failed')))
    assert (await enqueue(loop, c.actor, row['job_id']))['inbox_status'] == 'pending'
    assert model.calls == 0
    await loop.send('Handle the durable original result.', c.sid)
    assert model.calls == c.state.dispatched == 1
    assert (await c.manager.get_job(c.actor, row['job_id'])).result_status == 'handled'


async def test_processing_commit_ack_loss_recalls_same_decision_and_does_not_repeat_effect(pending_review):
    c, row, loop, model = pending_review
    commit = c.manager.commit_processing_decision
    fired = []

    async def lose_ack(*args, **kwargs):
        result = await commit(*args, **kwargs)
        if not fired:
            fired.append(True)
            raise OSError('test_processing_commit_ack_lost')
        return result

    c.manager.commit_processing_decision = lose_ack
    result = await loop.send('Handle the known result once.', c.sid)
    assert result.status == 'completed'
    assert (await c.manager.get_job(c.actor, row['job_id'])).supervision_status == 'closed'
    assert len(await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='processing_decision'", (row['job_id'],))) == 1
    assert model.calls == c.state.dispatched == c.model.calls == 1 and c.host.reserve_calls == 2
