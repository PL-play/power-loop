"""Actual main input/response, not queue acceptance, handles recovered outcomes."""

import pytest

from power_loop import AgentLoopConfig, HookPoint, InputRef, LLMResponse, LLMService, LLMTokenUsage, TargetRef
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.jobs import JobError
from power_loop.runtime.managed_completion import flush_local
from power_loop.runtime.managed_delivery import enqueue_runtime, identity
from power_loop.runtime.managed_effects import reconcile
from power_loop.tools.registry import ToolRegistry
from tests.control_faults import control_mutations
from tests.runtime_fixture import RunningAgent

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
    loop = RunningAgent(c.store, model, ToolRegistry(), task_runtime=c.runtime,
        config=AgentLoopConfig(model='test-model', max_tokens=20, max_rounds=2, compactor=None))
    loop.target = TargetRef(c.store.store_id, c.sid)
    if not sees:
        async def hide(ctx):
            ctx.messages = [m for m in ctx.messages if 'Original tool outcome:' not in str(m.get('content'))]
        loop.hooks.register(HookPoint.LLM_BEFORE, hide, name='test.hide-notice')
    return loop, model


async def consume_notice(loop, notification_id):
    await loop.start(target=loop.target)
    ref = InputRef(loop.target.store_id, loop.target.session_id, notification_id)
    await loop.runtime.client._notify(loop.target, notification_id, ref, ['work', 'activation'])
    result = await loop.runtime.client.wait(ref, actor=loop.actor, deadline_ms=now_ms()+15000)
    await loop.finish(ref, failed=result.status=='failed')
    return result


@pytest.mark.parametrize('unknown', [False, True])
async def test_main_actually_consumes_the_known_outcome_and_original_error_is_unchanged(registry_store, unknown, *, measure=None):
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
    from contextlib import nullcontext
    async with measure(c.store) if measure else nullcontext():
        try:
            first = await enqueue_runtime(c.runtime, c.actor, row['job_id'])
            again = await enqueue_runtime(c.runtime, c.actor, row['job_id'])
            assert first == again and not first['acknowledgement_created'] and model.calls == 0
            assert (await c.manager.get_job(c.actor, row['job_id'])).result_status == 'pending_delivery'
            with pytest.raises(JobError):
                await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
            result = await consume_notice(loop, first['notification_id'])
            assert result.status == 'completed' and model.calls == 1
            current = await c.manager.get_job(c.actor, row['job_id'])
            assert current.result_status == 'handled'
            await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
            assert (await c.manager.get_job(c.actor, row['job_id'])).supervision_status == 'closed'
            for raw in original:
                assert await c.manager.db.fetchone(f'SELECT * FROM {c.store.t.messages} WHERE id=?', (raw['id'],)) == raw
            assert c.state.dispatched == c.state.prepared == c.model.calls == model.calls == 1
            assert c.state.lookups == (1 if unknown else 0)
            inbox = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.follow_up_queue} WHERE item_id=?', (identity(c.manager, row),))
            assert len(inbox) == 1 and inbox[0]['status'] == 'delivered'
        finally:
            await loop.aclose()



@pytest.mark.parametrize('failure', ['hidden', 'model_error'])
async def test_queued_or_transcribed_but_unprocessed_outcome_keeps_scope_open(registry_store, failure):
    c, row = await failure_case(registry_store)
    await flush_local(c.runtime, c.actor, row['job_id'])
    loop, model = await controller(c, sees=failure != 'hidden', fail=failure == 'model_error')
    try:
        notice = await enqueue_runtime(c.runtime, c.actor, row['job_id'])
        result = await consume_notice(loop, notice['notification_id'])
        assert result.status != 'completed' if failure == 'model_error' else result.status == 'completed'
        assert (await c.manager.get_job(c.actor, row['job_id'])).result_status != 'handled'
        with pytest.raises(JobError):
            await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
        assert c.state.dispatched == c.model.calls == 1
    finally:
        await loop.aclose()


async def test_unknown_business_outcome_is_not_published_as_a_known_resolution(registry_store):
    c = await scenario(registry_store, budgeted=True, pending=True)
    [row] = await c.manager.db.fetchall(f"SELECT * FROM {c.manager.jobs} WHERE kind='background_tool'")
    loop, model = await controller(c)
    try:
        with pytest.raises(JobError):
            await enqueue_runtime(c.runtime, c.actor, row['job_id'])
        assert model.calls == 0 and c.state.lookups == 0
        assert not await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE event_type='managed_effect_delivery'")
    finally:
        await loop.aclose()


@pytest.fixture
async def pending_review(registry_store):
    c, row = await failure_case(registry_store)
    await flush_local(c.runtime, c.actor, row['job_id'])
    loop, model = await controller(c)
    await enqueue_runtime(c.runtime, c.actor, row['job_id'])
    try:
        yield c, row, loop, model
    finally:
        await loop.aclose()






async def test_duplicate_queue_after_handling_does_not_create_another_notification_or_model(pending_review):
    c, row, loop, model = pending_review
    await consume_notice(loop, identity(c.manager,row))
    await c.manager.finalize_supervision(c.actor, row['job_id'], generation=1)
    before = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq')
    receipt = await enqueue_runtime(c.runtime, c.actor, row['job_id'])
    assert receipt['inbox_status'] == 'delivered'
    assert await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.messages} ORDER BY seq') == before
    assert model.calls == c.state.dispatched == 1


async def test_foreign_session_and_child_context_cannot_redirect_the_original_main(pending_review):
    from power_loop.runtime.jobs import JobActor
    from power_loop.runtime.task_runtime import _SEND

    c, row, loop, model = pending_review
    outsider = await c.store.create_session()
    with pytest.raises(JobError):
        await enqueue_runtime(c.runtime, JobActor(c.actor.tenant_id, c.actor.principal_id, outsider), row['job_id'])
    token = _SEND.set(object())
    try:
        with pytest.raises(JobError, match='requires_host'):
            await enqueue_runtime(c.runtime, c.actor, row['job_id'])
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

    @asynccontextmanager
    async def write():
        flags = {'target': False}

        async def intercept(execute, statement, params):
            is_inbox = statement.startswith('INSERT') and c.store.t.follow_up_queue in statement
            is_notice = statement.startswith(f'INSERT INTO {c.manager.events} ') and len(params) > 2 and params[2] == 'managed_effect_delivery'
            is_projection = "SET result_status='pending_delivery'" in statement
            if not fired and ((stage == 'inbox' and is_inbox) or (stage in ('notice', 'commit_ack') and is_notice)
                              or (stage == 'projection' and is_projection)):
                flags['target'] = True
                if stage != 'commit_ack':
                    fired.append(stage)
                    raise OSError('test_managed_delivery_' + stage)
            return await execute(statement, params)

        async with original_write() as tx:
            async with control_mutations(tx, intercept):
                yield tx
        if flags['target'] and not fired:
            fired.append(stage)
            raise OSError('test_managed_delivery_' + stage)

    c.manager._write = write
    try:
        with pytest.raises(OSError, match='test_managed_delivery_'):
            await enqueue_runtime(c.runtime, c.actor, row['job_id'])
        assert fired == [stage]
        rows = await c.manager.db.fetchall(f'SELECT * FROM {c.store.t.follow_up_queue} WHERE item_id=?', (identity(c.manager, row),))
        notices = await c.manager.db.fetchall(f"SELECT event_id FROM {c.manager.events} WHERE job_id=? AND event_type='managed_effect_delivery'", (row['job_id'],))
        assert len(rows) == len(notices) == (1 if stage == 'commit_ack' else 0)
        assert await enqueue_runtime(c.runtime, c.actor, row['job_id']) == await enqueue_runtime(c.runtime, c.actor, row['job_id'])
        assert model.calls == 0 and c.state.dispatched == c.model.calls == 1
        assert (await c.manager.get_job(c.actor, row['job_id'])).result_status == 'pending_delivery'
    finally:
        await loop.aclose()


async def consume_queued(loop):
    rows=await loop.runtime.store._db.fetchall(f"SELECT item_id FROM {loop.runtime.store.t.follow_up_queue} "
        "WHERE session_id=? AND status='pending' ORDER BY id",(loop.target.session_id,))
    assert rows, 'the original notification must already be accepted'
    result=None
    for row in rows:
        result=await consume_notice(loop,row['item_id'])
    return result

