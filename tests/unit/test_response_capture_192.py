"""Capture is original-attempt evidence, independent of observers and billing."""
import asyncio
from dataclasses import asdict, replace

import pytest

from power_loop import AgentEvent, AgentEventType, LlmResponseReceivedPayload
from power_loop.runtime import model_attempts as attempts
from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.jobs import JobError
from power_loop.runtime.response_capture import ModelResponseCapture, ResponseCaptureReceipt, record

from .test_job_registry_192 import registry_store
from .test_model_attempt_events_192 import event_case
from .test_model_budgets_192 import budget_case, counters, start

__all__ = ['registry_store', 'budget_case', 'event_case']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
FACTS = dict(usage={'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10},
             cost_amount=None, response_id='original-response')


def receipt(capture):
    return ResponseCaptureReceipt(capture.capture_id, capture.fingerprint, 'original-host-record')


async def test_capture_precedes_observation_and_is_bound_before_dispatch(budget_case):
    c = budget_case
    seen = []
    async def capture(packet):
        assert await attempts._read(c.manager, c.manager.db, 'finished', packet.grant.attempt_id, required=False) is None
        state = await budgets.get_budget(c.manager, c.actor, packet.grant.attempt_id)
        assert state['phase'] == 'dispatch_authorized' and state['observation'] is None
        assert packet.context == state['context'] and asdict(packet.grant) == state['grant']
        assert packet.started_at_ms == state['started_at_ms']
        seen.append(packet)
        return receipt(packet)
    c.host.capture_response = capture
    a = await start(c)
    c.host.capture_response = lambda _: pytest.fail('callback must be frozen before physical dispatch')
    await record(a, reported=FACTS, observation=FACTS, estimated=False)
    [packet] = seen
    assert packet.usage == FACTS['usage'] and packet.response_id == FACTS['response_id']
    assert packet.grant.attempt_id == a.attempt_id
    assert (await counters(c))['inflight'] == 0 and c.host.reserve_calls == c.host.apply_calls == 1
    assert not c.runtime.budget_control_tasks()


@pytest.mark.parametrize('failure', ['raise', 'cancel', 'none', 'wrong_id', 'wrong_hash', 'mutate'])
async def test_bad_capture_cannot_prevent_observation_or_repeat_charge(budget_case, failure, caplog):
    c = budget_case
    async def capture(packet):
        if failure == 'raise':
            raise OSError('PRIVATE secret must not be logged')
        if failure == 'cancel':
            raise asyncio.CancelledError('PRIVATE secret must not be logged')
        if failure == 'none':
            return None
        if failure == 'wrong_id':
            return replace(receipt(packet), capture_id='another-call')
        if failure == 'wrong_hash':
            return replace(receipt(packet), fingerprint='f' * 64)
        packet.usage['total_tokens'] = 999
        packet.context['job_id'] = 'mutated'
        return receipt(packet)
    c.host.capture_response = capture
    a = await start(c)
    await record(a, reported=FACTS, observation=FACTS, estimated=False)
    state = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert state['observation']['usage'] == FACTS['usage']
    assert state['context']['job_id'] == c.binding.lease.job_id
    assert not state['pending'] and c.host.reserve_calls == c.host.apply_calls == 1
    assert 'unconfirmed' in caplog.text and 'PRIVATE' not in caplog.text


async def test_capture_does_not_depend_on_post_response_store_availability(budget_case, monkeypatch):
    c = budget_case
    seen = []
    async def capture(packet):
        seen.append(packet)
        return receipt(packet)
    c.host.capture_response = capture
    a = await start(c)
    original = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    def unavailable():
        raise OSError('original store offline')
    with monkeypatch.context() as patch:
        patch.setattr(c.manager, '_write', unavailable)
        await record(a, reported=FACTS, observation=FACTS, estimated=False)
    assert len(seen) == 1 and c.host.apply_calls == 0
    assert await budgets.get_budget(c.manager, c.actor, a.attempt_id) == original
    assert c.binding.resource_evidence['resource_state'] == 'unknown'


@pytest.mark.parametrize('waiter', ['cancel', 'timeout'])
async def test_waiter_exit_retains_actual_capture_and_observation(budget_case, waiter, monkeypatch):
    c = budget_case
    if waiter == 'timeout':
        config = replace(c.config, timeout_s=0.1)
        c.runtime.budget_config = c.runtime._original_budget_config = config
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []
    async def capture(packet):
        entered.set()
        await release.wait()
        seen.append(packet)
        return receipt(packet)
    c.host.capture_response = capture
    a = await start(c)
    task = asyncio.create_task(record(a, reported=FACTS, observation=FACTS, estimated=False))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if waiter == 'cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await asyncio.wait_for(task, 2)
        pending = c.runtime.budget_control_tasks()
        assert pending and all(not t.cancelled() for t in pending)
        release.set()
        await asyncio.wait_for(asyncio.gather(*pending), 3)
        assert len(seen) == 1
        state = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
        assert state['observation']['status'] == 'returned' and not state['pending']
        assert c.host.reserve_calls == c.host.apply_calls == 1
    finally:
        release.set()
        await asyncio.gather(task, *c.runtime.budget_control_tasks(), return_exceptions=True)


async def test_missing_usage_is_not_estimated_in_capture(budget_case):
    c = budget_case
    seen = []
    async def capture(packet):
        seen.append(packet)
        return receipt(packet)
    c.host.capture_response = capture
    a = await start(c)
    await record(a, reported=dict(usage={}, response_id=None, cost_amount=None), observation=FACTS, estimated=True)
    assert seen[0].usage == {} and seen[0].cost_amount is None
    saved = await budgets.get_budget(c.manager, c.actor, a.attempt_id)
    assert saved['observation']['estimated'] and saved['observation']['usage'] == FACTS['usage']


@pytest.mark.parametrize('changes', [dict(usage={'total_tokens': True}), dict(usage={'unexpected': 1}),
    dict(usage={'total_tokens': -1}), dict(cost_amount='NaN'), dict(cost_amount='-1'),
    dict(response_id='bad\nvalue'), dict(started_at_ms=True), dict(received_at_ms=-1),
    dict(context={'store_id': 'other'}), dict(context={'password': 'secret'})])
async def test_capture_rejects_invalid_facts(budget_case, changes):
    c = budget_case
    c.host.capture_response = lambda _: None
    a = await start(c)
    binding = a.capture_binding
    value = ModelResponseCapture(binding.context, binding.grant, binding.started_at_ms, 1, **FACTS)
    with pytest.raises(JobError):
        replace(value, **changes)


@pytest.mark.parametrize('failure', [None, RuntimeError, asyncio.CancelledError, 'mutate'])
async def test_new_event_precedes_capture_and_cannot_change_actual_observation(event_case, failure):
    c, order = event_case, []
    def observer(event):
        order.append('event')
        assert isinstance(event.data, LlmResponseReceivedPayload)
        assert AgentEvent.from_dict(event.to_dict()).payload == event.payload
        if failure == 'mutate':
            event.data.usage['total_tokens'] = 999
        elif failure:
            raise failure('observer is not authority')
    async def capture(packet):
        order.append('capture')
        assert await attempts._read(c.manager, c.manager.db, 'finished', packet.grant.attempt_id, required=False) is None
        assert packet.usage['total_tokens'] == 10
        return receipt(packet)
    c.host.capture_response = capture
    c.loop.event_bus.subscribe(AgentEventType.LLM_RESPONSE_RECEIVED, observer)
    sid = await c.loop.new_session()
    await c.loop.send('Reply 42', sid)
    assert order == ['event', 'capture'] and c.model.calls == c.host.apply_calls == 1
    [done] = [e for e in c.events if e.type == AgentEventType.LLM_CALL_COMPLETED]
    assert done.data.total_tokens == 10 and done.data.model_attempt_id


async def test_no_response_event_or_capture_on_supplier_error(event_case):
    c = event_case
    c.model.failure = True
    async def forbidden(_):
        pytest.fail('No complete response was received')
    c.host.capture_response = forbidden
    sid = await c.loop.new_session()
    await c.loop.send('Reply 42', sid)
    assert not [e for e in c.events if e.type == AgentEventType.LLM_RESPONSE_RECEIVED]
    assert c.model.calls == 1


@pytest.mark.parametrize('event_case', ['legacy', 'tracked', 'budget'], indirect=True)
@pytest.mark.parametrize('kind', [AgentEventType.LLM_CALL_COMPLETED, AgentEventType.STREAM_COMPLETED])
async def test_post_response_subscriber_failure_never_retries_physical_call(event_case, kind):
    from power_loop.runtime.retry import LLMRetryPolicy
    c = event_case
    c.loop.config.retry_policy = LLMRetryPolicy(max_attempts=3)
    def broken(event):
        raise RuntimeError('post-response telemetry failed')
    c.loop.event_bus.subscribe(kind, broken)
    await c.loop.send('Reply 42', await c.loop.new_session())
    assert c.model.calls == 1


async def test_invalid_capture_provider_blocks_before_dispatch(budget_case):
    c = budget_case
    c.host.capture_response = 'not callable'
    with pytest.raises(JobError, match='invalid_response_capture_provider'):
        await start(c)
    rows = await c.manager.db.fetchall(f"SELECT event_type FROM {c.manager.events}")
    assert not any(r['event_type'] == 'model_attempt_dispatched' for r in rows)


async def test_shutdown_tracks_capture_until_actual_callback_exit(budget_case):
    c = budget_case
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def capture(packet):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()  # an actual transport may suppress cancellation
        return receipt(packet)
    c.host.capture_response = capture
    a = await start(c)
    task = asyncio.create_task(record(a, reported=FACTS, observation=FACTS, estimated=False))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await c.runtime.shutdown(timeout_s=0.05)
        await asyncio.wait_for(cancelled.wait(), 2)
        pending = c.runtime.budget_control_tasks()
        assert pending and all(not t.done() for t in pending)
        release.set()
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 3)
        assert not c.runtime.budget_control_tasks()
    finally:
        release.set()
        await asyncio.gather(task, *c.runtime.budget_control_tasks(), return_exceptions=True)


async def test_pipeline_event_and_capture_keep_missing_provider_usage_unknown(event_case, monkeypatch):
    c, seen = event_case, []
    complete = c.model.complete
    async def missing(*args, **kwargs):
        response = await complete(*args, **kwargs)
        response.token_usage = None
        return response
    async def capture(packet):
        seen.append(packet)
        return receipt(packet)
    monkeypatch.setattr(c.model, 'complete', missing)
    c.host.capture_response = capture
    await c.loop.send('Reply 42', await c.loop.new_session())
    [early] = [e for e in c.events if e.type == AgentEventType.LLM_RESPONSE_RECEIVED]
    [done] = [e for e in c.events if e.type == AgentEventType.LLM_CALL_COMPLETED]
    assert early.data.usage == seen[0].usage == {}
    assert done.data.estimated and done.data.total_tokens > 0
