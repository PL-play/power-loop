"""Original persisted tools reserve, settle and retry without re-execution."""
import json

import pytest

from power_loop import ToolDefinition, ToolRegistry
from power_loop.runtime import model_budgets as budgets
from power_loop.runtime import tool_attempts as tools
from power_loop.runtime.budget_contracts import BudgetPlan
from power_loop.runtime.jobs import JobError
from power_loop.runtime.pricing import Free
from power_loop.runtime.task_runtime import _SEND

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case, counters

__all__ = ['budget_case', 'registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def prepared(c, call_id='call-1'):
    reg = await c.manager.register_job(c.runtime.actor(c.sid), kind='send', session_id=c.sid,
        system_key='tool-send', request={'model_attempt_protocol': 1, 'send_index': 1})
    binding = await c.runtime.claim(c.runtime.actor(c.sid), reg)
    from power_loop.runtime.model_attempts import enable
    await enable(binding)
    seq = await c.store.append_message(c.sid, role='assistant', send_index=1, tool_calls=[
        {'id': call_id, 'type': 'function', 'function': {'name': 'echo', 'arguments': json.dumps({'text': 'hello'})}}])
    ref = await c.store.message_ref(c.sid, seq)
    registry = ToolRegistry()
    registry.register(ToolDefinition('echo', '', pricing=Free('local')), lambda **kwargs: kwargs['text'])
    async def plan(context, call, *, pricing):
        assert call.name == 'echo' and context['kind'] == 'tool'
        return BudgetPlan(0, {'attempt_id': context['attempt_id'], 'origin': context})
    c.host.prepare_tool = plan
    token = _SEND.set(binding)
    try:
        attempt = await tools.prepare(binding, ref, call_id, registry.get('echo'), {'text': 'hello'})
    finally:
        _SEND.reset(token)
    return attempt, registry


async def start(c, attempt):
    token = _SEND.set(attempt.binding)
    try:
        assert await tools.reserve(attempt)
        async with c.manager._write() as tx:
            job = await c.manager._fence(tx, attempt.binding.actor, attempt.binding.lease)
            await tools.authorize_dispatch(attempt, tx, job)
    finally:
        _SEND.reset(token)


async def settle(c, attempt, error=None):
    token = _SEND.set(attempt.binding)
    try:
        tools.capture(attempt, None, error)
    finally:
        _SEND.reset(token)
    return await tools.finish(attempt)


async def test_original_tool_uses_own_identity_zero_tokens_and_frozen_charge(budget_case):
    c = budget_case
    attempt, registry = await prepared(c)
    await start(c, attempt)
    assert await registry.invoke_async('echo', {'text': 'hello'}) == 'hello'
    ack = await settle(c, attempt)
    state = await budgets.get_budget(c.manager, c.actor, attempt.attempt_id)
    assert ack['request_hash'] == state['ack']['request_hash']
    assert state['context']['kind'] == 'tool' and 'model' not in state['context']
    charge = state['submission']['request']['evidence']['charge']
    assert charge['amount'] == '0' and charge['quality'] == 'free'
    assert (await counters(c))['used_calls'] == 1 and (await counters(c))['held_tokens'] == 0
    assert (await counters(c))['inflight'] == 0


async def test_apply_ack_loss_reconciles_saved_charge_without_price_or_handler(budget_case):
    c = budget_case
    attempt, _ = await prepared(c)
    await start(c, attempt)
    c.host.apply_lost_ack = True
    with pytest.raises(OSError):
        await settle(c, attempt)
    state = await budgets.get_budget(c.manager, c.actor, attempt.attempt_id)
    assert state['pending'] and len(c.host.applied) == 1
    c.host.apply_lost_ack = False
    result = await budgets.reconcile_usage(c.runtime, c.actor, attempt.binding.lease.job_id, attempt.attempt_id)
    assert result['usage_acknowledged'] and not result['execution_authority']
    assert len(c.host.applied) == 1 and c.host.reserve_calls == 1
    assert (await counters(c))['used_calls'] == 1


async def test_dispatch_without_host_grant_and_second_dispatch_are_rejected(budget_case):
    c = budget_case
    attempt, _ = await prepared(c)
    token = _SEND.set(attempt.binding)
    try:
        async with c.manager._write() as tx:
            job = await c.manager._fence(tx, attempt.binding.actor, attempt.binding.lease)
            with pytest.raises(JobError, match='budget_attempt_missing'):
                await tools.authorize_dispatch(attempt, tx, job)
    finally:
        _SEND.reset(token)
    await start(c, attempt)
    token = _SEND.set(attempt.binding)
    try:
        async with c.manager._write() as tx:
            job = await c.manager._fence(tx, attempt.binding.actor, attempt.binding.lease)
            with pytest.raises(JobError, match='already_dispatched'):
                await tools.authorize_dispatch(attempt, tx, job)
    finally:
        _SEND.reset(token)
    await settle(c, attempt, RuntimeError('local tool failed'))


@pytest.mark.parametrize('failed', [False, True])
@pytest.mark.parametrize('charged_error', [False, True])
async def test_real_agent_loop_bills_original_tool_and_does_not_reexecute_on_delivery_retry(registry_store, failed, charged_error):
    from decimal import Decimal

    from power_loop import (
        AgentLoopConfig,
        LLMResponse,
        LLMService,
        LLMTokenUsage,
        StatefulAgentLoop,
        ToolResult,
        record_billing_receipt,
    )
    from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.pricing import BillingReceipt, Charge, Estimate, Money
    from power_loop.runtime.task_runtime import TaskRuntime

    from .test_model_budgets_192 import Host
    store, _ = registry_store
    host = Host()
    class TestPrice:
        rule_id = 'isolated-provider-tool'
        def version(self):
            return 'frozen-test-rule-v1'
        def estimate(self, call):
            return Estimate(Money(Decimal('0.5'), 'CNY'), {'calls': 1})
        def settle(self, call, receipt, error):
            assert receipt.provider_amount.amount == Decimal('0.25')
            return Charge(receipt.provider_amount, 'provider_reported', receipt.units,
                          self.rule_id, self.version())

    async def plan(context, call, *, pricing):
        assert context['kind'] == 'tool' and pricing is registry.get('echo').pricing
        return BudgetPlan(0, {'attempt_id': context['attempt_id'], 'origin': context})
    host.prepare_tool = plan

    class Model(LLMService):
        calls = 0
        async def complete(self, request, **kwargs):
            self.calls += 1
            tool_calls = ([{'id': 'real-echo', 'type': 'function', 'function': {
                'name': 'echo', 'arguments': '{"text":"hello"}'}}] if self.calls == 1 else None)
            return LLMResponse(raw_text='' if tool_calls else 'DONE', tool_calls=tool_calls,
                token_usage=LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
        def stream(self, request):
            raise AssertionError('complete only')
        async def close(self):
            pass

    calls = []
    async def echo(text):
        from power_loop import tool_billing_managed
        assert tool_billing_managed('echo') and not tool_billing_managed('nested-other-tool')
        calls.append(text)
        host.apply_unavailable = failed
        receipt = BillingReceipt(provider_amount=Money(Decimal('0.25'), 'CNY'),
                                 provider_request_id='original-provider-request')
        if charged_error:
            await record_billing_receipt(receipt)
            raise ValueError('provider charged; response decoding failed')
        return ToolResult('actual tool result', receipt)
    registry = ToolRegistry()
    registry.register(ToolDefinition('echo', '', pricing=TestPrice()), echo)
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id='tool-loop', principal_id='host', executor_id='executor',
        boot_id='boot', lease_ms=300000, boot_contract_hash='c'*64,
        budget_config=BudgetConfig('tools-196', (LocalBudgetScope('shared', max_calls=8),), host))
    model = Model()
    host.service = model
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(model='test-model', max_tokens=20, max_rounds=3, delegation_policy='single_level'))
    sid = await loop.new_session()
    try:
        result = await loop.send('echo hello', sid)
        states = await manager.db.fetchall(f'SELECT state_json FROM {manager.prefix}task_budget_attempts')
        tool_states = [json.loads(row['state_json']) for row in states
                       if json.loads(row['state_json'])['context'].get('kind') == 'tool']
        assert len(tool_states) == 1 and calls == ['hello']
        state = tool_states[0]
        assert state['observation']['charge']['quality'] == 'provider_reported'
        assert state['observation']['charge']['amount'] == '0.25'
        assert state['observation']['outcome'] == ('failed' if charged_error else 'returned')
        assert state['phase'] == 'settling'
        host.apply_unavailable = False
        if failed:
            assert state['pending']
            await budgets.reconcile_usage(runtime, runtime.actor(sid), state['job_id'], state['attempt_id'])
        else:
            assert not state['pending'] and result.status == 'completed'
        assert calls == ['hello']
        assert state['attempt_id'] in host.applied
    finally:
        host.apply_unavailable = False
        await runtime.shutdown(timeout_s=1)
        await loop.aclose(drain_timeout_s=.1)


async def test_unstarted_tool_refunds_once_and_cannot_dispatch_after_release(budget_case):
    c = budget_case
    attempt, _ = await prepared(c)
    token = _SEND.set(attempt.binding)
    try:
        await tools.reserve(attempt)
        first = await budgets.release_unstarted(c.runtime, c.actor, attempt.binding.lease.job_id, attempt.attempt_id)
        assert await budgets.release_unstarted(c.runtime, c.actor, attempt.binding.lease.job_id, attempt.attempt_id) == first
        assert len(c.host.released) == 1 and (await counters(c))['held_calls'] == 0
        async with c.manager._write() as tx:
            job = await c.manager._fence(tx, attempt.binding.actor, attempt.binding.lease)
            with pytest.raises(JobError, match='dispatch_not_authorized'):
                await tools.authorize_dispatch(attempt, tx, job)
    finally:
        _SEND.reset(token)


async def test_tool_release_ack_loss_replays_original_release_and_refunds_once(budget_case):
    c = budget_case
    attempt, _ = await prepared(c)
    token = _SEND.set(attempt.binding)
    try:
        await tools.reserve(attempt)
    finally:
        _SEND.reset(token)
    original = c.host.release
    attempts = []
    async def lose_ack(submission):
        attempts.append(submission.request_hash)
        result = await original(submission)
        if len(attempts) == 1:
            raise OSError('release committed; ACK lost')
        return result
    c.host.release = lose_ack
    with pytest.raises(OSError):
        await budgets.release_unstarted(c.runtime, c.actor, attempt.binding.lease.job_id, attempt.attempt_id)
    assert (await counters(c))['held_calls'] == 1
    await budgets.release_unstarted(c.runtime, c.actor, attempt.binding.lease.job_id, attempt.attempt_id)
    assert len(set(attempts)) == 1 and (await counters(c))['held_calls'] == 0


async def test_tool_dispatch_and_budget_authorization_rollback_together(budget_case):
    c = budget_case
    attempt, _ = await prepared(c)
    token = _SEND.set(attempt.binding)
    try:
        c.host.reserve_lost_ack = True
        await tools.reserve(attempt)
        with pytest.raises(OSError):
            async with c.manager._write() as tx:
                job = await c.manager._fence(tx, attempt.binding.actor, attempt.binding.lease)
                await tools.authorize_dispatch(attempt, tx, job)
                raise OSError('dispatch transaction rolled back')
    finally:
        _SEND.reset(token)
    assert c.host.reserve_calls == 1
    state = await budgets.get_budget(c.manager, c.actor, attempt.attempt_id)
    assert state['phase'] == 'host_reserved'
    await budgets.release_unstarted(c.runtime, c.actor, attempt.binding.lease.job_id, attempt.attempt_id)
    assert (await counters(c))['used_calls'] == 0


async def test_rehashed_settlement_tampering_is_rejected_before_host_delivery(budget_case):
    from power_loop.runtime.jobs import _hash, _json
    c = budget_case
    attempt, _ = await prepared(c)
    await start(c, attempt)
    c.host.apply_unavailable = True
    with pytest.raises(OSError):
        await settle(c, attempt)
    c.host.apply_unavailable = False
    key = tools.event_id(c.manager, 'finished', attempt.attempt_id)
    row = await c.manager.db.fetchone(f'SELECT payload_json FROM {c.manager.events} WHERE event_id=?', (key,))
    payload = json.loads(row['payload_json'])
    payload['charge']['amount'] = '999'
    await c.manager.db.execute(f'UPDATE {c.manager.events} SET payload_json=?,request_hash=? WHERE event_id=?',
                              (_json(payload), _hash(payload), key))
    before = c.host.apply_calls
    with pytest.raises(JobError, match='evidence_changed'):
        await budgets.reconcile_usage(c.runtime, c.actor, attempt.binding.lease.job_id, attempt.attempt_id)
    assert c.host.apply_calls == before
