import asyncio
import json
from decimal import Decimal

import pytest

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    StatefulAgentLoop,
    ToolDefinition,
    ToolRegistry,
    invoke_priced_tool,
    record_billing_receipt,
    tool_billing_active,
    tool_billing_managed,
)
from power_loop.runtime import model_budgets as budgets
from power_loop.runtime.budget_contracts import BudgetConfig, BudgetPlan, LocalBudgetScope
from power_loop.runtime.jobs import JobError, JobManager
from power_loop.runtime.pricing import BillingReceipt, Charge, Estimate, Free, Money
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Paid:
    rule_id = 'inner-image'
    def version(self):
        return 'fixed-inner-price'
    def estimate(self, call):
        return Estimate(Money(Decimal('1'), 'CNY'), {'calls': 1})
    def settle(self, call, receipt, error):
        if receipt is None:
            raise JobError('missing_supplier_receipt')
        return Charge(receipt.provider_amount, 'provider_reported', receipt.units,
            self.rule_id, self.version())


@pytest.mark.parametrize('fault', ['success', 'paid_error', 'ack_lost', 'parallel', 'reserve_lost',
    'reserve_unconfirmed', 'receipt_disk', 'finished_disk', 'detached', 'argument_mutation'])
async def test_composite_uses_separate_original_paid_attempts_and_only_retries_bookkeeping(registry_store, monkeypatch, fault):
    store, _ = registry_store
    host, calls, child_contexts = Host(), [], []
    original_arguments = {'text': '0', 'metadata': {'value': 'original'}}
    from power_loop.runtime import nested_tools, tool_attempts
    if fault == 'receipt_disk':
        original_persist, failures = nested_tools.persist_receipt, []
        async def persist(slot):
            if not failures:
                failures.append(True)
                raise OSError('receipt checkpoint disk failure')
            return await original_persist(slot)
        monkeypatch.setattr(nested_tools, 'persist_receipt', persist)
    if fault == 'finished_disk':
        original_save, failures = tool_attempts._save, []
        async def save(manager, tx, job, phase, attempt_id, value):
            if phase == 'finished' and value['charge']['rule_id'] == 'inner-image' and not failures:
                failures.append(True)
                raise OSError('settlement checkpoint disk failure')
            return await original_save(manager, tx, job, phase, attempt_id, value)
        monkeypatch.setattr(tool_attempts, '_save', save)
    supplier_started = asyncio.Event()
    async def plan(context, call, *, pricing):
        if call.name == 'inner':
            child_contexts.append(context)
            host.reserve_lost_ack = fault in ('reserve_lost', 'reserve_unconfirmed')
            if fault == 'argument_mutation':
                original_arguments['metadata']['value'] = 'changed during reservation'
        return BudgetPlan(0, {'attempt_id': context['attempt_id'], 'origin': context})
    host.prepare_tool = plan
    original_lookup = host.lookup
    async def lookup(plan):
        if fault == 'reserve_unconfirmed' and plan.request['origin'].get('tool_name') == 'inner':
            return None
        return await original_lookup(plan)
    host.lookup = lookup
    class Model(LLMService):
        calls = 0
        async def complete(self, request, **kw):
            self.calls += 1
            return LLMResponse(raw_text='DONE' if self.calls > 1 else '', tool_calls=(None if self.calls > 1 else
                [{'id': 'composite-call', 'type': 'function', 'function': {'name': 'composite', 'arguments': '{}'}}]),
                token_usage=LLMTokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))
        def stream(self, request):
            raise AssertionError('complete only')
        async def close(self):
            pass
    async def paid(text, metadata=None):
        assert tool_billing_active() and tool_billing_managed('inner') and not tool_billing_managed('composite')
        if fault == 'argument_mutation':
            assert metadata == {'value': 'original'}
        calls.append(text)
        supplier_started.set()
        if fault == 'detached':
            await asyncio.sleep(.02)
        await record_billing_receipt(BillingReceipt(provider_amount=Money(Decimal('0.25'), 'CNY'),
            provider_request_id='supplier-' + text))
        if fault == 'paid_error':
            raise ValueError('failed after supplier billed')
        if fault == 'ack_lost':
            host.apply_lost_ack = True
        return 'image-' + text
    inner = ToolDefinition('inner', '', required_params=('text',), pricing=Paid())
    async def composite():
        assert tool_billing_managed('composite')
        try:
            if fault == 'argument_mutation':
                return await invoke_priced_tool(inner, original_arguments, paid)
            if fault == 'detached':
                asyncio.create_task(invoke_priced_tool(inner, {'text': '0'}, paid))
                await supplier_started.wait()
                return 'created task'
            if fault == 'parallel':
                result = await asyncio.gather(*(invoke_priced_tool(inner, {'text': str(i)}, paid) for i in range(2)))
                assert result == ['image-0', 'image-1']
                return '\n'.join(result)
            return await invoke_priced_tool(inner, {'text': '0'}, paid)
        finally:
            host.apply_lost_ack = host.reserve_lost_ack = False
    registry = ToolRegistry()
    registry.register(ToolDefinition('composite', '', pricing=Free('composite')), composite)
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id='nested-tools', principal_id='host', executor_id='executor',
        boot_id='boot', lease_ms=300000, boot_contract_hash='c'*64,
        budget_config=BudgetConfig('nested-tools-196', (LocalBudgetScope('shared', max_calls=20, max_inflight=8),), host))
    model = Model()
    host.service = model
    loop = StatefulAgentLoop(llm=model, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(model='test', max_tokens=20, max_rounds=3, delegation_policy='single_level'))
    try:
        sid = await loop.new_session()
        result = await loop.send('make image', sid)
        if fault in ('success', 'ack_lost', 'parallel', 'reserve_lost', 'detached', 'argument_mutation'):
            assert result.status == 'completed', result
        states = [json.loads(row['state_json']) for row in await manager.db.fetchall(
            f'SELECT state_json FROM {manager.prefix}task_budget_attempts')]
        inner_states = [state for state in states if state['context'].get('tool_name') == 'inner']
        count = 2 if fault == 'parallel' else 1
        assert len(inner_states) == count
        assert len({state['attempt_id'] for state in inner_states}) == count
        if fault == 'reserve_unconfirmed':
            assert calls == [] and len(host.released) == 1
        else:
            assert len(calls) == count
            for state in inner_states:
                assert state['observation']['charge']['amount'] == '0.25'
                assert state['observation']['charge']['rule_version'] == 'fixed-inner-price'
                assert state['observation']['provider_response_id'] in {'supplier-' + call for call in calls}
                host.apply_lost_ack = False
                await budgets.reconcile_usage(runtime, runtime.actor(sid), state['job_id'], state['attempt_id'])
                assert len(calls) == count
        assert runtime._native_returns.pending_count() == 0
        assert not runtime._native_returns.slots
        for state in states:
            if state['pending'] and state['context'].get('tool_name') != 'inner':
                host.apply_lost_ack = host.reserve_lost_ack = False
                await budgets.reconcile_usage(runtime, runtime.actor(sid), state['job_id'], state['attempt_id'])
    finally:
        host.apply_lost_ack = host.reserve_lost_ack = False
        await loop.aclose()


async def test_internal_paid_tool_requires_original_callback_before_any_handler():
    async def never(**kw):
        pytest.fail('supplier invoked outside original callback')
    with pytest.raises(JobError, match='nested_tool_original_required'):
        await invoke_priced_tool(ToolDefinition('inner', '', pricing=Paid()), {}, never)
