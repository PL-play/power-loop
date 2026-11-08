import asyncio
from decimal import Decimal

import pytest

from power_loop import (
    AgentLoopConfig,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    ToolDefinition,
    ToolRegistry,
    invoke_priced_tool,
    record_billing_receipt,
    tool_billing_active,
    tool_billing_managed,
)
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


@pytest.mark.parametrize('fault', ['success', 'paid_error', 'parallel', 'detached', 'argument_mutation'])
async def test_composite_uses_separate_original_paid_attempts_and_only_retries_bookkeeping(registry_store, monkeypatch, fault, runtime_agent):
    store, _ = registry_store
    host, calls, child_contexts = Host(), [], []
    original_arguments = {'text': '0', 'metadata': {'value': 'original'}}
    supplier_started = asyncio.Event()
    async def plan(context, call, *, pricing):
        if call.name == 'inner':
            child_contexts.append(context)
            if fault == 'argument_mutation':
                original_arguments['metadata']['value'] = 'changed during reservation'
        return BudgetPlan(0, {'attempt_id': context['attempt_id'], 'origin': context})
    host.prepare_tool = plan
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
    observations = {}
    prepare_usage = host.prepare_usage
    async def observe(context, grant, observation):
        observations[context['attempt_id']] = observation
        return await prepare_usage(context, grant, observation)
    host.prepare_usage = observe
    model = Model()
    host.service = model
    loop = await runtime_agent(model=model, store=store, task_runtime=runtime, registry=registry,
        config=AgentLoopConfig(model='test', max_tokens=20, max_rounds=3, delegation_policy='single_level'))
    try:
        result = await loop.send('make image')
        if fault in ('success', 'ack_lost', 'parallel', 'reserve_lost', 'detached', 'argument_mutation'):
            assert result.status == 'completed', result
        inner_ids = {context['attempt_id'] for context in child_contexts}
        count = 2 if fault == 'parallel' else 1
        assert len(inner_ids) == count and len(calls) == count
        for identity in inner_ids:
            evidence = observations[identity]
            assert evidence['charge']['amount'] == '0.25'
            assert evidence['charge']['rule_version'] == 'fixed-inner-price'
            assert evidence['provider_response_id'] in {'supplier-' + call for call in calls}
        assert not runtime._native_returns.slots
        assert (await manager.db.fetchone(f'SELECT COUNT(*) AS n FROM {manager.prefix}task_budget_attempts'))['n'] == 0
    finally:
        host.apply_lost_ack = host.reserve_lost_ack = False
        await loop.aclose()


async def test_internal_paid_tool_requires_original_callback_before_any_handler():
    async def never(**kw):
        pytest.fail('supplier invoked outside original callback')
    with pytest.raises(JobError, match='nested_tool_original_required'):
        await invoke_priced_tool(ToolDefinition('inner', '', pricing=Paid()), {}, never)
