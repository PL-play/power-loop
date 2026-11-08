"""Concurrent current workers share one budget policy without repeating calls."""
import asyncio

import pytest

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMService,
    LLMTokenUsage,
    RegistryRuntimeFactory,
    Runtime,
    RuntimeDescriptor,
    TrustedActor,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_transport import MemoryMessageTransport
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.session_ownership import MemorySessionOwnershipCoordinator
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

async def test_two_actual_sends_share_first_policy_without_repeating_models(registry_store):
    store, connect = registry_store
    stores = [store, await connect()]
    entered = []

    class Model(LLMService):
        calls = 0
        async def complete(self, request, **kwargs):
            self.calls += 1
            return LLMResponse(raw_text='done', token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))
        async def close(self):
            pass

    class RacingHost(Host):
        async def prepare(self, context, logical_request, *, service):
            entered.append(context['attempt_id'])
            return await super().prepare(context, logical_request, service=service)

    host, model = RacingHost(), Model()
    host.service = model
    budget = BudgetConfig('first-policy', (LocalBudgetScope('shared', max_calls=10,
        max_tokens=10000, max_inflight=10),), host, max_pending=16)
    transport, coordinator = MemoryMessageTransport(), MemorySessionOwnershipCoordinator()
    runtimes = []
    for index, active in enumerate(stores):
        factory = RegistryRuntimeFactory()
        managed = TaskRuntime(JobManager(active), tenant_id='policy-race', principal_id='host',
            executor_id=f'executor-{index}', boot_id='boot', boot_contract_hash='a'*64, budget_config=budget)
        async def build(descriptor, context, managed=managed):
            return ExecutionEnvironment(model, None, AgentLoopConfig(compactor=None, max_tokens=20), task_runtime=managed)
        factory.register('policy@1', build)
        runtime = Runtime(store=active, factory=factory, node_id=f'worker-{index}', capacity=1,
            coordinator=coordinator, transport=transport, timers_enabled=False, recovery_interval_s=None)
        runtimes.append(runtime)
    actor = TrustedActor('policy-race', 'host', None)
    descriptor = RuntimeDescriptor(1, 'agent', 'policy@1', None, 'builtin.empty@1', 'builtin.none@1', 'builtin.none@1', {'max_tokens': 20}, {})
    try:
        for runtime in runtimes:
            await runtime.worker.start()
        targets = [await runtime.open_session(descriptor, actor=actor) for runtime in runtimes]
        receipts = await asyncio.gather(*(runtime.client.send(target, 'run once', message_id='start', actor=actor)
            for runtime, target in zip(runtimes, targets, strict=True)))
        results = await asyncio.gather(*(runtime.client.wait(receipt.input_ref, actor=actor, deadline_ms=now_ms()+15000)
            for runtime, receipt in zip(runtimes, receipts, strict=True)))
        assert all(result.status == 'completed' and result.final_text == 'done' for result in results), results
        assert model.calls == host.reserve_calls == host.apply_calls == 2
        assert len(set(entered)) == 2
        assert (await store._db.fetchone(f'SELECT COUNT(*) AS n FROM {store.table_prefix}task_budget_policies'))['n'] == 1
        for active, target in zip(stores, targets, strict=True):
            assert [row.role for row in await active.load_all_messages(target.session_id)] == ['user', 'assistant']
    finally:
        for runtime in reversed(runtimes):
            await runtime.worker.shutdown(timeout_s=2, cancel_running=True)
            await runtime.client.aclose()
        await transport.aclose()
