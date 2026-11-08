"""Current worker enforces a principal's mandatory policy before model dispatch."""
from dataclasses import replace

import pytest

from power_loop import (
    AgentLoopConfig,
    ExecutionEnvironment,
    LLMResponse,
    LLMTokenUsage,
    RegistryRuntimeFactory,
    Runtime,
)
from power_loop.agent.runtime_store import now_ms
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_runtime import TaskRuntime
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host
from .test_stateful_loop import _Scripted

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def policy_worker(registry_store):
    store, _ = registry_store
    free_model, paid_model, host = _Scripted(), _Scripted([LLMResponse(raw_text='done', token_usage=LLMTokenUsage(prompt_tokens=3,completion_tokens=2,total_tokens=5))]), Host()
    host.service = paid_model
    budget = BudgetConfig('first-policy', (LocalBudgetScope('shared', max_calls=10,
        max_tokens=10000, max_inflight=10),), host, max_pending=16)
    manager = JobManager(store)
    managed = [TaskRuntime(manager, tenant_id='policy-admission', principal_id='host',
        executor_id='worker', boot_id='boot', boot_contract_hash='a'*64, budget_config=value) for value in (None, budget)]
    shell = RunningAgent(store, free_model, task_runtime=managed[0])
    factory = RegistryRuntimeFactory()
    async def build(descriptor, context):
        index = int(descriptor.config.get('paid', False))
        return ExecutionEnvironment((free_model, paid_model)[index], None,
            AgentLoopConfig(compactor=None, max_tokens=20), task_runtime=managed[index])
    factory.register('fixture.agent@1', build)
    runtime = Runtime(store=store, factory=factory, node_id='policy-worker', capacity=2,
        timers_enabled=False, recovery_interval_s=None)
    await runtime.worker.start()
    targets = [await runtime.open_session(replace(shell.descriptor, config={'paid': paid, 'max_tokens':20}),
        actor=shell.actor) for paid in (False, True)]
    counter = 0
    async def send(paid):
        nonlocal counter
        counter += 1
        receipt = await runtime.client.send(targets[int(paid)], 'once', message_id=str(counter), actor=shell.actor)
        result = await runtime.client.wait(receipt.input_ref, actor=shell.actor, deadline_ms=now_ms()+10000)
        while runtime.worker._tasks:
            import asyncio
            await asyncio.gather(*tuple(runtime.worker._tasks), return_exceptions=True)
        return result
    try:
        yield store, managed, free_model, paid_model, host, send
    finally:
        await runtime.worker.shutdown(timeout_s=2, cancel_running=True)
        await runtime.client.aclose()
        await runtime.transport.aclose()


async def test_warm_free_execution_observes_the_new_mandatory_policy(policy_worker):
    store, managed, free, paid, host, send = policy_worker
    assert (await send(False)).status == 'completed'
    paid_result = await send(True)
    assert paid_result.status == 'completed', paid_result
    refused = await send(False)
    assert refused.status == 'failed' and refused.error_code == 'budget_provider_required'
    assert len(free.calls) == len(paid.calls) == 1
    assert host.reserve_calls == host.apply_calls == 1
    assert (await store._db.fetchone(f'SELECT COUNT(*) AS n FROM {store.table_prefix}task_budget_policies'))['n'] == 1
