"""A cold budget admission borrows its policy original outside its tree writer."""
from __future__ import annotations

import asyncio

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.runtime import budget_policy_sources
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.response_capture import ResponseCaptureReceipt
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host, budget_case

__all__ = ['budget_case', 'registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_two_actual_sends_reprepare_a_concurrently_created_policy_without_repeating_models(registry_store, monkeypatch):
    store, connect = registry_store
    stores = [store, await connect()]
    host = Host()

    async def capture(packet):
        return ResponseCaptureReceipt(packet.capture_id, packet.fingerprint, 'fixture-original')

    host.capture_response = capture

    class Model(LLMService):
        calls = 0
        async def complete(self, request, **kwargs):
            self.calls += 1
            return LLMResponse(raw_text='done', token_usage=LLMTokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))
        def stream(self, request):
            raise AssertionError('nonstream fixture')
        async def close(self):
            pass

    model = Model()
    host.service = model
    config = BudgetConfig('first-policy', (LocalBudgetScope('shared', max_calls=10,
        max_tokens=10000, max_inflight=10),), host, max_pending=16)
    runtimes = [TaskRuntime(JobManager(active), tenant_id='policy-race', principal_id='host',
        executor_id=f'executor-{index}', boot_id='boot', boot_contract_hash='a'*64, budget_config=config)
        for index, active in enumerate(stores)]
    loops = [StatefulAgentLoop(llm=model, store=active, task_runtime=runtime,
        config=AgentLoopConfig(compactor=None, delegation_policy='single_level', max_tokens=20))
        for active, runtime in zip(stores, runtimes, strict=True)]
    sessions = [await loop.new_session() for loop in loops]
    for runtime in runtimes:
        await runtime.ensure_started()
    read = budget_policy_sources._read
    seen, ready = [], asyncio.Event()

    async def both_absent(runtime):
        value = await read(runtime)
        if value is None and not ready.is_set():
            seen.append(runtime.executor_id)
            if len(seen) == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 5)
        return value

    try:
        with monkeypatch.context() as patch:
            patch.setattr(budget_policy_sources, '_read', both_absent)
            results = await asyncio.wait_for(asyncio.gather(*(loop.send('run once', sid)
                for loop, sid in zip(loops, sessions, strict=True))), 15)
        assert sorted(seen) == ['executor-0', 'executor-1']
        assert all(result.status == 'completed' and result.final_text == 'done' for result in results)
        assert model.calls == 2 and host.reserve_calls == 2 and host.apply_calls == 2
        manager = runtimes[0].manager
        assert (await manager.db.fetchone(f'SELECT COUNT(*) AS n FROM {manager.prefix}task_budget_policies'))['n'] == 1
        assert (await manager.db.fetchone(f"SELECT COUNT(*) AS n FROM {manager.events} WHERE event_type='model_budget_policy'"))['n'] == 0
        assert (await manager.db.fetchone(f"SELECT COUNT(*) AS n FROM {manager.jobs} WHERE kind='send'"))['n'] == 2
        for active, sid in zip(stores, sessions, strict=True):
            assert [row.role for row in await active.load_all_messages(sid)] == ['user', 'assistant']
    finally:
        for runtime, loop in zip(runtimes, loops, strict=True):
            await runtime.shutdown(timeout_s=1)
            await loop.aclose(drain_timeout_s=.1)
