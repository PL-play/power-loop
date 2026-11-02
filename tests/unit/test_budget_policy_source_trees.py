"""A cold budget admission borrows its policy original outside its tree writer."""
from __future__ import annotations

import asyncio

import pytest

from power_loop import AgentLoopConfig, LLMResponse, LLMService, LLMTokenUsage, StatefulAgentLoop
from power_loop.runtime import budget_policy_sources, model_attempts, model_budgets
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.response_capture import ResponseCaptureReceipt
from power_loop.runtime.session_controls import ControlReadView
from power_loop.runtime.task_runtime import TaskRuntime

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import Host, budget_case

__all__ = ['budget_case', 'registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('port', ['open-control', 'ensure-policy'])
async def test_cold_budget_admission_reads_another_roots_original_without_a_writer(budget_case, monkeypatch, port):
    c = budget_case
    first_root = c.binding.lease.job_id
    marker_id = 'mb_policy_'+model_budgets.policy_key(c.runtime)[3:]
    marker = await c.manager.db.fetchone(f'SELECT * FROM {c.manager.events} WHERE event_id=?', (marker_id,))
    assert marker['job_id'] == first_root
    cold_store = await c.connect()
    manager = JobManager(cold_store)
    runtime = TaskRuntime(manager, tenant_id=c.runtime.tenant_id, principal_id=c.runtime.principal_id,
        executor_id=c.runtime.executor_id, boot_id=c.runtime.boot_id, lease_ms=300000,
        boot_contract_hash='a'*64, budget_config=c.config)
    sid = await cold_store.create_session()
    actor = runtime.actor(sid)
    registration = await manager.register_job(actor, kind='send', session_id=sid, system_key='cold-second-root',
        request={'model_attempt_protocol':1})
    binding = await runtime.claim(actor, registration)
    assert registration.job.root_job_id != first_root
    assert not manager.__dict__.get('_budget_markers')
    view_one, direct_one = ControlReadView.fetchone, manager.db.fetchone
    observed = []

    def note(params):
        if marker_id in params:
            outside = manager._control_intents.get() is None
            observed.append(outside)
            assert outside, 'foreign policy original was read while the second root writer was held'

    async def view_read(view, sql, params=()):
        note(params)
        return await view_one(view, sql, params)

    async def direct_read(sql, params=()):
        note(params)
        return await direct_one(sql, params)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(ControlReadView, 'fetchone', view_read)
            patch.setattr(manager.db, 'fetchone', direct_read)
            if port == 'open-control':
                await model_attempts.enable(binding)
            else:
                assert await model_budgets.ensure_policy(binding)
        assert observed and all(observed)
        assert await manager.db.fetchone(f'SELECT * FROM {manager.events} WHERE event_id=?', (marker_id,)) == marker
        assert c.host.reserve_calls == 0 and c.host.apply_calls == 0
    finally:
        await runtime.shutdown(timeout_s=1)


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
        assert (await manager.db.fetchone(f"SELECT COUNT(*) AS n FROM {manager.events} WHERE event_type='model_budget_policy'"))['n'] == 1
        assert (await manager.db.fetchone(f"SELECT COUNT(*) AS n FROM {manager.jobs} WHERE kind='send'"))['n'] == 2
        for active, sid in zip(stores, sessions, strict=True):
            assert [row.role for row in await active.load_all_messages(sid)] == ['user', 'assistant']
    finally:
        for runtime, loop in zip(runtimes, loops, strict=True):
            await runtime.shutdown(timeout_s=1)
            await loop.aclose(drain_timeout_s=.1)
