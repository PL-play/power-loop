"""An actual process dies with two original calls; original budgets govern review."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.execution_membership import scope
from power_loop.runtime.jobs import JobActor, JobError, JobLease, JobManager
from power_loop.runtime.managed_completion import restore_result
from power_loop.runtime.pending_calls import retire_pending
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.runtime.tool_call_plans import reconcile_pending
from power_loop.tools.registry import ToolRegistry

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_budget_tree_192 import RealBudgetHost

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('phase', ['before_callbacks', 'after_first_exit'])
@pytest.mark.parametrize('allowed_calls', [2, 1])
async def test_real_original_two_call_process_exit_is_recovered_without_replay(tmp_path, record_property, phase, allowed_calls):
    database, target = tmp_path / 'batch.db', tmp_path / 'observations.json'
    child = store = provider = runtime = loop = None
    observations = {}
    try:
        child = await asyncio.create_subprocess_exec(sys.executable,
            str(Path(__file__).with_name('_managed_batch_crash_worker_192.py')),
            str(database), str(target), str(allowed_calls), phase,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        record_property('original_batch_worker_pid', child.pid)
        stdout, stderr = await asyncio.wait_for(child.communicate(), 150)
        assert child.returncode == 75, stderr.decode()[-2000:]
        observations = json.loads(target.read_text())
        expected_effects = 1 if phase == 'after_first_exit' else 0
        assert len(observations['effects']) == len(observations['preparations']) == expected_effects
        assert observations['real_provider_requests'] == observations['budget_reservations'] == 1
        store = await SessionStore.open(str(database), schema='compatible')
        manager, actor = JobManager(store), JobActor(**observations['actor'])
        raw = await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq')
        assert not any(row['role'] == 'tool' for row in raw)
        for job in observations['jobs']:
            if job['generation'] == 1:
                await restore_result(manager, actor, job['job_id'])
        provider, host = BoundedProvider(limit=1), RealBudgetHost()
        host.service = provider
        runtime = TaskRuntime(manager, tenant_id=actor.tenant_id, principal_id=actor.principal_id,
            executor_id='original-review', boot_id='review-boot', boot_contract_hash='b' * 64,
            budget_config=BudgetConfig('batch-crash-v1', (LocalBudgetScope('authority', max_calls=allowed_calls),), host))
        loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=ToolRegistry(),
            config=AgentLoopConfig(max_tokens=2048, max_rounds=2, delegation_policy='single_level',
                representation=ProjectedRepresentation(), retry_policy=LLMRetryPolicy(max_attempts=1)))
        pending = (await store.get_state(actor.session_id)).pending
        assert len(pending['tool_call_ids']) == 2
        source = await manager.get_job(actor, observations['source_job_id'])
        with pytest.raises(JobError):
            await reconcile_pending(loop, actor, source.job_id)
        assert (await store.get_state(actor.session_id)).pending == pending and provider.calls == 0
        # This parent owns and has wait/reaped the exact original worker.
        async with manager._write() as tx:
            original_scope = await scope(manager, tx, source.job_id, source.generation)
        assert original_scope['lease']['executor_id'] == 'original' and original_scope['lease']['boot_id'] == 'crashed-boot'
        await manager.close_model_execution(actor, JobLease(**original_scope['lease']),
            evidence={'exit': 'interrupted', 'physical_process_exit': {'pid': child.pid, 'returncode': child.returncode}})
        staged = await reconcile_pending(loop, actor, source.job_id)
        assert len(staged['notices']) == 2 and not staged['execution_authority'] and not staged['acknowledgement_created']
        assert await reconcile_pending(loop, actor, source.job_id) == staged
        disposed = await retire_pending(loop, actor, source.job_id)
        assert not disposed['acknowledgement_created'] and not disposed['billing_settled']
        assert (await store.get_state(actor.session_id)).pending is None
        with pytest.raises(JobError):
            await manager.finalize_supervision(actor, source.job_id, generation=1)

        async def honest_request(ctx):
            text = '\n'.join(str(message.get('content', '')) for message in ctx.messages)
            assert text.count('Original tool outcome:') == 2
            assert all(job['job_id'] in text for job in observations['jobs'])
            assert not any(message.get('role') == 'tool' or message.get('tool_calls') for message in ctx.messages)
            ctx.tools = []
        loop.hooks.register(HookPoint.LLM_BEFORE, honest_request, name='test.real-batch-honest-review')
        result = await loop.send('Read both original outcomes supplied here. Do not repeat any action or call tools. '
                                 'Reply exactly ACKNOWLEDGED.', actor.session_id)
        if allowed_calls == 1:
            assert result.status == 'degraded' and 'budget_max_calls_exhausted' in result.final_text
            assert provider.calls == host.reserve_calls == len(host.applied) == 0
        else:
            assert result.status == 'completed' and result.final_text.strip() == 'ACKNOWLEDGED'
            assert provider.calls == host.reserve_calls == len(host.applied) == 1
        for original in observations['jobs']:
            current = await manager.get_job(actor, original['job_id'])
            decisions = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='processing_decision'", (current.job_id,))
            assert len(decisions) == (1 if allowed_calls == 2 else 0)
            if original['generation'] == 0:
                assert current.generation == 0 and current.result_status == 'suppressed' and current.supervision_status == 'closed'
            else:
                assert current.result_status == ('handled' if allowed_calls == 2 else 'pending_delivery')
            if decisions:
                decision = json.loads(decisions[0]['payload_json'])
                assert decision['response_ref']['session_id'] == actor.session_id
                assert decision['event']['managed_delivery']['pending_disposition'] == disposed['disposition']
        assert ((await manager.get_job(actor, source.job_id)).supervision_status == 'closed') == (allowed_calls == 2)
        for message in raw:
            assert await manager.db.fetchone(f'SELECT * FROM {store.t.messages} WHERE id=?', (message['id'],)) == message
        assert not await manager.db.fetchall(f"SELECT id FROM {store.t.messages} WHERE role='tool'")
        record_property('original_effect_executions', expected_effects)
    finally:
        usage = {}
        if provider is not None:
            provider.report(lambda key, value: usage.__setitem__(key, value))
        for key in ['real_provider_requests', 'reported_tokens', 'local_provider_calls_refused']:
            record_property(key, int(observations.get(key, 0)) + int(usage.get(key, 0)))
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        if runtime is not None:
            await runtime.shutdown(timeout_s=1, grace_s=.01)
        if loop is not None:
            await loop.aclose(drain_timeout_s=.1)
        if provider is not None:
            await provider.close()
        if store is not None:
            await store.close()
