"""A real main model reads the restored outcome; the original action is not replayed."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobActor, JobError, JobLease, JobManager
from power_loop.runtime.managed_completion import restore_result
from power_loop.runtime.managed_delivery import enqueue
from power_loop.runtime.pending_calls import retire_pending
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_budget_tree_192 import RealBudgetHost

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.parametrize('allowed_calls', [2, 1])
async def test_real_crashed_original_is_handled_by_original_main_without_repeating_tool(tmp_path, record_property, allowed_calls):
    database, observations_path = tmp_path / 'original.db', tmp_path / 'original-observations.json'
    child = provider = loop = runtime = store = None
    original_usage = {}
    try:
        child = await asyncio.create_subprocess_exec(sys.executable,
            str(Path(__file__).with_name('_managed_pending_crash_worker_192.py')),
            str(database), str(observations_path), str(allowed_calls), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        record_property('original_delivery_worker_pid', child.pid)
        stdout, stderr = await asyncio.wait_for(child.communicate(), 150)
        assert child.returncode == 75, stderr.decode()[-2000:]
        original_usage = json.loads(observations_path.read_text())
        assert original_usage['effect_executions'] == original_usage['real_provider_requests'] == 1
        actor, job = JobActor(**original_usage['actor']), original_usage['job_id']
        store = await SessionStore.open(str(database), schema='compatible')
        manager = JobManager(store)
        original = await manager.db.fetchall(f'SELECT * FROM {store.t.messages} ORDER BY seq')
        assert not any(row['role'] == 'tool' for row in original)
        await restore_result(manager, actor, job)
        provider, host = BoundedProvider(limit=1), RealBudgetHost()
        host.service = provider
        runtime = TaskRuntime(manager, tenant_id=actor.tenant_id, principal_id=actor.principal_id,
            executor_id='original-main-reviewer', boot_id='review-boot', boot_contract_hash='b' * 64,
            budget_config=BudgetConfig('crash-v1', (LocalBudgetScope('authority', max_calls=allowed_calls),), host))
        loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=ToolRegistry(),
            config=AgentLoopConfig(max_tokens=2048, max_rounds=2, delegation_policy='single_level',
                representation=ProjectedRepresentation(), retry_policy=LLMRetryPolicy(max_attempts=1)))

        async def schema(ctx):
            assert any('Original tool outcome:' in str(m.get('content')) for m in ctx.messages)
            # Honest projection exposes the original tool's missing raw reply;
            # it does not insert a fabricated role=tool message into history.
            assert not any(m.get('tool_calls') for m in ctx.messages)
            ctx.tools = []

        loop.hooks.register(HookPoint.LLM_BEFORE, schema, name='test.real-original-outcome-review')
        first = await enqueue(loop, actor, job)
        assert await enqueue(loop, actor, job) == first
        assert provider.calls == 0 and not first['acknowledgement_created']
        assert (await manager.get_job(actor, job)).result_status == 'pending_delivery'
        # The parent owns and has reaped this exact worker process. Canonical
        # callback completion alone must not clear the crashed send's pending.
        original_pending = (await store.get_state(actor.session_id)).pending
        source = await manager.get_job(actor, (await manager.get_job(actor, job)).parent_job_id)
        with pytest.raises(JobError):
            await retire_pending(loop, actor, source.job_id)
        assert (await store.get_state(actor.session_id)).pending == original_pending
        assert child.returncode == 75
        from power_loop.runtime.execution_membership import scope
        async with manager._write() as tx:
            original_scope = await scope(manager, tx, source.job_id, source.generation)
        assert original_scope['lease']['executor_id'] == 'original'
        assert original_scope['lease']['boot_id'] == 'crashed-boot'
        await manager.close_model_execution(actor,
            JobLease(**original_scope['lease']),
            evidence={'exit': 'interrupted', 'physical_process_exit': {'pid': child.pid, 'returncode': child.returncode}})
        disposed = await retire_pending(loop, actor, source.job_id)
        assert await retire_pending(loop, actor, source.job_id) == disposed
        assert not disposed['acknowledgement_created'] and not disposed['billing_settled']
        assert (await store.get_state(actor.session_id)).pending is None
        assert (await manager.get_job(actor, job)).result_status == 'pending_delivery'
        result = await loop.send('Review the original tool outcome already supplied in this session. '
            'Do not repeat its action or call tools. Reply exactly ACKNOWLEDGED.', actor.session_id)
        if allowed_calls == 1:
            assert result.status == 'degraded' and 'budget_max_calls_exhausted' in result.final_text
            assert provider.calls == host.reserve_calls == len(host.applied) == 0
            current = await manager.get_job(actor, job)
            assert current.result_status == 'pending_delivery' and current.supervision_status != 'closed'
            assert not await manager.db.fetchall(f"SELECT event_id FROM {manager.events} WHERE job_id=? AND event_type='processing_decision'", (job,))
            for row in original:
                assert await manager.db.fetchone(f'SELECT * FROM {store.t.messages} WHERE id=?', (row['id'],)) == row
            assert not await manager.db.fetchall(f"SELECT id FROM {store.t.messages} WHERE role='tool'")
            return
        assert result.status == 'completed' and result.final_text.strip() == 'ACKNOWLEDGED'
        assert provider.calls == host.reserve_calls == len(host.applied) == 1
        current = await manager.get_job(actor, job)
        assert current.result_status == 'handled' and current.supervision_status == 'closed'
        assert original_usage['effect_executions'] == 1
        for row in original:
            assert await manager.db.fetchone(f'SELECT * FROM {store.t.messages} WHERE id=?', (row['id'],)) == row
        assert not await manager.db.fetchall(f"SELECT id FROM {store.t.messages} WHERE role='tool'")
        [decision] = await manager.db.fetchall(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='processing_decision'", (job,))
        handling = json.loads(decision['payload_json'])
        assert handling['response_ref']['session_id'] == actor.session_id
        assert handling['event']['managed_delivery']['event_id'] == first['notification_id']
        assert handling['event']['managed_delivery']['pending_disposition'] == disposed['disposition']
        assert await retire_pending(loop, actor, source.job_id) == disposed
        record_property('review_budget_reservations', host.reserve_calls)
    finally:
        usage = {}
        if provider is not None:
            provider.report(lambda k, v: usage.__setitem__(k, v))
        for key in ['real_provider_requests', 'reported_tokens', 'local_provider_calls_refused']:
            record_property(key, int(original_usage.get(key, 0)) + int(usage.get(key, 0)))
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        if runtime is not None:
            await runtime.shutdown(timeout_s=1, grace_s=0.01)
        if loop is not None:
            await loop.aclose(drain_timeout_s=0.1)
        if provider is not None:
            await provider.close()
        if store is not None:
            await store.close()
