"""A real main model reads the restored outcome; the original action is not replayed."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobActor, JobManager
from power_loop.runtime.managed_completion import restore_result
from power_loop.runtime.managed_delivery import enqueue
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry

from .test_real_async_boundaries_192 import BoundedProvider
from .test_real_budget_tree_192 import RealBudgetHost

pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_crashed_original_is_handled_by_original_main_without_repeating_tool(tmp_path, record_property):
    database, observations_path = tmp_path / 'original.db', tmp_path / 'original-observations.json'
    child = provider = loop = runtime = store = None
    original_usage = {}
    try:
        child = await asyncio.create_subprocess_exec(sys.executable,
            str(Path(__file__).with_name('_managed_completion_crash_worker_192.py')),
            str(database), str(observations_path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
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
            budget_config=BudgetConfig('review-authority-v1', (LocalBudgetScope('review', max_calls=1),), host))
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
        result = await loop.send('Review the original tool outcome already supplied in this session. '
            'Do not repeat its action or call tools. Reply exactly ACKNOWLEDGED.', actor.session_id)
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
