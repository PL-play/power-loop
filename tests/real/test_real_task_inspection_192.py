"""A real main model reads an alert and inspects the original unknown task."""
import asyncio

import pytest

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop.core.agent_context import get_session_id
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.task_inspection import CONSTRAINTS, inspect_tasks
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.jobs import register_jobs_tools
from power_loop.tools.registry import ToolRegistry
from tests.unit.test_job_registry_192 import registry_store, runtime
from tests.unit.test_task_inspection_192 import POLICY, troubled

from .test_real_async_boundaries_192 import BoundedProvider

__all__ = ['registry_store', 'runtime']
pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


async def test_real_main_reads_original_alert_and_queries_without_replaying_unknown_task(runtime, record_property):
    store, manager, actor, root, lease, connect, now = runtime
    target = await troubled(runtime)
    # A fresh connection reconstructs the alert from persisted task state.
    reader = JobManager(await connect(), now_ms=manager.now)
    [key] = (await inspect_tasks(reader, actor, root.job.job_id, policy=POLICY))['notification_ids']
    before = await manager.get_job(actor, target.job_id)
    provider = BoundedProvider(limit=5)
    registry = ToolRegistry()
    register_jobs_tools(registry)
    execution = TaskRuntime(manager, tenant_id=actor.tenant_id, principal_id=actor.principal_id,
        executor_id='alert-main', boot_id='original-budget', lease_ms=300000, boot_contract_hash='e'*64)
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=execution, tool_registry=registry,
        config=AgentLoopConfig(max_rounds=4, max_tokens=2048, delegation_policy='single_level', compactor=None, system_prompt=(
            'You supervise original tasks. When a task alert arrives, call the offered jobs function '
            'with action=get and job_id equal to the affected job_id, exactly once using native function calling. '
            'Then explain its unknown state briefly. Never start, stop, or replace work.')))
    from power_loop.runtime.retry import LLMRetryPolicy
    loop.config.retry_policy = LLMRetryPolicy(max_attempts=1)
    seen = []
    from power_loop import HookPoint

    async def actual_request(ctx):
        assert get_session_id() == actor.session_id
        seen.extend(m.get('content', '') for m in ctx.messages if isinstance(m.get('content'), str))
        if any(m.get('role') == 'tool' and target.job_id in m.get('content', '') for m in ctx.messages):
            # Two actual phases: inspect once, then summarize without more tools.
            ctx.tools = []
            ctx.system_prompt += '\nThe original task inspection has returned. Give a short final text response now.'

    loop.hooks.register(HookPoint.LLM_BEFORE, actual_request, name='test.actual-alert-request')
    try:
        async with asyncio.timeout(150):
            result = await loop.send('Handle the queued task alert by inspecting its original task.', actor.session_id, tools=['jobs'])
            assert result.status == 'completed' and result.final_text
            assert provider.calls >= 2
            rows = await store._db.fetchall(f"SELECT content,tool_call_id FROM {store.t.messages} WHERE session_id=? AND role='tool'", (actor.session_id,))
            assert len(rows) == 1 and target.job_id in rows[0]['content']
            assert any(CONSTRAINTS in text and target.job_id in text for text in seen)
            assert await manager.get_job(actor, target.job_id) == before
            event = await store._db.fetchone(f'SELECT event_type FROM {manager.events} WHERE event_id=?', (key,))
            assert event['event_type'] == 'task_alert'
    finally:
        provider.report(record_property)
        await execution.shutdown(timeout_s=3)
        await loop.aclose(drain_timeout_s=.1)
        await provider.close()
