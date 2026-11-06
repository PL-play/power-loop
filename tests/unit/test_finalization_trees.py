"""Supervision closes against actual work, handled results and unsettled fees."""

import pytest

from power_loop.runtime.jobs import JobError

from .test_job_registry_192 import registry_store
from .test_model_budgets_192 import budget_case, observed, start

__all__ = ["registry_store", "budget_case"]
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_unacknowledged_fee_blocks_tree_close(budget_case):
    c = budget_case
    c.host.apply_lost_ack = True
    attempt = await start(c)
    await observed(c, attempt)
    await c.manager.close_model_execution(c.actor, c.binding.lease, evidence={"exit": "completed"})
    with pytest.raises(JobError, match="budget_reconciliation_required"):
        await c.manager.finalize_supervision(c.actor, c.binding.lease.job_id, generation=1)
    row = await c.manager.get_job(c.actor, c.binding.lease.job_id)
    assert row.supervision_status == "open"
    pending = await c.manager.db.fetchall(
        f"SELECT pending FROM {c.manager.prefix}task_budget_attempts WHERE job_id=?", (row.job_id,)
    )
    assert pending and pending[0]["pending"]


async def test_synchronous_managed_effect_uses_caller_session(registry_store):
    from power_loop import AgentLoopConfig, LLMResponse, LLMService, ToolDefinition
    from power_loop.runtime.jobs import JobManager
    from power_loop.runtime.managed_effects import (
        FunctionManagedTool,
        ManagedToolContract,
        ManagedToolPreparation,
        ManagedToolResult,
    )
    from power_loop.runtime.pricing import Free
    from power_loop.runtime.task_runtime import TaskRuntime
    from power_loop.tools.registry import ToolRegistry
    from tests.runtime_fixture import RunningAgent
    store, _ = registry_store
    runtime = TaskRuntime(JobManager(store), tenant_id='test', principal_id='host', executor_id='managed', boot_id='boot')
    executed = []
    async def prepare(context, arguments):
        return ManagedToolPreparation({'call':context.tool_call_id})
    async def execute(context, reference):
        executed.append(reference['call'])
        return ManagedToolResult('effect done', {'receipt':'saved'})
    async def recover(context, reference):
        return None
    definition = ToolDefinition('effect', 'finite synchronous effect', input_schema={'type':'object'}, pricing=Free('local'))
    adapter = FunctionManagedTool(ManagedToolContract('p4.effect', 'a'*64),prepare,execute,recover)
    registry = ToolRegistry()
    registry.register(definition, adapter.bind(runtime,definition))
    class Model(LLMService):
        async def complete(self, request, **kwargs):
            if any(m.get('role')=='tool' for m in request.messages):
                return LLMResponse(raw_text='done')
            return LLMResponse(raw_text='',tool_calls=[{'id':'effect','type':'function','function':{'name':'effect','arguments':'{}'}}])
        async def close(self):
            pass
    runner = await RunningAgent(store,Model(),registry,AgentLoopConfig(compactor=None),task_runtime=runtime).start()
    try:
        result = await runner.send('do it')
        assert result.status == 'completed' and executed == ['effect'], (result, [m.content for m in await store.load_all_messages(runner.target.session_id)])
        jobs = await store._db.fetchall(f"SELECT session_id,supervision_status FROM {store.table_prefix}jobs WHERE kind='background_tool'")
        assert len(jobs) == 1 and jobs[0]['session_id'] == runner.target.session_id
        assert jobs[0]['supervision_status'] == 'closed'
    finally:
        await runner.aclose()
