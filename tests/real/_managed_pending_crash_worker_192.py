"""Real model; exit the process after durable callback exit, before result write."""
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.managed_completion import validate_exit
from power_loop.runtime.managed_effects import (
    FunctionManagedTool,
    ManagedToolContract,
    ManagedToolPreparation,
    ManagedToolResult,
    descriptor,
)
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.real.test_real_async_boundaries_192 import BoundedProvider
from tests.real.test_real_budget_tree_192 import RealBudgetHost


async def main():
    store = await SessionStore.open(sys.argv[1], schema='tasks')
    provider, host = BoundedProvider(limit=1), RealBudgetHost()
    host.service = provider
    manager = JobManager(store)
    runtime = TaskRuntime(manager, tenant_id='crash-proof', principal_id='host', executor_id='original',
        boot_id='crashed-boot', boot_contract_hash='d' * 64,
        budget_config=BudgetConfig('crash-v1', (LocalBudgetScope('authority', max_calls=int(sys.argv[3])),), host))
    assert sys.argv[3] in ('1', '2')
    effects = []

    async def prepare(context, arguments):
        assert arguments == {'value': 42}
        return ManagedToolPreparation({'original': context.tool_call_id})

    async def execute(context, reference):
        effects.append(reference)
        return ManagedToolResult('Original beacon set', {'receipt_id': 'original-42'})

    async def query(context, reference):
        raise AssertionError('Crash fixture must not query or repeat the effect')

    implementation = FunctionManagedTool(ManagedToolContract('example.crash', 'a' * 64), prepare, execute, query)
    definition = ToolDefinition('set_beacon', 'Set the beacon once with value 42.', input_schema={
        'type': 'object', 'properties': {'value': {'type': 'integer', 'enum': [42]}}, 'required': ['value']})
    registry = ToolRegistry()
    registry.register(definition, implementation.bind(runtime, definition))
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(max_tokens=2048, max_rounds=2, delegation_policy='single_level',
                              retry_policy=LLMRetryPolicy(max_attempts=1)))
    sid = await loop.new_session()
    original = manager._record_result_tx

    async def crash_before_result(tx, actor, lease, **kwargs):
        row = await manager._row(tx, lease.job_id)
        if row['kind'] == 'background_tool':
            await validate_exit(manager, tx, row, await descriptor(manager, tx, row))
            observations = {}
            provider.report(lambda k, v: observations.__setitem__(k, v))
            observations.update(actor=asdict(runtime.actor(sid)), job_id=lease.job_id,
                effect_executions=len(effects), budget_reservations=host.reserve_calls,
                budget_usage_observations=len(host.applied), exit_code=75)
            with Path(sys.argv[2]).open('x') as file:
                json.dump(observations, file)
                file.flush()
                os.fsync(file.fileno())
            os._exit(75)
        return await original(tx, actor, lease, **kwargs)

    manager._record_result_tx = crash_before_result

    async def schema(ctx):
        ctx.tools = [definition.to_openai_tool()]

    loop.hooks.register(HookPoint.LLM_BEFORE, schema, name='test.crash-original-call')
    await loop.send('Call set_beacon once with value=42. Never call another tool or repeat the call.', sid)
    raise AssertionError('Fixture did not reach its required crash point')


asyncio.run(main())
