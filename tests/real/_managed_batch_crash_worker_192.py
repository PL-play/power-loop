"""Real ordered two-call response; die before callbacks or after the first exit."""

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
from power_loop.runtime.pricing import Free
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.real.test_real_async_boundaries_192 import BoundedProvider
from tests.real.test_real_budget_tree_192 import RealBudgetHost


async def main():
    database, target, allowed, phase = sys.argv[1:]
    assert allowed in ('1', '2') and phase in ('before_callbacks', 'after_first_exit')
    store = await SessionStore.open(database, schema='tasks')
    provider, host, manager = BoundedProvider(limit=1), RealBudgetHost(), JobManager(store)
    host.service = provider
    runtime = TaskRuntime(manager, tenant_id='batch-crash-proof', principal_id='host', executor_id='original',
        boot_id='crashed-boot', boot_contract_hash='d' * 64,
        budget_config=BudgetConfig('batch-crash-v1', (LocalBudgetScope('authority', max_calls=int(allowed)),), host))
    effects, preparations = [], []
    names = ['set_first_beacon', 'set_second_beacon']

    async def prepare(context, arguments):
        assert arguments == {'value': 42}
        preparations.append(context.tool_call_id)
        return ManagedToolPreparation({'original': context.tool_call_id})

    async def execute(context, reference):
        effects.append(context.tool_call_id)
        return ManagedToolResult('Original beacon set', {'receipt_id': 'original-42'})

    async def query(context, reference):
        raise AssertionError('The crash fixture must never query or replay an effect')

    implementation = FunctionManagedTool(ManagedToolContract('example.batch-crash', 'a' * 64), prepare, execute, query)
    definitions = [ToolDefinition(name, 'Set this beacon once with value42.', input_schema={
        'type': 'object', 'properties': {'value': {'type': 'integer', 'enum': [42]}}, 'required': ['value']}, pricing=Free("test_fixture")) for name in names]
    registry = ToolRegistry()
    for definition in definitions:
        registry.register(definition, implementation.bind(runtime, definition))
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(max_tokens=2048, max_rounds=2, delegation_policy='single_level',
                              retry_policy=LLMRetryPolicy(max_attempts=1)))
    sid = await loop.new_session()

    async def dump(tx):
        source = _SEND.get().lease.job_id
        [plan] = await tx.fetchall(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='tool_call_plan'", (source,))
        entries = json.loads(plan['payload_json'])['entries']
        assert [entry['tool_name'] for entry in entries] == names
        jobs = []
        for entry in entries:
            row = await manager._row(tx, entry['job_id'])
            jobs.append({'job_id': row['job_id'], 'generation': row['generation'], 'call_id': entry['call_id']})
        assert [row['generation'] for row in jobs] == ([0, 0] if phase == 'before_callbacks' else [1, 0])
        observations = {}
        provider.report(lambda key, value: observations.__setitem__(key, value))
        observations.update(actor=asdict(runtime.actor(sid)), jobs=jobs, source_job_id=source,
            effects=effects, preparations=preparations, phase=phase, exit_code=75,
            budget_reservations=host.reserve_calls, budget_usage_observations=len(host.applied))
        with Path(target).open('x') as file:
            json.dump(observations, file)
            file.flush()
            os.fsync(file.fileno())
        os._exit(75)

    original = manager._record_result_tx
    async def crash_after_first_exit(tx, actor, lease, **kwargs):
        row = await manager._row(tx, lease.job_id)
        if row['kind'] == 'background_tool':
            assert phase == 'after_first_exit' and len(effects) == len(preparations) == 1
            await validate_exit(manager, tx, row, await descriptor(manager, tx, row))
            await dump(tx)
        return await original(tx, actor, lease, **kwargs)
    manager._record_result_tx = crash_after_first_exit

    async def schema(ctx):
        ctx.tools = [definition.to_openai_tool() for definition in definitions]
    async def batch(ctx):
        assert [call['function']['name'] for call in ctx.tool_calls] == names
        if phase == 'before_callbacks':
            async with manager._write() as tx:
                await dump(tx)
    loop.hooks.register(HookPoint.LLM_BEFORE, schema, name='test.original-two-call-schema')
    loop.hooks.register(HookPoint.TOOLS_BATCH_BEFORE, batch, name='test.original-batch-crash')
    await loop.send('In your VERY FIRST response, issue exactly TWO tool calls in this order: '
                    'set_first_beacon(value=42), set_second_beacon(value=42). Put BOTH calls in the SAME response. '
                    'Do not wait between calls, do not write prose, do not call anything else or repeat a call.', sid)
    raise AssertionError('Did not reach the required original-process crash point')


asyncio.run(main())
