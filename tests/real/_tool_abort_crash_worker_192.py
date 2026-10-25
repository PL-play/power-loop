"""Actual original process dies after its positive pre-admission checkpoint."""
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from power_loop import AgentLoopConfig, HookPoint, SessionStore, StatefulAgentLoop, ToolDefinition
from power_loop.runtime.budget_contracts import BudgetConfig, LocalBudgetScope
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.managed_effects import (
    FunctionManagedTool,
    ManagedToolContract,
    ManagedToolPreparation,
    ManagedToolResult,
)
from power_loop.runtime.retry import LLMRetryPolicy
from power_loop.runtime.task_runtime import _SEND, TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.real.test_real_async_boundaries_192 import BoundedProvider
from tests.real.test_real_budget_tree_192 import RealBudgetHost


async def main():
    database, target, allowed, phase = sys.argv[1:]
    assert allowed in ('1', '2') and phase == 'after_abort'
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
        'type': 'object', 'properties': {'value': {'type': 'integer', 'enum': [42]}}, 'required': ['value']}) for name in names]
    registry = ToolRegistry()
    for definition in definitions:
        registry.register(definition, implementation.bind(runtime, definition))
    loop = StatefulAgentLoop(llm=provider, store=store, task_runtime=runtime, tool_registry=registry,
        config=AgentLoopConfig(max_tokens=2048, max_rounds=2, delegation_policy='single_level',
                              retry_policy=LLMRetryPolicy(max_attempts=1)))
    sid = await loop.new_session()

    async def dump(tx):
        source = _SEND.get().lease.job_id
        [saved] = await tx.fetchall(f"SELECT payload_json FROM {manager.events} WHERE job_id=? AND event_type='tool_call_abort'", (source,))
        checkpoint = json.loads(saved['payload_json'])
        assert [entry['tool_name'] for entry in checkpoint['entries']] == names
        assert checkpoint['plan'] is None and not effects and not preparations
        jobs = []
        observations = {}
        provider.report(lambda key, value: observations.__setitem__(key, value))
        observations.update(actor=asdict(runtime.actor(sid)), jobs=jobs, source_job_id=source,
            effects=effects, preparations=preparations, phase=phase, exit_code=75, checkpoint=checkpoint,
            budget_reservations=host.reserve_calls, budget_usage_observations=len(host.applied))
        with Path(target).open('x') as file:
            json.dump(observations, file)
            file.flush()
            os.fsync(file.fileno())
        os._exit(75)

    from power_loop.runtime import tool_call_abort
    from tests.unit._tool_plan_faults_192 import fail_plan_once
    fail_plan_once(manager, 'seal')
    original = tool_call_abort._record
    async def crash_after_checkpoint(attempt):
        await original(attempt)
        async with manager._write() as tx:
            await dump(tx)
        raise AssertionError('Must exit the actual original process')
    tool_call_abort._record = crash_after_checkpoint

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
