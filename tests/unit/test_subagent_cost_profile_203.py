"""Full child, workflow and interaction flows, including acceptance, parent handling, and owner release."""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, ToolDefinition
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import InputRef, JobRef
from power_loop.runtime.pricing import Free
from power_loop.runtime.result_revisions import result_id
from power_loop.tools import create_default_tool_registry
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent
from tests.write_meter import WriteMeter

from .test_background_runtime_203 import BackgroundParent
from .test_job_registry_192 import registry_store
from .test_send_row_profile import _snapshot
from .test_send_sql_profile import Profiled, _counts
from .test_subagent_runtime_203 import ParentAndChild

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('flow', ['subagent', 'background', 'workflow'])
async def test_complete_flow_cost(registry_store, tmp_path, flow):
    store, _ = registry_store
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    gate = asyncio.Event()
    if flow == 'workflow':
        from power_loop import LLMResponse
        from power_loop.workflow import register_workflow_tool

        from ._workflow_fixture_203 import Model, leaf
        register_workflow_tool(registry)
        class WorkflowParent(Model):
            async def complete(self, request, **kwargs):
                if request.system_prompt.startswith(('A', 'B')):
                    return LLMResponse(raw_text='leaf done')
                if not any(m.get('role') == 'tool' for m in request.messages):
                    return LLMResponse(raw_text='', tool_calls=[{'id':'workflow', 'type':'function', 'function':{
                        'name':'workflow', 'arguments':json.dumps({'action':'create','spec':{
                            'name':'cost','root':{'type':'sequence','steps':[leaf('A'),leaf('B')]}}})}}])
                return LLMResponse(raw_text='parent done')
        model = WorkflowParent()
    elif flow == 'subagent':
        model = ParentAndChild()
        model.child_gate = gate
        register_spawn_agent(registry)
    elif flow == 'background':
        model = BackgroundParent()
        async def slow():
            await gate.wait()
            return 'background result'
        registry.register(ToolDefinition('slow', 'read', async_capable=True, pricing=Free('local')), slow)
    original = store._db
    meter = await WriteMeter.install(original, store.table_prefix) if os.environ.get('P0_WRITE_METER') else None
    store._db = profiled = Profiled(original, meter)
    runner = await RunningAgent(store, model, registry, AgentLoopConfig(compactor=None)).start()
    try:
        before = await _snapshot(original, store.table_prefix)
        profiled.records.clear()
        profiled.tx_sites.clear()
        if meter:
            await meter.reset()
        started = time.monotonic()
        await runner.send('start')
        rows = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(m.content for m in rows if m.tool_call_id == ('spawn' if flow == 'subagent' else flow)))
        gate.set()
        result = await runner.runtime.client.wait(JobRef(**receipt['job_ref']), actor=runner.actor, deadline_ms=now_ms()+10000)
        assert result.status == 'completed'
        parent_input = InputRef(store.store_id, runner.target.session_id,
            hashlib.sha256(('result:' + result_id(result.job_ref.job_id, result.job_ref.generation)).encode()).hexdigest())
        await runner.drain(after_input=parent_input)
        elapsed = time.monotonic()-started
        records, tx_sites = list(profiled.records), list(profiled.tx_sites)
        sql = {**_counts(records, tx_sites), 'total_sql':len(records), 'transactions':len(tx_sites)}
        writes = await meter.report() if meter else None
        after = await _snapshot(original, store.table_prefix)
        assert sql['total_sql'] == len(records)
        assert sql['transactions'] == len(tx_sites)
        delta = {t:[after[t][0]-before[t][0],after[t][1]-before[t][1]] for t in after if after[t] != before[t]}
        assert after.get('execution_batches', (0,0))[0] == 0
        for table in ('execution_checkpoints', 'execution_allocations', 'task_budget_attempts'):
            assert after[table][0] == 0
        if output := os.environ.get('P3_FLOW_OUT'):
            with Path(output).open('a') as file:
                file.write(json.dumps({'db':original.dialect.name,'flow':flow,'sql':sql,'tables':delta,
                    'writes':writes,'elapsed_s':elapsed,'records':records,'tx_sites':tx_sites,
                    'B':None,'comparison':'no equivalent fixed-B flow case; ordinary-send B comparison is reported separately'})+'\n')
    finally:
        gate.set()
        await runner.aclose()
        store._db = original
        if meter:
            await meter.close()


async def test_drain_waits_for_parent_not_yet_scheduled(registry_store, tmp_path):
    from power_loop.agent.runtime_transport import PublishReceipt

    store, _ = registry_store
    model = ParentAndChild()
    model.child_gate = asyncio.Event()
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    register_spawn_agent(registry)
    runner = await RunningAgent(store, model, registry).start()
    publish = runner.runtime.transport.publish
    held = []

    async def delay_parent(packet):
        if (packet.target == runner.target and packet.channel in ('work', 'activation')
                and packet.input_ref and packet.input_ref.message_id != 'm1'):
            held.append(packet)
            return PublishReceipt(True)
        return await publish(packet)

    runner.runtime.transport.publish = delay_parent
    try:
        await runner.send('start')
        model.child_gate.set()
        await runner.drain()
        assert held and not runner.runtime.worker._tasks
        parent_ref = held[0].input_ref
        assert parent_ref not in runner.executions
        pending = asyncio.create_task(runner.drain(after_input=parent_ref))
        await asyncio.sleep(0)
        assert not pending.done()
        for packet in held:
            await publish(packet)
        await asyncio.wait_for(pending, 15)
        assert not runner.runtime.worker._tasks
        rows = await store.load_all_messages(runner.target.session_id)
        assert any('[subagent_result]' in (row.content or '') for row in rows)
    finally:
        model.child_gate.set()
        await runner.aclose()
