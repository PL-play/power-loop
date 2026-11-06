"""Full P3 flows, including acceptance, parent handling, and owner release."""
import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from power_loop import AgentLoopConfig, ToolDefinition
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import JobRef
from power_loop.runtime.pricing import Free
from power_loop.tools import create_default_tool_registry
from power_loop.tools.spawn_agent import register_spawn_agent
from tests.runtime_fixture import RunningAgent
from tests.write_meter import WriteMeter

from .test_background_runtime_203 import BackgroundParent
from .test_interactions_runtime_203 import AskingModel
from .test_job_registry_192 import registry_store
from .test_send_row_profile import _snapshot
from .test_send_sql_profile import Profiled, _counts
from .test_subagent_runtime_203 import ParentAndChild

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('flow', ['subagent', 'background', 'answer'])
async def test_p3_complete_flow_cost(registry_store, tmp_path, flow):
    store, _ = registry_store
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    gate = asyncio.Event()
    if flow == 'subagent':
        model = ParentAndChild()
        model.child_gate = gate
        register_spawn_agent(registry)
    elif flow == 'background':
        model = BackgroundParent()
        async def slow():
            await gate.wait()
            return 'background result'
        registry.register(ToolDefinition('slow', 'read', async_capable=True, pricing=Free('local')), slow)
    else:
        model = AskingModel()
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
        if flow == 'answer':
            receipt = await runner.runtime.client.send(runner.target, 'ask', message_id='ask', actor=runner.actor)
            while receipt.input_ref not in runner.executions:
                await asyncio.sleep(.01)
            await runner.finish(receipt.input_ref)
            status = await runner.runtime.client.inspect(receipt.input_ref, actor=runner.actor)
            state = await store.get_state(runner.target.session_id)
            pending = state.pending
            await runner.runtime.client.answer(status.job_ref,
                interaction_id=pending['pending_interactions'][0]['interaction_id'], wait_round=pending['round_index'],
                value='blue', message_id='answer', actor=runner.actor)
            result = await runner.runtime.client.wait(receipt.input_ref, actor=runner.actor, deadline_ms=now_ms()+10000)
        else:
            await runner.send('start')
            rows = await store.load_all_messages(runner.target.session_id)
            receipt = json.loads(next(m.content for m in rows if m.tool_call_id == ('spawn' if flow == 'subagent' else 'background')))
            gate.set()
            result = await runner.runtime.client.wait(JobRef(**receipt['job_ref']), actor=runner.actor, deadline_ms=now_ms()+10000)
        assert result.status == 'completed'
        await runner.drain()
        elapsed = time.monotonic()-started
        sql = {**_counts(profiled.records, profiled.tx_sites), 'total_sql':len(profiled.records), 'transactions':len(profiled.tx_sites)}
        writes = await meter.report() if meter else None
        after = await _snapshot(original, store.table_prefix)
        delta = {t:[after[t][0]-before[t][0],after[t][1]-before[t][1]] for t in after if after[t] != before[t]}
        assert after.get('execution_batches', (0,0))[0] == 0
        for table in ('execution_checkpoints', 'execution_allocations', 'task_budget_attempts'):
            assert after[table][0] == 0
        if output := os.environ.get('P3_FLOW_OUT'):
            with Path(output).open('a') as file:
                file.write(json.dumps({'db':original.dialect.name,'flow':flow,'sql':sql,'tables':delta,
                    'writes':writes,'elapsed_s':elapsed,'B':None,'comparison':'new P3 flow; no equivalent fixed-B case'})+'\n')
    finally:
        gate.set()
        await runner.aclose()
        store._db = original
        if meter:
            await meter.close()
