"""Canonical output survives caller/client recreation and is delivered compactly."""
import json

import pytest

from power_loop import AgentLoopConfig, RuntimeClient
from power_loop.agent.runtime_store import now_ms
from power_loop.agent.runtime_types import JobRef
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.background import RESULT_PREFIX, load_record
from power_loop.runtime.pricing import Free
from power_loop.tools import create_default_tool_registry
from tests.runtime_fixture import RunningAgent

from .test_background_runtime_203 import BackgroundParent
from .test_job_registry_192 import registry_store

__all__ = ['registry_store']
pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_full_result_identity_survives_client_recreation(registry_store, tmp_path):
    store, _ = registry_store
    output = 'Error: ' + 'x' * 12000 + 'END'
    registry = create_default_tool_registry(workspace_dir=str(tmp_path))
    registry.register(ToolDefinition('slow', 'test', async_capable=True, pricing=Free('read')),
                      lambda **args: output)
    runner = await RunningAgent(store, BackgroundParent(), registry, AgentLoopConfig(compactor=None)).start()
    recovered = RuntimeClient(store=store, transport=runner.runtime.transport)
    try:
        await runner.send('start background')
        rows = await store.load_all_messages(runner.target.session_id)
        receipt = json.loads(next(m.content for m in rows if m.tool_call_id=='background'))
        result = await recovered.wait(JobRef(**receipt['job_ref']), actor=runner.actor, deadline_ms=now_ms()+10000)
        assert result.status=='failed' and result.final_text==output
        record = await load_record(store,runner.target.session_id,receipt['job_id'])
        assert record['tool']=='slow' and record['result']==output
        await runner.drain()
        rows = await store.load_all_messages(runner.target.session_id)
        notifications = [m for m in rows if RESULT_PREFIX in (m.content or '')]
        assert len(notifications)==1 and len(notifications[0].content)<2000
    finally:
        await recovered.aclose()
        await runner.aclose()
