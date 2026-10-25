import asyncio
import json
from types import SimpleNamespace

import pytest

from power_loop import SessionStore
from power_loop.contracts.tools import ToolDefinition
from power_loop.core import agent_context
from power_loop.runtime.background import RESULT_PREFIX, record_key
from power_loop.runtime.runtime_state import BackgroundRuntimeProjector
from power_loop.tools import default_tools as dt
from power_loop.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_full_result_identity_survives_manager_recreation(monkeypatch):
    store = await SessionStore.open(':memory:')
    sid = await store.create_session()
    sid = sid.session_id if hasattr(sid, 'session_id') else sid
    output = 'Error: ' + 'x' * 12000 + 'END'

    registry = ToolRegistry()
    registry.register(ToolDefinition('scene_pkg', 'test', async_capable=True), lambda **args: output)

    monkeypatch.setattr(agent_context, 'get_current_loop', lambda: SimpleNamespace(tool_registry=registry))
    monkeypatch.setattr(dt, '_current_store_and_session', lambda: (store, sid))
    monkeypatch.setattr(dt, '_TOOL_TASK_ON_COMPLETE', None)
    manager = dt.BackgroundManager()
    try:
        receipt = await manager.run_tool('scene_pkg', {'action': 'sprite', 'frames': ['a.png']})
        tid = receipt.split('task_id=')[1].split('（')[0]
        await asyncio.gather(*list(manager._workers.values()))
        record = await store.get_runtime_state(sid, record_key(tid))
        assert record['result'] == output and record['status'] == 'failed'
        assert record['args']['frames'] == ['a.png']
        recovered = dt.BackgroundManager()
        for text in (await recovered.check(tid), await recovered.wait([tid], 0.01)):
            data = json.loads(text.split(RESULT_PREFIX)[1])['tasks'][0]
            assert data['result'] == output and data['tool'] == 'scene_pkg'
        messages = await BackgroundRuntimeProjector(mark_seen=False).project(
            store=store, session_id=sid, round_index=0, context=None)
        assert messages[0]['_persist'] and '_ack' not in messages[0]
        assert len(messages[0]['content']) < 2000
    finally:
        await store.close()
