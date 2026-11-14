"""First execution records mutable inspection data outside the frozen descriptor."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from power_loop import AgentLoopConfig
from power_loop.runtime.history_projector import DefaultDeterministicProjector
from tests.runtime_fixture import RunningAgent, ScriptedModel

from .session_native_fixture import DESCRIPTOR

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize('projected', [False, True])
async def test_first_send_records_history_mode_once_without_changing_descriptor(registry_store, projected, monkeypatch):
    store, connect = registry_store
    config = AgentLoopConfig(compactor=None,
        **({"representation": DefaultDeterministicProjector()} if projected else {}))
    agent = await RunningAgent(store, ScriptedModel([]), config=config).start()
    try:
        sid = agent.target.session_id
        before = (await store.get_session(sid)).descriptor
        await store.set_runtime_state(sid, 'meta', {'customer': 'Ada'})
        writes = []
        original = store.set_runtime_state
        async def record(sid, key, value):
            if key == 'meta':
                writes.append(value)
            await original(sid, key, value)
        monkeypatch.setattr(store, 'set_runtime_state', record)
        assert (await agent.send('first')).status == 'completed'
        peer = await connect()
        meta = await peer.get_runtime_state(sid, 'meta')
        assert meta['history_mode'] == ('projection' if projected else 'default')
        assert meta['projector_version'] == (int(config.projection_representation.version) if projected else None)
        assert meta['customer'] == 'Ada'
        assert (await agent.send('second')).status == 'completed'
        assert await peer.get_runtime_state(sid, 'meta') == meta
        assert len(writes) == 1
        assert (await peer.get_session(sid)).descriptor == before
    finally:
        await agent.aclose()


@pytest.mark.parametrize('config,name', [({'name': 'alice'}, 'alice'),
    ({'name': 'parent', 'child_spec': {'name': 'child'}}, 'child')])
async def test_blackboard_author_uses_frozen_descriptor_name(monkeypatch, config, name):
    from power_loop.tools import blackboard
    row = SimpleNamespace(descriptor=replace(DESCRIPTOR, config=config))
    store = SimpleNamespace(get_session=AsyncMock(return_value=row))
    monkeypatch.setattr(blackboard, 'get_tool_runtime_context',
                        lambda: SimpleNamespace(store=store, session_id='session'))
    assert await blackboard._author() == name
    store.get_session.assert_awaited_once_with('session')
