"""The execution records no history-mode metadata; the real projection version is the fact.

Design 216 removed ``_record_history_mode``: after the projection-only unification the mode is
constant, so the loop must not touch user runtime metadata on every send. What remains observable
is the actual ``projector_version`` stamped on the projection rows.
"""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from power_loop import AgentLoopConfig
from tests.runtime_fixture import RunningAgent, ScriptedModel

from .session_native_fixture import DESCRIPTOR

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_user_metadata_survives_sends_without_mode_writes(registry_store, monkeypatch):
    store, connect = registry_store
    config = AgentLoopConfig()
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
        assert (await agent.send('second')).status == 'completed'
        peer = await connect()
        meta = await peer.get_runtime_state(sid, 'meta')
        assert meta == {'customer': 'Ada'}
        assert writes == []  # the loop never rewrites mode/trigger/keep metadata
        assert (await peer.get_session(sid)).descriptor == before
        # The version that actually rendered history comes from the projection rows.
        rows = await peer.load_project_messages(sid)
        assert rows and {r.projector_version for r in rows} == {config.representation.version}
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