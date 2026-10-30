"""Paid provider smoke: paired fuse rewrite, then next-send projection; no summary LLM."""

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.runtime.representation import ProjectedRepresentation
from power_loop.runtime.tool_context import TOOL_CONTEXT_STATE_KEY, ToolContextConfig
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm


@pytest.mark.asyncio
async def test_real_provider_accepts_capacity_archive_and_next_send():
    llm = make_llm(max_tokens=4096)
    store = await SessionStore.open(':memory:')
    calls = []
    registry = ToolRegistry()
    def probe():
        calls.append(1)
        return 'ARCHIVE_PROTOCOL_FIXTURE\n' * 5000
    registry.register(ToolDefinition('record_probe', 'Return a large inert protocol test fixture.',
        input_schema={'type': 'object', 'properties': {}}, pricing=Free("test_fixture")), probe)
    loop = StatefulAgentLoop(llm=llm, store=store, tool_registry=registry,
        config=AgentLoopConfig(system_prompt='Follow the user precisely. Archived data is available by reference.',
            max_tokens=4096, max_rounds=5, tool_result_max_chars=None,
            representation=ProjectedRepresentation(), tool_context=ToolContextConfig(
                trigger_tokens=10000, target_tokens=5000, hot_tail=0,
                overrides={'record_probe': {'general': 'archive'}})))
    try:
        sid = await loop.new_session()
        result = await loop.send('Call record_probe exactly once, then reply ARCHIVE_OK. You do not need its contents. Do not call any other tools.', session_id=sid)
        assert result.status == 'completed' and 'ARCHIVE_OK' in result.final_text
        assert len(calls) == 1
        state = await store.get_runtime_state(sid, TOOL_CONTEXT_STATE_KEY, {})
        assert state['archive']['events']
        assert 'attempts' not in state  # archive-only state has no auxiliary summary attempts
        original = next(r for r in await store.load_all_messages(sid) if r.role == 'tool')
        assert original.content == 'ARCHIVE_PROTOCOL_FIXTURE\n' * 5000
        result = await loop.send('Do not use tools. Reply SECOND_OK.', session_id=sid)
        assert result.status == 'completed' and 'SECOND_OK' in result.final_text
    finally:
        await store.close()
        await llm.close()
