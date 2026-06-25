"""Real-LLM companion for the LLM-call/stream-event + usage unit tests
(test_llm_call_events.py, test_usage_and_heal.py).

Against a live provider: STREAM_STARTED/COMPLETED and LLM_CALL_STARTED/COMPLETED stay paired, token
usage is reported, and each send carries its own usage.
"""

from __future__ import annotations

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    SessionStore,
    StatefulAgentLoop,
)

from ._llm import make_llm


@pytest.mark.asyncio
async def test_real_send_emits_paired_events_and_usage() -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        counts: dict = {}
        for et in (AgentEventType.STREAM_STARTED, AgentEventType.STREAM_COMPLETED,
                   AgentEventType.LLM_CALL_STARTED, AgentEventType.LLM_CALL_COMPLETED):
            bus.subscribe(et, lambda e: counts.__setitem__(e.type, counts.get(e.type, 0) + 1))
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=128, temperature=0), store=store, event_bus=bus,
            config=AgentLoopConfig(system_prompt="Reply in one short sentence.",
                                   max_rounds=1, max_tokens=128, temperature=0, compactor=None),
        )
        sid = await loop.new_session()
        r1 = await loop.send("Say hi.", session_id=sid)
        assert r1.status == "completed"

        # paired stream + call events (>=1 each, equal counts — no dangling 'started')
        assert counts.get(AgentEventType.STREAM_STARTED, 0) >= 1
        assert counts[AgentEventType.STREAM_STARTED] == counts[AgentEventType.STREAM_COMPLETED]
        assert counts[AgentEventType.LLM_CALL_STARTED] == counts[AgentEventType.LLM_CALL_COMPLETED]

        # the live provider reports token usage
        assert (r1.usage or {}).get("total_tokens", 0) > 0

        # a second send gets its own (independent) usage row
        r2 = await loop.send("Say bye.", session_id=sid)
        assert (r2.usage or {}).get("total_tokens", 0) > 0
    finally:
        await store.close()
