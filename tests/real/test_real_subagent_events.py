"""Real-LLM companion for sub-agent lifecycle events (test_subagent_events.py).

A real sub-agent run must emit SUBAGENT_TASK_START … SUBAGENT_COMPLETED on the parent bus.

(The heal-pending / abort-then-resend area's real companion is
``test_real_pending_resume.py``; it is provider-limited on endpoints that reject the
abort-then-resend message shape — the same `llm-transport` strictness tracked elsewhere — so it is
not duplicated here.)
"""

from __future__ import annotations

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentLoopConfig,
    AgentSpec,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.spec import run_agent_spec

from ._llm import make_llm


@pytest.mark.asyncio
async def test_real_subagent_emits_lifecycle_events() -> None:
    store = await SessionStore.open(":memory:")
    try:
        bus = AgentEventBus()
        seen: list = []
        bus.subscribe(None, lambda e: seen.append(e.type) if e.source == "subagent" else None)
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0), store=store, event_bus=bus,
            config=AgentLoopConfig(system_prompt="parent", max_rounds=2, max_tokens=256,
                                   temperature=0, compactor=None),
        )
        parent_sid = await loop.new_session()
        tl, ts = set_current_loop(loop), set_session_id(parent_sid)
        try:
            res = await run_agent_spec(
                AgentSpec(name="child", system_prompt="Answer in one short sentence."),
                "Name the capital of France.", parent_loop=loop,
            )
        finally:
            reset_current_loop(tl)
            reset_session_id(ts)
        assert res["status"] == "completed"
        # lifecycle events fired on the parent bus, tagged source="subagent"
        assert AgentEventType.SUBAGENT_TASK_START in seen
        assert AgentEventType.SUBAGENT_COMPLETED in seen
    finally:
        await store.close()
