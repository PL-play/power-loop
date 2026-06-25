"""Real-LLM companion for the shared-blackboard unit tests (test_blackboard.py).

The live model, given the ``board_*`` tools and a board in its runtime env, must actually post a note
to the shared board — verifying the coordination back-channel works end-to-end through a real run.
"""

from __future__ import annotations

import pytest

from power_loop import (
    AgentLoopConfig,
    SessionStore,
    SqliteBlackboard,
    StatefulAgentLoop,
    register_blackboard_tools,
)
from power_loop.runtime.env import RuntimeEnv, runtime_env_context
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm


@pytest.mark.asyncio
async def test_real_agent_posts_to_shared_board() -> None:
    store = await SessionStore.open(":memory:")
    try:
        reg = ToolRegistry()
        register_blackboard_tools(reg)
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=512, temperature=0), store=store, tool_registry=reg,
            config=AgentLoopConfig(
                system_prompt=("You coordinate with a teammate through a shared board. To record "
                               "anything, you MUST call the board_post tool — that is the only way "
                               "to write to the board."),
                max_rounds=4, max_tokens=512, temperature=0, compactor=None),
        )
        bb = SqliteBlackboard(store)
        sid = await loop.new_session(metadata={"spec_name": "alice"})

        with runtime_env_context(RuntimeEnv(blackboard=bb, blackboard_id="conv-real")):
            r = await loop.send(
                "Post a task to the shared board with the text: draft the introduction.",
                session_id=sid,
            )
        assert r.status == "completed"

        entries = await bb.read("conv-real")
        assert entries, "the agent did not post anything to the board"
        assert any("introduction" in e.text.lower() or "intro" in e.text.lower() for e in entries)
        assert entries[0].author == "alice"  # author resolved from session metadata (spec_name)
    finally:
        await store.close()
