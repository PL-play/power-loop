"""Real-LLM companion for the shared-blackboard unit tests (test_blackboard.py).

The live model, given the ``board_*`` tools and a board in its runtime env, must actually post a note
to the shared board — verifying the coordination back-channel works end-to-end through a real run.
"""

from __future__ import annotations

import pytest

from power_loop import (
    AgentLoopConfig,
    SessionStore,
        StoreBlackboard,
    register_blackboard_tools,
)
from power_loop.runtime.env import RuntimeEnv
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm


@pytest.mark.asyncio
async def test_real_agent_posts_to_shared_board(runtime_agent) -> None:
    store = await SessionStore.open(":memory:")
    try:
        reg = ToolRegistry()
        register_blackboard_tools(reg)
        bb = StoreBlackboard(store)
        loop = await runtime_agent(runtime_env=RuntimeEnv(blackboard=bb, blackboard_id="conv-real"),
            model=make_llm(max_tokens=512, temperature=0), store=store, registry=reg,
            config=AgentLoopConfig(
                system_prompt=("You coordinate with a teammate through a shared board. To record "
                               "anything, you MUST call the board tool with action='post' — that "
                               "is the only way to write to the board."),
                max_rounds=4, max_tokens=512, temperature=0, compactor=None),
        )
        sid = loop.target.session_id
        await store.merge_session_metadata(sid, {"spec_name":"alice"})

        r = await loop.send(
                "Post a task to the shared board with the text: draft the introduction.",
            )
        assert r.status == "completed"

        entries = await bb.read("conv-real")
        assert entries, "the agent did not post anything to the board"
        assert any("introduction" in e.text.lower() or "intro" in e.text.lower() for e in entries)
        assert entries[0].author == "alice"  # author resolved from session metadata (spec_name)
    finally:
        await loop.aclose()
        await store.close()
