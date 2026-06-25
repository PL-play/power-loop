"""Real-LLM companion for the follow-up / steering unit tests (test_follow_up.py).

A follow-up message must reach the live model and steer the reply. We exercise both the
idle-delivery path (follow_up on an idle session falls through to a send) and the in-flight
queue+merge path (follow_up while a tool-using send is mid-run drains at the next round).
"""

from __future__ import annotations

import asyncio

import pytest

from power_loop import (
    AgentLoopConfig,
    FollowUpQueued,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.agent.follow_up import FOLLOW_UP_MESSAGE_NAME
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm
from .judge import assert_passes


@pytest.mark.asyncio
async def test_idle_follow_up_steers_the_reply() -> None:
    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0),
            store=store,
            config=AgentLoopConfig(system_prompt="Answer briefly in English.",
                                   max_rounds=1, max_tokens=256, temperature=0, compactor=None),
        )
        sid = await loop.new_session()
        await loop.send("Name a primary color.", session_id=sid)
        # follow_up on an idle session delivers as a send; the steering must reach the model.
        steer = "Forget the previous color. Reply with exactly one word — the color 'teal'."
        r2 = await loop.follow_up(steer, session_id=sid)
        assert not isinstance(r2, FollowUpQueued)  # idle → delivered, not queued
        await assert_passes(question=steer, answer=r2.final_text,
                            rubric="The reply contains the word 'teal' (case-insensitive).")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_inflight_follow_up_is_queued_merged_and_steers() -> None:
    # While a tool-using send is mid-run, a follow_up is QUEUED and merged into the next round so
    # the live model sees it. Uses a slow tool to keep the send in-flight long enough to enqueue.
    store = await SessionStore.open(":memory:")
    try:
        reg = ToolRegistry()

        async def wait_tool(**kwargs):
            await asyncio.sleep(1.5)
            return "waited"

        reg.register(ToolDefinition(
            name="wait", description="Wait briefly before continuing.",
            input_schema={"type": "object", "properties": {}}), wait_tool)

        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=384, temperature=0), store=store, tool_registry=reg,
            config=AgentLoopConfig(
                system_prompt="First call the `wait` tool exactly once, then answer the user.",
                max_rounds=4, max_tokens=384, temperature=0, compactor=None),
        )
        sid = await loop.new_session()
        send_task = asyncio.create_task(loop.send("Name a fruit.", session_id=sid))
        # Wait for the send to be in-flight (lock held), then steer.
        for _ in range(400):
            if loop._lock_for(sid).locked():
                break
            await asyncio.sleep(0.01)
        queued = await loop.follow_up("Make sure the fruit you name is specifically 'mango'.", session_id=sid)
        result = await send_task

        assert result.status == "completed"
        rows = await store.load_active_messages(sid)
        # Either it was genuinely queued (merged follow_up row present), or it raced to idle and was
        # delivered — both paths must get the steering text in front of the model.
        if isinstance(queued, FollowUpQueued):
            assert any(r.name == FOLLOW_UP_MESSAGE_NAME for r in rows)
        await assert_passes(
            question="Name a fruit (steered to mango via a follow-up).",
            answer=result.final_text or "",
            rubric="The reply names the fruit 'mango' (case-insensitive). If it named a different "
                   "fruit, FAIL — the follow-up steering was not applied.",
        )
    finally:
        await store.close()
