"""Real-LLM check that spawn_agent / run_agent route through StatefulAgentLoop
correctly: parent delegates → child session is created with parent linking →
child's answer flows back as a tool result → parent integrates it.

Quality of the child's answer is rated by the judge agent.
"""

from __future__ import annotations

import pytest

from power_loop import (
    AgentLoopConfig,
    AgentSpec,
    SessionStore,
    StatefulAgentLoop,
    SubagentLifecycle,
)
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.spec import run_agent_spec
from power_loop.tools.registry import ToolRegistry
from power_loop.tools.spawn_agent import register_spawn_agent

from ._llm import make_llm
from .judge import assert_passes


@pytest.mark.asyncio
async def test_parent_can_spawn_subagent_via_tool() -> None:
    """End-to-end: parent LLM should choose to call spawn_agent and surface
    the child's answer. We verify that a child session was created and
    cleaned up (EPHEMERAL default = delete on completed)."""
    store = SessionStore.open(":memory:")
    try:
        registry = ToolRegistry()
        register_spawn_agent(registry)
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=512, temperature=0),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "You delegate research tasks. For any factual or "
                    "research-style question, call the `spawn_agent` tool "
                    "with a clear task description; do not answer from "
                    "memory. After the sub-agent replies, summarize its "
                    "answer for the user."
                ),
                max_rounds=5, max_tokens=512, temperature=0,
                compactor=None,
            ),
            tool_registry=registry,
        )
        q = "Delegate this and report back: what is the capital of Japan?"
        sid = loop.new_session()
        r = await loop.send(q, session_id=sid)
        assert r.status == "completed"
        # All EPHEMERAL children should be cleaned up by now.
        assert store.list_children(r.session_id) == []
        await assert_passes(
            question=q,
            answer=r.final_text,
            rubric=(
                "Answer identifies Tokyo as the capital of Japan "
                "(case-insensitive match of 'Tokyo' or '东京')."
            ),
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_run_agent_spec_with_linked_lifecycle_persists_child() -> None:
    """Direct `run_agent_spec` call with LINKED lifecycle: the child session
    should survive past return so the parent can audit it."""
    store = SessionStore.open(":memory:")
    try:
        parent_loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=128, temperature=0),
            store=store,
            config=AgentLoopConfig(
                system_prompt="Reply briefly.",
                max_rounds=1, max_tokens=128, temperature=0,
                compactor=None,
            ),
        )
        # Run one parent turn to get a parent session id in place.
        parent_sid = parent_loop.new_session()
        pr = await parent_loop.send("hi", session_id=parent_sid)
        assert pr.session_id == parent_sid

        # Now invoke run_agent_spec directly, simulating what the meta-tool
        # does internally. Use a child LLM with the same provider.
        spec = AgentSpec(
            name="math-helper",
            system_prompt=(
                "You compute small math problems. Reply with the number only."
            ),
            max_rounds=1,
            max_tokens=64,
            lifecycle=SubagentLifecycle.LINKED.value,
        )

        tok_loop = set_current_loop(parent_loop)
        tok_sid = set_session_id(parent_sid)
        try:
            result = await run_agent_spec(spec, "What is 17 + 25?", parent_loop=parent_loop)
        finally:
            reset_current_loop(tok_loop)
            reset_session_id(tok_sid)

        assert result["status"] == "completed"
        assert result["session_id"] is not None  # LINKED → preserved
        children = store.list_children(parent_sid)
        assert len(children) == 1
        assert children[0].lifecycle is SubagentLifecycle.LINKED

        await assert_passes(
            question="What is 17 + 25?",
            answer=result["final_text"],
            rubric="Answer contains the number 42.",
        )
    finally:
        store.close()
