"""Real-LLM check that spawn_agent / run_agent route through StatefulAgentLoop
correctly: parent delegates → child session is created with parent linking →
child's answer arrives through the inbox → parent integrates it.

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
from .test_real_async_boundaries_192 import settle_children


@pytest.mark.asyncio
async def test_parent_can_spawn_subagent_via_tool() -> None:
    """End-to-end: parent LLM should choose to call spawn_agent and surface
    the child's answer. The linked child remains available after completion."""
    store = await SessionStore.open(":memory:")
    try:
        registry = ToolRegistry()
        register_spawn_agent(registry)
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=2048, temperature=0),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "You delegate research tasks. For any factual or "
                    "research-style question, call the `spawn_agent` tool "
                    "exactly once with a clear task description; do not answer from "
                    "memory. After the sub-agent replies, summarize its "
                    "answer for the user."
                ),
                max_rounds=5, max_tokens=2048, temperature=0,
                compactor=None,
            ),
            tool_registry=registry,
        )
        q = "Delegate this and report back: what is the capital of Japan?"
        sid = await loop.new_session()
        r = await loop.send(q, session_id=sid)
        assert r.status == "completed"
        await settle_children()
        children = await store.list_children(sid)
        assert len(children) == 1
        assert children[0].lifecycle is SubagentLifecycle.LINKED
        continued = await loop.flush_follow_ups(sid, tools=[])
        if continued is not None:
            r = continued
        assert r.status == "completed"
        await assert_passes(
            question=q,
            answer=r.final_text,
            rubric=(
                "Answer identifies Tokyo as the capital of Japan "
                "(case-insensitive match of 'Tokyo' or '东京')."
            ),
        )
    finally:
        await loop.aclose(drain_timeout_s=0.1)
        await store.close()


@pytest.mark.asyncio
async def test_run_agent_spec_with_linked_lifecycle_persists_child() -> None:
    """Direct `run_agent_spec` call with LINKED lifecycle: the child session
    should survive past return so the parent can audit it."""
    store = await SessionStore.open(":memory:")
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
        parent_sid = await parent_loop.new_session()
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
        children = await store.list_children(parent_sid)
        assert len(children) == 1
        assert children[0].lifecycle is SubagentLifecycle.LINKED

        await assert_passes(
            question="What is 17 + 25?",
            answer=result["final_text"],
            rubric="Answer contains the number 42.",
        )
    finally:
        await store.close()


# ── design/126 §2: spawn_agent(output_schema=…) on a model without native json_schema ──

_TICKET = {
    "type": "object",
    "properties": {
        "priority": {"type": "string", "enum": ["low", "normal", "urgent"]},
        "team": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["priority", "team"],
}


@pytest.mark.asyncio
async def test_spawn_agent_returns_parsed_json_without_native_json_schema() -> None:
    """deepseek-flash rejects native json_schema; undeclared capability → the schema is written
    into the child's system prompt instead; its asynchronous result contains parsed JSON."""
    import json

    from power_loop.tools.spawn_agent import _handle_spawn_agent

    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=2048, temperature=0), store=store,
            config=AgentLoopConfig(max_rounds=3, max_tokens=2048, compactor=None),
        )
        parent = await loop.new_session()
        tok_loop, tok_sid = set_current_loop(loop), set_session_id(parent)
        try:
            out = await _handle_spawn_agent(
                task=("Triage this ticket: 'Checkout page returns 500 for every user since the "
                      "last deploy; payments are down.' Pick a priority and the owning team."),
                output_schema=json.dumps({"name": "Triage", "schema": _TICKET}),
            )
        finally:
            reset_current_loop(tok_loop)
            reset_session_id(tok_sid)
        accepted = json.loads(out)
        assert accepted["status"] == "queued"
        await settle_children()
        result = (await store.get_runtime_state(parent, "subagent:run:" + accepted["run_id"]))["result"]
        assert result["status"] == "completed", result
        value = result["payload"]
        assert value["priority"] == "urgent", value
        assert isinstance(value["team"], str) and value["team"].strip(), value
    finally:
        await loop.aclose(drain_timeout_s=0.1)
        await store.close()
