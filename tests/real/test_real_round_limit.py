"""Real-LLM companion for the round-limit unit tests (test_pipeline_round_limit_finalize.py).

When a send exhausts max_rounds, the loop must ask the live model for a wrap-up summary, return it
with status='interrupted', AND persist it as the assistant turn (M-stateful-loop-2).
"""

from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, SessionStore
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm


@pytest.mark.asyncio
async def test_hit_round_limit_returns_and_persists_summary(runtime_agent) -> None:
    store = await SessionStore.open(":memory:")
    try:
        reg = ToolRegistry()

        def counter(**kwargs):
            return "incremented; call again to continue"

        reg.register(ToolDefinition(
            name="step", description="Advance one step of a long multi-step process.",
            input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")), counter)

        loop = await runtime_agent(
            model=make_llm(max_tokens=384, temperature=0), store=store, registry=reg,
            config=AgentLoopConfig(
                system_prompt=("This is a long multi-step job. Call the `step` tool to make "
                               "progress, and keep calling it until the job is done."),
                max_rounds=1, max_tokens=384, temperature=0, compactor=None),
        )
        sid = loop.target.session_id
        r = await loop.send("Begin the multi-step job and keep going.", )

        # round 0 makes a tool call → max_rounds=1 is exhausted → wrap-up summary path
        assert r.status == "interrupted"
        assert r.final_text.strip()
        # The summary is recorded as an assistant turn (not just returned to the caller).
        msgs = [{'role':row.role,'content':row.content,'tool_calls':row.tool_calls} for row in await store.load_all_messages(sid)]
        assistant_texts = [m.get("content") for m in msgs if m.get("role") == "assistant"
                           and not m.get("tool_calls")]
        assert any(t and t.strip() for t in assistant_texts), msgs
    finally:
        await loop.aclose()
        await store.close()
