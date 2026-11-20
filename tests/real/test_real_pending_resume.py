"""Real-LLM check that the pending-state machine survives a simulated crash.

We simulate the crash by directly inserting an ``assistant(tool_calls)``
message + ``set_pending`` on the store, then verify:
  The next send closes stale calls with interrupted/unknown results and answers
  the new input against the real provider.
"""

from __future__ import annotations

import pytest

from power_loop import (
    AgentLoopConfig,
    SessionStore,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm
from .judge import assert_passes

pytestmark = pytest.mark.real_llm


def _noop_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo",
            description="Echo input.",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            required_params=(),
        pricing=Free("test_fixture")),
        lambda **kw: kw.get("text", ""),
    )
    return reg


@pytest.mark.asyncio
async def test_pending_is_interrupted_then_send_works(runtime_agent) -> None:
    store = await SessionStore.open(":memory:")
    try:
        loop = await runtime_agent(
            model=make_llm(max_tokens=2048, temperature=0.2),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "You are a calm assistant. If the previous turn was "
                    "aborted, simply acknowledge briefly and proceed with "
                    "the user's new request."
                ),
                max_rounds=2, max_tokens=2048, temperature=0.2,
                context_budget_tokens=10**9,
            ),
            registry=_noop_registry(),
        )

        sid = loop.target.session_id
        from power_loop.agent.runtime_store import RuntimeInputStore
        port = RuntimeInputStore(store)
        await port.accept(loop.target, "initial prompt", message_id="interrupted", actor=loop.actor)
        batch = await port.claim(loop.target, node_id="old", boot_id="old")
        seq = await store.append_message(
            sid,
            role="assistant",
            tool_calls=[{
                "id": "tc-stuck",
                "function": {"name": "echo", "arguments": '{"text":"x"}'},
            }],
            round_index=0, send_index=batch.send_index, pending_tool_calls=True,
        )
        assert (await store.get_state(sid)).pending == {"assistant_seq": seq}
        await port.final(batch, status="interrupted", error="crash")

        q = "In one short sentence, what does HTML stand for?"
        r = await loop.send(q)
        assert r.status == "completed"
        assert (await store.get_state(sid)).pending is None
        rows = await store.load_active_messages(sid)
        assert any(row.role == "tool" and "结果未知" in row.content for row in rows)
        await assert_passes(
            question=q,
            answer=r.final_text,
            rubric=(
                "Answer expands HTML correctly: contains the phrase "
                "'HyperText Markup Language' (case-insensitive, allow "
                "'hyper text' or 'hyper-text' variants)."
            ),
        )
    finally:
        await loop.aclose()
        await store.close()
