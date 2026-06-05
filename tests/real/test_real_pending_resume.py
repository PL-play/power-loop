"""Real-LLM check that the pending-state machine survives a simulated crash.

We simulate the crash by directly inserting an ``assistant(tool_calls)``
message + ``set_pending`` on the store, then verify:
  1. The next ``send`` raises :class:`SessionPendingError`.
  2. ``abort_pending`` synthesizes ``<aborted>`` tool messages.
  3. After abort, a subsequent send completes against the real LLM and the
     answer is judged coherent.
"""

from __future__ import annotations

import pytest

from power_loop import (
    AgentLoopConfig,
    SessionPendingError,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry

from ._llm import make_llm
from .judge import assert_passes


def _noop_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo",
            description="Echo input.",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            required_params=(),
        ),
        lambda **kw: kw.get("text", ""),
    )
    return reg


@pytest.mark.asyncio
async def test_pending_blocks_then_abort_clears_then_send_works() -> None:
    store = SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0.2),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "You are a calm assistant. If the previous turn was "
                    "aborted, simply acknowledge briefly and proceed with "
                    "the user's new request."
                ),
                max_rounds=2, max_tokens=256, temperature=0.2,
                compactor=None,
            ),
            tool_registry=_noop_registry(),
        )

        sid = store.create_session(system_prompt="S")
        store.append_message(sid, role="user", content="initial prompt")
        seq = store.append_message(
            sid,
            role="assistant",
            tool_calls=[{
                "id": "tc-stuck",
                "function": {"name": "echo", "arguments": '{"text":"x"}'},
            }],
            round_index=0,
        )
        store.set_pending(sid, {
            "assistant_seq": seq, "round_index": 0,
            "tool_call_ids": ["tc-stuck"],
            "tool_calls": [{
                "id": "tc-stuck",
                "function": {"name": "echo", "arguments": '{"text":"x"}'},
            }],
        })

        with pytest.raises(SessionPendingError):
            await loop.send("anything", session_id=sid)

        aborted = loop.abort_pending(sid, reason="crash-simulated")
        assert aborted == 1
        assert loop.get_pending(sid) is None

        q = "In one short sentence, what does HTML stand for?"
        r = await loop.send(q, session_id=sid)
        assert r.status == "completed"
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
        store.close()
