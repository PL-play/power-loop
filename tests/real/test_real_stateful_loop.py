"""Real-LLM smoke for the core StatefulAgentLoop send / multi-turn path.

Quality assertions go through the LLM-as-judge helper so the test stays
robust across model wording drift.
"""

from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, SessionStore, StatefulAgentLoop

from ._llm import make_llm
from .judge import assert_passes


@pytest.mark.asyncio
async def test_single_send_returns_meaningful_reply() -> None:
    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0.2),
            store=store,
            config=AgentLoopConfig(
                system_prompt="You are a concise assistant. Reply in English.",
                max_rounds=1, temperature=0.2, max_tokens=256,
                compactor=None,
            ),
        )
        q = "In one sentence, explain what HTTP stands for."
        sid = await loop.new_session()
        r = await loop.send(q, session_id=sid)
        assert r.status == "completed"
        assert r.final_text.strip()
        await assert_passes(
            question=q,
            answer=r.final_text,
            rubric=(
                "1) Mentions 'HyperText Transfer Protocol' (case-insensitive, "
                "with or without hyphen).\n"
                "2) Is a single sentence or compact phrase, not a long essay."
            ),
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_multi_turn_carries_session_history() -> None:
    """Second send must see the first turn — model is told a fact, then
    asked about it implicitly. The judge checks coherence."""
    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "You are an assistant with perfect memory of this chat. "
                    "Answer briefly in English."
                ),
                max_rounds=1, max_tokens=256, temperature=0,
                compactor=None,
            ),
        )
        sid = await loop.new_session()
        r1 = await loop.send("My favorite color is teal. Acknowledge briefly.", session_id=sid)
        assert r1.status == "completed"

        followup = "What did I just say my favorite color was?"
        r2 = await loop.send(followup, session_id=sid)
        assert r2.session_id == sid
        await assert_passes(
            question=followup,
            answer=r2.final_text,
            rubric=(
                "Answer explicitly identifies 'teal' as the user's favorite "
                "color (case-insensitive match for the word 'teal')."
            ),
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_close_session_removes_history_from_store() -> None:
    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=128, temperature=0),
            store=store,
            config=AgentLoopConfig(
                system_prompt="Reply in one word.",
                max_rounds=1, max_tokens=128, temperature=0,
                compactor=None,
            ),
        )
        sid = await loop.new_session()
        r = await loop.send("Say OK.", session_id=sid)
        assert await store.get_session(r.session_id) is not None
        deleted = await loop.close_session(r.session_id)
        assert deleted >= 1
        assert await store.get_session(r.session_id) is None
    finally:
        await store.close()
