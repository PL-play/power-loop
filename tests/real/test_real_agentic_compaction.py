"""Real-LLM end-to-end for the opt-in AgenticMemoryCompactor.

Verifies the differentiator vs DefaultCompactor: at the compaction boundary the agent runs a
bounded tool-loop that EXTRACTS durable facts into the session's notes (via the real note_add
tool, resolving the live session from the loop's contextvars) BEFORE folding the slice into the
compact_note — and the loop stays coherent afterwards.
"""

from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, MessageState, SessionStore, StatefulAgentLoop
from power_loop.runtime.compact import AgenticMemoryCompactor

from ._llm import make_llm
from .judge import assert_passes


@pytest.mark.asyncio
async def test_agentic_compaction_extracts_facts_to_notes_and_stays_coherent(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "500")  # force the fold on the seeded history
    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session(system_prompt="S")
        # Seed fat filler turns PLUS clear durable facts the compaction agent should remember.
        await store.append_message(
            sid, role="user", round_index=0,
            content="Important context to remember: my name is Mei, my project deadline is "
                    "March 3rd, and I always want answers in English. " + ("filler " * 60),
        )
        await store.append_message(
            sid, role="assistant", round_index=0, content="Understood. " + ("ack " * 80),
        )
        for i in range(1, 4):
            await store.append_message(sid, role="user", round_index=i, content="filler " + ("u" * 400))
            await store.append_message(sid, role="assistant", round_index=i, content="ok " + ("a" * 400))

        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=512, temperature=0),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "Earlier turns may have been summarized into a compact_note and durable facts "
                    "saved as notes. Answer the user's latest question concisely in English."
                ),
                max_rounds=1, max_tokens=512, temperature=0,
                compactor=AgenticMemoryCompactor(trigger_ratio=0.5, keep_last_n=1, max_rounds=4),
            ),
        )
        q = "Name the largest planet in our solar system."
        r = await loop.send(q, session_id=sid)
        assert r.status == "completed"

        # 1) compaction actually fired + persisted (same guarantees as the default compactor).
        assert len(await store.list_compactions(sid)) >= 1, "compactor never fired"
        all_rows = await store.load_all_messages(sid)
        assert any(row.name == "compact_note" for row in all_rows)
        assert any(row.state is MessageState.COMPACTED_OUT for row in all_rows)

        # 2) the differentiator: the agent extracted ≥1 durable fact into the session's notes.
        notes = await store.list_notes(sid)
        assert len(notes) >= 1, "agentic compactor wrote no notes — memory extraction did not run"

        # 3) the post-compaction loop is still coherent.
        await assert_passes(
            question=q,
            answer=r.final_text,
            rubric="Answer identifies Jupiter as the largest planet (case-insensitive).",
        )
    finally:
        await store.close()
