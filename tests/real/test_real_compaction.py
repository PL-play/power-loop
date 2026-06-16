"""Real-LLM check that compaction triggers at the configured threshold,
gets persisted, and the post-compaction loop still produces a coherent
answer judged by a peer agent.
"""

from __future__ import annotations

import pytest

from power_loop import (
    AgentLoopConfig,
    MessageState,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop.runtime.compact import DefaultCompactor

from ._llm import make_llm
from .judge import assert_passes


@pytest.mark.asyncio
async def test_compaction_triggers_persists_and_loop_still_coherent(monkeypatch) -> None:
    # Low env threshold so the heuristic estimator definitely fires on the
    # seeded fat history.
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "500")
    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session(system_prompt="S")
        # Seed several fat turns that the compactor must fold.
        for i in range(4):
            await store.append_message(
                sid, role="user", content="filler " + ("u" * 400), round_index=i
            )
            await store.append_message(
                sid, role="assistant",
                content="filler ack " + ("a" * 400),
                round_index=i,
            )
        # Final user prompt the model should still answer correctly.
        # (Loop seeds the latest user via send().)
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "Earlier turns may have been summarized into a "
                    "compact_note. Answer the user's latest question "
                    "concisely in English."
                ),
                max_rounds=1, max_tokens=256, temperature=0,
                compactor=DefaultCompactor(trigger_ratio=0.5, keep_last_n=1),
            ),
        )
        q = "Name the largest planet in our solar system."
        r = await loop.send(q, session_id=sid)
        assert r.status == "completed"

        # Compaction must have been recorded + rows marked.
        compactions = await store.list_compactions(sid)
        assert len(compactions) >= 1, "compactor never fired despite low threshold"
        all_rows = await store.load_all_messages(sid)
        assert any(row.state is MessageState.COMPACTED_OUT for row in all_rows)
        assert any(row.name == "compact_note" for row in all_rows)

        await assert_passes(
            question=q,
            answer=r.final_text,
            rubric=(
                "Answer identifies Jupiter as the largest planet in the "
                "solar system (case-insensitive)."
            ),
        )
    finally:
        await store.close()
