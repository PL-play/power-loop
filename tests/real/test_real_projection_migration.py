"""Real-LLM end-to-end for the mode-switch migration (the hardest case A): a session that the
in-place DefaultCompactor REALLY folded (a live compact_note) is reopened in projection mode, and
the one-time migration seeds a projection compact from that note so the session becomes
projection-native — without throwing and staying coherent.
"""

from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, MessageState, SessionStore, StatefulAgentLoop
from power_loop.runtime.compact import DefaultCompactor
from power_loop.runtime.history_projector import DefaultDeterministicProjector

from ._llm import make_llm
from .judge import assert_passes


@pytest.mark.asyncio
async def test_switch_compacted_session_to_projection_migrates(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "500")  # force the in-place fold
    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session(system_prompt="S")
        for i in range(4):  # fat history so the in-place compactor fires
            await store.append_message(sid, role="user", content="filler " + ("u" * 400), round_index=i)
            await store.append_message(sid, role="assistant", content="ack " + ("a" * 400), round_index=i)

        # Phase 1: run in DEFAULT mode with the in-place compactor → produces a real compact_note.
        base = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0), store=store,
            config=AgentLoopConfig(
                system_prompt="Answer concisely in English.", max_rounds=1, max_tokens=256,
                temperature=0, compactor=DefaultCompactor(trigger_ratio=0.5, keep_last_n=1),
            ),
        )
        await base.send("What is two plus two?", session_id=sid)
        all_rows = await store.load_all_messages(sid)
        assert any(r.name == "compact_note" for r in all_rows), "in-place compaction did not fire"
        assert any(r.state is MessageState.COMPACTED_OUT for r in all_rows)
        assert await store.load_project_messages(sid) == []  # nothing projected yet

        # Phase 2: reopen in PROJECTION mode → one-time migration seeds a projection compact from
        # the in-place note, and the loop stays coherent (never throws).
        proj = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0), store=store,
            config=AgentLoopConfig(
                system_prompt="Answer the user's latest question concisely in English.",
                max_rounds=1, max_tokens=8000, temperature=0, compactor=None,
                history_projector=DefaultDeterministicProjector(keep_last_sends=1),
            ),
        )
        q = "Name the largest planet in our solar system."
        r = await proj.send(q, session_id=sid)
        assert r.status == "completed"

        proj_rows = await store.load_project_messages(sid)
        assert any(p.kind == "compact" for p in proj_rows), "migration wrote no projection compact"
        assert (await store.get_session(sid)).metadata.get("projection_migrated") is not None

        await assert_passes(
            question=q, answer=r.final_text,
            rubric="Answer identifies Jupiter as the largest planet (case-insensitive).",
        )
    finally:
        await store.close()
