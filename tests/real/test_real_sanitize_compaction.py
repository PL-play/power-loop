"""Real-LLM check for the history self-heal × in-place compaction integration (review gap HS-001).

A corrupt ORPHAN tool-result row in pl_messages, in DEFAULT mode, while the in-place compactor
ALSO fires in the same run: the assembled prompt must be sanitized (the orphan never reaches the
provider → no 400) AND the compactor must fold the right rows (no desync from the sanitizer
dropping/realigning rows), with the loop staying coherent.
"""

from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, MessageState, SessionStore, StatefulAgentLoop
from power_loop.runtime.compact import DefaultCompactor

from ._llm import make_llm
from .judge import assert_passes


@pytest.mark.asyncio
async def test_orphan_row_sanitized_while_compaction_fires(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "500")  # force the in-place fold
    store = await SessionStore.open(":memory:")
    try:
        sid = await store.create_session(system_prompt="S")
        # A corrupt orphan tool-result row (no preceding assistant tool_call) — would 400 the
        # provider if it reached the prompt — plus fat history so compaction fires the same run.
        await store.append_message(sid, role="tool", tool_call_id="ghost", content="orphan " + ("g" * 200))
        for i in range(4):
            await store.append_message(sid, role="user", content="filler " + ("u" * 400), round_index=i)
            await store.append_message(sid, role="assistant", content="ack " + ("a" * 400), round_index=i)

        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0), store=store,
            config=AgentLoopConfig(
                system_prompt="Answer the user's latest question concisely in English.",
                max_rounds=1, max_tokens=256, temperature=0,
                compactor=DefaultCompactor(trigger_ratio=0.5, keep_last_n=1),
            ),
        )
        q = "Name the largest planet in our solar system."
        r = await loop.send(q, session_id=sid)
        assert r.status == "completed"  # orphan sanitized (no 400) and compaction didn't desync

        all_rows = await store.load_all_messages(sid)
        assert any(row.name == "compact_note" for row in all_rows)  # compaction fired
        assert any(row.state is MessageState.COMPACTED_OUT for row in all_rows)
        # the orphan row is still in the immutable audit (sanitize only filters the prompt)
        assert any(row.tool_call_id == "ghost" for row in all_rows)

        await assert_passes(
            question=q, answer=r.final_text,
            rubric="Answer identifies Jupiter as the largest planet (case-insensitive).",
        )
    finally:
        await store.close()
