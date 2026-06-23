"""Real-LLM validation of the built-in MemoryRecallHook (3.1.0).

Proves end-to-end against a live model that a note recalled by NoteMemory is
injected into the request (at the tail) and actually reaches + is used by the
model — i.e. the hook-based ephemeral injection works, not just in unit tests.
"""

from __future__ import annotations

import pytest

from power_loop import AgentLoopConfig, NoteMemory, SessionStore, StatefulAgentLoop

from ._llm import make_llm
from .judge import assert_passes


@pytest.mark.asyncio
async def test_recalled_note_reaches_and_is_used_by_live_model() -> None:
    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "You are a concise assistant. You are given persistent notes "
                    "about the user — use them when answering. Reply in English."
                ),
                max_rounds=1,
                temperature=0,
                max_tokens=256,
                compactor=None,
                memory=NoteMemory(store),
            ),
        )
        sid = await loop.new_session()
        # Seed a distinctive fact the model could not otherwise know.
        await store.add_note(sid, "The user's project codename is 'Bluefin'.")

        q = "What is my project's codename?"
        r = await loop.send(q, session_id=sid)
        assert r.status == "completed"
        assert r.final_text.strip()
        await assert_passes(
            question=q,
            answer=r.final_text,
            rubric="The answer states the project codename is 'Bluefin' (case-insensitive).",
        )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_note_added_midsession_surfaces_on_next_send() -> None:
    """Recall is once-per-send: a note added after send 1 must surface on send 2."""
    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=make_llm(max_tokens=256, temperature=0),
            store=store,
            config=AgentLoopConfig(
                system_prompt=(
                    "You are a concise assistant. Use the persistent notes you are "
                    "given. Reply in English."
                ),
                max_rounds=1,
                temperature=0,
                max_tokens=256,
                compactor=None,
                memory=NoteMemory(store),
            ),
        )
        sid = await loop.new_session()
        await loop.send("Hello.", session_id=sid)  # no note yet
        await store.add_note(sid, "The user's lucky number is 4127.")

        q = "What is my lucky number?"
        r = await loop.send(q, session_id=sid)
        assert r.status == "completed"
        await assert_passes(
            question=q,
            answer=r.final_text,
            rubric="The answer states the lucky number is 4127.",
        )
    finally:
        await store.close()
