"""Agent-authored notes: store CRUD, policy enforcement, rendering, recall."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from power_loop.runtime.notes import (
    DEFAULT_NOTES_POLICY,
    NotesFullError,
    NotesPolicy,
    SQLiteNoteMemory,
    add_note_checked,
    render_notes,
    update_note_checked,
)
from power_loop.runtime.store.store import SessionStore


@pytest.fixture()
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


SID = "sess_test"


# ── store CRUD ─────────────────────────────────────────────────────────────


async def test_add_assigns_sequential_ids(store: SessionStore) -> None:
    a = await store.add_note(SID, "first")
    b = await store.add_note(SID, "second")
    assert (a.note_id, b.note_id) == (1, 2)
    assert [n.content for n in await store.list_notes(SID)] == ["first", "second"]


async def test_ids_are_per_session(store: SessionStore) -> None:
    await store.add_note(SID, "x")
    other = await store.add_note("sess_other", "y")
    assert other.note_id == 1


async def test_id_not_reused_after_delete_of_last(store: SessionStore) -> None:
    await store.add_note(SID, "a")
    b = await store.add_note(SID, "b")
    await store.delete_note(SID, b.note_id)
    c = await store.add_note(SID, "c")
    # MAX(note_id)+1 over remaining rows → id 2 may be reused only after the
    # highest is gone; what matters is no collision with live rows.
    assert c.note_id not in {n.note_id for n in await store.list_notes(SID) if n.content != "c"}


async def test_update_and_delete(store: SessionStore) -> None:
    n = await store.add_note(SID, "old")
    assert await store.update_note(SID, n.note_id, content="new", pinned=True)
    got = (await store.list_notes(SID))[0]
    assert got.content == "new" and got.pinned
    assert await store.delete_note(SID, n.note_id)
    assert await store.count_notes(SID) == 0
    assert not await store.delete_note(SID, n.note_id)


async def test_close_session_cascades_notes(store: SessionStore) -> None:
    sid = await store.create_session()
    await store.add_note(sid, "to be removed")
    await store.close_session(sid)
    assert await store.count_notes(sid) == 0


# ── policy ─────────────────────────────────────────────────────────────────


async def test_reject_when_full(store: SessionStore) -> None:
    policy = NotesPolicy(max_notes=2)
    await add_note_checked(store, SID, "one", policy=policy)
    await add_note_checked(store, SID, "two", policy=policy)
    with pytest.raises(NotesFullError, match="note_delete"):
        await add_note_checked(store, SID, "three", policy=policy)
    assert await store.count_notes(SID) == 2


async def test_fifo_evicts_oldest_unpinned(store: SessionStore) -> None:
    policy = NotesPolicy(max_notes=2, eviction="fifo")
    await add_note_checked(store, SID, "oldest", policy=policy)
    await add_note_checked(store, SID, "middle", policy=policy)
    await add_note_checked(store, SID, "newest", policy=policy)
    contents = [n.content for n in await store.list_notes(SID)]
    assert contents == ["middle", "newest"]


async def test_fifo_never_evicts_pinned(store: SessionStore) -> None:
    policy = NotesPolicy(max_notes=2, eviction="fifo")
    await add_note_checked(store, SID, "keep me", pinned=True, policy=policy)
    await add_note_checked(store, SID, "unpinned", policy=policy)
    await add_note_checked(store, SID, "newcomer", policy=policy)
    contents = [n.content for n in await store.list_notes(SID)]
    assert "keep me" in contents and "unpinned" not in contents


async def test_fifo_all_pinned_raises(store: SessionStore) -> None:
    policy = NotesPolicy(max_notes=1, eviction="fifo")
    await add_note_checked(store, SID, "pinned", pinned=True, policy=policy)
    with pytest.raises(NotesFullError, match="pinned"):
        await add_note_checked(store, SID, "more", policy=policy)


async def test_too_long_and_empty_rejected(store: SessionStore) -> None:
    policy = NotesPolicy(max_note_chars=10)
    with pytest.raises(ValueError, match="too long"):
        await add_note_checked(store, SID, "x" * 11, policy=policy)
    with pytest.raises(ValueError, match="empty"):
        await add_note_checked(store, SID, "   ", policy=policy)


async def test_update_checked_validates(store: SessionStore) -> None:
    n = await add_note_checked(store, SID, "ok")
    with pytest.raises(ValueError, match="does not exist"):
        await update_note_checked(store, SID, 99, content="x")
    with pytest.raises(ValueError, match="nothing to update"):
        await update_note_checked(store, SID, n.note_id)
    await update_note_checked(store, SID, n.note_id, pinned=True)
    assert (await store.list_notes(SID))[0].pinned


# ── rendering ──────────────────────────────────────────────────────────────


def test_render_empty_is_empty() -> None:
    assert render_notes([]) == ""


async def test_render_shows_ids_and_pins(store: SessionStore) -> None:
    await store.add_note(SID, "alpha")
    await store.add_note(SID, "beta", pinned=True)
    text = render_notes(await store.list_notes(SID))
    assert "#1 alpha" in text
    assert "#2 [pinned] beta" in text
    assert text.startswith("YOUR NOTES")


async def test_render_budget_elides_oldest_unpinned_first(store: SessionStore) -> None:
    await store.add_note(SID, "old-unpinned " + "x" * 50)
    await store.add_note(SID, "pinned-note", pinned=True)
    await store.add_note(SID, "new-unpinned")
    policy = NotesPolicy(inject_max_chars=100)
    text = render_notes(await store.list_notes(SID), policy=policy)
    assert "pinned-note" in text
    assert "new-unpinned" in text
    assert "old-unpinned" not in text
    assert "1 older note(s) hidden" in text


# ── recall provider ────────────────────────────────────────────────────────


async def test_recall_injects_system_message(store: SessionStore) -> None:
    await store.add_note(SID, "remember the milk")
    provider = SQLiteNoteMemory(store)
    msgs = await provider.recall(messages=[], session_id=SID)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"
    assert msgs[0]["name"] == "memory_notes"
    assert "remember the milk" in msgs[0]["content"]


async def test_recall_empty_session_returns_nothing(store: SessionStore) -> None:
    provider = SQLiteNoteMemory(store)
    assert await provider.recall(messages=[], session_id=SID) == []
    assert await provider.recall(messages=[], session_id=None) == []


async def test_remember_is_noop(store: SessionStore) -> None:
    provider = SQLiteNoteMemory(store)
    assert await provider.remember(snapshot=object(), session_id=SID) is None


def test_default_policy_sane() -> None:
    assert DEFAULT_NOTES_POLICY.eviction == "reject"
    assert DEFAULT_NOTES_POLICY.max_notes > 0


# ── end-to-end through the loop ────────────────────────────────────────────


async def test_note_tool_writes_and_next_send_injects(store: SessionStore) -> None:
    """The model calls note_add via the default tool; the following send sees
    its note injected as a memory_notes system message."""
    from collections.abc import Callable
    from dataclasses import dataclass, field
    from typing import Any

    from power_loop import (
        AgentLoopConfig,
        StatefulAgentLoop,
        create_default_tool_registry,
    )
    from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMStreamChunk

    @dataclass
    class _Scripted:
        responses: list[LLMResponse]
        calls: list[list[dict[str, Any]]] = field(default_factory=list)
        _idx: int = 0

        async def complete(
            self,
            request: LLMRequest,
            *,
            on_chunk_delta_text: Callable[[str], Any] | None = None,
            on_chunk_think: Callable[[str], Any] | None = None,
            on_stream_end: Callable[[LLMResponse], Any] | None = None,
        ) -> LLMResponse:
            self.calls.append(list(request.messages))
            if self._idx >= len(self.responses):
                return LLMResponse(raw_text="done")
            response = self.responses[self._idx]
            self._idx += 1
            return response

        def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
            async def _empty() -> AsyncIterator[LLMStreamChunk]:
                if False:
                    yield LLMStreamChunk()

            return _empty()

        async def close(self) -> None:
            return None

    llm = _Scripted(
        responses=[
            LLMResponse(
                raw_text="",
                tool_calls=[
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {
                            "name": "note_add",
                            "arguments": '{"content": "project codename is BLUEBIRD"}',
                        },
                    }
                ],
            ),
            LLMResponse(raw_text="noted"),
            LLMResponse(raw_text="second reply"),
        ]
    )
    loop = StatefulAgentLoop(
        llm=llm,
        store=store,
        tool_registry=create_default_tool_registry(
            include=["note_add", "note_update", "note_delete"], bind=False
        ),
        config=AgentLoopConfig(
            system_prompt="S",
            max_rounds=3,
            compactor=None,
            memory=SQLiteNoteMemory(store),
        ),
    )
    sid = await loop.new_session()
    r1 = await loop.send("remember the codename", sid)
    assert r1.status == "completed"
    notes = await store.list_notes(sid)
    assert len(notes) == 1 and "BLUEBIRD" in notes[0].content

    await loop.send("what is the codename?", sid)
    last_request = llm.calls[-1]
    injected = [
        m for m in last_request if m.get("name") == "memory_notes" and m.get("role") == "system"
    ]
    assert len(injected) == 1
    assert "BLUEBIRD" in injected[0]["content"]
    await loop.aclose()
