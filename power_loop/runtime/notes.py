"""Agent-authored persistent notes ("self-managed memory").

Complements :mod:`power_loop.runtime.memory`:

* ``MemoryProvider`` is the *passive* seam — recall/remember at session
  boundaries, the provider decides what to keep (summaries, facts, RAG).
* Notes are *agent-driven* memory — the model itself decides mid-conversation
  what is worth keeping, via the ``note_add`` / ``note_update`` /
  ``note_delete`` default tools, persisted in the session store's ``notes``
  table.

The two meet in :class:`SQLiteNoteMemory`: a ``MemoryProvider`` whose
``recall()`` renders the session's notes into one system message injected at
every send (so the model always sees its own notes), and whose ``remember()``
is a no-op (writes already happened through the tools, in real time).

Capacity model
--------------
``NotesPolicy`` bounds everything. The default eviction mode is **reject**:
when the table is full, ``note_add`` fails with an instructive error telling
the model to delete or merge first. Silent loss is the worst failure mode for
an agent's memory — being forced to curate beats quietly forgetting.
``eviction="fifo"`` switches to queue semantics: the oldest unpinned note is
dropped to make room (pinned notes are never auto-evicted).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from power_loop.runtime.memory import LoopMessage
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.store.types import NoteRow


class NotesFullError(ValueError):
    """note_add was refused because the session is at max_notes (reject mode)."""


@dataclass(frozen=True)
class NotesPolicy:
    """Bounds for agent-authored notes.

    max_notes        : per-session note count cap.
    max_note_chars   : per-note content length cap (longer content is rejected,
                       never truncated silently).
    inject_max_chars : budget for the rendered recall message; oldest unpinned
                       notes are elided first, and the rendering says how many
                       were hidden so the model knows its memory view is partial.
    eviction         : "reject" (default) — note_add errors when full;
                       "fifo" — drop the oldest unpinned note to make room.
    """

    max_notes: int = 50
    max_note_chars: int = 1000
    inject_max_chars: int = 8000
    eviction: Literal["reject", "fifo"] = "reject"


DEFAULT_NOTES_POLICY = NotesPolicy()


async def add_note_checked(
    store: SessionStore,
    session_id: str,
    content: str,
    *,
    pinned: bool = False,
    policy: NotesPolicy = DEFAULT_NOTES_POLICY,
) -> NoteRow:
    """Policy-enforcing insert used by the ``note_add`` tool handler."""
    content = content.strip()
    if not content:
        raise ValueError("note content is empty")
    if len(content) > policy.max_note_chars:
        raise ValueError(
            f"note too long ({len(content)} chars > {policy.max_note_chars}); "
            "keep notes short — store details in a file instead"
        )
    if await store.count_notes(session_id) >= policy.max_notes:
        if policy.eviction == "fifo":
            for note in await store.list_notes(session_id):
                if not note.pinned:
                    await store.delete_note(session_id, note.note_id)
                    break
            else:
                raise NotesFullError(
                    f"notes are full ({policy.max_notes}) and all are pinned; "
                    "unpin or delete (note_delete) before adding"
                )
        else:
            raise NotesFullError(
                f"notes are full ({policy.max_notes}); delete stale notes with "
                "note_delete or merge related ones with note_update first"
            )
    return await store.add_note(session_id, content, pinned=pinned)


async def update_note_checked(
    store: SessionStore,
    session_id: str,
    note_id: int,
    *,
    content: str | None = None,
    pinned: bool | None = None,
    policy: NotesPolicy = DEFAULT_NOTES_POLICY,
) -> None:
    if content is not None:
        content = content.strip()
        if not content:
            raise ValueError("note content is empty; use note_delete to remove a note")
        if len(content) > policy.max_note_chars:
            raise ValueError(
                f"note too long ({len(content)} chars > {policy.max_note_chars})"
            )
    if content is None and pinned is None:
        raise ValueError("nothing to update — pass content and/or pinned")
    if not await store.update_note(session_id, note_id, content=content, pinned=pinned):
        raise ValueError(f"note #{note_id} does not exist")


def render_notes(notes: list[NoteRow], *, policy: NotesPolicy = DEFAULT_NOTES_POLICY) -> str:
    """Render notes as the text block injected at recall time.

    Pinned notes always survive the character budget; unpinned notes are
    elided oldest-first when over budget, with an explicit elision marker.
    """
    if not notes:
        return ""
    pinned = [n for n in notes if n.pinned]
    unpinned = [n for n in notes if not n.pinned]

    def line(n: NoteRow) -> str:
        flag = " [pinned]" if n.pinned else ""
        return f"#{n.note_id}{flag} {n.content}"

    budget = policy.inject_max_chars
    kept_unpinned: list[NoteRow] = []
    used = sum(len(line(n)) + 1 for n in pinned)
    # Keep the newest unpinned notes within budget (drop oldest first).
    for n in reversed(unpinned):
        cost = len(line(n)) + 1
        if used + cost > budget:
            break
        kept_unpinned.append(n)
        used += cost
    kept_unpinned.reverse()
    hidden = len(unpinned) - len(kept_unpinned)

    shown = sorted(pinned + kept_unpinned, key=lambda n: n.note_id)
    lines = [
        "YOUR NOTES (persistent memory you maintain with note_add / note_update / "
        "note_delete; survives context compaction):"
    ]
    lines.extend(line(n) for n in shown)
    if hidden > 0:
        lines.append(f"(… {hidden} older note(s) hidden by budget — consolidate or delete)")
    return "\n".join(lines)


class SQLiteNoteMemory:
    """``MemoryProvider`` that injects the session's own notes every send.

    ``remember()`` is a no-op: the notes table is mutated in real time by the
    note tools, so there is nothing to persist at session end.

    Pass the **same** :class:`SessionStore` the loop uses (or rely on the loop
    sharing its store): ``StatefulAgentLoop(store=store,
    config=AgentLoopConfig(memory=SQLiteNoteMemory(store)))``.
    """

    def __init__(
        self, store: SessionStore, *, policy: NotesPolicy = DEFAULT_NOTES_POLICY
    ) -> None:
        self._store = store
        self._policy = policy

    async def recall(
        self,
        *,
        messages: list[LoopMessage],
        session_id: str | None,
        budget_tokens: int = 1500,
    ) -> list[LoopMessage]:
        if session_id is None:
            return []
        notes = await self._store.list_notes(session_id)
        text = render_notes(notes, policy=self._policy)
        if not text:
            return []
        return [{"role": "system", "name": "memory_notes", "content": text}]

    async def remember(
        self,
        *,
        snapshot: Any,
        session_id: str | None = None,
    ) -> None:
        return None
