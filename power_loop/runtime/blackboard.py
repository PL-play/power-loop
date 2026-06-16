"""Scoped shared blackboard — a coordination surface for agents that share an id.

A *blackboard* is durable, shared, editable state that every agent bound to the
same ``board_id`` can read and append to. It's the generalization of two things
that are really the same: workflow sub-agents coordinating a fan-out, and
multiple long-lived agents in one conversation leaving each other notes. They
differ only in **scope** (a run id vs a conversation id) and lifetime — so the
primitive is keyed by an opaque ``board_id`` and the scope is the integrator's
choice.

Design (power-loop house style — Protocol + default impl + injectable):

* :class:`Blackboard` is the async Protocol a host implements. The default
  :class:`SqliteBlackboard` persists in the ``SessionStore``'s sqlite
  (``session_runtime_state``), good for in-process workflow sub-agents. A host
  whose board must outlive/span one process (e.g. a conversation-scoped board
  across requests or machines) injects its own implementation — the same way one
  injects a sandboxing ``ShellBackend`` for bash.
* Entries are **append + per-entry update/remove** (never whole-document
  overwrite), so concurrent authors don't clobber each other. Concurrency is the
  implementation's to guarantee; the default serializes read-modify-write per
  ``board_id``.
* It's reached by tools (``board_*``) and an optional context injection; both
  resolve the live board + id from :class:`~power_loop.runtime.env.RuntimeEnv`.

``board_id = None`` (the default) means *no board* — most sub-agents are
independent and should stay isolated; a board is opt-in for collaboration.
"""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "BlackboardEntry",
    "Blackboard",
    "SqliteBlackboard",
    "BlackboardError",
    "render_entries",
    "STORE_KEY_PREFIX",
]

STORE_KEY_PREFIX = "blackboard:"
DEFAULT_MAX_ENTRIES = 60
DEFAULT_MAX_TEXT_LEN = 2000


class BlackboardError(ValueError):
    """Raised on invalid board input (bad id, board full, unknown entry)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class BlackboardEntry:
    """One entry on a board. ``kind``/``status`` are opaque labels — what values
    are valid is the host's policy (enforced where the tools are registered)."""

    id: int
    text: str
    kind: str = "note"
    status: str | None = None
    author: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "text": self.text, "kind": self.kind, "status": self.status,
            "author": self.author, "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BlackboardEntry:
        return cls(
            id=int(d["id"]), text=str(d.get("text", "")), kind=str(d.get("kind", "note")),
            status=d.get("status"), author=d.get("author"),
            created_at=str(d.get("created_at", "")), updated_at=str(d.get("updated_at", "")),
        )

    def render(self) -> str:
        tag = f"{self.kind}·{self.status}" if self.status else self.kind
        who = f" ({self.author})" if self.author else ""
        return f"- #{self.id} [{tag}]{who} {self.text}"


def render_entries(entries: list[BlackboardEntry], *, header: str | None = None, empty: str = "") -> str:
    """Render a board snapshot for injection or a tool response."""
    if not entries:
        return (f"{header}\n{empty}".strip() if header else empty)
    body = "\n".join(e.render() for e in entries)
    return f"{header}\n{body}" if header else body


@runtime_checkable
class Blackboard(Protocol):
    """Async interface for a scoped shared board. Implement to back it with any
    store (the default uses the SessionStore; a host may use its own DB/api)."""

    async def read(self, board_id: str) -> list[BlackboardEntry]: ...

    async def post(
        self, board_id: str, *, text: str, kind: str = "note",
        status: str | None = None, author: str | None = None,
    ) -> BlackboardEntry: ...

    async def update(
        self, board_id: str, entry_id: int, *, text: str | None = None, status: str | None = None,
    ) -> BlackboardEntry: ...

    async def remove(self, board_id: str, entry_id: int) -> bool: ...


class SqliteBlackboard:
    """Default board, persisted in the ``SessionStore``'s sqlite via
    ``session_runtime_state`` (key ``blackboard:<board_id>``).

    Read-modify-write is serialized per ``board_id`` with an ``asyncio.Lock`` —
    safe for the in-process, single-event-loop case (workflow sub-agents sharing
    one store). It is NOT a cross-process writer; for that, inject a host
    implementation that funnels writes through your own single writer.
    """

    def __init__(
        self, store: Any, *, max_entries: int = DEFAULT_MAX_ENTRIES, max_text_len: int = DEFAULT_MAX_TEXT_LEN
    ) -> None:
        self._store = store
        self._max_entries = int(max_entries)
        self._max_text_len = int(max_text_len)
        # Weak-valued so a per-run/per-conversation board_id's lock is dropped once no
        # RMW holds it — otherwise a long-lived process driving many distinct board_ids
        # accumulated one Lock per id forever (C24). The `async with self._lock(id)`
        # critical section keeps a strong ref for its duration, so concurrent callers
        # still share one lock; it is only collectible while idle.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def _lock(self, board_id: str) -> asyncio.Lock:
        lock = self._locks.get(board_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[board_id] = lock
        return lock

    _DOC_KEY = "board"

    def _owner(self, board_id: str) -> str:
        if not board_id:
            raise BlackboardError("board_id is required")
        return f"{STORE_KEY_PREFIX}{board_id}"

    def _load_doc(self, board_id: str) -> dict[str, Any]:
        # Stored in the SessionStore's sqlite `shared_state` table, owned by the
        # board id (NOT a session — boards outlive any one agent session).
        doc = self._store.get_shared_state(self._owner(board_id), self._DOC_KEY, default=None)
        if not isinstance(doc, dict) or not isinstance(doc.get("entries"), list):
            return {"version": 1, "next_id": 1, "entries": []}
        doc.setdefault("version", 1)
        doc.setdefault("next_id", 1)
        return doc

    def _save_doc(self, board_id: str, doc: dict[str, Any]) -> None:
        self._store.set_shared_state(self._owner(board_id), self._DOC_KEY, doc)

    def _clean(self, text: str) -> str:
        return (text or "").strip()[: self._max_text_len]

    async def read(self, board_id: str) -> list[BlackboardEntry]:
        doc = self._load_doc(board_id)
        return [BlackboardEntry.from_dict(e) for e in doc.get("entries", [])]

    async def post(
        self, board_id: str, *, text: str, kind: str = "note",
        status: str | None = None, author: str | None = None,
    ) -> BlackboardEntry:
        body = self._clean(text)
        if not body:
            raise BlackboardError("text is required")
        async with self._lock(board_id):
            doc = self._load_doc(board_id)
            if len(doc["entries"]) >= self._max_entries:
                raise BlackboardError(
                    f"board is full ({self._max_entries} entries); remove stale entries first"
                )
            eid = int(doc.get("next_id", 1))
            doc["next_id"] = eid + 1
            now = _now_iso()
            entry = BlackboardEntry(
                id=eid, text=body, kind=kind, status=status, author=author,
                created_at=now, updated_at=now,
            )
            doc["entries"].append(entry.to_dict())
            self._save_doc(board_id, doc)
            return entry

    async def update(
        self, board_id: str, entry_id: int, *, text: str | None = None, status: str | None = None,
    ) -> BlackboardEntry:
        if text is None and status is None:
            raise BlackboardError("nothing to update: provide text and/or status")
        async with self._lock(board_id):
            doc = self._load_doc(board_id)
            for e in doc["entries"]:
                if int(e.get("id", -1)) == entry_id:
                    if text is not None:
                        body = self._clean(text)
                        if not body:
                            raise BlackboardError("text cannot be empty")
                        e["text"] = body
                    if status is not None:
                        e["status"] = status
                    e["updated_at"] = _now_iso()
                    self._save_doc(board_id, doc)
                    return BlackboardEntry.from_dict(e)
            raise BlackboardError(f"no board entry with id #{entry_id}")

    async def remove(self, board_id: str, entry_id: int) -> bool:
        async with self._lock(board_id):
            doc = self._load_doc(board_id)
            before = len(doc["entries"])
            doc["entries"] = [e for e in doc["entries"] if int(e.get("id", -1)) != entry_id]
            if len(doc["entries"]) == before:
                raise BlackboardError(f"no board entry with id #{entry_id}")
            self._save_doc(board_id, doc)
            return True
