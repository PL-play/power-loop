"""Scoped shared blackboard — a coordination surface for agents that share an id.

A *blackboard* is durable, shared, editable state that every agent bound to the
same ``board_id`` can read and append to. It's the generalization of two things
that are really the same: workflow sub-agents coordinating a fan-out, and
multiple long-lived agents in one conversation leaving each other notes. They
differ only in **scope** (a run id vs a conversation id) and lifetime — so the
primitive is keyed by an opaque ``board_id`` and the scope is the integrator's
choice.

The host selects an opaque board_id and authorizes access through RuntimeEnv.
StoreBlackboard serializes each mutation in the database across connections and
processes. Its owner calls delete when the board scope ends.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "BlackboardEntry",
    "Blackboard",
    "StoreBlackboard",
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


class StoreBlackboard:
    """Database-atomic board. SQLite serializes writers; server stores lock the
    unique owner/key row, including the first concurrent creation.

    The host owns board_id and must scope access before exposing it to tools.
    Call delete() when that scope ends; boards never create agent sessions.
    """

    def __init__(
        self, store: Any, *, max_entries: int = DEFAULT_MAX_ENTRIES, max_text_len: int = DEFAULT_MAX_TEXT_LEN
    ) -> None:
        self._store = store
        self._max_entries = int(max_entries)
        self._max_text_len = int(max_text_len)
    _DOC_KEY = "board"

    def _owner(self, board_id: str) -> str:
        if not board_id:
            raise BlackboardError("board_id is required")
        return f"{STORE_KEY_PREFIX}{board_id}"

    @staticmethod
    def _empty():
        return {"version": 1, "next_id": 1, "entries": []}

    async def _load_doc(self, board_id):
        return await self._store.get_shared_state(
            self._owner(board_id), self._DOC_KEY, default=self._empty())

    @asynccontextmanager
    async def _edit(self, board_id):
        db, table = self._store._db, self._store.t.shared_state
        owner = self._owner(board_id)
        async with db.transaction() as tx:
            insert = (f"INSERT INTO {table} (owner,state_key,value_json,updated_at) VALUES (?,?,?,?) "
                "ON DUPLICATE KEY UPDATE owner=VALUES(owner)" if db.dialect.name == "mysql" else
                db.dialect.insert_ignore(table, ("owner", "state_key", "value_json", "updated_at")))
            await tx.execute(insert,
                (owner, self._DOC_KEY, json.dumps(self._empty()), int(time.time()*1000)))
            suffix = "" if db.dialect.name == "sqlite" else " FOR UPDATE"
            row = await tx.fetchone(f"SELECT value_json FROM {table} "
                "WHERE owner=? AND state_key=?" + suffix, (owner, self._DOC_KEY))
            doc = json.loads(row["value_json"])
            yield doc
            await tx.execute(f"UPDATE {table} SET value_json=?,updated_at=? "
                "WHERE owner=? AND state_key=?",
                (json.dumps(doc, ensure_ascii=False), int(time.time()*1000), owner, self._DOC_KEY))

    async def delete(self, board_id):
        """Release a board whose host-owned scope has ended."""
        await self._store.delete_shared_state(self._owner(board_id), self._DOC_KEY)

    def _clean(self, text: str) -> str:
        return (text or "").strip()[: self._max_text_len]

    async def read(self, board_id: str) -> list[BlackboardEntry]:
        doc = await self._load_doc(board_id)
        return [BlackboardEntry.from_dict(e) for e in doc.get("entries", [])]

    async def post(
        self, board_id: str, *, text: str, kind: str = "note",
        status: str | None = None, author: str | None = None,
    ) -> BlackboardEntry:
        body = self._clean(text)
        if not body:
            raise BlackboardError("text is required")
        async with self._edit(board_id) as doc:
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
            return entry

    async def update(
        self, board_id: str, entry_id: int, *, text: str | None = None, status: str | None = None,
    ) -> BlackboardEntry:
        if text is None and status is None:
            raise BlackboardError("nothing to update: provide text and/or status")
        async with self._edit(board_id) as doc:
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
                    return BlackboardEntry.from_dict(e)
            raise BlackboardError(f"no board entry with id #{entry_id}")

    async def remove(self, board_id: str, entry_id: int) -> bool:
        async with self._edit(board_id) as doc:
            before = len(doc["entries"])
            doc["entries"] = [e for e in doc["entries"] if int(e.get("id", -1)) != entry_id]
            if len(doc["entries"]) == before:
                raise BlackboardError(f"no board entry with id #{entry_id}")
            return True
