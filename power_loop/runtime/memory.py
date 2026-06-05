"""MemoryProvider — pluggable long-term / cross-session memory.

Library scope
-------------
power-loop **does not implement a memory backend**. It defines:

* the ``MemoryProvider`` Protocol callers implement,
* a ``MemorySnapshot`` shape passed to ``remember``,
* the pipeline integration points (``MEMORY_RECALLED`` hook +
  ``MEMORY_RECALLED`` / ``MEMORY_FAILED`` events),
* the **inject position** invariant (after existing system messages,
  after compact_note, before the conversation history).

Concrete backends live in callers' code or in ``examples/`` — SQLite
fact store, HTTP API diary, vector DB RAG, etc. — none of them belong
in the library.

Failure model
-------------
* ``recall`` raises → treated as **no memory** (returns ``[]``) and emit
  ``MEMORY_FAILED``. Loop continues.
* ``remember`` raises → emit ``MEMORY_FAILED``. ``StatefulResult`` is
  still returned unchanged. Persisting memory must never block the user
  from getting a reply.
* Hook ``MEMORY_RECALLED`` returning ``HookDirective.SKIP`` → drop the
  recalled messages (do not inject).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

LoopMessage = dict[str, Any]


@dataclass
class MemorySnapshot:
    """What ``remember`` receives at session end.

    Includes the **full final history** (messages list as seen by the
    pipeline at SESSION_END time, after any compaction). Providers
    typically only persist a summary or selected facts; the full
    snapshot is supplied so the provider can decide.
    """

    session_id: str
    messages: list[LoopMessage] = field(default_factory=list)
    final_text: str = ""
    rounds: int = 0
    status: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemoryProvider(Protocol):
    """Caller-implemented memory backend.

    ``recall`` is called **once per send** (at SESSION_START, before the
    first round). The returned list is injected as ``role=system``
    messages with ``name`` prefixed ``memory_*`` (the library tags them
    automatically if you don't). Returning ``[]`` means "no memory this
    session".

    ``remember`` is called at SESSION_END regardless of status (including
    ``cancelled`` and ``degraded``); callers that only want to persist
    successful sessions should check ``snapshot.status`` themselves.
    """

    async def recall(
        self,
        *,
        messages: list[LoopMessage],
        session_id: str | None,
        budget_tokens: int = 1500,
    ) -> list[LoopMessage]:
        ...

    async def remember(
        self,
        *,
        snapshot: MemorySnapshot,
        session_id: str | None,
    ) -> None:
        ...


def tag_as_memory(messages: list[LoopMessage], *, prefix: str = "memory_") -> list[LoopMessage]:
    """Ensure every recalled message is a system message with a ``name``
    starting ``memory_*``. Idempotent; non-destructive (returns new dicts).

    The library calls this on the provider's output before injection so
    downstream code (hooks, compactor, audit) can identify memory rows
    by ``msg.get("name", "").startswith("memory_")``.
    """
    tagged: list[LoopMessage] = []
    for i, m in enumerate(messages):
        m2 = dict(m)
        m2["role"] = "system"
        name = str(m2.get("name") or "")
        if not name.startswith(prefix):
            m2["name"] = f"{prefix}{name or i}"
        tagged.append(m2)
    return tagged


__all__ = ["LoopMessage", "MemorySnapshot", "MemoryProvider", "tag_as_memory"]
