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

    ``recall`` is called **once per send** (memoized on the first round) by the
    built-in :class:`MemoryRecallHook` (an ``LLM_BEFORE`` hook). The returned
    list is injected — tagged ``role=system, name=memory_*`` — **ephemerally
    into the per-call request tail**, never into the loop's persisted history or
    the store, so the prompt prefix (system + prior rounds) stays append-only
    and prefix-cacheable. Returning ``[]`` means "no memory this session".

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


class MemoryRecallHook:
    """Built-in ``LLM_BEFORE`` hook that injects recalled memory EPHEMERALLY at
    the tail of the per-call message list.

    Replaces the old hardcoded index-0 ``self.history`` injection. Because it
    mutates only the per-call ``ctx.messages`` (the fresh ``[*self.history,
    *runtime_messages]`` list), the memory block:

    * never enters ``self.history`` / the store / the window cache → no seq map
      to realign, invisible to compaction, gone next round;
    * sits at the TAIL (default) so the whole prior-history prefix stays
      byte-identical across sends and is prefix-cacheable.

    Recall runs once per pipeline run (memoized at ``round_index == 0`` or on a
    session change) and the SAME block is re-appended every round, matching the
    historic once-per-send semantics — notes the agent writes mid-run surface on
    the NEXT run, keeping the tail stable within a run. NOTE: a ``resume()`` /
    ``submit_input()`` continuation is a FRESH run (round_index resets to 0), so
    it re-recalls — updated notes surface on resume too.

    Registered by ``StatefulAgentLoop`` under the name
    :pydata:`MemoryRecallHook.NAME`; a host can override it via
    ``hooks.replace(HookPoint.LLM_BEFORE, my_handler, name=MemoryRecallHook.NAME)``
    or disable it via ``hooks.remove(MemoryRecallHook.NAME)``.
    """

    NAME = "builtin.memory_recall"

    def __init__(
        self,
        provider: MemoryProvider,
        *,
        budget_tokens: int = 1500,
        position: str = "tail",
        hooks: Any = None,
        event_bus: Any = None,
    ) -> None:
        self._provider = provider
        self._budget = int(budget_tokens or 0)
        self._position = "front" if position == "front" else "tail"
        self._hooks = hooks  # for the MEMORY_RECALLED filtering hook (optional)
        self._event_bus = event_bus  # for telemetry (falls back to contextvar bus)
        self._memo_sid: str | None = None
        self._memo_block: list[LoopMessage] = []

    async def __call__(self, ctx: Any) -> None:
        # Recompute once per send: first round, or whenever the session changes
        # (a shared loop may interleave sessions). Otherwise reuse the memoized
        # block so the within-send request tail stays stable.
        if ctx.round_index == 0 or self._memo_sid != ctx.session_id:
            self._memo_block = await self._recall(ctx)
            self._memo_sid = ctx.session_id
        block = self._memo_block
        if not block:
            return
        msgs = ctx.messages
        # Append fresh COPIES so a later hook mutating an element can't reach the
        # memoized block or the loop's shared dicts.
        if self._position == "front":
            insert_at = 0
            while insert_at < len(msgs) and msgs[insert_at].get("role") == "system":
                insert_at += 1
            msgs[insert_at:insert_at] = [dict(m) for m in block]
        else:
            msgs.extend(dict(m) for m in block)

    async def _recall(self, ctx: Any) -> list[LoopMessage]:
        from power_loop.contracts.hook_contexts import MemoryRecalledCtx
        from power_loop.contracts.hooks import HookDirective, HookPoint

        try:
            recalled = await self._provider.recall(
                messages=ctx.messages, session_id=ctx.session_id, budget_tokens=self._budget,
            )
        except Exception as exc:  # noqa: BLE001 — recall is best-effort; never break the loop
            self._emit_failed(ctx, exc)
            return []
        recalled = list(recalled or [])
        returned = len(recalled)
        if self._hooks is not None and recalled:
            mr = MemoryRecalledCtx(
                recalled=recalled, session_id=ctx.session_id, budget_tokens=self._budget,
            )
            await self._hooks.run_typed_async(HookPoint.MEMORY_RECALLED, mr)
            if mr.directive == HookDirective.SKIP:
                self._emit_recalled(ctx, returned=returned, injected=0)
                return []
            recalled = list(mr.recalled or [])
        tagged = tag_as_memory(recalled)
        self._emit_recalled(ctx, returned=returned, injected=len(tagged))
        return tagged

    def _emit_recalled(self, ctx: Any, *, returned: int, injected: int) -> None:
        self._publish(ctx, "recalled", returned=returned, injected=injected)

    def _emit_failed(self, ctx: Any, exc: Exception) -> None:
        self._publish(ctx, "failed", error_type=type(exc).__name__, error_message=str(exc)[:500])

    def _publish(self, ctx: Any, kind: str, **fields: Any) -> None:
        """Best-effort telemetry on the contextvar event bus (never raises)."""
        try:
            from power_loop.contracts.event_payloads import (
                MemoryFailedPayload,
                MemoryRecalledPayload,
            )
            from power_loop.contracts.events import AgentEvent, AgentEventType
            from power_loop.core.agent_context import get_event_bus

            if kind == "recalled":
                etype = AgentEventType.MEMORY_RECALLED
                data = MemoryRecalledPayload(budget_tokens=self._budget, **fields)
            else:
                etype = AgentEventType.MEMORY_FAILED
                data = MemoryFailedPayload(phase="recall", **fields)
            bus = self._event_bus if self._event_bus is not None else get_event_bus()
            bus.publish(
                AgentEvent(
                    type=etype, data=data,
                    session_id=getattr(ctx, "session_id", None),
                    round_index=getattr(ctx, "round_index", 0),
                )
            )
        except Exception:  # noqa: BLE001 — telemetry must never break the loop
            pass


__all__ = [
    "LoopMessage",
    "MemorySnapshot",
    "MemoryProvider",
    "MemoryRecallHook",
    "tag_as_memory",
]
