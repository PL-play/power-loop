"""Unit tests for M1.9 — MemoryProvider Protocol + pipeline wiring.

Covers the four invariants from the ROADMAP §M1.9 "测试" block:

1. Fake provider injects messages in the correct position (after the
   leading system block, tagged ``role=system, name=memory_*``).
2. ``recall`` raise → empty injection + ``MEMORY_FAILED`` event, loop
   continues normally.
3. ``remember`` raise → does not affect the returned ``StatefulResult``;
   ``MEMORY_FAILED`` event fires with ``phase="remember"``.
4. ``MEMORY_RECALLED`` hook returning ``HookDirective.SKIP`` drops the
   injection entirely; the LLM does not see the recalled messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from power_loop import (
    AgentEventBus,
    AgentEventType,
    AgentHooks,
    AgentLoopConfig,
    HookDirective,
    HookPoint,
    MemorySnapshot,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.contracts.hook_contexts import MemoryRecalledCtx

# ── Fakes ────────────────────────────────────────────────────────────────


@dataclass
class _RecordingLLM:
    """Records the exact ``messages`` list the pipeline submitted."""
    response_text: str = "ok"
    seen: list[list[dict]] = field(default_factory=list)

    async def complete(self, request, **kwargs):
        self.seen.append([dict(m) for m in request.messages])
        return LLMResponse(raw_text=self.response_text)

    async def stream(self, request):
        raise NotImplementedError

    async def close(self):
        pass


@dataclass
class _FakeMemory:
    """Returns canned ``recall`` payload; records ``remember`` snapshots.

    Set ``recall_raises`` / ``remember_raises`` to force failure paths.
    """
    to_recall: list[dict] = field(default_factory=list)
    recall_raises: Exception | None = None
    remember_raises: Exception | None = None
    remembered: list[MemorySnapshot] = field(default_factory=list)
    recall_calls: int = 0

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        self.recall_calls += 1
        if self.recall_raises is not None:
            raise self.recall_raises
        return [dict(m) for m in self.to_recall]

    async def remember(self, *, snapshot, session_id):
        if self.remember_raises is not None:
            raise self.remember_raises
        self.remembered.append(snapshot)


def _new_loop(*, llm, store, bus, memory=None, hooks=None) -> StatefulAgentLoop:
    return StatefulAgentLoop(
        llm=llm, store=store, event_bus=bus, hooks=hooks,
        config=AgentLoopConfig(
            system_prompt="sp", max_rounds=1, compactor=None, memory=memory,
        ),
    )


# ── 1. Injection position + tagging ──────────────────────────────────────


@pytest.mark.asyncio
async def test_recalled_messages_injected_after_leading_system_and_tagged() -> None:
    store = await SessionStore.open(":memory:")
    try:
        llm = _RecordingLLM()
        bus = AgentEventBus()
        events: list = []
        bus.subscribe(None, lambda e: events.append(e))
        mem = _FakeMemory(to_recall=[
            {"content": "user prefers Chinese replies"},
            {"role": "user", "name": "leftover", "content": "user lives in Shanghai"},
        ])
        loop = _new_loop(llm=llm, store=store, bus=bus, memory=mem)

        sid = await loop.new_session()
        await loop.send("hi", session_id=sid)

        assert mem.recall_calls == 1
        sent = llm.seen[0]
        # Find the memory_* messages
        mem_msgs = [m for m in sent if str(m.get("name") or "").startswith("memory_")]
        assert len(mem_msgs) == 2
        assert all(m["role"] == "system" for m in mem_msgs)
        assert "Chinese" in mem_msgs[0]["content"]
        # The user's actual message must come AFTER all memory_* messages.
        first_user_idx = next(i for i, m in enumerate(sent) if m.get("role") == "user")
        last_mem_idx = max(i for i, m in enumerate(sent) if str(m.get("name") or "").startswith("memory_"))
        assert last_mem_idx < first_user_idx

        recall_events = [e for e in events if e.type == AgentEventType.MEMORY_RECALLED]
        assert len(recall_events) == 1
        assert recall_events[0].payload["returned"] == 2
        assert recall_events[0].payload["injected"] == 2
    finally:
        await store.close()


# ── 2. recall raise → soft fail, MEMORY_FAILED event ────────────────────


@pytest.mark.asyncio
async def test_recall_failure_is_soft_and_emits_event() -> None:
    store = await SessionStore.open(":memory:")
    try:
        llm = _RecordingLLM()
        bus = AgentEventBus()
        events: list = []
        bus.subscribe(None, lambda e: events.append(e))
        mem = _FakeMemory(recall_raises=RuntimeError("backend down"))
        loop = _new_loop(llm=llm, store=store, bus=bus, memory=mem)

        sid = await loop.new_session()
        r = await loop.send("hi", session_id=sid)
        assert r.status == "completed"

        # No memory_* messages reached the LLM.
        assert not any(str(m.get("name") or "").startswith("memory_") for m in llm.seen[0])
        failed = [e for e in events if e.type == AgentEventType.MEMORY_FAILED
                  and e.payload.get("phase") == "recall"]
        assert len(failed) == 1
        assert failed[0].payload["error_type"] == "RuntimeError"
    finally:
        await store.close()


# ── 3. remember raise → result unchanged, event fired ───────────────────


@pytest.mark.asyncio
async def test_remember_failure_does_not_affect_result() -> None:
    store = await SessionStore.open(":memory:")
    try:
        llm = _RecordingLLM()
        bus = AgentEventBus()
        events: list = []
        bus.subscribe(None, lambda e: events.append(e))
        mem = _FakeMemory(remember_raises=RuntimeError("disk full"))
        loop = _new_loop(llm=llm, store=store, bus=bus, memory=mem)

        sid = await loop.new_session()
        r = await loop.send("hi", session_id=sid)
        assert r.status == "completed"
        assert r.final_text == "ok"

        failed = [e for e in events if e.type == AgentEventType.MEMORY_FAILED
                  and e.payload.get("phase") == "remember"]
        assert len(failed) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_remember_receives_snapshot_with_history_and_status() -> None:
    store = await SessionStore.open(":memory:")
    try:
        llm = _RecordingLLM(response_text="done")
        bus = AgentEventBus()
        mem = _FakeMemory()
        loop = _new_loop(llm=llm, store=store, bus=bus, memory=mem)

        sid = await loop.new_session()
        await loop.send("hi", session_id=sid)
        assert len(mem.remembered) == 1
        snap = mem.remembered[0]
        assert snap.status == "completed"
        assert snap.final_text == "done"
        assert any(m.get("role") == "user" and m.get("content") == "hi" for m in snap.messages)
        assert snap.rounds >= 1
    finally:
        await store.close()


# ── 4. MEMORY_RECALLED hook SKIP → no injection ─────────────────────────


@pytest.mark.asyncio
async def test_memory_recalled_hook_skip_drops_injection() -> None:
    store = await SessionStore.open(":memory:")
    try:
        llm = _RecordingLLM()
        bus = AgentEventBus()
        events: list = []
        bus.subscribe(None, lambda e: events.append(e))
        hooks = AgentHooks()

        async def skip_all(ctx: MemoryRecalledCtx) -> None:
            ctx.directive = HookDirective.SKIP

        hooks.register(HookPoint.MEMORY_RECALLED, skip_all)

        mem = _FakeMemory(to_recall=[{"content": "secret to redact"}])
        loop = _new_loop(llm=llm, store=store, bus=bus, memory=mem, hooks=hooks)

        sid = await loop.new_session()
        await loop.send("hi", session_id=sid)
        assert not any(str(m.get("name") or "").startswith("memory_") for m in llm.seen[0])
        recall_events = [e for e in events if e.type == AgentEventType.MEMORY_RECALLED]
        assert recall_events[0].payload["returned"] == 1
        assert recall_events[0].payload["injected"] == 0
    finally:
        await store.close()


# ── 5. tag_as_memory utility ─────────────────────────────────────────────


def test_tag_as_memory_is_idempotent_and_non_destructive() -> None:
    from power_loop import tag_as_memory
    original = [{"content": "fact A"}, {"role": "user", "name": "memory_x", "content": "already tagged"}]
    out = tag_as_memory(original)
    # original untouched
    assert original[0] == {"content": "fact A"}
    assert original[1]["role"] == "user"
    # output normalised
    assert all(m["role"] == "system" for m in out)
    assert out[0]["name"].startswith("memory_")
    assert out[1]["name"] == "memory_x"  # already-tagged name preserved
