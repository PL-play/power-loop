"""Unit tests for the built-in MemoryRecallHook + the AgentHooks name/replace/remove API.

These exercise the hook in isolation (a fake LlmBefore-like ctx) so we can assert
position, once-per-send memoization, and ephemerality without a full loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from power_loop import AgentHooks, AgentLoopConfig, HookPoint, MemoryRecallHook


@dataclass
class _FakeProvider:
    to_recall: list[dict] = field(default_factory=list)
    calls: int = 0

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        self.calls += 1
        return [dict(m) for m in self.to_recall]

    async def remember(self, *, snapshot, session_id):  # pragma: no cover - unused
        return None


def _ctx(round_index: int, session_id: str, messages: list[dict]):
    # Mirrors the fields the real LlmBeforeCtx exposes to the hook.
    return SimpleNamespace(round_index=round_index, session_id=session_id, messages=messages)


# ── position ──────────────────────────────────────────────────────────────


async def test_tail_position_appends_after_history() -> None:
    hook = MemoryRecallHook(_FakeProvider(to_recall=[{"content": "fact"}]), position="tail")
    msgs = [{"role": "system", "content": "sp"}, {"role": "user", "content": "hi"}]
    await hook(_ctx(0, "s1", msgs))
    assert msgs[-1]["name"].startswith("memory_") and msgs[-1]["content"] == "fact"
    assert [m.get("role") for m in msgs] == ["system", "user", "system"]


async def test_front_position_inserts_after_leading_system() -> None:
    hook = MemoryRecallHook(_FakeProvider(to_recall=[{"content": "fact"}]), position="front")
    msgs = [{"role": "system", "content": "sp"}, {"role": "user", "content": "hi"}]
    await hook(_ctx(0, "s1", msgs))
    # after the leading system block, before the user message
    assert msgs[1]["name"].startswith("memory_")
    assert [m.get("role") for m in msgs] == ["system", "system", "user"]


# ── once-per-send memoization ───────────────────────────────────────────────


async def test_recall_runs_once_per_send_memoized_across_rounds() -> None:
    prov = _FakeProvider(to_recall=[{"content": "fact"}])
    hook = MemoryRecallHook(prov)
    # round 0 recomputes; subsequent rounds reuse the memoized block
    for r in range(4):
        msgs = [{"role": "user", "content": "hi"}]
        await hook(_ctx(r, "s1", msgs))
        assert msgs[-1]["content"] == "fact"  # injected every round
    assert prov.calls == 1  # but recalled only once


async def test_new_send_round0_recomputes() -> None:
    prov = _FakeProvider(to_recall=[{"content": "fact"}])
    hook = MemoryRecallHook(prov)
    await hook(_ctx(0, "s1", [{"role": "user", "content": "a"}]))
    await hook(_ctx(1, "s1", [{"role": "user", "content": "a"}]))
    await hook(_ctx(0, "s1", [{"role": "user", "content": "b"}]))  # next send → round 0
    assert prov.calls == 2


async def test_session_change_recomputes() -> None:
    prov = _FakeProvider(to_recall=[{"content": "fact"}])
    hook = MemoryRecallHook(prov)
    await hook(_ctx(0, "s1", [{"role": "user", "content": "a"}]))
    await hook(_ctx(1, "s2", [{"role": "user", "content": "a"}]))  # different session, round>0
    assert prov.calls == 2


async def test_concurrent_sessions_do_not_evict_each_others_memo() -> None:
    # memory-hooks-2: a SHARED hook interleaving sessions must memoize each independently. Pre-fix
    # the single scalar memo was evicted on every session switch → redundant provider.recall calls.
    prov = _FakeProvider(to_recall=[{"content": "fact"}])
    hook = MemoryRecallHook(prov)
    await hook(_ctx(0, "s1", [{"role": "user", "content": "a"}]))  # send s1 → recall
    await hook(_ctx(0, "s2", [{"role": "user", "content": "b"}]))  # send s2 → recall
    assert prov.calls == 2
    # Interleaved later rounds reuse each session's own memo — no extra recalls (pre-fix: > 2).
    await hook(_ctx(1, "s1", [{"role": "user", "content": "a"}]))
    await hook(_ctx(1, "s2", [{"role": "user", "content": "b"}]))
    await hook(_ctx(2, "s1", [{"role": "user", "content": "a"}]))
    assert prov.calls == 2


async def test_empty_recall_injects_nothing() -> None:
    hook = MemoryRecallHook(_FakeProvider(to_recall=[]))
    msgs = [{"role": "user", "content": "hi"}]
    await hook(_ctx(0, "s1", msgs))
    assert msgs == [{"role": "user", "content": "hi"}]


async def test_injected_dicts_are_copies_not_shared() -> None:
    hook = MemoryRecallHook(_FakeProvider(to_recall=[{"content": "fact"}]))
    msgs = [{"role": "user", "content": "hi"}]
    await hook(_ctx(0, "s1", msgs))
    injected = msgs[-1]
    injected["content"] = "MUTATED"  # mutating the injected copy...
    msgs2 = [{"role": "user", "content": "hi"}]
    await hook(_ctx(1, "s1", msgs2))  # ...must not corrupt the memoized block
    assert msgs2[-1]["content"] == "fact"


# ── AgentHooks name / replace / remove / has ───────────────────────────────


def test_hooks_register_replace_remove_has() -> None:
    hooks = AgentHooks()

    def h1(ctx):  # noqa: ANN001
        return None

    def h2(ctx):  # noqa: ANN001
        return None

    hooks.register(HookPoint.LLM_BEFORE, h1, name="x")
    assert hooks.has("x")
    assert hooks.has("x", HookPoint.LLM_BEFORE)
    # replace by name → still exactly one entry of that name
    hooks.replace(HookPoint.LLM_BEFORE, h2, name="x")
    entries = hooks._handlers[str(HookPoint.LLM_BEFORE)]
    assert len([e for e in entries if e.name == "x"]) == 1
    assert entries[-1].handler is h2
    # remove
    assert hooks.remove("x") == 1
    assert not hooks.has("x")
    assert hooks.remove("x") == 0  # idempotent


def test_register_replace_false_keeps_both() -> None:
    hooks = AgentHooks()
    hooks.register(HookPoint.LLM_BEFORE, lambda c: None, name="x")
    hooks.register(HookPoint.LLM_BEFORE, lambda c: None, name="x")  # replace defaults False
    entries = hooks._handlers[str(HookPoint.LLM_BEFORE)]
    assert len([e for e in entries if e.name == "x"]) == 2


# ── effective_context_budget headroom ───────────────────────────────────────


def test_effective_context_budget_reserves_memory() -> None:
    prov = _FakeProvider()
    cfg = AgentLoopConfig(max_tokens=8000, memory=prov, memory_budget_tokens=1500)
    assert cfg.effective_context_budget() == 6500
    # no memory → unchanged
    cfg2 = AgentLoopConfig(max_tokens=8000)
    assert cfg2.effective_context_budget() == 8000
    # reserve never goes below 1 even if budget < memory_budget
    cfg3 = AgentLoopConfig(max_tokens=1000, memory=prov, memory_budget_tokens=5000)
    assert cfg3.effective_context_budget() == 1


# ── auto-registration + override/disable through a real loop ────────────────


@dataclass
class _RecLLM:
    seen: list[list[dict]] = field(default_factory=list)

    async def complete(self, request, **kwargs):
        from power_loop._vendor.llm_client.interface import LLMResponse

        self.seen.append([dict(m) for m in request.messages])
        return LLMResponse(raw_text="ok")

    async def stream(self, request):  # pragma: no cover
        raise NotImplementedError

    async def close(self):  # pragma: no cover
        pass


async def _run(loop, text="hi"):
    sid = await loop.new_session()
    await loop.send(text, session_id=sid)


@pytest.mark.asyncio
async def test_builtin_hook_auto_registered_when_memory_set() -> None:
    from power_loop import SessionStore, StatefulAgentLoop

    store = await SessionStore.open(":memory:")
    try:
        llm = _RecLLM()
        loop = StatefulAgentLoop(
            llm=llm, store=store,
            config=AgentLoopConfig(
                max_rounds=1, compactor=None,
                memory=_FakeProvider(to_recall=[{"content": "fact"}]),
            ),
        )
        assert loop.hooks.has(MemoryRecallHook.NAME)
        await _run(loop)
        assert any(str(m.get("name") or "").startswith("memory_") for m in llm.seen[0])
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_host_can_disable_builtin_hook_via_remove() -> None:
    from power_loop import SessionStore, StatefulAgentLoop

    store = await SessionStore.open(":memory:")
    try:
        llm = _RecLLM()
        loop = StatefulAgentLoop(
            llm=llm, store=store,
            config=AgentLoopConfig(
                max_rounds=1, compactor=None,
                memory=_FakeProvider(to_recall=[{"content": "fact"}]),
            ),
        )
        loop.hooks.remove(MemoryRecallHook.NAME)
        await _run(loop)
        assert not any(str(m.get("name") or "").startswith("memory_") for m in llm.seen[0])
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_builtin_memory_hook_flag_off_skips_registration() -> None:
    from power_loop import SessionStore, StatefulAgentLoop

    store = await SessionStore.open(":memory:")
    try:
        loop = StatefulAgentLoop(
            llm=_RecLLM(), store=store,
            config=AgentLoopConfig(
                max_rounds=1, compactor=None, builtin_memory_hook=False,
                memory=_FakeProvider(to_recall=[{"content": "fact"}]),
            ),
        )
        assert not loop.hooks.has(MemoryRecallHook.NAME)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_builtin_memory_hook_runs_after_host_llm_before_hook() -> None:
    """P1: order=100 makes the builtin memory hook run AFTER a host's default-order
    LLM_BEFORE hook, so memory lands at the TRUE tail (after host-appended content)."""
    from power_loop import HookPoint, SessionStore, StatefulAgentLoop

    store = await SessionStore.open(":memory:")
    try:
        llm = _RecLLM()
        loop = StatefulAgentLoop(
            llm=llm, store=store,
            config=AgentLoopConfig(
                max_rounds=1, compactor=None,
                memory=_FakeProvider(to_recall=[{"content": "the-note"}]),
            ),
        )

        def host_hook(ctx):  # default order 0 → runs before the builtin (order 100)
            ctx.messages.append({"role": "system", "name": "host_sentinel", "content": "HOST"})

        loop.hooks.register(HookPoint.LLM_BEFORE, host_hook, name="host.sentinel")
        await _run(loop)

        sent = llm.seen[0]
        host_idx = next(i for i, m in enumerate(sent) if m.get("name") == "host_sentinel")
        mem_idx = next(i for i, m in enumerate(sent) if str(m.get("name") or "").startswith("memory_"))
        assert host_idx < mem_idx  # memory appended AFTER the host hook's content
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_host_can_preregister_under_builtin_name_and_loop_does_not_clobber() -> None:
    """P1: if the host already registered a handler under MemoryRecallHook.NAME, the
    loop's has()-guard must NOT overwrite it — the host's override wins."""
    from power_loop import AgentHooks, HookPoint, SessionStore, StatefulAgentLoop

    store = await SessionStore.open(":memory:")
    try:
        hooks = AgentHooks()

        def host_override(ctx):
            ctx.messages.append({"role": "system", "name": "memory_host_override", "content": "HOST-MEM"})

        hooks.register(HookPoint.LLM_BEFORE, host_override, name=MemoryRecallHook.NAME)

        llm = _RecLLM()
        loop = StatefulAgentLoop(
            llm=llm, store=store, hooks=hooks,
            config=AgentLoopConfig(
                max_rounds=1, compactor=None,
                memory=_FakeProvider(to_recall=[{"content": "builtin-note"}]),
            ),
        )
        # exactly one handler under the name (loop didn't add the builtin alongside)
        entries = [e for e in loop.hooks._handlers[str(HookPoint.LLM_BEFORE)]
                   if e.name == MemoryRecallHook.NAME]
        assert len(entries) == 1 and entries[0].handler is host_override

        await _run(loop)
        sent = llm.seen[0]
        assert any(m.get("name") == "memory_host_override" for m in sent)  # host's ran
        assert not any(m.get("content") == "builtin-note" for m in sent)  # builtin did NOT
    finally:
        await store.close()
