"""Reconstructable runtime executions and the internal active-window cache.

Resumable, lightweight StatefulAgentLoop + the per-session active-window cache.

The crown-jewel invariant: the cache is a *rebuildable accelerator*, never a source of
truth — so a loop WITH caching feeds the LLM byte-for-byte the same prompts as a loop with
caching OFF, under every regime that mutates the in-memory working history WITHOUT changing
the durable store (memory recall, microcompaction) or that reshuffles the active set
(compaction), plus a session-prompt edit between sends. We assert that equivalence directly by
comparing the recorded LLM requests of a cache-on run and a cache-off run on separate stores.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import (
    AgentLoopConfig,
    RequestOptions,
    RuntimeRequestError,
    SchemaPolicy,
    StoreSchemaError,
)
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.agent.stateful_loop import _ExecutionLoop
from power_loop.runtime.compact import CompactionPlan
from power_loop.runtime.jobs import JobManager
from power_loop.runtime.memory import MemoryProvider
from power_loop.runtime.pricing import Free
from power_loop.runtime.store.store import SessionStore
from power_loop.runtime.task_runtime import TaskRuntime
from power_loop.tools.registry import ToolRegistry
from tests.runtime_fixture import RunningAgent

from .test_job_registry_192 import registry_store

__all__ = ["registry_store"]


pytestmark = pytest.mark.unit


# ── recording fake LLMs ──────────────────────────────────────────────────────


@dataclass
class _SeqLLM(LLMService):
    """Returns scripted responses in order (then 'done'); records a deep snapshot of every
    request (system_prompt + messages) so two runs can be compared prompt-for-prompt."""

    responses: list[LLMResponse] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    _i: int = 0

    async def complete(self, request: LLMRequest, **_: Any) -> LLMResponse:
        self.calls.append(
            {"system_prompt": request.system_prompt, "messages": copy.deepcopy(request.messages)}
        )
        if self._i >= len(self.responses):
            return LLMResponse(raw_text="done")
        r = self.responses[self._i]
        self._i += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


@dataclass
class _FoldLLM(LLMService):
    """Emits ``tool_rounds`` fat tool calls (each tool result keeps tokens above the tiny
    compaction threshold), then a final answer — so one send spans rounds and folds. A
    tool-less request is the compaction summary call → returns a summary. Records requests."""

    tool_rounds: int = 3
    calls: list[dict[str, Any]] = field(default_factory=list)
    _i: int = 0

    async def complete(self, request: LLMRequest, **_: Any) -> LLMResponse:
        self.calls.append(
            {"system_prompt": request.system_prompt, "messages": copy.deepcopy(request.messages)}
        )
        if not request.tools:
            return LLMResponse(raw_text="<summary>folded</summary>")
        i = self._i
        self._i += 1
        if i < self.tool_rounds:
            return LLMResponse(
                raw_text="",
                tool_calls=[{"id": f"tc{i}", "type": "function",
                             "function": {"name": "blob", "arguments": "{}"}}],
            )
        return LLMResponse(raw_text="final answer")

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _tool_resp(call_id: str, name: str, args: str = "{}") -> LLMResponse:
    return LLMResponse(
        raw_text="",
        tool_calls=[{"id": call_id, "type": "function",
                     "function": {"name": name, "arguments": args}}],
    )


def _echo_registry() -> ToolRegistry:
    from power_loop.contracts.tools import ToolDefinition
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="echo", description="echo",
                       input_schema={"type": "object", "properties": {"x": {"type": "string"}}}, pricing=Free("test_fixture")),
        lambda **kw: f"echoed:{kw.get('x', '')}",
    )
    return reg


def _blob_registry() -> ToolRegistry:
    from power_loop.contracts.tools import ToolDefinition
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="blob", description="fat", input_schema={"type": "object", "properties": {}}, pricing=Free("test_fixture")),
        lambda **kw: "T" * 4000,
    )
    return reg


class _FixedMemory(MemoryProvider):
    """Injects one deterministic ``memory_*`` system message on every recall (NEVER persisted —
    exactly the non-durable in-memory injection the cache must not capture)."""

    async def recall(self, *, messages, session_id, budget_tokens=1500):
        return [{"role": "system", "name": "memory_fixed", "content": "REMEMBERED-FACT"}]

    async def remember(self, *, snapshot, session_id=None):
        return None


class _EveryRoundCompactor:
    """Folds the front of history (keeping the last 2) once there's enough — so a single
    multi-round send folds repeatedly (the double-fold regime)."""

    async def maybe_compact(self, messages, *, llm, max_tokens, round_index, context=None):
        if len(messages) < 4:
            return None
        return CompactionPlan(fold_start_idx=0, fold_end_idx=len(messages) - 3,
                              summary_text="folded", before_tokens=100, after_tokens=10)


def _cache_loop(store, *, size=256):
    # Pure cache units do not execute inputs or use a removed public constructor.
    return _ExecutionLoop(llm=_SeqLLM(), store=store, session_cache_size=size, config=AgentLoopConfig(delegation_policy="single_level"),
        task_runtime=TaskRuntime(JobManager(store), tenant_id="test", principal_id="host",
                                 executor_id="fixture", boot_id="boot"))


@pytest.mark.parametrize("mode", ["plain", "recall", "tool", "fold", "microcompact", "fold_once"])
async def test_rebuilt_worker_preserves_prompts_and_history(mode, tmp_path, monkeypatch):
    from power_loop.runtime.env import RuntimeEnv
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "100")
    monkeypatch.setenv("CONTEXT_MICRO_HOT_TAIL", "0")
    monkeypatch.setenv("CONTEXT_MICRO_SIZE_LIMIT", "100")
    results = []
    class OneShot(_EveryRoundCompactor):
        done = False
        async def maybe_compact(self, messages, **kwargs):
            if self.done:
                return None
            result = await super().maybe_compact(messages, **kwargs)
            self.done = result is not None
            return result
    for rebuild in (False, True):
        store = await SessionStore.open(str(tmp_path / f"{rebuild}.db"))
        config = AgentLoopConfig(system_prompt="S", max_rounds=8, compactor=None)
        registry, options = None, None
        if mode in {"fold", "fold_once"}:
            model = _FoldLLM(tool_rounds=3)
            config.compactor = _EveryRoundCompactor() if mode == "fold" else OneShot()
            registry, options = _blob_registry(), RequestOptions(tools=["blob"])
        elif mode in {"tool", "microcompact"}:
            name = "echo" if mode == "tool" else "blob"
            model = _SeqLLM(responses=[_tool_resp("c1", name), LLMResponse(raw_text="d1"),
                                     _tool_resp("c2", name), LLMResponse(raw_text="d2")])
            registry = _echo_registry() if mode == "tool" else _blob_registry()
            options = RequestOptions(tools=[name])
            config.microcompact_enabled = mode == "microcompact"
        else:
            model = _SeqLLM(responses=[LLMResponse(raw_text=f"r{i}") for i in range(3)])
            if mode == "recall":
                config.memory = _FixedMemory()
        # Resource ports are reconstructed by the current factory each time.
        env = RuntimeEnv(workspace_dir=tmp_path, home_dir=tmp_path)
        agent = await RunningAgent(store, model, registry, config, runtime_env=env).start()
        try:
            for i, body in enumerate(("one", "two", "three")):
                if rebuild and i:
                    target = agent.target
                    await agent.aclose()
                    agent = await RunningAgent(store, model, registry, config, runtime_env=env).start(target=target)
                result = await agent.send(body, options=options, message_id=f"input-{i}")
                assert result.status == "completed"
            rows = await store.load_active_messages(agent.target.session_id)
            results.append((model.calls, [(row.role, row.content, row.name) for row in rows]))
            assert not agent.runtime.worker._tasks
        finally:
            await agent.aclose()
            await store.close()
    assert results[0] == results[1]


async def test_fresh_worker_and_other_writer_preserve_full_history(registry_store):
    store, connect = registry_store
    model = _SeqLLM(responses=[LLMResponse(raw_text="a1"), LLMResponse(raw_text="a2")])
    config = AgentLoopConfig(compactor=None)
    first = await RunningAgent(store, model, config=config).start()
    target = first.target
    await first.send("u1", message_id="first")
    await first.aclose()
    other = await connect(schema=SchemaPolicy.VERIFY)
    second = await RunningAgent(other, _SeqLLM(responses=[LLMResponse(raw_text="b1")]), config=config).start(target=target)
    await second.send("u2", message_id="other")
    await second.aclose()
    last = await RunningAgent(store, model, config=config).start(target=target)
    try:
        assert (await last.send("u3", message_id="last")).status == "completed"
        assert [(m["role"], m["content"]) for m in model.calls[-1]["messages"]] == [
            ("user", "u1"), ("assistant", "a1"), ("user", "u2"), ("assistant", "b1"), ("user", "u3")]
    finally:
        await last.aclose()


async def test_unknown_pending_tool_alignment_survives_worker_rebuild(runtime_agent):
    first = await runtime_agent(model=_SeqLLM(), config=AgentLoopConfig(compactor=None))
    store, target = first.runtime.store, first.target
    await store.append_message(target.session_id, role="user", content="prior")
    await store.append_message(target.session_id, role="assistant", content="", tool_calls=[{
        "id":"old-call", "type":"function", "function":{"name":"echo", "arguments":"{}"}}])
    await first.aclose()
    called = []
    registry = _echo_registry()
    registry.register(registry.get("echo").definition, lambda **kw: called.append(kw), overwrite=True)
    model = _SeqLLM(responses=[LLMResponse(raw_text="continued")])
    fresh = await runtime_agent(model=model, store=store, registry=registry,
                                config=AgentLoopConfig(compactor=None), target=target)
    assert (await fresh.send("new input")).final_text == "continued"
    assert not called
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "old-call" and
               "结果未知" in m["content"] for m in model.calls[0]["messages"])


async def test_appended_input_is_visible_after_worker_rebuild(runtime_agent):
    import asyncio

    from power_loop.agent.runtime_store import now_ms
    entered, release = asyncio.Event(), asyncio.Event()
    class Blocked(_SeqLLM):
        async def complete(self, request, **kwargs):
            if not self.calls:
                entered.set()
                await release.wait()
            return await super().complete(request, **kwargs)
    model = Blocked(responses=[LLMResponse(raw_text="r1"), LLMResponse(raw_text="r2")])
    agent = await runtime_agent(model=model, config=AgentLoopConfig(compactor=None))
    task = asyncio.create_task(agent.send("first"))
    await asyncio.wait_for(entered.wait(), 5)
    appended = await agent.runtime.client.append(agent.target, "steer", message_id="steer", actor=agent.actor)
    release.set()
    await task
    await agent.runtime.client.wait(appended.input_ref, actor=agent.actor, deadline_ms=now_ms()+10000)
    await agent.drain()
    await agent.aclose()
    next_model = _SeqLLM(responses=[LLMResponse(raw_text="done")])
    fresh = await runtime_agent(model=next_model, store=agent.runtime.store, target=agent.target,
                                config=AgentLoopConfig(compactor=None))
    await fresh.send("second", message_id="second")
    assert any(m["role"] == "user" and "steer" in m["content"] for m in next_model.calls[0]["messages"])


@pytest.mark.parametrize("size", [0, 2])
async def test_internal_cache_remains_bounded_and_prewarm_handles_missing_session(tmp_path, size):
    store = await SessionStore.open(str(tmp_path / "cache.db"))
    loop = _cache_loop(store, size=size)
    try:
        assert not await loop.prewarm("missing")
        for i in range(4):
            sid = await store.create_session()
            await store.append_message(sid, role="user", content=str(i))
            await loop.prewarm(sid)
        assert loop.cache_stats["entries"] <= size
        if size:
            assert loop.cache_stats["evictions"] >= 2
        else:
            assert loop.cache_stats == {"hits": 0, "misses": 0, "evictions": 0, "entries": 0}
    finally:
        await loop.aclose()
        await store.close()


async def test_internal_cache_invalidates_folded_rows(tmp_path):
    store = await SessionStore.open(str(tmp_path / "cache.db"))
    loop = _cache_loop(store)
    try:
        sid = await store.create_session()
        for i in range(4):
            await store.append_message(sid, role="user", content=f"m{i}")
        assert await loop.prewarm(sid)
        state = await store.get_state(sid)
        assert loop._cache_get(sid, state.next_seq, state.last_compact_seq) is not None
        assert loop._cache_get(sid, state.next_seq, state.last_compact_seq+1) is None
        await loop.prewarm(sid)
        await store.record_compaction(sid, from_seq=2, to_seq=3, note_content="sum", before_tokens=9,
            after_tokens=1, round_index=0, fold_seqs=[2,3], order_key=2)
        await loop._refresh_window_cache_after_send(sid, store)
        assert sid not in loop._session_cache
        active = [m.content for m in await store.load_active_messages(sid)]
        assert "m1" not in active and "m2" not in active and "sum" in active
    finally:
        await loop.aclose()
        await store.close()


async def test_store_schema_verify_and_reopen_through_current_runtime(tmp_path):
    path = str(tmp_path / "store.db")
    with pytest.raises(StoreSchemaError) as error:
        await SessionStore.open(path, schema=SchemaPolicy.VERIFY)
    assert "not initialized" in str(error.value) and error.value.ddl
    store = await SessionStore.open(path)
    agent = await RunningAgent(store, _SeqLLM(responses=[LLMResponse(raw_text="hi")])).start()
    target = agent.target
    await agent.send("hello", message_id="first")
    await agent.aclose()
    await store.close()
    reopened = await SessionStore.open(path, schema=SchemaPolicy.VERIFY)
    agent = await RunningAgent(reopened, _SeqLLM(responses=[LLMResponse(raw_text="again")])).start(target=target)
    try:
        assert [m.content for m in await reopened.load_active_messages(target.session_id)] == ["hello", "hi"]
        assert (await agent.send("next", message_id="second")).status == "completed"
    finally:
        await agent.aclose()
        await reopened.close()


async def test_store_table_prefix_isolation_and_wrong_store_target(tmp_path):
    path = str(tmp_path / "namespaces.db")
    a, b = await SessionStore.open(path, table_prefix="alpha_"), await SessionStore.open(path, table_prefix="beta_")
    first = await RunningAgent(a, _SeqLLM()).start()
    target = first.target
    await first.send("alpha")
    await first.aclose()
    second = await RunningAgent(b, _SeqLLM()).start()
    try:
        with pytest.raises(RuntimeRequestError) as error:
            await second.runtime.client.send(target, "wrong store", message_id="wrong", actor=second.actor)
        assert error.value.code == "wrong_store"
        assert await b.get_session(target.session_id) is None
    finally:
        await second.aclose()
        await a.close()
        await b.close()
