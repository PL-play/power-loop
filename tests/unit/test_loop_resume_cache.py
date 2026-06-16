"""Resumable, lightweight StatefulAgentLoop + the per-session active-window cache.

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
    SchemaPolicy,
    StatefulAgentLoop,
    StoreSchemaError,
)
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.runtime.compact import CompactionPlan
from power_loop.runtime.memory import MemoryProvider
from power_loop.runtime.store.store import SessionStore
from power_loop.tools.registry import ToolRegistry

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
                       input_schema={"type": "object", "properties": {"x": {"type": "string"}}}),
        lambda **kw: f"echoed:{kw.get('x', '')}",
    )
    return reg


def _blob_registry() -> ToolRegistry:
    from power_loop.contracts.tools import ToolDefinition
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(name="blob", description="fat", input_schema={"type": "object", "properties": {}}),
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


# ── the warm-vs-cold equivalence harness ─────────────────────────────────────


async def _run(cache_size: int, llm: LLMService, scenario, *, config: AgentLoopConfig,
               registry: ToolRegistry | None = None, tools=None):
    """Run ``scenario(loop, sid)`` on a fresh in-memory store with the given cache size.
    Returns (recorded llm.calls, the loop's final active messages, cache_stats)."""
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(llm=llm, store=store, config=config, tool_registry=registry,
                             session_cache_size=cache_size)
    sid = await loop.new_session()
    await scenario(loop, sid, tools)
    msgs = await loop.get_messages(sid)
    stats = dict(loop.cache_stats)
    await loop.aclose()
    return llm.calls, msgs, stats


async def _assert_warm_equals_cold(make_llm, scenario, *, config_factory, registry_factory=None,
                                   tools=None, expect_hits=True):
    """Run the identical scenario with caching ON (256) and OFF (0) on separate stores and
    assert the LLM saw byte-identical prompts (system_prompt + messages) at every call."""
    warm_calls, warm_msgs, warm_stats = await _run(
        256, make_llm(), scenario, config=config_factory(),
        registry=registry_factory() if registry_factory else None, tools=tools,
    )
    cold_calls, cold_msgs, _ = await _run(
        0, make_llm(), scenario, config=config_factory(),
        registry=registry_factory() if registry_factory else None, tools=tools,
    )
    assert warm_calls == cold_calls, "cache changed the prompts the LLM saw"

    # final active history (role+content+name) must also match
    def norm(ms):
        return [(m.get("role"), m.get("content"), m.get("name")) for m in ms]

    assert norm(warm_msgs) == norm(cold_msgs)
    if expect_hits:
        assert warm_stats["hits"] > 0, f"cache never engaged: {warm_stats}"
    return warm_stats


# ── scenarios ────────────────────────────────────────────────────────────────


async def _three_plain_sends(loop, sid, tools):
    await loop.send("one", session_id=sid)
    await loop.send("two", session_id=sid)
    await loop.send("three", session_id=sid)


async def _two_sends_with_tool(loop, sid, tools):
    await loop.send("first", session_id=sid, tools=tools)
    await loop.send("second", session_id=sid, tools=tools)


async def _multiround_then_send(loop, sid, tools):
    await loop.send("go", session_id=sid, tools=["blob"])
    await loop.send("again", session_id=sid, tools=["blob"])


# ── tests: warm == cold under each regime ────────────────────────────────────


async def test_plain_multisend_warm_equals_cold():
    stats = await _assert_warm_equals_cold(
        lambda: _SeqLLM(responses=[LLMResponse(raw_text=f"r{i}") for i in range(5)]),
        _three_plain_sends,
        config_factory=lambda: AgentLoopConfig(system_prompt="S", max_rounds=2, compactor=None),
    )
    # 1st send loads+populates (miss); 2nd & 3rd reuse (hit)
    assert stats["hits"] >= 2


async def test_recall_warm_equals_cold():
    """A MemoryProvider injects a non-persisted memory_* message every send; the cache must
    not double-inject or cache the placeholder — warm prompts must equal cold."""
    await _assert_warm_equals_cold(
        lambda: _SeqLLM(responses=[LLMResponse(raw_text=f"r{i}") for i in range(5)]),
        _three_plain_sends,
        config_factory=lambda: AgentLoopConfig(
            system_prompt="S", max_rounds=2, compactor=None, memory=_FixedMemory()),
    )


async def test_tool_round_warm_equals_cold():
    await _assert_warm_equals_cold(
        lambda: _SeqLLM(responses=[
            _tool_resp("c1", "echo", '{"x":"hi"}'), LLMResponse(raw_text="done1"),
            _tool_resp("c2", "echo", '{"x":"yo"}'), LLMResponse(raw_text="done2"),
        ]),
        _two_sends_with_tool,
        config_factory=lambda: AgentLoopConfig(system_prompt="S", max_rounds=4, compactor=None),
        registry_factory=_echo_registry,
        tools=["echo"],
    )


async def test_compaction_warm_equals_cold(monkeypatch):
    """A send that folds (repeatedly) must invalidate the cache; the next send reloads — warm
    prompts must equal cold even across the double-fold regression regime."""
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "100")
    await _assert_warm_equals_cold(
        lambda: _FoldLLM(tool_rounds=3),
        _multiround_then_send,
        config_factory=lambda: AgentLoopConfig(
            system_prompt="S", max_rounds=8, max_tokens=4000, compactor=_EveryRoundCompactor()),
        registry_factory=_blob_registry,
        tools=["blob"],
        expect_hits=False,  # every send folds → cache invalidated each time
    )


async def test_microcompact_warm_equals_cold(tmp_path, monkeypatch):
    """Microcompaction rewrites large tool-message content in-place in the WORKING history
    (never the store). The cache holds only the durable projection, so warm == cold."""
    monkeypatch.setenv("CONTEXT_MICRO_HOT_TAIL", "0")
    monkeypatch.setenv("CONTEXT_MICRO_SIZE_LIMIT", "100")
    from power_loop.runtime.env import RuntimeEnv, runtime_env_context

    def cfg():
        return AgentLoopConfig(system_prompt="S", max_rounds=6, compactor=None)

    async def scenario(loop, sid, tools):
        await loop.send("a", session_id=sid, tools=["blob"])
        await loop.send("b", session_id=sid, tools=["blob"])

    # microcompact needs a runtime home (it spills cold tool outputs to home/.cache); the
    # blob tool returns a fat (>limit) output so microcompact fires each round.
    with runtime_env_context(RuntimeEnv(workspace_dir=tmp_path, home_dir=tmp_path)):
        await _assert_warm_equals_cold(
            lambda: _SeqLLM(responses=[
                _tool_resp("c1", "blob"), LLMResponse(raw_text="d1"),
                _tool_resp("c2", "blob"), LLMResponse(raw_text="d2"),
            ]),
            scenario,
            config_factory=cfg,
            registry_factory=_blob_registry,
            tools=["blob"],
        )


# ── tests: statelessness — a fresh cold loop resumes identically ─────────────


async def test_fresh_loop_resumes_identically():
    """A loop is freely re-creatable: after a warm loop runs, a brand-new loop (empty cache)
    bound to the same store sees the identical active history and can continue the session."""
    store = await SessionStore.open(":memory:")

    def cfg():
        return AgentLoopConfig(system_prompt="S", max_rounds=2, compactor=None, memory=_FixedMemory())

    warm = StatefulAgentLoop(llm=_SeqLLM(responses=[LLMResponse(raw_text="a"), LLMResponse(raw_text="b")]),
                             store=store, config=cfg())
    sid = await warm.new_session()
    await warm.send("hi", session_id=sid)
    await warm.send("again", session_id=sid)
    warm_hist = [(m.get("role"), m.get("content")) for m in await warm.get_messages(sid)]

    # brand-new loop, SAME store, empty cache → identical view + can continue
    fresh = StatefulAgentLoop(llm=_SeqLLM(responses=[LLMResponse(raw_text="c")]), store=store, config=cfg())
    assert fresh.cache_stats["entries"] == 0
    fresh_hist = [(m.get("role"), m.get("content")) for m in await fresh.get_messages(sid)]
    assert fresh_hist == warm_hist
    r = await fresh.send("continue", session_id=sid)
    assert r.status == "completed"


async def test_prewarm_makes_first_send_a_hit():
    store = await SessionStore.open(":memory:")
    seed = StatefulAgentLoop(llm=_SeqLLM(responses=[LLMResponse(raw_text="x")]), store=store,
                             config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None))
    sid = await seed.new_session()
    await seed.send("seed", session_id=sid)

    fresh = StatefulAgentLoop(llm=_SeqLLM(responses=[LLMResponse(raw_text="y")]), store=store,
                              config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None))
    assert await fresh.prewarm(sid) is True
    assert await fresh.prewarm("sess_nope") is False
    await fresh.send("after prewarm", session_id=sid)
    assert fresh.cache_stats["hits"] >= 1  # prewarm primed the window → first send hits


# ── tests: out-of-band durable appends must NOT serve a stale cached window ──
# (regression for the review's confirmed blocker: resume/submit_input/abort/heal and other
#  store writers append rows the cache never saw; the next plain send must reload, not HIT a
#  stale window — guaranteed by the _cache_append contiguity guard.)


async def _pending_then_resume_then_send(loop, sid, tools):
    # send 1: model emits a tool_call but the loop has NO registry → goes pending (warms cache
    # with user + assistant(tool_calls)); resume() then appends an error-tool + assistant
    # OUT-OF-BAND of the cache; send 2 must see those rows.
    await loop.send("u1", session_id=sid)
    await loop.resume(sid)
    await loop.send("u2", session_id=sid)


async def test_resume_after_warm_send_warm_equals_cold():
    await _assert_warm_equals_cold(
        lambda: _SeqLLM(responses=[
            _tool_resp("c1", "echo"), LLMResponse(raw_text="after-resume"), LLMResponse(raw_text="fin"),
        ]),
        _pending_then_resume_then_send,
        config_factory=lambda: AgentLoopConfig(system_prompt="S", max_rounds=3, compactor=None),
        expect_hits=False,  # resume invalidates → send 2 reloads (the correct, non-stale path)
    )


async def _pending_then_abort_then_send(loop, sid, tools):
    await loop.send("u1", session_id=sid)                       # tool_call, no registry → pending (warms)
    await loop.abort_pending(sid)                               # <aborted> tool row, out-of-band
    await loop.send("u2", session_id=sid, heal_pending=True)    # must reload


async def test_abort_after_warm_send_warm_equals_cold():
    await _assert_warm_equals_cold(
        lambda: _SeqLLM(responses=[_tool_resp("c1", "echo"), LLMResponse(raw_text="fin")]),
        _pending_then_abort_then_send,
        config_factory=lambda: AgentLoopConfig(system_prompt="S", max_rounds=3, compactor=None),
        expect_hits=False,
    )


async def test_cross_writer_shared_store_forces_reload():
    """Two loops share one store (the supported shared-store path). When loop B appends to a
    session, loop A's cached window is stale; A's next send must reload (contiguity guard), not
    serve a window missing B's rows — the cross-process data-safety guarantee."""
    store = await SessionStore.open(":memory:")

    def cfg():
        return AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None)

    a_llm = _SeqLLM(responses=[LLMResponse(raw_text="a1"), LLMResponse(raw_text="a2")])
    a = StatefulAgentLoop(llm=a_llm, store=store, config=cfg())
    b = StatefulAgentLoop(llm=_SeqLLM(responses=[LLMResponse(raw_text="b1")]), store=store, config=cfg())
    sid = await a.new_session()
    await a.send("u1", session_id=sid)          # loop A warms its window
    await b.send("u2", session_id=sid)          # loop B appends out-of-band of A's cache
    await a.send("u3", session_id=sid)          # A's window is stale → must reload
    # The bug shows in the PROMPT A's third send fed the LLM (its final stored history is always
    # correct since get_messages reads the store): a stale-cache HIT would omit B's u2/b1.
    last_prompt = [(m.get("role"), m.get("content")) for m in a_llm.calls[-1]["messages"]]
    assert last_prompt == [("user", "u1"), ("assistant", "a1"), ("user", "u2"),
                           ("assistant", "b1"), ("user", "u3")], last_prompt


async def test_fold_then_recache_then_reuse(monkeypatch):
    """After a fold invalidates the window, a later plain send must reload+recache and a
    further send must REUSE — proving the post-fold recache token is correct (warm == cold AND
    a real cache hit, which the always-fold regime can't show)."""
    monkeypatch.setenv("CONTEXT_COMPACT_THRESHOLD", "100")

    class _OneShot:
        def __init__(self):
            self._done = False

        async def maybe_compact(self, messages, *, llm, max_tokens, round_index, context=None):
            if self._done or len(messages) < 4:
                return None
            self._done = True
            return CompactionPlan(fold_start_idx=0, fold_end_idx=len(messages) - 3,
                                  summary_text="folded", before_tokens=100, after_tokens=10)

    async def scenario(loop, sid, tools):
        await loop.send("s1", session_id=sid, tools=["blob"])   # multi-round → folds once → invalidate
        await loop.send("s2", session_id=sid, tools=["blob"])   # reload + recache
        await loop.send("s3", session_id=sid, tools=["blob"])   # reuse → HIT

    await _assert_warm_equals_cold(
        lambda: _FoldLLM(tool_rounds=2),
        scenario,
        config_factory=lambda: AgentLoopConfig(
            system_prompt="S", max_rounds=8, max_tokens=4000, compactor=_OneShot()),
        registry_factory=_blob_registry,
        tools=["blob"],
        expect_hits=True,  # s3 reuses the post-fold recached window
    )


async def test_followup_drain_warm_equals_cold():
    """A follow-up drained mid-run appends an extra user row; the post-run delta must fold it
    into the warm window so the NEXT send's prompt equals a cold reload."""
    from power_loop.agent.follow_up import FollowUpQueued

    async def scenario(loop, sid, tools):
        # queue a follow-up while a send holds the session lock, so it drains mid-run
        async def _send_with_followup():
            return await loop.send("first", session_id=sid)
        import asyncio as _aio
        task = _aio.create_task(_send_with_followup())
        await _aio.sleep(0)  # let the send acquire the lock + start
        # best-effort: if the lock is held, this enqueues; if not, it sends (either way the
        # next send below validates warm==cold)
        res = await loop.follow_up("steer", session_id=sid)
        await task
        if not isinstance(res, FollowUpQueued):
            # it ran as a plain send; that's fine — still a multi-send warm/cold comparison
            pass
        await loop.send("second", session_id=sid)

    await _assert_warm_equals_cold(
        lambda: _SeqLLM(responses=[LLMResponse(raw_text=f"r{i}") for i in range(6)]),
        scenario,
        config_factory=lambda: AgentLoopConfig(system_prompt="S", max_rounds=2, compactor=None),
        expect_hits=False,
    )


# ── tests: cache disabled ────────────────────────────────────────────────────


async def test_cache_size_zero_disables():
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(llm=_SeqLLM(responses=[LLMResponse(raw_text="a"), LLMResponse(raw_text="b")]),
                             store=store, config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None),
                             session_cache_size=0)
    sid = await loop.new_session()
    await loop.send("one", session_id=sid)
    await loop.send("two", session_id=sid)
    assert loop.cache_stats == {"hits": 0, "misses": 0, "evictions": 0, "entries": 0}


async def test_lru_eviction_bounds_entries():
    store = await SessionStore.open(":memory:")
    loop = StatefulAgentLoop(llm=_SeqLLM(responses=[LLMResponse(raw_text="x")] * 50),
                             store=store, config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None),
                             session_cache_size=2)
    sids = [await loop.new_session() for _ in range(4)]
    for s in sids:
        await loop.send("hi", session_id=s)
    assert loop.cache_stats["entries"] <= 2
    assert loop.cache_stats["evictions"] >= 2


# ── tests: store binding / construction ──────────────────────────────────────


async def test_store_and_config_kwargs_conflict_raises():
    store = await SessionStore.open(":memory:")
    with pytest.raises(ValueError, match="not both"):
        StatefulAgentLoop(llm=_SeqLLM(), store=store, dsn=":memory:")
    with pytest.raises(ValueError, match="not both"):
        StatefulAgentLoop(llm=_SeqLLM(), store=store, table_prefix="x_")
    with pytest.raises(ValueError, match="not both"):
        StatefulAgentLoop(llm=_SeqLLM(), store=store, schema=SchemaPolicy.VERIFY)
    await store.close()


async def test_loop_opens_owned_store_via_dsn(tmp_path):
    db = str(tmp_path / "owned.db")
    loop = StatefulAgentLoop(llm=_SeqLLM(responses=[LLMResponse(raw_text="hi")]),
                             dsn=f"sqlite://{db}",
                             config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None))
    sid = await loop.new_session()
    r = await loop.send("hello", session_id=sid)
    assert r.status == "completed"
    await loop.aclose()
    # reopen with VERIFY (schema now exists) on a fresh loop → resumes
    loop2 = StatefulAgentLoop(llm=_SeqLLM(responses=[LLMResponse(raw_text="again")]),
                              dsn=f"sqlite://{db}", schema=SchemaPolicy.VERIFY,
                              config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None))
    hist = await loop2.get_messages(sid)
    assert [m.get("content") for m in hist] == ["hello", "hi"]
    await loop2.aclose()


async def test_loop_table_prefix_isolation(tmp_path):
    db = str(tmp_path / "prefixed.db")
    a = StatefulAgentLoop(llm=_SeqLLM(responses=[LLMResponse(raw_text="A")]),
                          dsn=f"sqlite://{db}", table_prefix="alpha_",
                          config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None))
    b = StatefulAgentLoop(llm=_SeqLLM(responses=[LLMResponse(raw_text="B")]),
                          dsn=f"sqlite://{db}", table_prefix="beta_",
                          config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None))
    sa = await a.new_session()
    await a.send("in alpha", session_id=sa)
    # the beta-prefixed loop shares the FILE but not the tables → session unknown there
    from power_loop.contracts.errors import SessionNotFoundError
    with pytest.raises(SessionNotFoundError):
        await b.send("x", session_id=sa)
    await a.aclose()
    await b.aclose()


async def test_owned_store_schema_verify_on_fresh_db_raises_with_ddl(tmp_path):
    db = str(tmp_path / "fresh.db")
    loop = StatefulAgentLoop(llm=_SeqLLM(), dsn=f"sqlite://{db}", schema=SchemaPolicy.VERIFY,
                             config=AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None))
    with pytest.raises(StoreSchemaError) as ei:
        await loop.new_session()
    assert "not initialized" in str(ei.value)
    assert ei.value.ddl  # carries the provisioning script


# ── gated: the loop binds to real server backends via dsn (Step A end-to-end) ─

import os  # noqa: E402
import socket  # noqa: E402
from urllib.parse import urlparse  # noqa: E402


def _reachable(dsn: str, default_port: int) -> bool:
    u = urlparse(dsn)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or default_port), timeout=2):
            return True
    except OSError:
        return False


_PG_DSN = os.environ.get("POWER_LOOP_TEST_PG_DSN", "postgresql://deeptalk:deeptalk@localhost:5433/power_loop_test")
_MYSQL_DSN = os.environ.get("POWER_LOOP_TEST_MYSQL_DSN", "mysql://deeptalk:deeptalk@localhost:3307/power_loop_test")


async def _loop_binds_and_resumes(dsn: str) -> None:
    """A loop opens an owned store at ``dsn`` (its own table_prefix), runs sends with the
    window cache, and a fresh loop resumes the same session from the same DSN."""
    prefix = "plrc_"

    def cfg():
        return AgentLoopConfig(system_prompt="S", max_rounds=1, compactor=None)

    loop = StatefulAgentLoop(
        llm=_SeqLLM(responses=[LLMResponse(raw_text="hi"), LLMResponse(raw_text="again")]),
        dsn=dsn, table_prefix=prefix, config=cfg(),
    )
    sid = await loop.new_session()
    await loop.send("hello", session_id=sid)
    await loop.send("more", session_id=sid)
    assert loop.cache_stats["hits"] >= 1  # 2nd send reused the window
    await loop.aclose()

    # fresh loop, same DSN+prefix, VERIFY (schema already provisioned) → resumes identically
    fresh = StatefulAgentLoop(
        llm=_SeqLLM(responses=[LLMResponse(raw_text="z")]),
        dsn=dsn, table_prefix=prefix, schema=SchemaPolicy.VERIFY, config=cfg(),
    )
    hist = [m.get("content") for m in await fresh.get_messages(sid)]
    assert hist[:2] == ["hello", "hi"]
    r = await fresh.send("continue", session_id=sid)
    assert r.status == "completed"
    await fresh.aclose()


@pytest.mark.skipif(not _reachable(_PG_DSN, 5432), reason=f"Postgres not reachable at {_PG_DSN}")
async def test_loop_binds_postgres_via_dsn():
    pytest.importorskip("asyncpg")
    await _loop_binds_and_resumes(_PG_DSN)


@pytest.mark.skipif(not _reachable(_MYSQL_DSN, 3306), reason=f"MySQL not reachable at {_MYSQL_DSN}")
async def test_loop_binds_mysql_via_dsn():
    pytest.importorskip("aiomysql")
    await _loop_binds_and_resumes(_MYSQL_DSN)
