"""Configurable MAX_SPAWN_DEPTH.

The depth ceiling is no longer a hard module constant: it lives on the
``SessionStore`` instance (default :data:`MAX_SPAWN_DEPTH`) and can be set via
``SessionStore.open(max_spawn_depth=...)`` or the ``StatefulAgentLoop``
``max_spawn_depth`` argument. Both enforcement points honor it:

* ``SessionStore.create_session`` raises past the limit;
* ``run_agent_spec`` pre-checks and returns ``status="rejected"``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass, field

import pytest

from power_loop import (
    MAX_SPAWN_DEPTH,
    AgentLoopConfig,
    AgentSpec,
    SessionStore,
    StatefulAgentLoop,
)
from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.core.agent_context import (
    reset_current_loop,
    reset_session_id,
    set_current_loop,
    set_session_id,
)
from power_loop.runtime.spec import run_agent_spec

pytestmark = pytest.mark.unit


@pytest.fixture
def store() -> Generator[SessionStore, None, None]:
    s = SessionStore.open(":memory:")
    yield s
    s.close()


# ── store-level config ────────────────────────────────────────────────────


def test_default_is_module_constant() -> None:
    s = SessionStore.open(":memory:")
    try:
        assert s.max_spawn_depth == MAX_SPAWN_DEPTH
    finally:
        s.close()


def test_open_accepts_custom_limit() -> None:
    s = SessionStore.open(":memory:", max_spawn_depth=1)
    try:
        assert s.max_spawn_depth == 1
        root = s.create_session()
        child = s.create_session(parent_session_id=root)  # depth 1 == limit: ok
        with pytest.raises(ValueError, match="exceeds max 1"):
            s.create_session(parent_session_id=child)  # depth 2 > 1: rejected
    finally:
        s.close()


def test_higher_limit_allows_deeper_chain() -> None:
    s = SessionStore.open(":memory:", max_spawn_depth=5)
    try:
        cur = s.create_session()
        for _ in range(5):  # would blow the default of 3
            cur = s.create_session(parent_session_id=cur)
        assert s.max_spawn_depth == 5
    finally:
        s.close()


@pytest.mark.parametrize("bad", [0, -1, "x", None])
def test_invalid_limit_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="max_spawn_depth"):
        SessionStore.open(":memory:", max_spawn_depth=bad)  # type: ignore[arg-type]


def test_property_setter_validates(store: SessionStore) -> None:
    store.max_spawn_depth = 7
    assert store.max_spawn_depth == 7
    with pytest.raises(ValueError, match="max_spawn_depth"):
        store.max_spawn_depth = 0


# ── loop-level convenience param ──────────────────────────────────────────


@dataclass
class _LLM(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    _i: int = 0

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        if self._i >= len(self.responses):
            return LLMResponse(raw_text="done", content_text="done")
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


def test_loop_forwards_limit_to_owned_store() -> None:
    loop = StatefulAgentLoop(llm=_LLM(), db_path=":memory:", max_spawn_depth=9)
    try:
        assert loop.store.max_spawn_depth == 9
    finally:
        loop.close()


def test_loop_overrides_shared_store_limit(store: SessionStore) -> None:
    assert store.max_spawn_depth == MAX_SPAWN_DEPTH
    StatefulAgentLoop(llm=_LLM(), store=store, max_spawn_depth=2)
    assert store.max_spawn_depth == 2  # explicit arg overrode the shared store


def test_loop_leaves_shared_store_when_unset(store: SessionStore) -> None:
    store.max_spawn_depth = 4
    StatefulAgentLoop(llm=_LLM(), store=store)  # no max_spawn_depth arg
    assert store.max_spawn_depth == 4  # untouched


async def test_run_agent_spec_honors_configured_limit(store: SessionStore) -> None:
    """run_agent_spec's pre-check reads the store's configurable ceiling."""
    store.max_spawn_depth = 1
    parent_loop = StatefulAgentLoop(
        llm=_LLM(responses=[LLMResponse(raw_text="p", content_text="p")]),
        store=store, config=AgentLoopConfig(max_rounds=1),
    )
    # Build a parent chain already at depth 1 (== limit) so the next spawn bounces.
    root = store.create_session()
    at_limit = store.create_session(parent_session_id=root)  # depth 1

    parent_loop.llm = _LLM(responses=[LLMResponse(raw_text="child", content_text="child")])
    tok_loop = set_current_loop(parent_loop)
    tok_sid = set_session_id(at_limit)
    try:
        result = await run_agent_spec(AgentSpec(name="c", system_prompt="P"), "go",
                                      parent_loop=parent_loop)
    finally:
        reset_current_loop(tok_loop)
        reset_session_id(tok_sid)

    assert result["status"] == "rejected"
    assert "exceeds max 1" in result["final_text"]


async def test_run_agent_spec_falls_back_when_store_lacks_attr() -> None:
    """A custom/legacy store with no `max_spawn_depth` falls back to the constant.

    Every real store sets the attribute, so the `getattr(..., MAX_SPAWN_DEPTH)`
    fallback in run_agent_spec's pre-check is otherwise unexercised.
    """

    class _Row:
        spawn_depth = MAX_SPAWN_DEPTH  # already at the default ceiling

    class _StoreNoAttr:  # duck-typed: only what the pre-check touches, no max_spawn_depth
        def get_session(self, sid: str) -> _Row:
            return _Row()

    class _FakeLoop:
        store = _StoreNoAttr()

    tok = set_session_id("parent")
    try:
        result = await run_agent_spec(
            AgentSpec(name="c", system_prompt="P"), "go", parent_loop=_FakeLoop(),
        )
    finally:
        reset_session_id(tok)

    # depth+1 (4) > fallback ceiling (MAX_SPAWN_DEPTH=3) → rejected with the default.
    assert result["status"] == "rejected"
    assert f"exceeds max {MAX_SPAWN_DEPTH}" in result["final_text"]
