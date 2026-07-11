"""S3 (3.14): ``AgentLoopConfig.subagent_config_factory`` — host seam rewriting
the default child config built by ``run_agent_spec``."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from power_loop import AgentLoopConfig, AgentSpec, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.runtime.spec import run_agent_spec


@dataclass
class _Capturing(LLMService):
    """Records each request's generation params so the child config is observable."""

    responses: list[LLMResponse] = field(default_factory=list)
    seen: list[LLMRequest] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        self.seen.append(request)
        if self._idx >= len(self.responses):
            return LLMResponse(raw_text="done")
        r = self.responses[self._idx]
        self._idx += 1
        return r

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


@pytest.fixture
async def store() -> AsyncIterator[SessionStore]:
    s = await SessionStore.open(":memory:")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_factory_rewrites_child_config(store: SessionStore) -> None:
    calls: list[tuple[str, AgentLoopConfig]] = []

    def factory(spec: AgentSpec, default: AgentLoopConfig) -> AgentLoopConfig:
        calls.append((spec.name, default))
        return replace(default, temperature=0.9, max_tokens=1234)

    llm = _Capturing(responses=[LLMResponse(raw_text="child out")])
    loop = StatefulAgentLoop(
        llm=llm, store=store,
        config=AgentLoopConfig(max_rounds=3, subagent_config_factory=factory),
    )
    res = await run_agent_spec(
        AgentSpec(name="kid", system_prompt="p", max_tokens=500),
        "task", parent_loop=loop,
    )
    assert res["status"] == "completed"
    # Factory saw the spec-derived default and its output was used as-is.
    assert calls[0][0] == "kid"
    assert calls[0][1].max_tokens == 500
    assert llm.seen[0].temperature == 0.9
    assert llm.seen[0].max_tokens == 1234


@pytest.mark.asyncio
async def test_no_factory_default_unchanged(store: SessionStore) -> None:
    llm = _Capturing(responses=[LLMResponse(raw_text="child out")])
    loop = StatefulAgentLoop(
        llm=llm, store=store, config=AgentLoopConfig(max_rounds=3),
    )
    await run_agent_spec(
        AgentSpec(name="kid", system_prompt="p", max_tokens=500, temperature=0.2),
        "task", parent_loop=loop,
    )
    assert llm.seen[0].max_tokens == 500
    assert llm.seen[0].temperature == 0.2


@pytest.mark.asyncio
async def test_factory_bad_return_type_raises_before_session(store: SessionStore) -> None:
    def factory(spec: AgentSpec, default: AgentLoopConfig) -> Any:
        return {"not": "a config"}

    llm = _Capturing()
    loop = StatefulAgentLoop(
        llm=llm, store=store,
        config=AgentLoopConfig(max_rounds=3, subagent_config_factory=factory),
    )
    with pytest.raises(TypeError, match="subagent_config_factory must return"):
        await run_agent_spec(
            AgentSpec(name="kid", system_prompt="p"), "task", parent_loop=loop,
        )
    # Raised BEFORE the child session was created — nothing leaked.
    assert llm.seen == []


@pytest.mark.asyncio
async def test_factory_exception_propagates_without_leaking(store: SessionStore) -> None:
    def factory(spec: AgentSpec, default: AgentLoopConfig) -> AgentLoopConfig:
        raise RuntimeError("host factory bug")

    llm = _Capturing()
    loop = StatefulAgentLoop(
        llm=llm, store=store,
        config=AgentLoopConfig(max_rounds=3, subagent_config_factory=factory),
    )
    with pytest.raises(RuntimeError, match="host factory bug"):
        await run_agent_spec(
            AgentSpec(name="kid", system_prompt="p"), "task", parent_loop=loop,
        )
    assert llm.seen == []
