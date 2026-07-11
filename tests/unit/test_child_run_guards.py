"""S1 (3.14): child-run guards — host per-send state suspended around inline
child runs (``StatefulAgentLoop.register_child_run_guard``)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from power_loop import AgentLoopConfig, AgentSpec, SessionStore, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.spec import run_agent_spec


@dataclass
class _Scripted(LLMService):
    responses: list[LLMResponse] = field(default_factory=list)
    _idx: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
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


def _loop(store: SessionStore, llm: LLMService) -> StatefulAgentLoop:
    return StatefulAgentLoop(llm=llm, store=store, config=AgentLoopConfig(max_rounds=3))


def test_register_and_remove_by_name(store: SessionStore) -> None:
    llm = _Scripted()
    loop = _loop(store, llm)

    @contextlib.contextmanager
    def g() -> Iterator[None]:
        yield

    loop.register_child_run_guard(g, name="one")
    loop.register_child_run_guard(g)  # anonymous
    assert loop.remove_child_run_guard("one") is True
    assert loop.remove_child_run_guard("one") is False
    assert loop.remove_child_run_guard("missing") is False
    assert len(loop._child_run_guards) == 1


@pytest.mark.asyncio
async def test_guards_wrap_child_run_in_order(store: SessionStore) -> None:
    events: list[str] = []

    def make_guard(tag: str):
        @contextlib.contextmanager
        def g() -> Iterator[None]:
            events.append(f"enter:{tag}")
            try:
                yield
            finally:
                events.append(f"exit:{tag}")

        return g

    llm = _Scripted(responses=[LLMResponse(raw_text="child out")])
    loop = _loop(store, llm)
    loop.register_child_run_guard(make_guard("a"), name="a")
    loop.register_child_run_guard(make_guard("b"), name="b")

    res = await run_agent_spec(
        AgentSpec(name="kid", system_prompt="p"), "task", parent_loop=loop,
    )
    assert res["status"] == "completed"
    # Entered in registration order, exited in reverse, around the whole run.
    assert events == ["enter:a", "enter:b", "exit:b", "exit:a"]


@pytest.mark.asyncio
async def test_guards_restore_on_child_failure(store: SessionStore) -> None:
    """A raising child run still exits the guards (state restored)."""
    state = {"suspended": False}

    @contextlib.contextmanager
    def g() -> Iterator[None]:
        state["suspended"] = True
        try:
            yield
        finally:
            state["suspended"] = False

    class _Boom(_Scripted):
        async def complete(self, request: LLMRequest, **_kw: Any) -> LLMResponse:
            raise RuntimeError("provider down")

    loop = _loop(store, _Boom())
    loop.register_child_run_guard(g)
    with pytest.raises(RuntimeError, match="provider down"):
        await run_agent_spec(
            AgentSpec(name="kid", system_prompt="p"), "task", parent_loop=loop,
        )
    assert state["suspended"] is False


@pytest.mark.asyncio
async def test_guards_reenter_for_grandchild(store: SessionStore) -> None:
    """Nested child runs enter the same guards again (re-entrancy contract)."""
    depth = {"cur": 0, "max": 0}

    @contextlib.contextmanager
    def g() -> Iterator[None]:
        depth["cur"] += 1
        depth["max"] = max(depth["max"], depth["cur"])
        try:
            yield
        finally:
            depth["cur"] -= 1

    async def spawnling(**_kw: Any) -> str:
        from power_loop.core.agent_context import get_current_loop

        res = await run_agent_spec(
            AgentSpec(name="grandkid", system_prompt="p"),
            "task", parent_loop=get_current_loop(),
        )
        return str(res["status"])

    llm = _Scripted(responses=[
        LLMResponse(raw_text="", tool_calls=[{
            "id": "t1", "type": "function",
            "function": {"name": "spawnling", "arguments": "{}"},
        }]),                                  # child round 1 → spawn grandchild
        LLMResponse(raw_text="grandchild"),   # grandchild run
        LLMResponse(raw_text="child done"),   # child round 2
    ])
    from power_loop.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="spawnling", description="spawn",
            input_schema={"type": "object", "properties": {}},
        ),
        spawnling,
    )
    loop = StatefulAgentLoop(
        llm=llm, store=store, tool_registry=reg, config=AgentLoopConfig(max_rounds=3),
    )
    loop.register_child_run_guard(g)
    res = await run_agent_spec(
        AgentSpec(name="kid", system_prompt="p"), "task", parent_loop=loop,
    )
    assert res["status"] == "completed"
    assert depth["max"] == 2  # child + nested grandchild
    assert depth["cur"] == 0
