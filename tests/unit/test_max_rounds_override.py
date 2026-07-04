"""Per-call ``max_rounds`` override on send()/follow_up() — run a bounded continuation with a
different round budget than config.max_rounds (e.g. a short "finalize" turn)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.tools.registry import ToolRegistry


@dataclass
class _AlwaysTool(LLMService):
    """Fake LLM that always calls the (registered, resolvable) echo tool → the loop advances
    round after round until it hits its round cap. ``calls`` = number of LLM invocations."""

    calls: int = 0

    async def complete(
        self,
        request: LLMRequest,
        *,
        on_chunk_delta_text: Callable[[str], Any] | None = None,
        on_chunk_think: Callable[[str], Any] | None = None,
        on_stream_end: Callable[[LLMResponse], Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            raw_text="",
            tool_calls=[{
                "id": f"c{self.calls}", "type": "function",
                "function": {"name": "echo", "arguments": '{"text": "x"}'},
            }],
        )

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _empty() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()

        return _empty()

    async def close(self) -> None:
        return None


def _echo_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo", description="echo back",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}},
                          "required": ["text"]},
            required_params=("text",),
        ),
        lambda **kw: kw.get("text", ""),
    )
    return reg


def _loop(db, max_rounds):
    return StatefulAgentLoop(
        llm=_AlwaysTool(),
        db_path=str(db),
        config=AgentLoopConfig(system_prompt="t", max_rounds=max_rounds),
        tool_registry=_echo_registry(),
    )


async def _run_rounds(loop, *, override=None):
    sid = await loop.new_session()
    await loop.send("hi", sid, max_rounds=override)
    return loop.llm.calls


async def test_send_max_rounds_override_matches_that_config(tmp_path) -> None:
    # send(max_rounds=2) on a config.max_rounds=10 loop runs as many rounds as a
    # config.max_rounds=2 loop — i.e. the override, not the config, governs.
    with_override = await _run_rounds(_loop(tmp_path / "a.db", 10), override=2)
    as_config = await _run_rounds(_loop(tmp_path / "b.db", 2))
    assert with_override == as_config


async def test_override_caps_below_config(tmp_path) -> None:
    small = await _run_rounds(_loop(tmp_path / "c.db", 10), override=2)
    big = await _run_rounds(_loop(tmp_path / "d.db", 10))  # no override → config's 10
    assert small < big


async def test_no_override_uses_config(tmp_path) -> None:
    three = await _run_rounds(_loop(tmp_path / "e.db", 3))
    five = await _run_rounds(_loop(tmp_path / "f.db", 5))
    assert three < five  # config governs when no override is passed


async def test_follow_up_idle_honors_override(tmp_path) -> None:
    loop = _loop(tmp_path / "g.db", 10)
    sid = await loop.new_session()
    await loop.follow_up("finalize", sid, max_rounds=1)  # idle → send fallback → override applies
    ref = _loop(tmp_path / "h.db", 1)
    ref_sid = await ref.new_session()
    await ref.send("finalize", ref_sid)
    assert loop.llm.calls == ref.llm.calls
