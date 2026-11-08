"""RequestOptions bound a new turn; append retains the configured execution limits."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from power_loop import AgentLoopConfig, RequestOptions
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.pricing import Free
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
        pricing=Free("test_fixture")),
        lambda **kw: kw.get("text", ""),
    )
    return reg


async def _run_rounds(runtime_agent, *, configured, override=None, append=False):
    from power_loop.agent.runtime_store import now_ms
    model = _AlwaysTool()
    agent = await runtime_agent(model=model, registry=_echo_registry(),
        config=AgentLoopConfig(system_prompt="t", max_rounds=configured))
    options = RequestOptions(max_rounds=override) if override is not None else None
    if append:
        receipt = await agent.runtime.client.append(agent.target, "finalize", message_id="append",
            actor=agent.actor)
        await agent.runtime.client.wait(receipt.input_ref, actor=agent.actor, deadline_ms=now_ms()+10000)
        await agent.drain()
    else:
        await agent.send("hi", options=options)
    return model.calls


async def test_send_max_rounds_override_matches_that_config(runtime_agent):
    assert await _run_rounds(runtime_agent, configured=10, override=2) == await _run_rounds(runtime_agent, configured=2)


async def test_override_caps_below_config(runtime_agent):
    assert await _run_rounds(runtime_agent, configured=10, override=2) < await _run_rounds(runtime_agent, configured=10)


async def test_no_override_uses_config(runtime_agent):
    assert await _run_rounds(runtime_agent, configured=3) < await _run_rounds(runtime_agent, configured=5)


async def test_idle_append_uses_frozen_configuration(runtime_agent):
    assert await _run_rounds(runtime_agent, configured=1, append=True) == await _run_rounds(runtime_agent, configured=1)
