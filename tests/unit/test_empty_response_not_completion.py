"""A provider's empty response (no text AND no tool calls) must NOT be treated as completion.

Observed in production (DeepTalk conv, deepseek-v4): after 29 rounds of real tool work the model
returned one empty chunk — no content, no tool_calls — and the loop finalized status="completed"
with final_text="". A whole investigation's worth of work produced a blank final answer. An empty
turn is a provider hiccup, not "I am done": a finishing agent either says something (non-empty
text) or calls a terminal tool (pass_turn). This test pins the corrected behavior.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from power_loop import AgentLoopConfig, StatefulAgentLoop
from power_loop._vendor.llm_client.interface import (
    LLMRequest,
    LLMResponse,
    LLMService,
    LLMStreamChunk,
)
from power_loop.tools.registry import ToolRegistry


@dataclass
class _ScriptedLLM(LLMService):
    """Returns each scripted LLMResponse in turn; after the script is exhausted, returns a
    non-empty final text (a well-behaved completion) so a fixed loop can terminate cleanly."""

    script: list[LLMResponse] = field(default_factory=list)
    calls: int = 0

    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                        on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        i = self.calls
        self.calls += 1
        if i < len(self.script):
            return self.script[i]
        return LLMResponse(raw_text="done for real", tool_calls=[])

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        async def _e() -> AsyncIterator[LLMStreamChunk]:
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


def _empty() -> LLMResponse:
    return LLMResponse(raw_text="", tool_calls=[])


def _loop(db, llm, max_rounds=10):
    return StatefulAgentLoop(
        llm=llm, db_path=str(db),
        config=AgentLoopConfig(system_prompt="t", max_rounds=max_rounds),
        tool_registry=ToolRegistry(),
    )


async def test_single_empty_response_is_not_completion(tmp_path) -> None:
    # One empty turn, then a real answer. The empty turn must NOT end the send with "".
    llm = _ScriptedLLM(script=[_empty()])
    loop = _loop(tmp_path / "a.db", llm)
    sid = await loop.new_session()
    result = await loop.send("investigate", sid)
    assert result.final_text == "done for real", (
        f"empty turn was treated as completion; got final_text={result.final_text!r}"
    )
    assert llm.calls >= 2, "loop stopped after the empty turn instead of retrying"


async def test_non_empty_no_tool_still_completes(tmp_path) -> None:
    # A NORMAL completion (agent says something, no tool call) must be unaffected by the fix.
    llm = _ScriptedLLM(script=[LLMResponse(raw_text="here is my report", tool_calls=[])])
    loop = _loop(tmp_path / "b.db", llm)
    sid = await loop.new_session()
    result = await loop.send("hi", sid)
    assert result.final_text == "here is my report"
    assert llm.calls == 1, "a normal one-shot completion should not retry"


async def test_persistent_empty_terminates_bounded(tmp_path) -> None:
    # If the provider keeps returning empty forever, the loop must still terminate (bounded),
    # not spin until the round cap silently. It should end degraded/completed, never hang.
    llm = _ScriptedLLM(script=[_empty()] * 50)
    loop = _loop(tmp_path / "c.db", llm, max_rounds=6)
    sid = await loop.new_session()
    result = await loop.send("hi", sid)
    assert result.status in ("degraded", "completed")
    assert llm.calls <= 8, f"unbounded empty-retry loop: {llm.calls} calls"
