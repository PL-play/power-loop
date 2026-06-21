"""AgenticMemoryCompactor: the opt-in, memory-aware compactor.

Unit-level: drives the bounded tool-loop with a fake LLM + a stub note registry (no runtime
context needed), and checks the fall-back-to-single-call safety net. End-to-end note-writing
against a real session + provider lives in tests/real/test_real_agentic_compaction.py.
"""

from __future__ import annotations

import pytest

from power_loop._vendor.llm_client.interface import LLMRequest, LLMResponse
from power_loop.contracts.tools import ToolDefinition
from power_loop.runtime.compact import (
    DEFAULT_COMPACTION_AGENT_PROMPT,
    AgenticMemoryCompactor,
    _strip_summary,
)
from power_loop.tools.registry import ToolRegistry


def _tc(call_id: str, name: str, arguments: str):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class _ScriptLLM:
    """Returns queued responses; records each request (to assert tools/system_prompt)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return self.responses.pop(0)


def _note_registry(recorded: list[tuple[str, dict]]) -> ToolRegistry:
    reg = ToolRegistry()

    async def fake_note_add(content: str, pinned: bool = False):
        recorded.append(("note_add", {"content": content, "pinned": pinned}))
        return f"noted: {content}"

    async def fake_note_update(note_id: int, content: str | None = None):
        recorded.append(("note_update", {"note_id": note_id, "content": content}))
        return "updated"

    reg.register(
        ToolDefinition(name="note_add", description="save a durable fact",
                       input_schema={"type": "object", "properties": {"content": {"type": "string"}}},
                       required_params=("content",)),
        fake_note_add,
    )
    reg.register(
        ToolDefinition(name="note_update", description="refine a note",
                       input_schema={"type": "object", "properties": {"note_id": {"type": "integer"}}},
                       required_params=("note_id",)),
        fake_note_update,
    )
    return reg


SLICE = [{"role": "user", "content": "I'm in Berlin and I prefer terse replies"},
         {"role": "assistant", "content": "Got it."}]


def test_strip_summary():
    assert _strip_summary("<summary>hi</summary>") == "hi"
    assert _strip_summary("  plain text  ") == "plain text"
    assert _strip_summary("") is None
    assert _strip_summary(None) is None
    assert _strip_summary("pre <summary> body </summary> post") == "body"


@pytest.mark.asyncio
async def test_extracts_facts_to_notes_then_summarizes():
    recorded: list = []
    reg = _note_registry(recorded)
    llm = _ScriptLLM([
        LLMResponse(raw_text="saving facts", tool_calls=[_tc("1", "note_add", '{"content": "lives in Berlin"}')]),
        LLMResponse(raw_text="<summary>User in Berlin; prefers terse replies.</summary>"),
    ])
    comp = AgenticMemoryCompactor(memory_tools=reg, max_rounds=4)
    out = await comp._summarize_async(SLICE, llm=llm)
    assert out == "User in Berlin; prefers terse replies."
    assert recorded == [("note_add", {"content": "lives in Berlin", "pinned": False})]
    # the compaction call carried the planned compaction system prompt + the note tools
    assert llm.calls[0].system_prompt == DEFAULT_COMPACTION_AGENT_PROMPT
    assert {t["function"]["name"] for t in (llm.calls[0].tools or [])} == {"note_add", "note_update"}


@pytest.mark.asyncio
async def test_multiple_tool_rounds_then_summary():
    recorded: list = []
    reg = _note_registry(recorded)
    llm = _ScriptLLM([
        LLMResponse(raw_text="", tool_calls=[_tc("1", "note_add", '{"content": "fact A"}')]),
        LLMResponse(raw_text="", tool_calls=[_tc("2", "note_add", '{"content": "fact B"}')]),
        LLMResponse(raw_text="<summary>two facts saved</summary>"),
    ])
    comp = AgenticMemoryCompactor(memory_tools=reg, max_rounds=5)
    out = await comp._summarize_async(SLICE, llm=llm)
    assert out == "two facts saved"
    assert [r[1]["content"] for r in recorded] == ["fact A", "fact B"]


@pytest.mark.asyncio
async def test_bad_tool_call_does_not_abort_compaction():
    recorded: list = []
    reg = _note_registry(recorded)
    # malformed arguments JSON → coerced to {} → handler raises on missing required → recorded as error,
    # but the loop continues and still returns the summary.
    llm = _ScriptLLM([
        LLMResponse(raw_text="", tool_calls=[_tc("1", "note_add", "not-json")]),
        LLMResponse(raw_text="<summary>done anyway</summary>"),
    ])
    comp = AgenticMemoryCompactor(memory_tools=reg, max_rounds=4)
    out = await comp._summarize_async(SLICE, llm=llm)
    assert out == "done anyway"


@pytest.mark.asyncio
async def test_falls_back_to_single_call_on_failure():
    # The agentic path raises whenever tools are offered; the compactor must fall back to the
    # plain single-call summary (which calls the LLM with NO tools) instead of failing the fold.
    class _FailingLLM:
        def __init__(self):
            self.calls = []

        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls.append(request)
            if request.tools:
                raise RuntimeError("provider rejected tools")
            return LLMResponse(raw_text="<summary>single-call fallback</summary>")

    reg = _note_registry([])
    llm = _FailingLLM()
    comp = AgenticMemoryCompactor(memory_tools=reg, max_rounds=4)
    out = await comp._summarize_async(SLICE, llm=llm)
    assert out == "single-call fallback"
    assert any(c.tools for c in llm.calls) and any(not c.tools for c in llm.calls)  # tried both


@pytest.mark.asyncio
async def test_exhausted_rounds_makes_final_toolfree_call():
    recorded: list = []
    reg = _note_registry(recorded)
    # Always tool-calls → never volunteers a summary; after max_rounds the compactor issues one
    # final tool-free request, which returns the summary.
    llm = _ScriptLLM([
        LLMResponse(raw_text="", tool_calls=[_tc("1", "note_add", '{"content": "x"}')]),
        LLMResponse(raw_text="", tool_calls=[_tc("2", "note_add", '{"content": "y"}')]),
        LLMResponse(raw_text="<summary>final forced</summary>"),  # the tool-free final call
    ])
    comp = AgenticMemoryCompactor(memory_tools=reg, max_rounds=2)
    out = await comp._summarize_async(SLICE, llm=llm)
    assert out == "final forced"
    assert not llm.calls[-1].tools  # the final call offered no tools


@pytest.mark.asyncio
async def test_final_tool_free_call_uses_content_text():
    # The forced final summary call (after rounds exhausted) must accept the summary from
    # content_text, not just raw_text (regression for the review's AGENTIC-001).
    recorded: list = []
    reg = _note_registry(recorded)
    llm = _ScriptLLM([
        LLMResponse(raw_text="", tool_calls=[_tc("1", "note_add", '{"content": "x"}')]),
        LLMResponse(raw_text="", content_text="<summary>from content_text</summary>"),  # final call
    ])
    comp = AgenticMemoryCompactor(memory_tools=reg, max_rounds=1)
    out = await comp._summarize_async(SLICE, llm=llm)
    assert out == "from content_text"
    assert not llm.calls[-1].tools  # final call offered no tools


@pytest.mark.asyncio
async def test_tool_call_with_no_name_is_answered_not_crashed():
    reg = _note_registry([])
    llm = _ScriptLLM([
        LLMResponse(raw_text="", tool_calls=[{"id": "1", "type": "function", "function": {}}]),  # no name
        LLMResponse(raw_text="<summary>ok</summary>"),
    ])
    comp = AgenticMemoryCompactor(memory_tools=reg, max_rounds=4)
    out = await comp._summarize_async(SLICE, llm=llm)
    assert out == "ok"  # nameless tool call answered with an error result, loop continued


def test_default_registry_is_note_tools():
    comp = AgenticMemoryCompactor()
    reg = comp._registry()
    assert set(reg.names()) == {"note_add", "note_update"}
