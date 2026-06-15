"""H2.6: property-based tests (hypothesis) for the hand-rolled parsers/state-machines.

These exercise the JSON-repair parser and the compactor span-selection over a wide
space of inputs, where example-based tests leave gaps.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_client.interface import LLMRequest, LLMResponse, LLMService, LLMStreamChunk
from power_loop.runtime.compact import DefaultCompactor
from power_loop.runtime.structured import StructuredOutputError, parse_structured

pytestmark = pytest.mark.unit


# ── parse_structured ───────────────────────────────────────────────────────

@given(s=st.text())
@settings(max_examples=300)
def test_parse_structured_is_total(s: str) -> None:
    """For ANY input, parse_structured either returns a dict or raises
    StructuredOutputError — never a different exception or return type."""
    try:
        result = parse_structured(s)
    except StructuredOutputError:
        return
    assert isinstance(result, dict)


# JSON-safe scalar values (no NaN/inf; floats excluded to keep == exact).
_json_scalars = st.none() | st.booleans() | st.integers() | st.text()


@given(d=st.dictionaries(st.text(min_size=1, max_size=8), _json_scalars, max_size=6))
def test_parse_structured_roundtrips_json_objects(d: dict) -> None:
    """A serialized JSON object parses back to an equal dict."""
    assert parse_structured(json.dumps(d)) == d


@given(v=st.lists(st.integers(), max_size=4) | st.integers() | st.text(max_size=6) | st.booleans())
def test_parse_structured_rejects_non_objects(v: Any) -> None:
    """A JSON value that is not an object must raise (reason not_object/no_json)."""
    with pytest.raises(StructuredOutputError):
        parse_structured(json.dumps(v))


# ── DefaultCompactor span selection ────────────────────────────────────────

class _FixedLLM(LLMService):
    async def complete(self, request: LLMRequest, *, on_chunk_delta_text=None,
                       on_chunk_think=None, on_stream_end=None) -> LLMResponse:
        return LLMResponse(raw_text="<summary>folded</summary>", content_text="<summary>folded</summary>")

    def stream(self, request: LLMRequest):
        async def _e():
            if False:
                yield LLMStreamChunk()
        return _e()

    async def close(self) -> None:
        return None


@st.composite
def _valid_history(draw: Any) -> list[dict]:
    """A protocol-valid history: leading system, then exchanges where any
    assistant(tool_calls) is immediately followed by its matching tool replies."""
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    n = draw(st.integers(min_value=1, max_value=8))
    for i in range(n):
        msgs.append({"role": "user", "content": draw(st.text(min_size=1, max_size=20))})
        if draw(st.booleans()):  # tool round
            k = draw(st.integers(min_value=1, max_value=2))
            ids = [f"call_{i}_{j}" for j in range(k)]
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": cid, "type": "function",
                                "function": {"name": "f", "arguments": "{}"}} for cid in ids],
            })
            for cid in ids:
                msgs.append({"role": "tool", "tool_call_id": cid, "name": "f", "content": "r"})
        else:
            msgs.append({"role": "assistant", "content": draw(st.text(max_size=20))})
    return msgs


def _has_orphan_tool(msgs: list[dict]) -> bool:
    """A `tool` message is valid only immediately after an assistant(tool_calls)
    or another `tool` (the contiguous tool-reply block). Anything else is orphaned."""
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            if i == 0:
                return True
            prev = msgs[i - 1]
            if not (prev.get("role") == "tool"
                    or (prev.get("role") == "assistant" and prev.get("tool_calls"))):
                return True
    return False


@given(history=_valid_history())
@settings(max_examples=200, deadline=None)
def test_compaction_preserves_protocol_validity_and_system(history: list[dict]) -> None:
    """Folding a valid history never produces an orphan tool and never drops the
    leading system message (the M1.7a invariants), for any history shape."""
    assert not _has_orphan_tool(history)  # sanity: generator produces valid input

    cmp = DefaultCompactor(absolute_threshold=1, keep_last_n=1, summary_llm=_FixedLLM())
    plan = asyncio.run(cmp.maybe_compact(history, llm=_FixedLLM(), max_tokens=10**9, round_index=0))
    if plan is None:
        return
    folded = (
        history[: plan.fold_start_idx]
        + [{"role": "system", "name": "compact_note", "content": plan.summary_text}]
        + history[plan.fold_end_idx + 1:]
    )
    assert not _has_orphan_tool(folded), folded
    assert folded[0].get("role") == "system", "leading system message must survive"


def test_orphan_tool_checker_has_teeth() -> None:
    """Guard against a vacuous property: the checker must flag a real orphan and
    accept a valid pair."""
    orphan = [{"role": "user", "content": "hi"}, {"role": "tool", "tool_call_id": "x", "content": "r"}]
    valid = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function",
                                                             "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "x", "content": "r"},
    ]
    assert _has_orphan_tool(orphan) is True
    assert _has_orphan_tool(valid) is False
