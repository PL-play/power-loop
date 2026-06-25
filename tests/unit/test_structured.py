"""Unit tests for M1.3 — StructuredOutputSpec + parse_structured.

Covers:
- Spec → OpenAI response_format dict shape (strict toggle, description, examples).
- parse_structured: clean JSON, fenced JSON, prose-with-embedded-JSON,
  trailing comma, prose-only (no JSON at all), wrong top-level type.
- Schema validation: required-key missing → precise error reason.
"""

from __future__ import annotations

import pytest

from power_loop import StructuredOutputError, StructuredOutputSpec, parse_structured
from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.runtime.structured import _extract_first_json_object


def test_extract_skips_stray_closing_brace_before_json() -> None:
    """Regression: a '}' in prose before the real object used to drive the brace
    depth negative so the object was never returned (valid output rejected)."""
    assert _extract_first_json_object('Use the }} placeholder. Result: {"ok": true}') == '{"ok": true}'
    assert _extract_first_json_object('set x = a} ; then {"a": 1}') == '{"a": 1}'
    assert _extract_first_json_object('{"a": {"b": 1}}') == '{"a": {"b": 1}}'  # nested
    assert _extract_first_json_object("no json here") is None


def test_parse_structured_recovers_after_stray_brace() -> None:
    out = parse_structured('thoughts}} then {"label": "urgent"}', schema={"type": "object"})
    assert out == {"label": "urgent"}


def test_examples_folded_into_response_format() -> None:
    # exec-skills-structured-5: the public `examples` field is no longer silently dropped — it's
    # folded into the json_schema description so it actually reaches the model.
    spec = StructuredOutputSpec(
        name="S", schema={"type": "object"}, description="D", examples=[{"a": 1}, {"a": 2}]
    )
    desc = spec.to_openai_response_format()["json_schema"]["description"]
    assert "D" in desc and "Examples:" in desc and '{"a": 1}' in desc and '{"a": 2}' in desc


def test_tool_message_always_emits_tool_call_id() -> None:
    # exec-skills-structured-7: a tool-role message must always carry tool_call_id (empty if unset),
    # never omit it (an omitted id is a structurally-invalid tool message).
    from power_loop.contracts.messages import AgentMessage
    assert AgentMessage(role="tool", content="r", tool_call_id=None).to_openai_message().get(
        "tool_call_id") == ""
    assert AgentMessage(role="tool", content="r", tool_call_id="tc1").to_openai_message()[
        "tool_call_id"] == "tc1"


def test_extract_handles_odd_quote_count_in_prose() -> None:
    # M-exec-skills-structured-2: an unbalanced (odd) '"' in the prose preamble must NOT strand the
    # scanner "in string" so the real object's '{' is read as string content and never found.
    assert _extract_first_json_object('I used the "search tool: {"answer": 42}') == '{"answer": 42}'
    assert _extract_first_json_object('a " b " c " d {"x": 1}') == '{"x": 1}'  # 3 stray quotes


def test_parse_structured_recovers_after_odd_quote_prose() -> None:
    out = parse_structured('Result for "urgent: {"label": "urgent"}', schema={"type": "object"})
    assert out == {"label": "urgent"}


# ── Spec ────────────────────────────────────────────────────────────────


def test_spec_strict_true_renders_full_payload() -> None:
    spec = StructuredOutputSpec(
        name="UserCard",
        schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        description="A user profile card",
    )
    fmt = spec.to_openai_response_format()
    assert fmt["type"] == "json_schema"
    js = fmt["json_schema"]
    assert js["name"] == "UserCard"
    assert js["strict"] is True
    assert js["description"] == "A user profile card"
    assert js["schema"]["required"] == ["name"]


def test_spec_strict_false_omits_strict_key() -> None:
    spec = StructuredOutputSpec(name="X", schema={"type": "object"}, strict=False)
    assert "strict" not in spec.to_openai_response_format()["json_schema"]


# ── parse_structured: happy paths ───────────────────────────────────────


def test_parse_clean_json_string() -> None:
    assert parse_structured('{"a": 1}') == {"a": 1}


def test_parse_llm_response_object() -> None:
    assert parse_structured(LLMResponse(raw_text='{"a": 1}')) == {"a": 1}


def test_parse_strips_markdown_fences() -> None:
    text = "```json\n{\"name\": \"alan\", \"age\": 30}\n```"
    assert parse_structured(text) == {"name": "alan", "age": 30}


def test_parse_extracts_embedded_json_from_prose() -> None:
    text = "Sure, here is the card:\n{\"k\": \"v\"}\nThanks!"
    assert parse_structured(text) == {"k": "v"}


def test_parse_repairs_trailing_comma() -> None:
    text = '{"a": 1, "b": [1, 2,],}'
    assert parse_structured(text) == {"a": 1, "b": [1, 2]}


# ── parse_structured: error paths ──────────────────────────────────────


def test_parse_prose_only_raises_no_json() -> None:
    with pytest.raises(StructuredOutputError) as exc:
        parse_structured("Sorry, I cannot do that.")
    assert exc.value.reason == "no_json"


def test_parse_malformed_object_raises_invalid_json() -> None:
    with pytest.raises(StructuredOutputError) as exc:
        parse_structured('{"a": 1, "b": <not json>}')
    # repair doesn't fix this — reason should reflect we found {...} but couldn't parse
    assert exc.value.reason == "invalid_json"


def test_parse_top_level_array_raises_not_object() -> None:
    with pytest.raises(StructuredOutputError) as exc:
        parse_structured("[1, 2, 3]")
    # json.loads accepts the array, but our contract requires a dict.
    assert exc.value.reason == "not_object"


# ── Schema validation (minimal local check) ─────────────────────────────


def test_schema_missing_required_raises_with_field_in_reason() -> None:
    schema = {"type": "object", "required": ["name", "age"]}
    with pytest.raises(StructuredOutputError) as exc:
        parse_structured('{"name": "alan"}', schema=schema)
    assert exc.value.reason == "missing_required:age"


def test_schema_all_required_present_passes() -> None:
    schema = {"type": "object", "required": ["name"]}
    assert parse_structured('{"name": "x"}', schema=schema) == {"name": "x"}


def test_schema_no_required_field_passes() -> None:
    schema = {"type": "object"}
    assert parse_structured('{"x": 1}', schema=schema) == {"x": 1}


# ── error carries raw_text for debugging ────────────────────────────────


def test_error_preserves_truncated_raw_text() -> None:
    long_prose = "x" * 5000
    with pytest.raises(StructuredOutputError) as exc:
        parse_structured(long_prose)
    assert len(exc.value.raw_text) <= 1000
    assert exc.value.raw_text.startswith("x")
