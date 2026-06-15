"""Structured output helpers (M1.3).

Two things, kept deliberately small:

1. :class:`StructuredOutputSpec` —— declarative bundle of (name, JSON
   schema, strict-mode) that knows how to render itself into the
   OpenAI-compatible ``response_format`` dict the provider expects.
2. :func:`parse_structured` —— extracts a JSON object from any LLM
   response text, repairs common defects (markdown fences, trailing
   commas), and optionally validates against a minimal
   subset of JSON Schema (``type=="object"`` + ``required`` keys).

We do **not** depend on jsonschema or pydantic for parsing. The schema
field is forwarded to the provider as-is for **server-side** validation
(strict mode); local parsing only enforces "shape is an object and
required keys are present", which is enough to catch the wedge cases
that bite real Agents: the model wrote prose instead of JSON, or omitted
a critical field.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from power_loop._vendor.llm_client.interface import LLMResponse
from power_loop.contracts.errors import PowerLoopError


class StructuredOutputError(PowerLoopError):
    """Raised when ``parse_structured`` cannot produce a valid object.

    ``raw_text`` is the original LLM output (truncated); ``reason``
    explains which check failed: ``no_json`` / ``invalid_json`` /
    ``not_object`` / ``missing_required:<field>`` / ``wrong_type:<field>``.
    """

    def __init__(self, *, reason: str, raw_text: str, detail: str = "") -> None:
        self.reason = reason
        self.raw_text = raw_text[:1000]
        self.detail = detail
        msg = f"structured output rejected ({reason})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


@dataclass
class StructuredOutputSpec:
    """Declarative spec for ``LLMRequest.response_format``.

    The OpenAI ``json_schema`` mode requires a ``name`` and a ``schema``.
    ``strict=True`` is the safer default (most providers will refuse
    extra/missing keys server-side); set to ``False`` for providers that
    don't yet support strict mode.
    """

    name: str
    schema: dict[str, Any]
    strict: bool = True
    description: str | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_openai_response_format(self) -> dict[str, Any]:
        """Render the ``response_format`` payload OpenAI-compatible APIs accept."""
        js: dict[str, Any] = {"name": self.name, "schema": self.schema}
        if self.strict:
            js["strict"] = True
        if self.description:
            js["description"] = self.description
        return {"type": "json_schema", "json_schema": js}


# ── JSON repair ─────────────────────────────────────────────────────────


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?|\n?```\s*$", re.MULTILINE)


def _strip_markdown_fences(text: str) -> str:
    """Drop leading/trailing ```json / ``` fences if present."""
    return _FENCE_RE.sub("", text).strip()


def _extract_first_json_object(text: str) -> str | None:
    """Find the first balanced ``{...}`` block. Skips string contents.

    Returns the substring or ``None`` if no balanced object exists.
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                # A stray '}' before any '{' (common in prose: "}}" placeholders,
                # code snippets). Ignore it instead of going negative — otherwise
                # the real object that follows is never balanced and we return None.
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def _try_repair_json(blob: str) -> str:
    """Apply cheap, reversible fixes that recover from typical LLM JSON noise.

    Currently this removes trailing commas before ``}`` / ``]`` — the defect
    that most often survives fence-stripping and balanced-object extraction.
    Markdown fences are stripped earlier by :func:`_strip_markdown_fences`, and
    this stays deliberately conservative: notably it does **not** rewrite
    single quotes to double quotes, because a regex can't do that without
    corrupting apostrophes inside string values. Only run after a direct
    ``json.loads`` has already failed."""
    # Trailing commas before } or ]
    return re.sub(r",(\s*[}\]])", r"\1", blob)


def parse_structured(
    output: LLMResponse | str,
    *,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract a dict from an LLM response, repairing common JSON defects.

    Steps:

    1. Pull text from ``LLMResponse`` (``raw_text`` / ``content_text``)
       or use the string directly.
    2. Strip markdown fences.
    3. Try ``json.loads`` directly; on failure, find the first balanced
       ``{...}`` and retry; on failure, apply trailing-comma repair and
       retry once more.
    4. If a ``schema`` is supplied, check ``type=="object"`` shape and
       all ``required`` keys are present. Mismatches raise
       :class:`StructuredOutputError` with a precise ``reason``.

    Args:
        output: An ``LLMResponse`` or the raw text string.
        schema: Optional JSON-Schema-flavoured dict. Only ``type`` and
            ``required`` (top-level) are enforced locally; the full
            schema is what the provider validates server-side when used
            via :class:`StructuredOutputSpec`.

    Returns:
        The parsed dict.

    Raises:
        StructuredOutputError: with ``reason`` in ``{"no_json",
        "invalid_json", "not_object", "missing_required:<field>"}``.
    """
    text = output if isinstance(output, str) else (
        getattr(output, "raw_text", "") or getattr(output, "content_text", "") or ""
    )
    text = text or ""
    cleaned = _strip_markdown_fences(text)

    parsed: Any = None
    for attempt in ("direct", "extract", "repair"):
        candidate = cleaned
        if attempt == "extract":
            extracted = _extract_first_json_object(cleaned)
            if extracted is None:
                continue
            candidate = extracted
        elif attempt == "repair":
            extracted = _extract_first_json_object(cleaned) or cleaned
            candidate = _try_repair_json(extracted)
        try:
            parsed = json.loads(candidate)
            break
        except Exception:
            continue
    else:
        # Distinguish "no object at all" from "object but malformed".
        reason = "no_json" if _extract_first_json_object(cleaned) is None else "invalid_json"
        raise StructuredOutputError(reason=reason, raw_text=text)

    if not isinstance(parsed, Mapping):
        raise StructuredOutputError(reason="not_object", raw_text=text,
                                    detail=f"got {type(parsed).__name__}")
    parsed = dict(parsed)

    if schema is not None:
        _check_top_level_schema(parsed, schema, raw_text=text)
    return parsed


def _check_top_level_schema(obj: dict[str, Any], schema: dict[str, Any], *, raw_text: str) -> None:
    """Minimal local validation: top-level type=object + required keys present.

    Anything deeper (per-field type checking, enum, regex, nested
    schemas) is intentionally delegated to the provider's strict-mode
    server-side validation. Doing it here would duplicate a real
    json-schema implementation and silently disagree with the provider
    in edge cases.
    """
    expected_type = schema.get("type")
    if expected_type not in (None, "object"):
        raise StructuredOutputError(
            reason="not_object", raw_text=raw_text,
            detail=f"schema requires type={expected_type!r}",
        )
    for field_name in schema.get("required") or []:
        if field_name not in obj:
            raise StructuredOutputError(
                reason=f"missing_required:{field_name}", raw_text=raw_text,
            )


__all__ = [
    "StructuredOutputError",
    "StructuredOutputSpec",
    "parse_structured",
]
