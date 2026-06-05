"""Token budgeting utilities — heuristic, vendor-neutral.

These are deliberately approximate: we want a single number that's cheap to
compute, monotonic with content size, and works without a tokenizer
dependency. Used by the compactor to decide *when* to trigger; not used for
billing.

Rule of thumb: ~4 chars per token for English-heavy LLM transcripts. Adjust
via :data:`CHARS_PER_TOKEN` if needed.
"""

from __future__ import annotations

import json
from typing import Any

CHARS_PER_TOKEN = 4
"""Approximate chars-per-token used by :func:`estimate_tokens`."""


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Heuristic token count for a single message dict.

    Counts content (string or JSON-serialized non-string), tool_calls
    arguments, name fields, and a small per-message overhead for the role
    framing.
    """
    overhead = 4  # role tag + delimiters
    content = message.get("content")
    if isinstance(content, str):
        body = content
    elif content is None:
        body = ""
    else:
        body = json.dumps(content, ensure_ascii=False)
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        body += json.dumps(tool_calls, ensure_ascii=False)
    name = message.get("name") or ""
    return overhead + estimate_text_tokens(body) + estimate_text_tokens(name)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)
