"""Token estimates for calls whose real usage the provider never reported (design/124 §10).

Providers put usage in the LAST stream chunk only. A call that fails mid-way, times out, or is
aborted (steering, stop) never sees that chunk — yet it was billed: live tests (design/124
§11.2) show the prompt is charged in full once processing started, plus whatever was generated
before the disconnect, and nothing after. So such a call is recorded with ESTIMATED tokens
(flagged ``estimated``) instead of silently counting as zero.

Heuristic, not a tokenizer: CJK characters ≈ 0.7 token each, everything else ≈ 1 token per 3.6
characters, plus small per-message / per-image overheads. Calibrated against DeepSeek usage in
``tests/real/test_real_usage_estimate.py``; good to tens of percent, which is what "we paid for
roughly this much" needs. :mod:`power_loop.runtime.budget` keeps its own coarser estimate for
context budgeting — the two answer different questions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

#: CJK ideographs, kana, hangul, full-width forms and CJK punctuation.
_CJK = re.compile(r"[⺀-鿿가-힯豈-﫿＀-￯　-〿]")
CJK_TOKENS_PER_CHAR = 0.7
OTHER_CHARS_PER_TOKEN = 3.6
MESSAGE_OVERHEAD_TOKENS = 4
#: A picture's cost varies by provider and size; one middling tile-set is a fair stand-in.
IMAGE_BLOCK_TOKENS = 765


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return int(round(cjk * CJK_TOKENS_PER_CHAR + other / OTHER_CHARS_PER_TOKEN))


def _content_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_text_tokens(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += estimate_text_tokens(block)
            elif isinstance(block, Mapping):
                btype = block.get("type")
                if btype == "text":
                    total += estimate_text_tokens(str(block.get("text") or ""))
                elif btype in ("image_url", "image", "attachment", "input_image"):
                    total += IMAGE_BLOCK_TOKENS
                else:
                    total += estimate_text_tokens(json.dumps(block, ensure_ascii=False))
        return total
    return estimate_text_tokens(json.dumps(content, ensure_ascii=False, default=str))


def estimate_messages_tokens(messages: Iterable[Mapping[str, Any]]) -> int:
    total = 0
    for m in messages:
        total += MESSAGE_OVERHEAD_TOKENS + _content_tokens(m.get("content"))
        for tc in m.get("tool_calls") or ():
            fn = tc.get("function") if isinstance(tc, Mapping) else None
            if isinstance(fn, Mapping):
                total += estimate_text_tokens(str(fn.get("name") or ""))
                total += estimate_text_tokens(str(fn.get("arguments") or ""))
        for key in ("reasoning_content", "name"):
            v = m.get(key)
            if isinstance(v, str):
                total += estimate_text_tokens(v)
    return total


def estimate_prompt_tokens(
    *,
    messages: Iterable[Mapping[str, Any]],
    system_prompt: str | None = None,
    tools: Iterable[Any] | None = None,
) -> int:
    """What the provider billed as prompt for this request (estimate)."""
    total = estimate_messages_tokens(messages)
    if system_prompt:
        total += MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(system_prompt)
    if tools:
        total += estimate_text_tokens(json.dumps(list(tools), ensure_ascii=False, default=str))
    return total


def estimate_completion_tokens(*, text: str = "", think: str = "",
                               tool_calls: Iterable[Mapping[str, Any]] | None = None) -> int:
    """What the provider billed as output for what it had produced (estimate)."""
    total = estimate_text_tokens(text) + estimate_text_tokens(think)
    for tc in tool_calls or ():
        fn = tc.get("function") if isinstance(tc, Mapping) else None
        if isinstance(fn, Mapping):
            total += estimate_text_tokens(str(fn.get("name") or ""))
            total += estimate_text_tokens(str(fn.get("arguments") or ""))
    return total


__all__ = [
    "estimate_completion_tokens",
    "estimate_messages_tokens",
    "estimate_prompt_tokens",
    "estimate_text_tokens",
]
