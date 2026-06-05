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


def trim_history(
    messages: list[dict[str, Any]],
    max_tokens: int,
    *,
    keep_system: bool = True,
    keep_last_n: int = 2,
) -> list[dict[str, Any]]:
    """Drop messages from the middle until the token budget fits.

    Used by callers that want to cap context before calling the LLM
    directly (without the full compactor pipeline). This is a **pure
    trim** — no summarisation, no LLM calls. For LLM-backed summary
    compaction, use :class:`power_loop.runtime.compact.DefaultCompactor`.

    Preserves:
    * All ``role=system`` messages at the front (when ``keep_system=True``).
    * The last ``keep_last_n`` exchanges (user-bounded). An exchange is
      a ``user`` message followed by everything up to (but not including)
      the next ``user`` — so ``assistant(tool_calls) ↔ tool`` pairs stay
      together.
    * The ``assistant(tool_calls)`` ↔ ``tool`` atomic pair near the cut
      boundary: if the first removed message is a ``tool``, we walk back
      to include the preceding ``assistant(tool_calls)`` so the protocol
      isn't broken.

    Returns a new list (does not mutate the input). If the budget
    already fits, the original list is returned unchanged.
    """
    if not messages or max_tokens <= 0:
        return []

    if estimate_tokens(messages) <= max_tokens:
        return list(messages)

    # Partition: leading system block, body, tail.
    n = len(messages)
    sys_end = 0
    if keep_system:
        while sys_end < n and messages[sys_end].get("role") == "system":
            sys_end += 1

    # Find tail exchanges (keep_last_n) by walking backwards from the end.
    exchanges = 0
    tail_start = n
    for i in range(n - 1, sys_end - 1, -1):
        if messages[i].get("role") == "user":
            exchanges += 1
            if exchanges >= keep_last_n:
                tail_start = i
                break
    else:
        tail_start = sys_end  # not enough exchanges — keep everything after system

    # Walk the cut boundary forward to preserve tool_calls ↔ tool atomicity.
    # If the first message we'd drop is a ``tool``, extend the tail back
    # until we include the preceding ``assistant(tool_calls)``.
    while tail_start > sys_end and messages[tail_start - 1].get("role") == "tool" \
            and messages[tail_start - 1].get("tool_call_id"):
        # Walk back to find the matching assistant(tool_calls).
        tid = messages[tail_start - 1].get("tool_call_id")
        extended = tail_start - 1
        while extended > sys_end:
            extended -= 1
            role = messages[extended].get("role")
            if role == "assistant" and messages[extended].get("tool_calls"):
                # Check if any of its tool_calls match this tool_call_id.
                tcs = messages[extended].get("tool_calls") or []
                if any(tc.get("id") == tid for tc in tcs):
                    tail_start = extended
                    break
            elif role == "user":
                # We hit a user boundary without finding the pair — stop.
                break
        else:
            break  # safety: cannot find pair, stop extending

    # Build result: system + trimmed body + tail.
    budget_tail = estimate_tokens(messages[tail_start:])
    remaining = max_tokens - budget_tail
    system_tokens = estimate_tokens(messages[:sys_end])
    body_budget = remaining - system_tokens

    if body_budget < 0:
        # Can't even fit system + tail. Degrade: drop system, keep only tail.
        if estimate_tokens(messages[tail_start:]) > max_tokens:
            # Tail alone exceeds budget — last resort: trim from tail end too.
            result: list[dict[str, Any]] = []
            spent = 0
            orphan_tool_ids: set[str] = set()  # tool msgs in result whose assistant is not yet in
            for m in reversed(messages[tail_start:]):
                cost = estimate_message_tokens(m)
                tid = m.get("tool_call_id") or ""
                if spent + cost > max_tokens:
                    # Atomicity: if this is an assistant(tool_calls) whose
                    # tool msg is already in the result, include it anyway.
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        tcs = m.get("tool_calls") or []
                        if any(tc.get("id") in orphan_tool_ids for tc in tcs):
                            result.insert(0, m)
                            spent += cost
                    # Also: tool message whose assistant is not yet in result
                    # — skip it (will be pulled in by the assistant if it fits).
                    elif m.get("role") == "tool" and tid:
                        pass  # leave orphan_tool_ids entry; don't advance
                    break
                result.insert(0, m)
                spent += cost
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in (m.get("tool_calls") or []):
                        orphan_tool_ids.discard(tc.get("id") or "")
                elif m.get("role") == "tool" and tid:
                    orphan_tool_ids.add(tid)
            return result
        return list(messages[tail_start:])

    # Take body messages from index sys_end to tail_start, keep as many
    # from the front as fit.
    body: list[dict[str, Any]] = []
    spent = system_tokens
    for i in range(sys_end, tail_start):
        cost = estimate_message_tokens(messages[i])
        if spent + cost > max_tokens - budget_tail:
            break
        body.append(messages[i])
        spent += cost

    return list(messages[:sys_end]) + body + list(messages[tail_start:])
